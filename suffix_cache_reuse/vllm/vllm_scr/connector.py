# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SCRConnector: the worker-side half of Suffix Cache Reuse for vLLM 0.19.1.

A KV connector is used only as vLLM's supported channel from the scheduler to the workers:
the scheduler (vllm_scr.scheduler) attaches, for every request that jumped over a relocated
span in this step, the slot ids of the held rows (source), the slot ids of the span's new
positions (destination) and the position shift. Before the step's forward pass,
start_load_kv() copies K and V of every attention layer of the group from source to
destination slots and re-rotates the keys by the position shift (keys are stored after RoPE).
No KV is loaded from outside vLLM: get_num_new_matched_tokens() always returns 0.

    --kv-transfer-config '{"kv_connector":"SCRConnector",
                           "kv_connector_module_path":"vllm_scr.connector","kv_role":"kv_both"}'
"""
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole, SupportsHMA)


def _log(msg):
    print(f"[scr] {msg}", file=sys.stderr, flush=True)


@dataclass
class SCRMeta(KVConnectorMetadata):
    # attention: (kv cache group id, src slots, dst slots, new_pos - old_pos)
    ops: list = field(default_factory=list)
    # recurrent state (mamba/GDN groups): (group id, src block id, dst block id)
    ssm_restore: list = field(default_factory=list)   # before the forward (after preprocess_mamba)
    ssm_save: list = field(default_factory=list)      # after the forward

    def add(self, gid, src, dst, delta):
        self.ops.append((int(gid), np.asarray(src, np.int64), np.asarray(dst, np.int64),
                         np.asarray(delta, np.int64)))


def find_rotary(head_size=None):
    """The model's rotary embedding, from vLLM's get_rope() instance cache."""
    from vllm.model_executor.layers import rotary_embedding as re_mod
    cands = []
    for r in list(getattr(re_mod, "_ROPE_DICT", {}).values()):
        if head_size is not None and getattr(r, "head_size", None) != head_size:
            continue
        if all(r is not c for c in cands):
            cands.append(r)
    if not cands:
        return None
    sig = {(type(c).__name__, c.rotary_dim, float(c.base), bool(c.is_neox_style)) for c in cands}
    if len(sig) != 1:
        raise RuntimeError(f"vllm_scr: several distinct rotary embeddings {sig}; unsupported")
    return cands[0]


def rotary_inv_freq(rot):
    """Inverse frequencies the model uses (validated against its cos/sin cache)."""
    inv = rot._compute_inv_freq(rot.base).double().cpu()
    cache = rot.cos_sin_cache.detach().float().cpu()
    rd2 = rot.rotary_dim // 2
    pos = torch.tensor([0, 1, 7, 100, 1000, min(30000, cache.shape[0] - 1)], dtype=torch.float64)
    cos = torch.cos(pos[:, None] * inv[None, :]).float()
    err = (cache[pos.long(), :rd2] - cos).abs().max().item()
    if err > 2e-2:  # bf16 cache: ~4e-3
        raise RuntimeError(f"vllm_scr: inv_freq does not reproduce cos_sin_cache (err {err:.3g})")
    return inv, err


def rerotate_(k, delta, inv_freq, rotary_dim, neox):
    """In place: k[n, H, D] (post-RoPE keys) -> keys rotated by `delta` more positions."""
    ang = delta.double()[:, None] * inv_freq[None, :].to(delta.device)
    cos = torch.cos(ang).float()[:, None, :]  # [n, 1, rd/2]
    sin = torch.sin(ang).float()[:, None, :]
    kr = k[..., :rotary_dim].float()
    if neox:
        x1, x2 = kr[..., : rotary_dim // 2], kr[..., rotary_dim // 2:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    else:
        x1, x2 = kr[..., 0::2], kr[..., 1::2]
        out = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2)
    k[..., :rotary_dim] = out.to(k.dtype)
    return k


class SCRConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        self._no_rerot = os.environ.get("KVREUSE_NO_REROTATE", "0") == "1"
        self._verify = os.environ.get("KVREUSE_VERIFY", "0") == "1"
        self._layers = None      # gid -> [(K view, V view)]
        self._rot = None
        self.n_ops = 0
        self.n_tokens = 0
        self.n_restore = 0
        self.n_save = 0
        self._mamba = {}

    # ------------------------------------------------------------------ worker side
    def register_kv_caches(self, kv_caches):
        cfg = self._kv_cache_config
        from vllm.v1.kv_cache_interface import AttentionSpec
        from vllm.v1.kv_cache_interface import MambaSpec
        self._layers = {}
        self._mamba = {}
        for gid, grp in enumerate(cfg.kv_cache_groups):
            spec = grp.kv_cache_spec
            if isinstance(spec, MambaSpec):
                self._mamba[gid] = [list(kv_caches[name]) for name in grp.layer_names]
                continue
            if not isinstance(spec, AttentionSpec):
                continue
            views = []
            for name in grp.layer_names:
                t = kv_caches[name]
                if isinstance(t, (list, tuple)):
                    raise RuntimeError(f"vllm_scr: unexpected kv cache type for {name}")
                if t.dim() == 5 and t.shape[0] == 2:      # FlashAttention: (2, nb, bs, H, D)
                    kv, vv = t[0], t[1]
                elif t.dim() == 5 and t.shape[1] == 2:    # FlashInfer: (nb, 2, bs, H, D)
                    kv, vv = t[:, 0], t[:, 1]
                else:
                    raise RuntimeError(f"vllm_scr: unsupported kv cache shape {tuple(t.shape)}")
                views.append((kv, vv))
            self._layers[gid] = views
            head = spec.head_size
        any_view = next(iter(self._layers.values()))[0][0]
        self._kbs = int(any_view.shape[1])
        self._rot = find_rotary(head)
        if self._rot is None:
            raise RuntimeError("vllm_scr: rotary embedding not found")
        self._inv, err = rotary_inv_freq(self._rot)
        self._inv = self._inv.to(any_view.device)
        _log(f"worker: groups={ {g: len(v) for g, v in self._layers.items()} } "
             f"k_cache={tuple(any_view.shape)} {any_view.dtype} kernel_block={self._kbs} "
             f"rotary={type(self._rot).__name__} rotary_dim={self._rot.rotary_dim} "
             f"head={head} neox={self._rot.is_neox_style} base={self._rot.base} "
             f"inv_freq_check_err={err:.2e} no_rerotate={self._no_rerot}")
        for g, layers in self._mamba.items():
            _log(f"worker: mamba group {g}: {len(layers)} layers, state tensors "
                 f"{[(tuple(t.shape), str(t.dtype)) for t in layers[0]]}")

    @torch.inference_mode()
    def start_load_kv(self, forward_context, **kwargs):
        meta = self._get_connector_metadata()
        if not isinstance(meta, SCRMeta):
            return
        timing = os.environ.get("KVREUSE_TIMING") == "1" and (meta.ops or meta.ssm_restore)
        if timing:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        for gid, src, dst, delta in meta.ops:
            views = self._layers[gid]
            dev = views[0][0].device
            s = torch.from_numpy(src).to(dev, non_blocking=True)
            d = torch.from_numpy(dst).to(dev, non_blocking=True)
            dl = torch.from_numpy(delta).to(dev, non_blocking=True)
            sb, so = s // self._kbs, s % self._kbs
            db, do = d // self._kbs, d % self._kbs
            for kv, vv in views:
                k = kv[sb, so]
                if not self._no_rerot:
                    rerotate_(k, dl, self._inv, self._rot.rotary_dim, self._rot.is_neox_style)
                kv[db, do] = k
                vv[db, do] = vv[sb, so]
            if self._verify:
                self._verify_op(views, sb, so, db, do, dl)
            self.n_ops += 1
            self.n_tokens += len(src)
        for gid, src, dst in meta.ssm_restore:
            self._copy_state(gid, src, dst)
            self.n_restore += 1
        if timing:
            torch.cuda.synchronize()
            _log(f"timing start_load_kv ops={len(meta.ops)} tokens={sum(len(o[1]) for o in meta.ops)} "
                 f"restore={len(meta.ssm_restore)} ms={1000 * (time.perf_counter() - t0):.2f}")

    def _verify_op(self, views, sb, so, db, do, dl):
        """KVREUSE_VERIFY=1 (debug): recompute the expected destination rows independently from
        the (unchanged) source rows on the first and last layer of the group and compare with
        what landed in the cache; both errors must be exactly 0. Works per rank under TP."""
        errs = []
        for li in sorted({0, len(views) - 1}):
            kv, vv = views[li]
            exp = kv[sb, so].clone()
            if not self._no_rerot:
                rerotate_(exp, dl, self._inv, self._rot.rotary_dim, self._rot.is_neox_style)
            ek = (kv[db, do].float() - exp.float()).abs().max().item()
            ev = (vv[db, do].float() - vv[sb, so].float()).abs().max().item()
            errs.append((li, ek, ev))
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = 0
        bad = any(ek or ev for _, ek, ev in errs)
        _log(f"verify rank={rank} tokens={len(sb)} {'FAIL' if bad else 'ok'} "
             + " ".join(f"layer{li}:k={ek:.3g},v={ev:.3g}" for li, ek, ev in errs))

    def _copy_state(self, gid, src, dst):
        for states in self._mamba[gid]:
            for t in states:
                t[dst].copy_(t[src])

    def wait_for_layer_load(self, layer_name):
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    @torch.inference_mode()
    def wait_for_save(self):
        meta = self._get_connector_metadata() if self.has_connector_metadata() else None
        if not isinstance(meta, SCRMeta):
            return
        for gid, src, dst in meta.ssm_save:
            self._copy_state(gid, src, dst)
            self.n_save += 1

    # ------------------------------------------------------------------ scheduler side
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return

    def build_connector_meta(self, scheduler_output):
        return SCRMeta()

    def request_finished(self, request, block_ids):
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        return False, None
