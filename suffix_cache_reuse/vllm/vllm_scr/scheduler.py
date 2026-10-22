# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Suffix Cache Reuse scheduler for vLLM 0.19.1 (scheduler side).

Loaded with `--scheduler-cls vllm_scr.scheduler.SCRAsyncScheduler` (or SCRScheduler when async
scheduling is off). With KVREUSE_ENABLED=0 the class schedules exactly like stock vLLM and only
writes the per-request accounting line; with KVREUSE_ENABLED=1 it also needs the SCRConnector
(`--kv-transfer-config`), which carries the copy/re-rotate ops to the worker.

Per request (SCR on):
  add_request      match the prompt to a tracked conversation (same rules as SGLang), diff it
                   against the prompt whose KV the conversation holds, plan up to K relocations.
  schedule         phase 1: prefill is capped at |A|+|B'| (first block's start), reusing the
                   ordinary prefix-cache hit for A. Before the next schedule() pass, the request
                   "jumps": num_computed_tokens += |C| (pre-hook), so vLLM allocates slots for C
                   and the rest in the same allocate_slots call and continues the prefill with
                   the next stretch; the next block's start caps it again, and so on.
  connector meta   for every request scheduled in a step after its jump, the slot ids of the
                   held rows (source) and of C's new positions (destination) go to the worker
                   connector, which copies K/V and re-rotates the keys before the forward pass.
  _free_request    the finished request's attention blocks are pinned (ref_cnt + 1) as the
                   conversation's "side buffer" for its next turn; the previous pin is released.
Fallback: a plan whose source is gone, or a preempted request, continues as normal prefill.

vLLM's scheduling loop is not modified: the per-request prefill cap is applied through the
existing `_mamba_block_aligned_split` hook (see "Integration seam" below).
"""
import collections
import inspect
import os
import sys
import time

import numpy as np

from vllm.logger import init_logger
from vllm.v1.core.sched import scheduler as _sched_mod
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

from vllm_scr import config as C
from vllm_scr import planner

logger = init_logger("vllm_scr")


def _log(msg):
    print(f"[scr] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------
# Integration seam (no source patching)
# --------------------------------------------------------------------------------------------
# SCR needs exactly one thing from vLLM's scheduling loop: a per-request upper bound on the
# number of prefill tokens scheduled in a step (so a prefill chunk can stop at |A|+|B'|, the
# start of a relocated span). In vLLM 0.19.x both branches of Scheduler.schedule (RUNNING and
# WAITING) already call `self._mamba_block_aligned_split(request, num_new_tokens, ...)` right
# after the token-budget / encoder clamps and right before allocate_slots, guarded by
# `self.need_mamba_block_aligned_split`. That method is the seam: the SCR scheduler sets the
# flag for every model and overrides the method to apply the cap first, then the upstream
# alignment only where upstream would have applied it (hybrid models in mamba_cache_mode=align).
# The seam is checked at import time (check_seam), so a vLLM that moved or renamed the call
# fails loudly here instead of silently serving without the cap.
SUPPORTED_VLLM = ("0.19.1",)


def check_seam():
    import vllm
    if vllm.__version__ not in SUPPORTED_VLLM and not C.get("KVREUSE_ALLOW_UNTESTED_VLLM", bool):
        raise RuntimeError(
            f"vllm_scr supports vLLM {', '.join(SUPPORTED_VLLM)}; found {vllm.__version__}. "
            "Set KVREUSE_ALLOW_UNTESTED_VLLM=1 to try anyway (the seam check below still runs).")
    src = inspect.getsource(Scheduler.schedule)
    n_call = src.count("self._mamba_block_aligned_split(")
    n_gate = src.count("if self.need_mamba_block_aligned_split:")
    sig = list(inspect.signature(Scheduler._mamba_block_aligned_split).parameters)
    want = ["self", "request", "num_new_tokens", "num_new_local_computed_tokens",
            "num_external_computed_tokens"]
    if n_call != 2 or n_gate != 2 or sig != want:
        raise RuntimeError(
            "vllm_scr: Scheduler.schedule no longer calls _mamba_block_aligned_split once per "
            f"branch (calls={n_call}, gates={n_gate}, signature={sig}); the SCR prefill cap would "
            "not be applied. This vLLM version is not supported.")


check_seam()


class SCRMixin:
    """See module docstring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scr_on = C.get("KVREUSE_ENABLED", bool)
        if self._scr_on:
            from vllm_scr.connector import SCRConnector
            if not isinstance(self.connector, SCRConnector):
                raise RuntimeError("KVREUSE_ENABLED=1 needs --kv-transfer-config with SCRConnector")
        self._scr_sessions = collections.OrderedDict()
        self._scr_sid = 0
        self._scr_stats = collections.Counter()
        self._scr_trace = collections.deque(maxlen=4096)  # executed spans (rid, pos0, row0, n)
        self._scr_max_blocks = C.get("KVREUSE_MAX_BLOCKS", int)
        self._scr_cache_reloc = C.get("KVREUSE_CACHE_RELOCATED", bool)
        self._scr_side_sessions = C.get("KVREUSE_SIDE_SESSIONS", int)
        self._scr_side_tokens = C.get("KVREUSE_SIDE_TOKENS", int)
        self._scr_max_sessions = C.get("KVREUSE_MAX_SESSIONS", int)
        nblk = int(self.cache_config.num_gpu_blocks or 0)
        self._scr_budget_blocks = int(nblk * C.get("KVREUSE_BUDGET_FRAC", float))
        self._scr_pinned_blocks = 0
        groups = self.kv_cache_config.kv_cache_groups
        self._scr_attn_gids = [g for g, grp in enumerate(groups)
                               if isinstance(grp.kv_cache_spec, AttentionSpec)]
        self._scr_mamba_gids = [g for g, grp in enumerate(groups)
                                if isinstance(grp.kv_cache_spec, MambaSpec)]
        self._scr_bs = {g: groups[g].kv_cache_spec.block_size for g in self._scr_attn_gids}
        self._scr_mbs = {g: groups[g].kv_cache_spec.block_size for g in self._scr_mamba_gids}
        if str(C.get("KVREUSE_THINK_IDS")) == "auto":
            try:
                from transformers import AutoTokenizer
                mc = self.vllm_config.model_config
                tok = AutoTokenizer.from_pretrained(mc.tokenizer, trust_remote_code=mc.trust_remote_code)
                ids = [tok.convert_tokens_to_ids(t) for t in ("<think>", "</think>")]
                ids = [i for i in ids if isinstance(i, int) and i >= 0 and i != tok.unk_token_id]
                os.environ["KVREUSE_THINK_IDS"] = ",".join(map(str, ids))
            except Exception as e:
                _log(f"think ids not resolved ({e}); cause label disabled")
                os.environ["KVREUSE_THINK_IDS"] = ""
        self._scr_t_last = time.time()
        self._scr_audit_every = C.get("KVREUSE_AUDIT", int)   # 0: off; N: refcount audit every N steps
        self._scr_steps = 0
        self._scr_deferred = []          # blocks kept alive for one extra step (see the jump)
        self._scr_mamba_mode = getattr(self.cache_config, "mamba_cache_mode", "none")
        self._scr_ssm_mode = C.get("KVREUSE_SSM_MODE")
        self._scr_ssm_save = C.get("KVREUSE_SSM_SAVE")          # end | boundary
        if self._scr_ssm_save not in ("end", "boundary"):
            raise RuntimeError(f"KVREUSE_SSM_SAVE={self._scr_ssm_save} not supported (end|boundary)")
        if self._scr_on and self._scr_mamba_gids:
            if self._scr_mamba_mode != "align":
                _log(f"hybrid model with mamba_cache_mode={self._scr_mamba_mode}: only 'align' is "
                     "implemented; SCR disabled, accounting only")
                self._scr_on = False
            elif self._scr_ssm_mode not in ("fork", "none"):
                raise RuntimeError(f"KVREUSE_SSM_MODE={self._scr_ssm_mode} not supported (fork|none)")
        # The upstream alignment applies only where upstream would apply it.
        self._scr_upstream_align = bool(self.need_mamba_block_aligned_split)
        why = self._scr_unsupported()
        if self._scr_on and why:
            _log(f"SCR disabled: {why}; accounting only")
            self._scr_on = False
        if self._scr_on:
            self.need_mamba_block_aligned_split = True   # route both branches through the seam
            self._scr_install_kvm_hooks()
        if self._scr_audit_every:
            self._scr_count_evictions()
        _log(f"init: on={self._scr_on} K={self._scr_max_blocks} attn_groups={self._scr_attn_gids} "
             f"block_size={self._scr_bs} num_gpu_blocks={nblk} budget_blocks={self._scr_budget_blocks} "
             f"side_sessions={self._scr_side_sessions} side_tokens={self._scr_side_tokens} "
             f"cache_relocated={int(self._scr_cache_reloc)} async={isinstance(self, AsyncScheduler)} "
             f"mamba_groups={self._scr_mamba_gids} mamba_block_size={self._scr_mbs} "
             f"mamba_cache_mode={self._scr_mamba_mode} ssm_mode={self._scr_ssm_mode} ssm_save={self._scr_ssm_save} "
             f"think_ids={os.environ.get('KVREUSE_THINK_IDS')}")

    def _scr_unsupported(self):
        """Configurations SCR has not been built for: serve them unchanged."""
        from vllm.v1.kv_cache_interface import (ChunkedLocalAttentionSpec, FullAttentionSpec,
                                                MLAAttentionSpec, SinkFullAttentionSpec,
                                                SlidingWindowSpec)
        vc = self.vllm_config
        if vc.speculative_config is not None:
            return "speculative decoding"
        if vc.parallel_config.pipeline_parallel_size > 1:
            return "pipeline parallelism"
        for g, grp in enumerate(self.kv_cache_config.kv_cache_groups):
            spec = grp.kv_cache_spec
            if isinstance(spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec, MLAAttentionSpec,
                                 SinkFullAttentionSpec)):
                return f"attention group {g} is {type(spec).__name__}"
            if isinstance(spec, AttentionSpec) and not isinstance(spec, FullAttentionSpec):
                return f"attention group {g} is {type(spec).__name__}"
        if self._scr_attn_gids == []:
            return "no attention layers"
        return None

    def _scr_count_evictions(self):
        """Debug (KVREUSE_AUDIT>0): count prefix-cache evictions, split by who allocated."""
        pool = self.kv_cache_manager.block_pool
        orig = pool._maybe_evict_cached_block
        self._scr_evict_src = "vllm"

        def counted(block):
            evicted = orig(block)
            if evicted:
                self._scr_stats[f"evict_{self._scr_evict_src}"] += 1
            return evicted
        pool._maybe_evict_cached_block = counted

    # ---------------------------------------------------------------- KV manager hooks
    def _scr_install_kvm_hooks(self):
        kvm = self.kv_cache_manager
        orig_alloc = kvm.allocate_slots

        def allocate_slots(request, *a, **kw):
            out = orig_alloc(request, *a, **kw)
            while out is None and self._scr_release_one(exclude=request):
                self._scr_stats["reclaim_retry"] += 1
                out = orig_alloc(request, *a, **kw)
            return out

        kvm.allocate_slots = allocate_slots
        for g in self._scr_mamba_gids:
            mgr = kvm.coordinator.single_type_managers[g]
            null = kvm.block_pool.null_block

            def mcache(request, num_tokens, _mgr=mgr, _orig=mgr.cache_blocks, _g=g, _null=null):
                # A recurrent-state block whose state is not the state at its own end (a
                # prefill cut at |A|+|B'|) must never be hashed into the prefix cache.
                dirty = (getattr(request, "_scr_dirty", None) or {}).get(_g)
                blocks = _mgr.req_to_blocks.get(request.request_id)
                if not dirty or blocks is None:
                    return _orig(request, num_tokens)
                saved = {}
                for i in dirty:
                    if i < len(blocks) and not blocks[i].is_null and blocks[i].block_hash is None:
                        saved[i] = blocks[i]
                        blocks[i] = _null
                try:
                    return _orig(request, num_tokens)
                finally:
                    for i, b in saved.items():
                        if blocks[i] is _null:
                            blocks[i] = b
                    if saved:
                        self._scr_stats["ssm_dirty_skipped"] += len(saved)

            mgr.cache_blocks = mcache
        if not self._scr_cache_reloc:
            coord = kvm.coordinator
            orig_cache = coord.cache_blocks

            def cache_blocks(request, num_tokens):
                cap = getattr(request, "_scr_cache_cap", None)
                if cap is not None:
                    num_tokens = min(num_tokens, cap)
                return orig_cache(request, num_tokens)

            coord.cache_blocks = cache_blocks

    # ---------------------------------------------------------------- side buffer (pinned blocks)
    def _scr_snap_release(self, snap):
        if snap is None or snap.get("freed"):
            return
        snap["dropped"] = True
        if snap["users"] > 0:
            return
        pool = self.kv_cache_manager.block_pool
        for blks in snap["blocks"].values():
            pool.free_blocks(reversed(blks))
        if snap.get("ssm"):
            pool.free_blocks(list(snap["ssm"].values()))
        self._scr_pinned_blocks -= snap["nblocks"]
        snap["freed"] = True

    def _scr_drop_snap(self, sess):
        snap = sess.get("snap")
        sess["snap"] = None
        self._scr_snap_release(snap)

    def _scr_release_one(self, exclude=None):
        """Release the least recently used held conversation that is not in flight."""
        for sid, sess in self._scr_sessions.items():
            if sess.get("snap") is not None and sess.get("inflight", 0) == 0:
                self._scr_drop_snap(sess)
                self._scr_stats["side_released"] += 1
                return True
        return False

    def _scr_enforce_limits(self):
        held = [s for s in self._scr_sessions.values() if s.get("snap") is not None]
        while (len(held) > self._scr_side_sessions
               or self._scr_pinned_blocks > self._scr_budget_blocks):
            if not self._scr_release_one():
                break
            held = [s for s in self._scr_sessions.values() if s.get("snap") is not None]
        while len(self._scr_sessions) > self._scr_max_sessions:
            victim = next((sid for sid, s in self._scr_sessions.items()
                           if s.get("inflight", 0) == 0), None)
            if victim is None:
                break
            self._scr_drop_snap(self._scr_sessions.pop(victim))

    def _scr_take_ssm(self, request):
        st = getattr(request, "_scr", None)
        ssm = st.pop("ssm_snap", None) if st else None
        return ssm

    def _scr_hold_boundary_state(self, request):
        """KVREUSE_SSM_SAVE=boundary: instead of copying the state at the end of the prompt into
        new blocks (which can evict prefix-cache entries), hold a reference on the state vLLM
        already caches at the prompt's last block boundary floor(P / bs) * bs."""
        bs = self.cache_config.block_size
        b = request.num_prompt_tokens // bs
        if b < 1 or len(request.block_hashes) < b:
            return None
        pool = self.kv_cache_manager.block_pool
        blks = pool.get_cached_block(request.block_hashes[b - 1], self._scr_mamba_gids)
        if not blks:
            self._scr_stats["ssm_boundary_miss"] += 1
            return None
        pool.touch(blks)
        self._scr_stats["ssm_boundary_hold"] += 1
        return dict(zip(self._scr_mamba_gids, blks))

    def _scr_free_ssm(self, request):
        ssm = self._scr_take_ssm(request)
        if ssm:
            self.kv_cache_manager.block_pool.free_blocks(list(ssm.values()))

    def _scr_pin(self, request, sess):
        if not self._scr_on:
            return
        P = request.num_prompt_tokens
        committed = request.num_computed_tokens
        if committed < P:  # aborted before its prompt was computed: keep the previous pin
            self._scr_stats["snap_skip_incomplete"] += 1
            return
        if committed > self._scr_side_tokens:
            self._scr_drop_snap(sess)
            self._scr_stats["snap_skip_long"] += 1
            return
        coord = self.kv_cache_manager.coordinator
        blocks, nb = {}, 0
        for g in self._scr_attn_gids:
            bl = list(coord.single_type_managers[g].req_to_blocks.get(request.request_id) or [])
            need = (P + self._scr_bs[g] - 1) // self._scr_bs[g]
            bl = bl[:need]
            if len(bl) < need or any(b.is_null for b in bl):
                self._scr_drop_snap(sess)
                self._scr_stats["snap_skip_blocks"] += 1
                return
            for b in bl:
                b.ref_cnt += 1
            blocks[g] = bl
            nb += len(bl)
        self._scr_drop_snap(sess)
        ssm = self._scr_take_ssm(request)
        if not ssm and self._scr_mamba_gids and self._scr_ssm_save == "boundary":
            ssm = self._scr_hold_boundary_state(request)
        if ssm:
            nb += len(ssm)
        sess["snap"] = {"blocks": blocks, "ids": list(request.prompt_token_ids), "n_held": P,
                        "users": 0, "nblocks": nb, "rid": request.request_id, "ssm": ssm}
        self._scr_pinned_blocks += nb
        self._scr_stats["snap"] += 1
        self._scr_enforce_limits()

    # ---------------------------------------------------------------- planning
    def add_request(self, request):
        is_new = request.request_id not in self.requests
        super().add_request(request)
        if not is_new:
            return
        try:
            self._scr_plan(request)
        except Exception as e:  # never break serving
            self._scr_stats["plan_error"] += 1
            _log(f"plan error ({type(e).__name__}: {e}); normal prefill")
            request._scr = None

    def _scr_plan(self, request):
        acct = {"reloc": 0, "blocks": 0, "fwd": 0, "sid": "-", "prev": 0, "planned": 0,
                "reason": "", "fallback": ""}
        request._scr_acct = acct
        request._scr = None
        if request.prompt_token_ids is None:
            return
        new_ids = list(request.prompt_token_ids)
        key = request.cache_salt
        if key is not None and str(key).startswith("kvreuse:off"):
            acct["reason"] = "switch_off"   # per-request switch, as in SGLang: plain serving
            return
        t0 = time.perf_counter()
        if self._scr_on:
            sid = planner.find_session(self._scr_sessions, new_ids, key, self._scr_stats)
        else:
            sid = planner.find_session(
                {k: dict(v, snap=True) for k, v in self._scr_sessions.items()}, new_ids, key)
        if sid is None:
            self._scr_sid += 1
            sid = f"s{self._scr_sid:06d}"
            self._scr_sessions[sid] = {"sid": sid, "key": key, "token_ids": [], "snap": None,
                                       "inflight": 0}
        else:
            self._scr_sessions.move_to_end(sid)
        sess = self._scr_sessions[sid]
        acct["sid"] = sid
        # Accounting only: the conversation's previous prompt by content, also when SCR could not
        # attach the request to a held conversation (prev/lcp feed the edited/append classes).
        ref = sess if sess["token_ids"] else None
        if ref is None and self._scr_on:
            rid2 = planner.find_session(
                {k: dict(v, snap=True) for k, v in self._scr_sessions.items() if k != sid}, new_ids, key)
            ref = self._scr_sessions.get(rid2) if rid2 else None
        old = ref["token_ids"] if ref else []
        acct["prev"] = len(old)
        if old:  # longest common prefix with the previous prompt (content, not cache)
            m = min(len(old), len(new_ids))
            neq = np.flatnonzero(np.asarray(old[:m], np.int64) != np.asarray(new_ids[:m], np.int64))
            acct["lcp"] = int(neq[0]) if neq.size else m
        sess["token_ids"] = new_ids
        sess["inflight"] = sess.get("inflight", 0) + 1
        st = {"sess": sess, "blocks": [], "bi": 0, "snap": None, "pending": None}
        request._scr = st
        if not self._scr_on:
            self._scr_enforce_limits()
            return
        snap = sess.get("snap")
        if snap is not None and not request.skip_reading_prefix_cache:
            blocks, info = planner.plan_relocation(snap["ids"], new_ids, snap["n_held"],
                                                   max_blocks=self._scr_max_blocks)
            acct["reason"] = info.get("reason", "")
            acct["cause"] = info.get("cause", "-")
            if blocks and self._scr_mamba_gids and self._scr_ssm_mode == "fork" and not snap.get("ssm"):
                acct["reason"] = "ssm_gated"   # no saved recurrent state for this conversation
                self._scr_stats["ssm_gated"] += 1
                blocks = []
            if blocks and self._scr_mamba_gids:
                blocks = self._scr_hybrid_trim(blocks, len(new_ids))
                if not blocks:
                    acct["reason"] = "hybrid_trim"
            if blocks:
                st["blocks"], st["snap"] = blocks, snap
                snap["users"] += 1
                acct["planned"] = sum(len(b["rows"]) for b in blocks)
                self._scr_stats["plans"] += 1
        else:
            acct["reason"] = "cold"
        acct["plan_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        self._scr_enforce_limits()
        if C.get("KVREUSE_DEBUG", bool):
            _log(f"plan rid={request.request_id} sid={sid} n_new={len(new_ids)} "
                 f"n_old={acct['prev']} blocks={[(b['bprime_end'], len(b['rows'])) for b in st['blocks']]} "
                 f"reason={acct['reason']} ms={acct['plan_ms']}")

    def _scr_hybrid_trim(self, blocks, P):
        """Hybrid models: keep the last block boundary of the prompt out of relocated spans.

        vLLM caches a recurrent state only where a prefill chunk ends on a block boundary, and a
        relocated span is never run through the linear-attention layers, so a boundary inside a
        span gets no state. If the prompt's last boundary b = floor(P / bs) * bs fell inside a
        span, the next turn of the conversation (usually an append) would find no cached state
        near its prefix and re-prefill from far back. Spans are therefore cut to end before b
        (at most one block of extra prefill on this turn); the chunk that crosses b is then cut
        at b and its state is cached.
        """
        bs = self.cache_config.block_size
        b = (P // bs) * bs
        mt = C.get("KVREUSE_V6_MIN_TAIL", int)
        out = []
        for blk in blocks:
            if blk["bprime_end"] >= b - 1:
                self._scr_stats["hybrid_trim_dropped"] += 1
                continue
            n_max = b - 1 - blk["bprime_end"]
            if len(blk["rows"]) > n_max:
                if n_max < mt:
                    self._scr_stats["hybrid_trim_dropped"] += 1
                    continue
                self._scr_stats["hybrid_trim_rows"] += len(blk["rows"]) - n_max
                blk = dict(blk, rows=blk["rows"][:n_max])
            out.append(blk)
        return out

    def _scr_abort(self, request, why):
        st = getattr(request, "_scr", None)
        if not st or not st["blocks"] and st["pending"] is None:
            return
        if st["bi"] < len(st["blocks"]) or st["pending"] is not None:
            self._scr_stats["abort_" + why] += 1
            request._scr_acct["fallback"] = why
        st["blocks"], st["pending"] = [], None

    # ---------------------------------------------------------------- the per-request cap
    def _scr_cap(self, request, num_new_tokens, num_computed):
        st = getattr(request, "_scr", None)
        if not st or st["bi"] >= len(st["blocks"]) or num_new_tokens <= 0:
            return num_new_tokens
        if st["snap"] is None or st["snap"].get("freed"):
            self._scr_abort(request, "source_gone")
            return num_new_tokens
        while st["bi"] < len(st["blocks"]):
            blk = st["blocks"][st["bi"]]
            if num_computed < blk["bprime_end"]:
                n = max(1, min(num_new_tokens, blk["bprime_end"] - num_computed))
                st["cap_end"] = blk["bprime_end"] if num_computed + n == blk["bprime_end"] else None
                return n
            # The prefix-cache hit reaches the block start (aligned A with B' empty, or a hit
            # into C): prefill one token past the hit and relocate the rest of the span.
            d = num_computed - blk["bprime_end"] + 1
            self._scr_stats["hit_overlap"] += 1
            blk["rows"] = blk["rows"][d:]
            blk["bprime_end"] += d
            if not blk["rows"]:
                st["bi"] += 1
        return num_new_tokens

    def _mamba_block_aligned_split(self, request, num_new_tokens,
                                   num_new_local_computed_tokens=0, num_external_computed_tokens=0):
        """The seam: SCR prefill cap, then (hybrid models only) block alignment."""
        st = getattr(request, "_scr", None) if self._scr_on else None
        nc = (request.num_computed_tokens + num_new_local_computed_tokens
              + num_external_computed_tokens)
        if st:
            num_new_tokens = self._scr_cap(request, num_new_tokens, nc)
        if not self._scr_upstream_align:
            return num_new_tokens
        if st and (st["blocks"] or st.get("jumped") or st.get("cut")):
            end = nc + num_new_tokens
            bs = self.cache_config.block_size
            if st.get("cap_end") is not None and end == st["cap_end"]:
                last_b = (end // bs) * bs
                if nc < last_b < end and end - nc > 0:
                    return last_b - nc     # first stop at the last boundary before the cut (cached)
                # The cut at |A|+|B'|: the state block written by this chunk holds the state at
                # a non-boundary position, so it must never enter the prefix cache, whatever
                # happens next (jump, or abort and normal prefill).
                dirty = request.__dict__.setdefault("_scr_dirty", {})
                for g in self._scr_mamba_gids:
                    dirty.setdefault(g, set()).add((end - 1) // self._scr_mbs[g])
                st["cut"] = True
                return num_new_tokens
            bs = self.cache_config.block_size
            if nc % bs and nc < max(request.num_prompt_tokens, request.num_tokens - 1):
                # Prefill from a non-aligned position (after a jump or an aborted cut): stop at
                # the next block boundary so the state there is cached; from the aligned position
                # vLLM's own split takes over (aligned chunks, and the last block boundary of the
                # prompt is always cut so its state is cached for the next turn).
                nxt = (nc // bs + 1) * bs
                return nxt - nc if end > nxt else num_new_tokens
        return super()._mamba_block_aligned_split(request, num_new_tokens,
                                                  num_new_local_computed_tokens,
                                                  num_external_computed_tokens)

    # ---------------------------------------------------------------- the jump
    def schedule(self):
        out = self._scr_schedule()
        n = self._scr_audit_every
        if n:
            self._scr_steps += 1
            if self._scr_steps % n == 0:
                self._scr_audit()
        return out

    def _scr_schedule(self):
        if not self._scr_on:
            return super().schedule()
        if self._scr_deferred:
            self.kv_cache_manager.block_pool.free_blocks(self._scr_deferred)
            self._scr_deferred = []
        for request in self.running:
            st = getattr(request, "_scr", None)
            if not st or st["pending"] is not None or st["bi"] >= len(st["blocks"]):
                continue
            blk = st["blocks"][st["bi"]]
            if request.num_computed_tokens != blk["bprime_end"]:
                continue
            snap = st["snap"]
            n = len(blk["rows"])
            if snap is None or snap.get("freed"):
                self._scr_abort(request, "source_gone")
                continue
            if request.num_computed_tokens + n >= request.num_prompt_tokens:
                self._scr_abort(request, "short")
                continue
            if not self._scr_cache_reloc:
                cap = getattr(request, "_scr_cache_cap", None)
                request._scr_cache_cap = blk["bprime_end"] if cap is None else min(cap, blk["bprime_end"])
            st["pending"] = {"rows": blk["rows"], "pos0": blk["bprime_end"]}
            if self._scr_mamba_gids:
                # The state block at the cut holds the state after A+B' at a non-boundary
                # position: never cache it, and keep it alive one more step because vLLM's
                # preprocess_mamba still reads it in the next forward.
                coord = self.kv_cache_manager.coordinator
                dirty = request.__dict__.setdefault("_scr_dirty", {})
                for g in self._scr_mamba_gids:
                    idx = (request.num_computed_tokens - 1) // self._scr_mbs[g]
                    bl = coord.single_type_managers[g].req_to_blocks.get(request.request_id) or []
                    dirty.setdefault(g, set()).add(idx)
                    if idx < len(bl) and not bl[idx].is_null:
                        bl[idx].ref_cnt += 1
                        self._scr_deferred.append(bl[idx])
                if st["bi"] == 0 and self._scr_ssm_mode == "fork":
                    st["restore"] = True
            st["jumped"] = True
            self._scr_trace.append((request.request_id, blk["bprime_end"], blk["rows"][0], n))
            st["bi"] += 1
            request.num_computed_tokens += n
            acct = request._scr_acct
            acct["reloc"] += n
            acct["blocks"] += 1
            self._scr_stats["spliced_blocks"] += 1
            self._scr_stats["spliced_tokens"] += n
            _log(f"splice OK: rid={request.request_id} sid={acct['sid']} A+B'={blk['bprime_end']} "
                 f"C'={n} live={request.num_prompt_tokens} blocks={st['bi']}/{len(st['blocks'])} "
                 f"rot_delta={blk['bprime_end'] - blk['rows'][0]}")
        return super().schedule()

    def _build_kv_connector_meta(self, connector, scheduler_output):
        meta = super()._build_kv_connector_meta(connector, scheduler_output)
        if not self._scr_on:
            return meta
        coord = self.kv_cache_manager.coordinator
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            st = getattr(request, "_scr", None) if request is not None else None
            if not st or st["pending"] is None:
                continue
            pend, snap = st["pending"], st["snap"]
            st["pending"] = None
            rows = np.asarray(pend["rows"], dtype=np.int64)
            newpos = np.arange(pend["pos0"], pend["pos0"] + len(rows), dtype=np.int64)
            for g in self._scr_attn_gids:
                bs = self._scr_bs[g]
                src_ids = np.asarray([b.block_id for b in snap["blocks"][g]], dtype=np.int64)
                dst_blocks = coord.single_type_managers[g].req_to_blocks[req_id]
                dst_ids = np.asarray([b.block_id for b in dst_blocks], dtype=np.int64)
                src = src_ids[rows // bs] * bs + rows % bs
                dst = dst_ids[newpos // bs] * bs + newpos % bs
                meta.add(g, src, dst, newpos - rows)
            if st.pop("restore", False):
                ssm = snap.get("ssm") or {}
                n_sched = scheduler_output.num_scheduled_tokens[req_id]
                for g in self._scr_mamba_gids:
                    idx = -(-(request.num_computed_tokens + n_sched) // self._scr_mbs[g]) - 1
                    dst_blk = coord.single_type_managers[g].req_to_blocks[req_id][idx]
                    meta.ssm_restore.append((g, ssm[g].block_id, dst_blk.block_id))
                self._scr_stats["ssm_restore"] += 1
        if self._scr_mamba_gids and self._scr_ssm_mode == "fork" and self._scr_ssm_save == "end":
            self._scr_emit_ssm_saves(meta, scheduler_output)
        return meta

    def _scr_emit_ssm_saves(self, meta, scheduler_output):
        """Snapshot the recurrent state at the end of each prompt (SGLang: prefill-end save)."""
        coord = self.kv_cache_manager.coordinator
        pool = self.kv_cache_manager.block_pool
        for req_id, n in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            st = getattr(request, "_scr", None) if request is not None else None
            if not st:
                continue
            nc, P = request.num_computed_tokens, request.num_prompt_tokens
            if not (nc < P == nc + n):
                continue
            self._scr_free_ssm(request)
            if pool.get_num_free_blocks() < len(self._scr_mamba_gids) + 1:
                self._scr_stats["ssm_save_nospace"] += 1
                continue

            snap = {}
            for g in self._scr_mamba_gids:
                idx = -(-(nc + n) // self._scr_mbs[g]) - 1
                src = coord.single_type_managers[g].req_to_blocks[req_id][idx]
                self._scr_evict_src = "ssm_save"
                try:
                    dst = pool.get_new_blocks(1)[0]
                finally:
                    self._scr_evict_src = "vllm"
                snap[g] = dst
                meta.ssm_save.append((g, src.block_id, dst.block_id))
            st["ssm_snap"] = snap
            self._scr_stats["ssm_save"] += 1

    def _preempt_request(self, request, timestamp):
        self._scr_abort(request, "preempted")
        self._scr_free_ssm(request)
        if getattr(request, "_scr_dirty", None):
            request._scr_dirty = None
        if getattr(request, "_scr_cache_cap", None) is not None:
            request._scr_cache_cap = None
        acct = getattr(request, "_scr_acct", None)
        if acct is not None:
            acct["preempt"] = acct.get("preempt", 0) + 1
        return super()._preempt_request(request, timestamp)

    # ---------------------------------------------------------------- audit
    def _scr_expected_refs(self):
        """Block id -> references the scheduler and SCR hold (requests, pins, saves, deferred)."""
        exp = collections.Counter()
        coord = self.kv_cache_manager.coordinator
        for mgr in coord.single_type_managers:
            for blocks in mgr.req_to_blocks.values():
                for b in blocks:
                    if not b.is_null:
                        exp[b.block_id] += 1
        snaps = {id(s["snap"]): s["snap"] for s in self._scr_sessions.values() if s.get("snap")}
        for r in self.requests.values():
            st = getattr(r, "_scr", None)
            if st and st.get("snap") is not None:
                snaps[id(st["snap"])] = st["snap"]
            for b in ((st or {}).get("ssm_snap") or {}).values():
                exp[b.block_id] += 1
        for sn in snaps.values():
            if sn.get("freed"):
                continue
            for bl in sn["blocks"].values():
                for b in bl:
                    exp[b.block_id] += 1
            for b in (sn.get("ssm") or {}).values():
                exp[b.block_id] += 1
        for b in self._scr_deferred:
            exp[b.block_id] += 1
        return exp

    def _scr_audit(self):
        """Every live block's ref_cnt must equal the references accounted for above, every
        block with ref_cnt 0 must be in the free queue, and no free block may be referenced."""
        pool = self.kv_cache_manager.block_pool
        exp = self._scr_expected_refs()
        bad = []
        n_zero = 0
        for b in pool.blocks:
            if b.is_null:
                continue
            if b.ref_cnt != exp.get(b.block_id, 0):
                bad.append((b.block_id, b.ref_cnt, exp.get(b.block_id, 0)))
            n_zero += b.ref_cnt == 0
        free = pool.get_num_free_blocks()
        ok = not bad and free == n_zero
        self._scr_stats["audit_ok" if ok else "audit_fail"] += 1
        if not ok:
            _log(f"audit FAIL: {len(bad)} refcount mismatches (block, actual, expected) {bad[:8]} "
                 f"free_queue={free} zero_ref_blocks={n_zero}")
            if C.get("KVREUSE_AUDIT_STRICT", bool):
                raise AssertionError(f"vllm_scr audit failed: {bad[:8]} free={free} zero={n_zero}")
        elif not self.requests:
            _log(f"audit ok (idle): free_blocks={free} pinned_blocks={self._scr_pinned_blocks} "
                 f"held_conversations={sum(1 for s in self._scr_sessions.values() if s.get('snap'))}")
        return ok, bad

    def scr_release_all(self):
        """Drop every held conversation (side buffer) and deferred block; used by tests/shutdown."""
        for sess in self._scr_sessions.values():
            self._scr_drop_snap(sess)
        if self._scr_deferred:
            self.kv_cache_manager.block_pool.free_blocks(self._scr_deferred)
            self._scr_deferred = []

    # ---------------------------------------------------------------- accounting
    def _update_after_schedule(self, scheduler_output):
        for req_id, n in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[req_id]
            acct = getattr(request, "_scr_acct", None)
            if acct is None:
                continue
            nc, P = request.num_computed_tokens, request.num_prompt_tokens
            acct["fwd"] += max(0, min(nc + n, P) - nc)
        return super()._update_after_schedule(scheduler_output)

    def _free_request(self, request, delay_free_blocks=False):
        st = getattr(request, "_scr", None)
        acct = getattr(request, "_scr_acct", None)
        try:
            if st is not None:
                sess = st["sess"]
                sess["inflight"] = max(0, sess.get("inflight", 1) - 1)
                if st.get("snap") is not None:
                    st["snap"]["users"] -= 1
                    if st["snap"].get("dropped"):
                        self._scr_snap_release(st["snap"])
                    st["snap"] = None
                self._scr_pin(request, sess)
                self._scr_free_ssm(request)
            if acct is not None and C.get("KVREUSE_TRACE", bool):
                P = request.num_prompt_tokens
                hit = max(0, int(request.num_cached_tokens))
                _log(f"fin rid={request.request_id} sid={acct['sid']} prev={acct['prev']} lcp={acct.get('lcp', 0)} prompt={P} "
                     f"hit={hit} reloc={acct['reloc']} fwd={acct['fwd']} decode={request.num_output_tokens} "
                     f"blocks={acct['blocks']} planned={acct['planned']} preempt={acct.get('preempt', 0)} "
                     f"reason={acct['reason'] or '-'} cause={acct.get('cause', '-')} fallback={acct['fallback'] or '-'} "
                     f"plan_ms={acct.get('plan_ms', 0)} lat={time.time() - request.arrival_time:.3f} "
                     f"status={request.status.name}")
        except Exception as e:
            _log(f"finish hook error ({type(e).__name__}: {e})")
        request._scr = None
        now = time.time()
        if now - self._scr_t_last > 60:
            self._scr_t_last = now
            _log(f"stats {dict(self._scr_stats)} pinned_blocks={self._scr_pinned_blocks} "
                 f"sessions={len(self._scr_sessions)} free_blocks={self.kv_cache_manager.block_pool.get_num_free_blocks()}")
        out = super()._free_request(request, delay_free_blocks)
        if self._scr_audit_every and not self.requests:
            self._scr_audit()      # idle: everything still referenced is a pin or a save
        return out


class SCRScheduler(SCRMixin, Scheduler):
    pass


class SCRAsyncScheduler(SCRMixin, AsyncScheduler):
    pass
