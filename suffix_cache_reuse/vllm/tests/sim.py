# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU simulation of the SCR scheduler inside a real vLLM Scheduler (no GPU, no model weights).

The real SCRScheduler (vLLM's KV cache manager, block pool, prefix cache, chunked prefill,
preemption) is driven by a fake worker that mirrors what the GPU worker does with the
scheduler's output:

  attention KV   one entry per slot, (token id, position). A forward pass over positions
                 [nc, nc+n) writes them; SCR copy ops copy entries with position + delta.
                 Before each forward the worker checks that every position < nc holds exactly
                 (token[p], p) -- prefix-cache hits and relocated spans included.
  recurrent      (hybrid, mamba_cache_mode="align") one state per block, mirroring
  state          preprocess_mamba / postprocess_mamba of vLLM 0.19.1: the state is a rolling
                 hash of the tokens consumed plus its position. SCR fork-restores and saves are
                 applied as the connector would. Checks: a state is never read from a freed
                 block; every state read is either exact (equals the hash of the request's own
                 prefix at that position) or an SCR fork state at the right position; and every
                 recurrent-state block that carries a prefix-cache hash holds the state at the
                 end of its own block (pos == (i + 1) * block_size) -- dirty cut blocks never
                 enter the prefix cache.
  sampling       greedy, token = f(context tokens, recurrent state): identical across SCR
                 on/off for dense models whenever the KV checks pass.

At the end every request is finished, the side buffer is released and the block pool must be
whole again (no leaks); KVREUSE_AUDIT=1 additionally audits all reference counts every step.
"""
import os
import random
import sys
import tempfile
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASK = (1 << 61) - 1
BASE = 1_000_003


def _cfg_dir():
    d = HERE / "fixtures" / "tiny-llama"
    if not (d / "config.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64,
            "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
            "num_hidden_layers": 2, "vocab_size": 32000, "max_position_embeddings": 65536,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "torch_dtype": "bfloat16",
            "bos_token_id": 1, "eos_token_id": 2}))
    return str(d)


def make_scheduler(hybrid=False, scr=True, num_blocks=2000, block_size=16, budget=512,
                   max_seqs=32, env=None, stock=False, async_sched=False):
    env = dict(env or {})
    os.environ.update({"KVREUSE_ENABLED": "1" if scr else "0", "KVREUSE_THINK_IDS": "",
                       "KVREUSE_TRACE": "0", "KVREUSE_AUDIT": "1", "KVREUSE_AUDIT_STRICT": "1"})
    os.environ.update(env)
    import torch
    from vllm.config import (CacheConfig, DeviceConfig, KVTransferConfig, ModelConfig,
                             SchedulerConfig, VllmConfig)
    from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
                                            MambaSpec)
    from vllm.v1.structured_output import StructuredOutputManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.utils.hashing import sha256_cbor
    from vllm_scr.scheduler import SCRAsyncScheduler, SCRScheduler

    m = ModelConfig(model=_cfg_dir(), skip_tokenizer_init=True, max_model_len=65536, dtype="bfloat16")
    cc = CacheConfig(block_size=block_size, enable_prefix_caching=True,
                     mamba_cache_mode="align" if hybrid else "none")
    sc = SchedulerConfig(max_num_batched_tokens=budget, max_num_seqs=max_seqs, max_model_len=65536,
                         is_encoder_decoder=False, enable_chunked_prefill=True,
                         long_prefill_token_threshold=0)
    kt = KVTransferConfig(kv_connector="SCRConnector", kv_connector_module_path="vllm_scr.connector",
                          kv_role="kv_both") if scr and not stock else None
    vc = VllmConfig(model_config=m, cache_config=cc, scheduler_config=sc, kv_transfer_config=kt,
                    device_config=DeviceConfig(device="cpu"))
    vc.cache_config.num_gpu_blocks = num_blocks
    groups = [KVCacheGroupSpec(["model.layers.0.self_attn.attn"],
                               FullAttentionSpec(block_size=block_size, num_kv_heads=2, head_size=16,
                                                 dtype=torch.bfloat16))]
    if hybrid:
        groups.append(KVCacheGroupSpec(["model.layers.1.linear_attn"],
                                       MambaSpec(block_size=block_size, shapes=((4,),),
                                                 dtypes=(torch.float32,), mamba_cache_mode="align")))
    kvc = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    if async_sched:
        cls = AsyncScheduler if stock else SCRAsyncScheduler
    else:
        cls = Scheduler if stock else SCRScheduler
    s = cls(vllm_config=vc, kv_cache_config=kvc,
                     structured_output_manager=StructuredOutputManager(vc), block_size=block_size)
    init_none_hash(sha256_cbor)
    hasher = get_request_block_hasher(block_size, sha256_cbor)
    return s, hasher


class SimError(AssertionError):
    pass


class Worker:
    """Mirrors the GPU worker's KV writes, SCR copies and the align-mode recurrent state."""

    def __init__(self, sched, vocab=30000):
        self.s = sched
        self.vocab = vocab
        self.slots = {}        # attention slot -> (token, pos)
        self.state = {}        # mamba block id -> dict(pos, h, exact)
        self.state_idx = {}    # req id -> current state block index (vLLM's mamba_state_idx)
        self.pref = {}         # req id -> prefix hashes
        self.wblocks = {}      # req id -> per group list of block ids (worker view)
        self.last_end = {}     # req id -> end of its previous chunk
        groups = sched.kv_cache_config.kv_cache_groups
        from vllm.v1.kv_cache_interface import MambaSpec
        self.attn = [g for g, x in enumerate(groups) if not isinstance(x.kv_cache_spec, MambaSpec)]
        self.mamba = [g for g, x in enumerate(groups) if isinstance(x.kv_cache_spec, MambaSpec)]
        self.bs = groups[0].kv_cache_spec.block_size
        self.stats = {"steps": 0, "verified_tokens": 0, "state_reads": 0, "fork_reads": 0,
                      "hashed_state_checks": 0, "reloc_ops": 0, "reloc_tokens": 0}

    def _blocks(self, g, rid):
        """The worker's view of the block table (vLLM appends new block ids; it never learns
        that the scheduler dropped a block, so it may still read it -- hence the ref check)."""
        return self.wblocks[rid][g]

    def _slot(self, g, rid, p):
        return self._blocks(g, rid)[p // self.bs] * self.bs + p % self.bs

    def _sync_tables(self, so):
        for nr in so.scheduled_new_reqs:
            self.wblocks[nr.req_id] = [list(x) for x in nr.block_ids]
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            nb = cr.new_block_ids[i]
            if rid in cr.resumed_req_ids:
                self.wblocks[rid] = [list(x) for x in nb]
            elif nb is not None:
                for cur, new in zip(self.wblocks[rid], nb):
                    cur.extend(new)
        for b in (so.new_block_ids_to_zero or ()):
            self.state.pop(b, None)

    def _prefix(self, req, upto):
        h = self.pref.setdefault(req.request_id, [0])
        toks = req.all_token_ids
        while len(h) <= upto:
            h.append((h[-1] * BASE + toks[len(h) - 1] + 1) & MASK)
        return h[upto]

    def step(self, so):
        s = self.s
        pool = s.kv_cache_manager.block_pool
        meta = so.kv_connector_metadata
        self._sync_tables(so)
        for rid in list(so.finished_req_ids) + list(so.preempted_req_ids or ()):
            self.state_idx.pop(rid, None)
        for rid in so.scheduled_cached_reqs.resumed_req_ids:
            self.state_idx.pop(rid, None)
        work = []
        for rid, n in so.num_scheduled_tokens.items():
            req = s.requests[rid]
            work.append((rid, req, req.num_computed_tokens - n, n))
        # (1) preprocess_mamba: carry the running state into the current state block
        for rid, req, nc, n in work:
            for g in self.mamba:
                bl = self._blocks(g, rid)
                prev = self.state_idx.get(rid, (nc - 1) // self.bs)
                curr = -(-(nc + n) // self.bs) - 1
                if prev != -1 and prev != curr:
                    src = pool.blocks[bl[prev]]
                    if src.is_null or src.ref_cnt <= 0:
                        raise SimError(f"{rid}: state read from a freed block (idx {prev}, nc {nc})")
                    self.state[bl[curr]] = dict(self.state.get(src.block_id, {"pos": -1}))
                self.state_idx[rid] = curr
        # (2) connector: relocation copies and fork restores
        restored = set()
        if meta is not None and getattr(meta, "ops", None) is not None:
            for gid, src, dst, delta in meta.ops:
                self.stats["reloc_ops"] += 1
                self.stats["reloc_tokens"] += len(src)
                vals = [self.slots.get(int(a)) for a in src]
                for a, b, d, v in zip(src, dst, delta, vals):
                    if v is None:
                        raise SimError(f"relocation source slot {a} never written")
                    self.slots[int(b)] = (v[0], v[1] + int(d))
            for gid, sb, db in meta.ssm_restore:
                st = self.state.get(sb)
                if st is None:
                    raise SimError(f"fork restore from an unwritten state block {sb}")
                self.state[db] = dict(st, exact=False, fork=True)
                restored.add(db)
        # (3) verify the attention context, (4) forward, (5) postprocess_mamba
        for rid, req, nc, n in work:
            toks = req.all_token_ids
            for g in self.attn:
                for p in range(nc):
                    v = self.slots.get(self._slot(g, rid, p))
                    if v != (toks[p], p):
                        raise SimError(f"{rid}: KV at position {p} is {v}, expected {(toks[p], p)} "
                                       f"(nc={nc}, n={n})")
                self.stats["verified_tokens"] += nc
                for p in range(nc, nc + n):
                    self.slots[self._slot(g, rid, p)] = (toks[p], p)
            for g in self.mamba:
                bl = self._blocks(g, rid)
                cur = pool.blocks[bl[self.state_idx[rid]]]
                st = self.state.get(cur.block_id, {"pos": -1}) if nc > 0 else {"pos": 0, "h": 0, "exact": True}
                self.stats["state_reads"] += 1
                jumped_now = rid in self.last_end and nc != self.last_end[rid]
                if st.get("fork"):
                    # an SCR fork state (restored, or derived from one): position consistency,
                    # except across a jump over a relocated span (the linear layers skip it)
                    self.stats["fork_reads"] += 1
                    if cur.block_id not in restored and not jumped_now and st["pos"] != nc:
                        raise SimError(f"{rid}: fork state at pos {st['pos']} read at nc {nc}")
                    new = {"pos": nc + n, "h": None, "exact": False, "fork": True}
                elif st.get("exact") and st["pos"] == nc and st["h"] == self._prefix(req, nc):
                    new = {"pos": nc + n, "h": self._prefix(req, nc + n), "exact": True}
                elif jumped_now and (getattr(req, "_scr", None) or {}).get("jumped"):
                    # after a jump without a fork restore (KVREUSE_SSM_MODE=none, or a later span)
                    self.stats["fork_reads"] += 1
                    new = {"pos": nc + n, "h": None, "exact": False, "fork": True}
                else:
                    raise SimError(f"{rid}: wrong recurrent state at nc {nc}: {st}")
                self.state[cur.block_id] = new
                aligned = (nc + n) // self.bs * self.bs
                if aligned >= nc + n:
                    self.state[bl[aligned // self.bs - 1]] = dict(new)
        for rid, req, nc, n in work:
            self.last_end[rid] = nc + n
        # (6) wait_for_save: SCR prefill-end state saves
        if meta is not None and getattr(meta, "ssm_save", None) is not None:
            for gid, sb, db in meta.ssm_save:
                self.state[db] = dict(self.state[sb])
        # pool-wide: every recurrent-state block that carries a prefix-cache hash (in use or
        # free-but-cached) holds the state at the end of a block
        if self.mamba:
            from vllm.v1.core.kv_cache_utils import get_group_id
            for b in pool.blocks:
                h = b.block_hash
                if h is None or b.is_null or get_group_id(h) not in self.mamba:
                    continue
                st = self.state.get(b.block_id)
                self.stats["hashed_state_checks"] += 1
                if st is None or st["pos"] <= 0 or st["pos"] % self.bs:
                    raise SimError(f"prefix-cached state block {b.block_id} holds the state at "
                                   f"{None if st is None else st['pos']} (dirty block cached)")
        # invariants on hashed blocks of running requests
        for rid, req, nc, n in work:
            for g in self.mamba:
                sched_bl = self.s.kv_cache_manager.coordinator.single_type_managers[g].req_to_blocks[rid]
                for i, b in enumerate(sched_bl):
                    if b.is_null or b.block_hash is None:
                        continue
                    self.stats["hashed_state_checks"] += 1
                    st = self.state.get(b.block_id)
                    if st is None or st["pos"] != (i + 1) * self.bs:
                        raise SimError(f"{rid}: prefix-cached state block {i} holds the state at "
                                       f"{None if st is None else st['pos']} (dirty block cached)")
        # sampling
        sampled = []
        for rid, req, nc, n in work:
            if req.num_computed_tokens >= req.num_tokens:
                toks = req.all_token_ids
                h = 0
                for t in toks[-6:]:
                    h = (h * 1000003 + t) & 0xFFFFFFFF
                if self.mamba:
                    st = self.state[self._blocks(self.mamba[0], rid)[self.state_idx[rid]]]
                    h ^= (st.get("h") or 7777) & 0xFFFFFFFF
                sampled.append([3 + h % (self.vocab - 3)])
            else:
                sampled.append([])
        self.stats["steps"] += 1
        return sampled


def run(convs=24, turns=6, hybrid=False, scr=True, seed=0, num_blocks=2000, block_size=16,
        budget=512, max_out=24, concurrency=12, env=None, stock=False, patch=None,
        edit_rate=0.65, max_seqs=32, async_sched=False):
    """Multi-turn editing conversations. Each conversation has its own RNG, so its prompts
    depend only on its own outputs (identical across runs when the outputs are).
    Returns (stats, outputs {(conv, turn): (tokens, acct)}, schedule trace)."""
    from vllm import SamplingParams
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    s, hasher = make_scheduler(hybrid=hybrid, scr=scr, num_blocks=num_blocks, block_size=block_size,
                               budget=budget, env=env, stock=stock, max_seqs=max_seqs,
                               async_sched=async_sched)
    if patch is not None:
        patch(s)
    w = Worker(s)
    rngs = {c: random.Random(seed * 100003 + c) for c in range(convs)}
    tok = lambda c, k: [rngs[c].randrange(3, 30000) for _ in range(k)]
    conv = {c: {"segs": [tok(c, rngs[c].randrange(600, 1400))]
                + [tok(c, rngs[c].randrange(80, 400)) for _ in range(4)], "turn": 0}
            for c in range(convs)}
    outputs, live, trace, npre = {}, {}, [], [0]
    queue = list(range(convs))
    random.Random(seed).shuffle(queue)

    def submit(c):
        cv = conv[c]
        prompt = [t for seg in cv["segs"] for t in seg]
        rid = f"c{c}-t{cv['turn']}"
        r = Request(rid, prompt, SamplingParams(max_tokens=rngs[c].randrange(4, max_out),
                                                temperature=0, ignore_eos=True),
                    None, block_hasher=hasher)
        s.add_request(r)
        live[rid] = c

    def edit(c, out):
        cv, r = conv[c], rngs[c]
        segs = cv["segs"]
        segs.append(out + tok(c, r.randrange(40, 200)))     # assistant output + tool result
        u = r.random() / max(edit_rate, 1e-9) * 0.65
        if u < 0.35 and len(segs) > 3:                       # delete a middle segment
            del segs[r.randrange(1, len(segs) - 1)]
        elif u < 0.55 and len(segs) > 3:                     # replace a middle segment
            segs[r.randrange(1, len(segs) - 1)] = tok(c, r.randrange(20, 120))
        elif u < 0.65 and len(segs) > 4:                     # delete two
            for _ in range(2):
                del segs[r.randrange(1, len(segs) - 1)]
        cv["turn"] += 1                                      # else: pure append

    for _ in range(min(concurrency, len(queue))):
        submit(queue.pop(0))
    guard = 0
    while live:
        guard += 1
        if guard > 200000:
            raise SimError("no progress")
        so = s.schedule()
        trace.append(tuple(sorted(so.num_scheduled_tokens.items())))
        npre[0] += len(so.preempted_req_ids or ())
        sampled = w.step(so)
        rids = list(so.num_scheduled_tokens)
        mro = ModelRunnerOutput(req_ids=rids, req_id_to_index={r: i for i, r in enumerate(rids)},
                                sampled_token_ids=sampled)
        before = {r: s.requests.get(r) for r in live}
        s.update_from_output(so, mro)
        for rid in list(live):
            if rid not in s.requests:
                c = live.pop(rid)
                req = before[rid]
                out = list(req.output_token_ids)
                acct = dict(getattr(req, "_scr_acct", None) or {})
                acct["num_cached_tokens"] = int(req.num_cached_tokens)
                acct["prompt"] = len(req.prompt_token_ids)
                outputs[(c, conv[c]["turn"])] = (out, acct)
                if conv[c]["turn"] + 1 < turns:
                    edit(c, out)
                    queue.append(c)
                if queue:
                    submit(queue.pop(0))
    so = s.schedule()
    w.step(so)
    pool = s.kv_cache_manager.block_pool
    st = dict(w.stats, requests=len(outputs))
    if not stock:
        st["pinned_at_end"] = s._scr_pinned_blocks
        ok, bad = s._scr_audit()
        if not ok:
            raise SimError(f"audit failed at end: {bad[:5]}")
        s.scr_release_all()
        st["scr"] = dict(s._scr_stats)
    free_after = pool.get_num_free_blocks()
    if free_after != num_blocks - 1:
        raise SimError(f"block leak: {free_after} free of {num_blocks - 1} after releasing all")
    if any(b.ref_cnt for b in pool.blocks if not b.is_null):
        raise SimError("nonzero ref_cnt after releasing all")
    st["preemptions"] = npre[0]
    return st, outputs, trace
