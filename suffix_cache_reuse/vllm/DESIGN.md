# Suffix Cache Reuse on vLLM 0.19.1: design

This document describes how Suffix Cache Reuse (SCR) fits into vLLM's v1 engine:

- where it plugs in, and why there;
- the per-request data flow;
- what is approximated;
- the fallback paths and memory use;
- hybrid (linear-attention) models;
- what an upstream vLLM hook would need to provide.

The algorithm and its defaults are those of the [SGLang version](../sglang/README.md).

## 1. vLLM v1 internals that matter

| piece | where | relevant behaviour |
|---|---|---|
| prefix cache | `v1/core/kv_cache_manager.py`, `block_pool.py`, `single_type_kv_cache_manager.py` | A hash chain over `block_size`-token blocks (`request.block_hashes`). `get_computed_blocks` returns a block-aligned hit capped at `num_tokens-1`. `allocate_slots(request, n)` allocates blocks to cover `num_computed_tokens + n`, then hashes and caches every block that becomes full (`coordinator.cache_blocks`). Blocks are reference-counted. Blocks with `ref_cnt == 0` stay in the free queue and can still be hit until they are evicted. |
| scheduler | `v1/core/sched/scheduler.py`; `AsyncScheduler` when async scheduling is on, the 0.19 default | There are no separate prefill and decode phases. In each step a request is given `num_new_tokens = num_tokens - num_computed_tokens`, capped by the token budget. `num_computed_tokens` advances at schedule time. Every step the worker takes `num_computed_tokens` from the scheduler output, and positions and slots follow from it. Custom scheduler classes are supported (`--scheduler-cls`). |
| KV connector | `distributed/kv_transfer/kv_connector/v1/base.py` | Scheduler side: `get_num_new_matched_tokens` is asked only for waiting requests and only for tokens directly after the local hit; also `build_connector_meta` (once per step) and `request_finished`. Worker side: `register_kv_caches`; `start_load_kv` (after `preprocess_mamba`, before the forward); `wait_for_save` (after the forward). A connector must implement `SupportsHMA`, or vLLM turns the hybrid allocator off. |
| KV layout | attention backends | Per layer `(2, num_blocks, block_size, kv_heads, head_dim)`; FlashInfer uses `(num_blocks, 2, ...)`. Slot = `block_id * block_size + offset`. |
| RoPE | `model_executor/layers/rotary_embedding/` | RoPE is applied before the attention op writes K to the cache, so cached keys are post-RoPE. Rotary modules are memoised in `_ROPE_DICT`, and `cos_sin_cache` is stored in the model dtype. |
| hybrid state | `MambaManager`, `v1/worker/mamba_utils.py` | With prefix caching, the mamba cache mode is `all` or `align`. In `align`, a request's running recurrent state lives in its block `cdiv(num_computed + n_sched, block_size) - 1`. `preprocess_mamba` copies the state forward when that index changes. States are cached only where a step ends exactly on a block boundary. When the request advances, the previous state block is freed. |

## 2. Integration point

SCR consists of two pieces:

- a scheduler subclass, `vllm_scr.scheduler.SCRScheduler` or `SCRAsyncScheduler`, loaded with
  `--scheduler-cls`;
- a KV connector, `vllm_scr.connector.SCRConnector`, used only as the scheduler-to-worker
  channel for copy operations.

No vLLM source is modified or re-executed.

**Why the scheduler is needed.** A connector can only add externally loaded tokens directly
after the local prefix hit, for waiting requests, before anything is computed. SCR needs a
different order:

1. take A from the prefix cache;
2. compute B′;
3. place C from the held KV at the positions after B′;
4. compute what follows.

There can be up to K such C spans. The scheduler therefore has to do two things:

1. **The cap.** Stop a prefill chunk exactly at |A|+|B′|.
2. **The jump.** Advance `num_computed_tokens` over C without computing it.

The copy itself needs the worker's KV tensors. That is what the connector's `start_load_kv`
provides, and it runs at the right time.

**The seam.** In 0.19.x, both branches of `Scheduler.schedule` (RUNNING and WAITING) call
`self._mamba_block_aligned_split(request, num_new_tokens, ...)`. The call comes right after the
token-budget and encoder clamps and right before `allocate_slots`, and it is guarded by
`self.need_mamba_block_aligned_split`. That method is exactly a per-request "cap the number of
tokens scheduled this step" hook. The SCR scheduler uses it as follows:

- It sets the flag for every model and overrides the method.
- The override first applies the SCR cap. It then applies vLLM's own block alignment, but only
  where vLLM would have applied it (hybrid models in `mamba_cache_mode=align`).
- The jump happens in `schedule()` before calling `super().schedule()`.
- The copy operations are added in `_build_kv_connector_meta`.

`check_seam()` runs at import time and stops with a clear error in either of these cases:

- the vLLM version is not 0.19.1 (unless `KVREUSE_ALLOW_UNTESTED_VLLM=1`);
- `Scheduler.schedule` no longer calls the hook exactly once per branch under the same guard, or
  the hook's signature has changed.

`tests/test_sim.py` checks two properties of the seam:

- **Neutral when unused.** With SCR on and nothing to relocate, the schedule is identical, step
  for step, to vLLM's own `Scheduler`.
- **Neutral when off.** With SCR off, the same holds.

**Other instance-level hooks.** These are not source patches; they wrap one method each on the
scheduler's own KV manager objects:

- `kv_cache_manager.allocate_slots`: on failure, release held conversations (least recently
  used first) and retry before vLLM preempts anything.
- Each mamba manager's `cache_blocks`: keep a dirty recurrent-state block out of the prefix
  cache (§7).
- `coordinator.cache_blocks`, only with `KVREUSE_CACHE_RELOCATED=0`: cap caching at the first
  relocated position.

**What an upstream hook would need.** Four small APIs would make SCR, and similar KV-reuse
schemes, a plain plugin:

1. **A per-request prefill cap**, e.g. `Scheduler.limit_num_new_tokens(request,
   num_computed_tokens, num_new_tokens) -> int`, called in both branches where
   `_mamba_block_aligned_split` is called today. The cap must be applied before alignment.
2. **A skip API for running requests**, letting a connector declare that the next n tokens of
   a running request are provided externally:
   `get_num_new_matched_tokens(request, num_computed_tokens)` honoured for RUNNING requests at
   arbitrary positions, not only for waiting requests directly after the local hit. With it,
   the jump and the copy could live entirely in the connector.
3. **Block pinning**: `BlockPool.pin(blocks)` / `unpin(blocks)`, plus an "evictable before
   preemption" callback, instead of adjusting `ref_cnt` directly and wrapping `allocate_slots`.
4. **Hybrid state control**: a way to mark a recurrent-state block as not cacheable, and a
   supported state save/restore operation for connectors. Today these are done with the
   `cache_blocks` wrapper and direct copies on the worker's state tensors.

## 3. Per-request data flow (SCR on)

1. **Planning, in `add_request`.**
   - The prompt is matched to a tracked conversation using the same rules as the SGLang
     version: same `cache_salt`, a common prefix of at least 1024 tokens, prefix and overlap
     ratios, and an LRU of 16 conversations.
   - The new prompt is diffed against the prompt whose KV the conversation holds (a verified
     chunked diff, with difflib as fallback).
   - Up to K=6 surviving spans after the leading equal run are planned, longest first. Each
     span is at least 64 tokens, and its last 16 tokens are left for prefill.
2. **Phase 1, the cap.**
   - vLLM's prefix-cache hit covers A, block-aligned.
   - The cap stops the prefill at the first span's start, |A|+|B′|.
   - If the hit already reaches that point, the span start moves one token past the hit.
3. **The jump, at the start of the next `schedule()`.** For a running request whose
   `num_computed_tokens` equals the span start, `num_computed_tokens` is advanced by the span
   length. The RUNNING branch then allocates the span's blocks together with the next stretch,
   and the cap stops that stretch at the next span's start.
4. **The copy, in the step that first reads the span.**
   - `_build_kv_connector_meta` emits one operation per attention group: source slots (the
     held rows), destination slots (the new positions), and the position shift.
   - The worker copies K and V for every layer and re-rotates the keys by the shift. The angle
     is computed in float64, applied in float32, and stored in the cache dtype.
   - `inv_freq` is taken from the model's own rotary module and checked at start-up against
     its `cos_sin_cache`.
5. **After the last span**, the prefill continues normally (at least 16 tokens), then decoding
   follows.
6. **In `_free_request`.**
   - The finished request's blocks covering its prompt are pinned (`ref_cnt += 1`) as the
     conversation's held KV, the "side buffer".
   - The previous pin is released once no in-flight request still plans from it.

**Accounting.** Each request writes one `[scr] fin` line with its prompt, prefix hit,
relocated, forwarded and decoded tokens, and its fallback reason. `analysis/scr_report.py`
checks `prompt = hit + relocated + forwarded` for every request.

## 4. Side buffer: pinned blocks instead of a copy

The SGLang version copies the finished turn's KV into a separate GPU allocation. In vLLM, the
finished request's blocks are simply kept, by reference count, in the block pool. A held
conversation therefore costs no copy and no extra allocation, and blocks it shares with the
prefix cache are not duplicated. The limits are the same as in SGLang:

- at most `KVREUSE_SIDE_SESSIONS` (12) conversations are held;
- a conversation longer than `KVREUSE_SIDE_TOKENS` (30000) is not held;
- pins are capped at `KVREUSE_BUDGET_FRAC` (25 %) of the KV blocks.

When an allocation fails, held conversations that are not in flight are released first.

## 5. What is approximated (same as SGLang)

- **Stale context.** The relocated K/V of C were computed under the old context (with B, not
  B′). Only the RoPE position is corrected, and later tokens attend to these entries.
- **Compounding.** A held turn can already contain relocated rows from earlier turns, so
  staleness can compound across turns. It is bounded per edit by K.
- **Hybrid models.** The linear-attention layers resume from the recurrent state saved at the
  end of the previous prompt, so they do not see B′ (§7).
- **Prefix caching of relocated blocks.** With `KVREUSE_CACHE_RELOCATED=1` (the default), blocks
  holding relocated tokens enter the prefix cache like any other block. SGLang likewise inserts
  the finished request, relocated slots included, into its radix tree.

## 6. Fallback paths

| event | behaviour |
|---|---|
| no matching conversation, no held KV, the diff is a pure append, or no span qualifies | ordinary vLLM prefill (`reason=` in the fin line) |
| held KV released before the jump (`source_gone`) | remaining spans dropped; prefill continues |
| request preempted | plan dropped; vLLM recomputes the request as usual (`fallback=preempted`) |
| allocation fails | held conversations released, then vLLM's normal preemption |
| planning raises an error | request served normally; `plan_error` counted |
| `cache_salt` starting with `kvreuse:off` | SCR off for that request |
| hybrid model in a mamba cache mode other than `align` | SCR disabled at start-up, accounting only |
| speculative decoding, pipeline parallelism, sliding-window, chunked-local, MLA or sink attention groups | SCR disabled at start-up, accounting only |
| hybrid model, conversation without a saved recurrent state | no relocation (`ssm_gated`) |

## 7. Hybrid models (e.g. Qwen3.6-27B: 16 attention + 48 GatedDeltaNet layers)

**How vLLM serves the model.** vLLM serves Qwen3.6 with its hybrid KV cache manager:

- one attention group and three mamba groups, all drawing 784-token blocks from one pool;
- prefix caching in mamba cache mode `align`.

vLLM turns the hybrid manager off whenever a `--kv-transfer-config` is set, unless
`--no-disable-hybrid-kv-cache-manager` is also passed. `vllm-scr-serve` adds that flag.

**Save.** In the step where a request's prompt prefill completes:

- the scheduler takes one free block per mamba group;
- the worker copies the running state into it after the forward (`wait_for_save`);
- the saved state becomes part of the conversation's held snapshot.

A conversation without a saved state does not relocate.

**Restore.** In the step after the first jump, after `preprocess_mamba` and before the forward,
the saved state is copied into the request's running-state block. The linear-attention layers
thus continue from the state at the end of the previous prompt. Later spans of the same request
do not restore, as in SGLang.

**Cuts and the state cache.** The cut at |A|+|B′| is generally not block-aligned, so the
running-state block written by that chunk holds the state at |A|+|B′|, not at its block end.
Two safeguards handle this:

- **Never cached.** The block is marked dirty and never hashed into the prefix cache. In 0.19.1,
  vLLM's own align-mode bookkeeping already frees that block before it could be hashed when the
  request jumps. The skip protects against bookkeeping changes: `tests/test_sim.py` shows that
  without vLLM's freeing, it is this skip that keeps dirty states out of the cache.
- **Kept alive one step.** Because vLLM frees the block when the request jumps, but
  `preprocess_mamba` still copies from it in the next forward, SCR holds one extra reference on
  it for one step. `tests/test_sim.py` shows that without this, the worker reads a freed block.

After a jump, the next chunk is cut at the next block boundary, so all later states are cached
at their boundaries as usual.

**Keeping the next turn's prefix hit.** vLLM can resume a hybrid request only from a state
cached at a block boundary, and a relocated span is never run through the linear-attention
layers, so boundaries inside a span get no state. Two rules keep the state at the prompt's last
boundary, `floor(P / 784) * 784`, always cached:

- relocated spans are trimmed to end before that boundary, which costs at most one block of
  extra prefill;
- each prefill chunk that starts at a non-aligned position stops at the next boundary.

Boundaries inside relocated spans carry no cached state. When a later turn edits a point behind
such a span, vLLM resumes it from the last cached boundary before the span.

**Saving the state: copy or zero-copy.** `KVREUSE_SSM_SAVE` selects how the state at the end
of the previous prompt is kept for the next turn:

- `end` (default) copies the state at the end of each prompt into newly allocated blocks, as
  the SGLang version does. The blocks come from vLLM's free queue, which in a full pool also
  holds evictable prefix-cache entries.
- `boundary`, the **zero-copy** variant, takes no new block. When a conversation's turn
  finishes, it takes a reference on the recurrent state that vLLM already caches at the
  prompt's last block boundary, `floor(P / 784) * 784`, which the trimming rule above always
  keeps cached. At the first splice, it restores that state instead of the prompt-end state, so
  the linear-attention layers resume from that boundary; the attention layers keep the
  remaining `P mod 784` tokens.

Because zero-copy allocates nothing, it leaves the prefix cache untouched, which makes it the
recommended setting for hybrid models when many concurrent conversations share a tight KV
cache. `KVREUSE_AUDIT` counts prefix-cache evictions by cause (`evict_vllm`, `evict_ssm_save`).

## 8. Serving regime: prior reasoning

Many clients send an assistant turn's prior reasoning back as `reasoning_content`, but
vLLM 0.19.1 renders only `reasoning`, so without a shim prior reasoning does not reach the template.
`vllm-scr-serve` loads a one-function shim in both modes that copies `reasoning_content` into
`reasoning` (`vllm_scr/compat.py`, loaded via `vllm_scr/_site/sitecustomize.py`).
`tests/regime_probe.py` checks it against a running server: the prompt token count with prior
reasoning must exceed the count without it.

## 9. Differences from the SGLang implementation

- **No chunked-prefill slot limit.** vLLM prefills several requests at once, so any planned
  request can relocate. SGLang has a single chunked-prefill slot, so only one relocating
  request is prefilled at a time.
- **Prefix-cache granularity.** vLLM's prefix-cache hits are block-aligned: 16 tokens for
  dense models, and 784 tokens for Qwen3.6 in `align` mode. SGLang's radix cache is
  token-granular.

## 10. Verification

| check | what it shows |
|---|---|
| `tests/unit_test.py` (CPU) | planner, session matching, and RoPE re-rotation against a float64 reference |
| `tests/test_sim.py` (CPU) | the real `SCRScheduler` and `SCRAsyncScheduler` inside vLLM's `Scheduler` with a simulated worker, for dense and hybrid layouts, under concurrency, random edits, memory pressure and forced preemption mid-plan. See below. |
| `tests/exactness.py` (GPU, in process) | relocated K equals the re-rotated original K, and V equals the original V, on every layer (exactly); layer-0 keys match a cold prefill up to bf16 rounding; greedy continuations match a cold prefill |
| `KVREUSE_VERIFY=1` (GPU, any tensor-parallel size) | each worker re-derives every copied row from its source and compares it with the cache (exact match expected) |
| `KVREUSE_AUDIT=N` | the scheduler checks every KV block's reference count against the references held by requests, pins, saved states and deferred frees; it also counts prefix-cache evictions by cause (`evict_vllm`, `evict_ssm_save`) |
| `tests/smoke.py`, `tests/greedy_server.py` | end to end through the OpenAI API |

`tests/test_sim.py` checks, at every step:

- every position's KV equals its own token and position, including prefix hits and relocated
  spans;
- recurrent states are never read from freed blocks;
- prefix-cached recurrent states sit at block ends;
- all block reference counts are accounted for.

At the end it checks:

- the block pool is whole again;
- with SCR off, the schedule is identical to stock vLLM;
- dense models produce outputs identical to SCR off;
- negative controls (each safeguard removed) fail.
