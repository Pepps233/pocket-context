# Suffix Cache Reuse for vLLM

Suffix Cache Reuse (SCR) for vLLM 0.19.1 (v1 engine). It is a port of the
[SGLang version](../sglang/) of [Suffix Cache Reuse](../README.md).

When an agent edits its context, the prompt changes in the middle, not only at the end.
vLLM's prefix cache then re-prefills every token after the first change, including text
that did not change. SCR finds the spans that survived the edit and copies their cached keys
and values to the new positions. The keys are re-rotated to the new RoPE positions. Only the
new or changed tokens are prefilled.

vLLM's source is not modified. SCR plugs in through two supported extension points:

- a scheduler subclass, loaded with `--scheduler-cls`;
- a KV connector, loaded with `--kv-transfer-config`, which carries the copy operations to the
  GPU workers.

[DESIGN.md](DESIGN.md) covers the integration point, the per-request data flow, hybrid
(GatedDeltaNet) models, fallback paths, memory, and the hook that upstream vLLM would need.

## Install

```bash
pip install vllm==0.19.1
git clone https://github.com/facebookresearch/context-language-models.git
cd context-language-models
pip install -e suffix_cache_reuse/vllm      # provides the vllm-scr-serve command
```

## Run

```bash
# SCR on (default): same arguments as `vllm serve`
vllm-scr-serve Qwen/Qwen3.6-27B --served-model-name qwen36-27b --enable-prefix-caching \
    --language-model-only --reasoning-parser qwen3

# Same server and scheduler class with SCR off (stock vLLM scheduling, same accounting log)
KVREUSE_ENABLED=0 vllm-scr-serve Qwen/Qwen3.6-27B ...
```

`vllm-scr-serve` (or `python -m vllm_scr.serve`) adds the following to your arguments:

- `--scheduler-cls`;
- the connector config;
- `--no-disable-hybrid-kv-cache-manager`, needed for hybrid models.

It then starts `vllm serve`. The command can be used as a drop-in replacement for `vllm serve`.

A single request can opt out of SCR with `"cache_salt": "kvreuse:off..."`.

Every finished request writes one accounting line, `[scr] fin ...`, to the server log.
`analysis/scr_report.py server.log` reads these lines and splits each prompt into
prefix-cache hit, relocated, and forwarded (prefilled) tokens.

## Settings

All settings are environment variables. They have the same names and defaults as the SGLang
version where the meaning carries over.

| variable | default | meaning |
|---|---|---|
| `KVREUSE_ENABLED` | 1 | 0: stock vLLM scheduling (accounting only) |
| `KVREUSE_MAX_BLOCKS` | 6 | K, the maximum number of surviving spans relocated per request, longest first (minimum 1) |
| `KVREUSE_V6_MIN_TAIL` | 64 | minimum span length (tokens) worth relocating |
| `KVREUSE_V6_MIN_EXTEND` | 16 | number of tokens at the end of each span that are prefilled rather than relocated |
| `KVREUSE_MAX_SESSIONS` | 16 | number of conversations tracked for prompt matching (LRU) |
| `KVREUSE_SIDE_SESSIONS` | 12 | number of conversations whose previous-turn KV is held for relocation |
| `KVREUSE_SIDE_TOKENS` | 30000 | conversations longer than this are not held |
| `KVREUSE_BUDGET_FRAC` | 0.25 | maximum fraction of KV blocks the held conversations may pin |
| `KVREUSE_MATCH_*` | see `config.py` | thresholds for matching a prompt to a conversation (shared prefix of at least 1024 tokens, overlap ratios) |
| `KVREUSE_CACHE_RELOCATED` | 1 | 1: blocks containing relocated tokens enter the prefix cache like any other block (SGLang parity); 0: caching stops at the first relocated position |
| `KVREUSE_SSM_MODE` | fork | hybrid models: linear-attention layers resume from the state saved at the end of the previous prompt (`none`: no state restore) |
| `KVREUSE_SSM_SAVE` | end | hybrid models: `end` copies the state at the end of each prompt into new blocks (as SGLang does); `boundary` selects the **zero-copy** variant, which holds a reference on the state vLLM already caches at the prompt's last block boundary and allocates nothing. Recommended when many conversations share a tight KV cache (see Performance) |
| `KVREUSE_THINK_IDS` | auto | token ids of `<think>` and `</think>`, used only to label edit causes in the log; `auto` reads them from the tokenizer |
| `KVREUSE_TRACE` | 1 | write the per-request `[scr] fin` line |
| `KVREUSE_AUDIT` | 0 | N>0: check every KV block's reference count every N scheduler steps and whenever the server is idle (debugging) |
| `KVREUSE_VERIFY` | 0 | 1: each worker re-checks every relocation copy and logs the result per tensor-parallel rank (debugging) |
| `KVREUSE_ALLOW_UNTESTED_VLLM` | 0 | 1: skip the vLLM version pin (the integration-seam check still runs) |

## Supported versions and models

- **vLLM:** 0.19.1, checked at import time. On any other version the import fails with a
  clear error, unless `KVREUSE_ALLOW_UNTESTED_VLLM=1` is set. The scheduler hook that SCR
  relies on is also checked against the installed vLLM source at import time (see DESIGN.md §2).
- **Attention models:** models with standard full attention and RoPE. Tested on:
  - Qwen3-8B (neox RoPE);
  - Llama-3.1-8B-Instruct (llama3-scaled RoPE);
  - Qwen3.6-27B (partial, interleaved multimodal RoPE, used for text).

  The model's own rotary module provides the RoPE frequencies, and they are checked at
  start-up against its cos/sin cache.
- **Hybrid models:** hybrid linear-attention models such as Qwen3.6-27B (GatedDeltaNet) need
  vLLM's hybrid KV cache manager with `mamba_cache_mode=align`, which is vLLM's default for
  these models when prefix caching is on.
- **Tensor parallelism:** tested at TP=1 and TP=2.
- **Attention backends:** FlashAttention and FlashInfer KV layouts.

## Tests

CPU only, no GPU or model weights needed:

```bash
python tests/unit_test.py     # planner, session matching, RoPE re-rotation
python tests/test_sim.py      # the real SCRScheduler inside vLLM's Scheduler, driven by a simulated worker
```

`test_sim.py` runs many concurrent multi-turn conversations with random edits, for dense and
for hybrid KV-cache layouts. It checks every step:

- every request's attention KV equals its own (token, position) sequence, including prefix
  hits and relocated spans;
- the recurrent state is never read from a freed block;
- no prefix-cached recurrent-state block holds a state from a non-aligned prefill cut;
- every KV block's reference count matches the references that the scheduler and SCR hold.

At the end it checks:

- every block returns to the free pool once the held conversations are released;
- with SCR off, the schedule is identical, step for step, to vLLM's own `Scheduler`;
- dense models produce the same outputs with SCR on as with SCR off.

Negative controls disable each safeguard in turn and must fail.

GPU tests, each on a model of your choice:

```bash
python tests/exactness.py --model <path> [--hybrid]   # in-process; copy and RoPE checks, greedy agreement
vllm-scr-serve <model> --port 30000 ... &              # then, against the server:
python tests/smoke.py --port 30000
python tests/greedy_server.py --port 30000 --log server.log   # SCR vs cold prefill: agreement and answer correctness; any TP size
python tests/bench_conc.py --port 30000 --conc 8 16 32         # throughput and latency
```

## Performance

**Serving speed.** The numbers below come from `tests/bench_conc.py` on one H200 GPU per
server, with prefix caching enabled and otherwise default vLLM settings. Each conversation
starts from a document of about 11K tokens split into 24 sections and runs for six turns.
Every turn appends a fixed assistant reply and a tool-result block, and 60% of the turns after
the first delete or rewrite an earlier section. Every request decodes exactly 64 tokens, so
standard serving and SCR process the same tokens. Each run executes two conversations per
concurrent slot. TTFT is the median time to first token over turns 2 to 6. Zero-copy
(`KVREUSE_SSM_SAVE=boundary`) reuses the linear-attention state that vLLM already caches
instead of saving a copy; it does not apply to Qwen3-8B, which has no linear-attention layers.

| model | concurrency | requests/s, standard | requests/s, SCR | requests/s, SCR zero-copy | TTFT, standard | TTFT, SCR | TTFT, SCR zero-copy |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | 8 | 3.58 | **5.82** | – | 0.61 s | **0.28 s** | – |
| Qwen3-8B | 16 | 4.06 | **6.04** | – | 0.60 s | **0.29 s** | – |
| Qwen3-8B | 32 | 4.52 | **4.60** | – | 0.54 s | **0.53 s** | – |
| Qwen3.6-27B | 8 | 1.14 | 1.78 | **1.80** | 1.80 s | 1.09 s | **0.90 s** |
| Qwen3.6-27B | 16 | 1.28 | 1.79 | **1.83** | 2.12 s | 1.44 s | **1.35 s** |
| Qwen3.6-27B | 32 | **1.34** | 0.92 | 1.27 | **2.10 s** | 3.21 s | 2.20 s |

At 8 and 16 concurrent conversations, SCR raises throughput by 1.4 to 1.6 times and lowers the
median TTFT by 32% to 54% on both models.

**Tuning for many concurrent conversations.** Set `KVREUSE_MAX_SESSIONS` (and
`KVREUSE_SIDE_SESSIONS`) to at least the number of concurrent conversations, and for hybrid
models under KV-cache pressure use zero-copy (`KVREUSE_SSM_SAVE=boundary`).

**BrowseComp-Plus.** We also ran Qwen3.6-27B on all 830 BrowseComp-Plus questions with three
seeds, in the reasoning-stripping chat regime with a 28,672-token context budget. Each server
runs on one H200 GPU and serves 8 concurrent questions, and standard serving and SCR run side
by side on the same node. Pooled over the three seeds, SCR lowers prefix-reuse FLOPs per
question by 24.7% and the time each question takes from start to final answer by 12.5%, with
accuracy on par with standard serving (paired difference +0.08 ± 0.82 points, standard error
over questions). With SCR, 8.3% of prompt tokens are relocated instead of prefilled.

| serving | accuracy (%) | PFLOPs/question | time/question (s) | prefix hit (%) | relocated (%) | forwarded (%) |
|---|---|---|---|---|---|---|
| standard vLLM | 58.80 | 9.86 | 553.8 | 76.0 | 0.0 | 24.0 |
| vLLM with SCR | **58.88** | **7.42** | **484.5** | 73.2 | 8.3 | **18.5** |

Token columns are shares of all prompt tokens. Time per question covers the whole trajectory.
For the same comparison on SGLang, see the [SGLang version](../README.md#results).

## How it fits into vLLM

- **What is reused.** Relocated keys and values keep the values computed in the previous turn,
  and their positions are re-rotated to the new offsets, exactly as in the SGLang version. In
  hybrid models, the linear-attention layers continue from the state at the end of the previous
  prompt.
- **Memory.** The previous turn's KV of each held conversation stays in vLLM's own block pool.
  `KVREUSE_BUDGET_FRAC` sets how much of the pool these blocks may use, and they are the first to
  be released when an allocation needs room.
- **Hybrid models.**
  - For many concurrent conversations on a tight KV cache, use zero-copy
    (`KVREUSE_SSM_SAVE=boundary`, see Performance).
  - vLLM resumes a hybrid request from a recurrent state cached at a block boundary. A relocated
    span is copied as one piece, so when the next turn edits a point behind such a span, it
    resumes from the last boundary before the span (DESIGN.md §7).
  - SCR runs with `mamba_cache_mode=align`. With any other mode the server starts with SCR off
    and still writes the accounting log.
- **Supported serving setups.** SCR runs as the only KV connector, on a single pipeline stage,
  with prefill and decode on the same server, without speculative decoding, and on models whose
  attention groups are full attention or linear attention.
- **Reasoning field.** vLLM 0.19.1 renders an assistant turn's earlier reasoning from the
  `reasoning` field, while many clients send it back as `reasoning_content`. `vllm-scr-serve`
  loads a one-function shim that copies `reasoning_content` into `reasoning`, so earlier
  reasoning reaches the template either way (DESIGN.md §8).

## Layout

| path | contents |
|---|---|
| `vllm_scr/scheduler.py` | session matching, the prefill cap through the scheduler hook, skipping over relocated spans, side-buffer pins, hybrid safeguards, reference-count audit |
| `vllm_scr/planner.py` | prompt diff and span planning (same rules as the SGLang version) |
| `vllm_scr/connector.py` | worker side: K/V copy, RoPE re-rotation, recurrent-state save and restore |
| `vllm_scr/config.py`, `serve.py` | settings and launcher |
| `vllm_scr/compat.py`, `_site/` | the reasoning-field shim |
| `analysis/scr_report.py` | per-request token accounting and prefix-reuse FLOPs from the server log |
| `tests/` | the tests above |
