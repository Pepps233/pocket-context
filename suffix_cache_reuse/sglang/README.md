# Suffix Cache Reuse for SGLang

Suffix Cache Reuse (SCR) for SGLang 0.5.16, as a patch applied at start-up. See
[Suffix Cache Reuse](../README.md) for what SCR does, how it works, and its results; the
SGLang results and figures there were measured with this version. A version for vLLM is in [`../vllm`](../vllm/).

## Implementation

SCR is a monkeypatch applied at start-up to an installed SGLang
([`suffix_cache_reuse/overlay.py`](suffix_cache_reuse/overlay.py)); no SGLang source file is
modified. Per request:

```
previous prompt   [ A ][ B ][ C1 ][ x ][ C2 ][ ... ]
new prompt        [ A ][ B'][ C1 ][ C2 ][ ... ][ D ]
                   radix  |    |     |
                   cache  |  relocated spans (re-rotated K/V from the side buffer)
                        prefilled: B', the last few tokens of each span, D
```

| step | what the code does | setting |
|---|---|---|
| session match | attaches the request to the tracked conversation with the same `cache_salt`, the longest common prefix (at least 1024 tokens) and high token overlap; no session header needed | `KVREUSE_MAX_SESSIONS` |
| session-private slots | keeps each conversation's previous-turn K/V in a separate GPU side buffer | `KVREUSE_SIDE_SESSIONS`, `KVREUSE_SIDE_TOKENS` |
| diff and relocate | diffs the new prompt against the previous one and relocates the K longest surviving spans: copies their K/V into fresh KV-cache slots, re-rotates RoPE keys, and splices them into the request's prefix | `KVREUSE_MAX_BLOCKS` (K) |
| span tail | prefills the last few tokens of each span instead of relocating them | `KVREUSE_V6_MIN_EXTEND` |
| linear attention | restores the recurrent state saved at the end of the previous prompt when splicing the first span | `KVREUSE_SSM_MODE=fork` |
| fallback | serves the request with standard prefix caching when no plan applies or slots cannot be allocated; a request whose A came from the radix cache keeps its own recurrent state | — |

## Install

```bash
pip install sglang==0.5.16          # the SGLang version SCR is built for
git clone https://github.com/facebookresearch/context-language-models.git
cd context-language-models/suffix_cache_reuse/sglang
pip install -e .
```

## Run

```bash
bash examples/serve_qwen36.sh                        # Qwen3.6-27B, one GPU, SCR on
KVREUSE_ENABLED=0 bash examples/serve_qwen36.sh      # the same server with SCR off
```

The example is a thin wrapper around

```bash
python -m suffix_cache_reuse.serve --model-path Qwen/Qwen3.6-27B [any sglang.launch_server args]
```

`suffix_cache_reuse.serve` fills in the SCR environment defaults, puts
`suffix_cache_reuse/_site` (a `sitecustomize.py`) on `PYTHONPATH` so that every
SGLang process loads the patch, and then runs `python -m sglang.launch_server` with
your arguments. It adds `--log-requests --log-requests-level 0` if `--log-requests`
is not given; level 0 logs request metadata only (ids and token counts, no text),
which is what the analysis script reads.

The server prints one start-up line per scheduler process:

```
[kvreuse-site] ... max_blocks=6 mb_min_gap=0 splice_retry=8
```

and one line per relocated block:

```
[kvreuse] v6 splice OK: rid=<rid> sid=<session> slot=<n> A+B'=<prefix> C'=<relocated> extend=<n> live=<n> [...] blocks=<i>/<K>
```

## Configuration

All settings are environment variables read at server start. The defaults below
(`suffix_cache_reuse/config.py`) are the K=6 setting; a variable that is already
set in the environment takes precedence.

| variable | default | meaning |
|---|---|---|
| `KVREUSE_ENABLED` | `1` | `0` starts plain SGLang (nothing is patched) |
| `KVREUSE_MAX_BLOCKS` | `6` | K, the number of unchanged blocks relocated per request |
| `KVREUSE_SPLICE_RETRY` | `8` | scheduling rounds a ready plan waits for SGLang's chunked-prefill slot |
| `KVREUSE_SSM_MODE` | `fork` | recurrent-state handling for linear-attention layers (`fork`, `strict`, `none`) |
| `KVREUSE_V6_MIN_EXTEND` | `16` | tokens at the end of each block that are prefilled rather than relocated |
| `KVREUSE_SIDE_SESSIONS` | `12` | conversations held in the side buffer; set it to at least the number of concurrent conversations |
| `KVREUSE_SIDE_TOKENS` | `30000` | tokens held per conversation |
| `KVREUSE_MAX_SESSIONS` | `16` | conversations tracked for session matching (LRU) |
| `KVREUSE_TRACE` | `1` | one `kv6trace ev=req` line per request (session id, previous prompt length, and the first 48 prompt tokens as a tag) |
| `KVREUSE_DEBUG` | `1` | per-request planning lines |

The side buffer is a separate GPU allocation of `SIDE_SESSIONS x SIDE_TOKENS`
tokens of full-attention K/V; the server logs its size at the first request
(`side buffer: 12 slots x 30000 tokens (21.97 GiB, 64.0 KiB/token)` for
Qwen3.6-27B at the defaults). Leave room for it with `--mem-fraction-static` (the
example uses 0.75).

## Analysis

```bash
python -m suffix_cache_reuse.serve ... > server.log 2>&1
python analysis/scr_report.py server.log --csv requests.csv --json summary.json [--num-tasks N]
```

Every prompt token is served in exactly one of three ways, and the script reports
all three per request:

```
prompt_tokens = radix_matched + relocated + forwarded
```

| column | source |
|---|---|
| `rid`, `prompt_tokens`, `decode` | SGLang's `Finish:` line (`id`, `prompt_tokens`, `completion_tokens`) |
| `relocated`, `blocks` | sum of `C'` over the request's `v6 splice OK` lines, and their count |
| `radix_matched` | `cached_tokens - 2 x relocated` |
| `forwarded` | `prompt_tokens - radix_matched - relocated`, the tokens the model ran |
| `session`, `prev_prompt_tokens`, `class` | the `kv6trace ev=req` line; `class` is `edited` when the radix match stops more than 64 tokens short of the previous prompt, `append` otherwise, `first` for a new conversation |
| `prefix_reuse_pflops` | `2 x N_body x (forwarded + decode) / 1e15` |

On a request with relocated blocks SGLang's `cached_tokens` counts the relocated
rows twice (through the scheduler's prefix length and through SCR's own increment
when it splices a block), so the radix match is `cached_tokens - 2 x relocated`.

The summary gives the relocated share of prompt tokens and of non-radix tokens and
the prefix-reuse PFLOPs in total, per request and, with `--num-tasks`, per task.
`N_body` is the non-embedding parameter count (`--n-body-params`, default
24.3532e9 for Qwen3.6-27B). The script re-derives the token identity for every
request and the PFLOPs total two ways, and exits with status 2 if a check fails.
It reads logs from servers with SCR on or off, so the same command gives the
comparison row for plain SGLang.

## Using SCR with an agent

Point the agent's OpenAI-compatible client at the server
(`http://<host>:30000/v1`, model `qwen36-27b` in the example). The CLM agent and
other harnesses need no changes:

- **No session header is needed.** SCR attaches each request to its conversation
  by content: the tracked conversation with the longest common prefix (at least
  1024 tokens) and a high overall token overlap.
- **Request ids (optional).** To join server-side numbers with client-side logs,
  send a unique `rid` in the request body (`extra_body={"rid": "<conversation>-<turn>"}`
  with the OpenAI client). SGLang echoes it as the request id in the `Finish:` line
  and SCR uses it in its own log lines; it has no effect on scheduling or caching.
- **Per-request switch (optional).** `extra_body={"cache_salt": "kvreuse:off"}`
  serves that request as plain SGLang. `cache_salt` is also part of SGLang's radix
  cache key, so use it for whole conversations.
- **Isolation.** Like SGLang's radix cache, SCR keys conversations by `cache_salt`:
  requests with different salts never share relocated K/V or recurrent state.

## Supported versions and models

- SGLang 0.5.16.
- Qwen3.6-27B (hybrid GatedDeltaNet + full attention) with `KVREUSE_SSM_MODE=fork`,
  on one GPU (`--tp-size 1`), with the server arguments in
  `examples/serve_qwen36.sh`.

## Tests

```bash
python tests/unit_test.py                                   # CPU only, no server
python tests/smoke.py --port 30000                          # server with SCR on or off
python tests/regression.py --port 30000 --log server.log    # server with SCR on
python analysis/scr_report.py server.log
```

- `tests/unit_test.py` checks session matching by `cache_salt` and the recurrent state
  chosen when a planned relocation falls back to normal prefill.
- `tests/smoke.py` sends one three-turn conversation whose later turns delete sections
  from the middle of the first user message. With SCR on, the report shows relocated
  tokens on turns 2 and 3; with SCR off the same requests are served by the radix cache
  and prefill.
- `tests/regression.py` runs short conversations of the same shape and checks the server
  log: a turn under another `cache_salt` reuses nothing from the original conversation,
  a turn that finishes on its first token leaves no saved state for the next turn to
  reuse, and an ordinary conversation is relocated as usual.

## Layout

| path | contents |
|---|---|
| `suffix_cache_reuse/overlay.py` | the SGLang patch: session matching, prompt diff and span planning, K/V relocation with RoPE re-rotation, recurrent-state handling, fallback |
| `suffix_cache_reuse/config.py` | settings and their defaults |
| `suffix_cache_reuse/serve.py`, `_site/` | launcher and the `sitecustomize.py` that loads the patch in every SGLang process |
| `analysis/scr_report.py` | per-request token accounting and prefix-reuse FLOPs from the server log |
| `examples/serve_qwen36.sh` | Qwen3.6-27B server on one GPU |
| `tests/` | the tests above |

## License

See the repository [LICENSE](../../LICENSE).
