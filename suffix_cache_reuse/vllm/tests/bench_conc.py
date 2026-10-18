#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Concurrent multi-turn editing benchmark against a running OpenAI-compatible server.

    python tests/bench_conc.py --port P --conc 8 16 32 [--convs-per-slot 2] [--turns 6] [--json out.json]

Each conversation starts from a ~--words-word document split into sections; every turn appends a
fixed assistant reply and a tool-result-like block, and with probability --edit-rate deletes or
rewrites one earlier section (the edit SCR targets). Assistant replies are fixed text, and
every request decodes exactly --max-tokens tokens (ignore_eos), so the token workload is
identical across servers and runs; only the serving differs. The seed fixes the workload.

Per concurrency level, `conc` workers each run conversations back to back until
conc * convs-per-slot conversations are done. Reported: wall time, requests/s, output tokens/s,
TTFT and end-to-end latency (p50/p90/p99), split into first turns and later turns.
"""
import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

W = ("amber basalt cobalt delta ember fjord granite harbor indigo juniper kelp lagoon meadow "
     "nickel onyx prairie quartz river summit tundra umber valley willow xenon").split()


def stream(port, model, msgs, max_tokens, extra):
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, usage = None, {}
    with urllib.request.urlopen(req, timeout=1800) as f:
        for line in f:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            ch = ev.get("choices") or []
            if ttft is None and ch and (ch[0].get("delta") or {}).get("content") is not None:
                ttft = time.perf_counter() - t0
            if ttft is None and ch and (ch[0].get("delta") or {}).get("reasoning_content") is not None:
                ttft = time.perf_counter() - t0
    e2e = time.perf_counter() - t0
    return ttft if ttft is not None else e2e, e2e, usage


def make_conv(rng, words, n_sec):
    return ["## Section %d\n" % i + " ".join(rng.choice(W) + str(rng.randint(0, 999))
                                             for _ in range(words // n_sec)) for i in range(n_sec)]


def run_level(a, conc, seed):
    rng0 = random.Random(seed)
    n_conv = conc * a.convs_per_slot
    plans = []
    for c in range(n_conv):   # pre-generate every conversation: identical workload per seed
        rng = random.Random(rng0.random())
        secs = make_conv(rng, a.words, a.sections)
        turns = []
        for t in range(a.turns):
            ed = None
            if t > 0 and rng.random() < a.edit_rate:
                i = rng.randrange(1, len(secs) - 1)
                ed = ("del", i) if rng.random() < 0.6 else ("rw", i, "## Section rewritten\n" + " ".join(
                    rng.choice(W) for _ in range(rng.randint(20, 120))))
            tool = "Tool result: " + " ".join(rng.choice(W) + str(rng.randint(0, 99))
                                             for _ in range(rng.randint(80, 400)))
            turns.append((ed, tool))
        plans.append((secs, turns))
    recs, lock = [], threading.Lock()
    next_conv = [0]

    def worker(wid):
        while True:
            with lock:
                c = next_conv[0]
                next_conv[0] += 1
            if c >= n_conv:
                return
            secs, turns = plans[c]
            secs = list(secs)
            hist = []
            for t, (ed, tool) in enumerate(turns):
                if ed and ed[0] == "del" and ed[1] < len(secs):
                    del secs[ed[1]]
                elif ed and ed[0] == "rw" and ed[1] < len(secs):
                    secs[ed[1]] = ed[2]
                msgs = ([{"role": "system", "content": "You are a careful analyst."},
                         {"role": "user", "content": "\n\n".join(secs)}] + hist
                        + [{"role": "user", "content": f"Step {t}: summarise what changed."}])
                ttft, e2e, usage = stream(a.port, a.model, msgs, a.max_tokens, a.extra)
                with lock:
                    recs.append({"conv": c, "turn": t, "edited": bool(ed), "ttft": ttft, "e2e": e2e,
                                 "prompt": usage.get("prompt_tokens", 0),
                                 "completion": usage.get("completion_tokens", 0),
                                 "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                                 "t_end": time.perf_counter()})
                hist += [{"role": "user", "content": f"Step {t}: summarise what changed."},
                         {"role": "assistant", "content": f"Summary of step {t}: noted."},
                         {"role": "user", "content": tool}]

    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    wall = time.perf_counter() - t0

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")

    out = {"conc": conc, "conversations": n_conv, "requests": len(recs), "wall_s": round(wall, 2),
           "req_per_s": round(len(recs) / wall, 3),
           "out_tok_per_s": round(sum(r["completion"] for r in recs) / wall, 1),
           "prompt_tok": sum(r["prompt"] for r in recs)}
    for name, sel in (("first", lambda r: r["turn"] == 0), ("later", lambda r: r["turn"] > 0),
                      ("edited", lambda r: r["edited"])):
        rr = [r for r in recs if sel(r)]
        for m in ("ttft", "e2e"):
            v = [r[m] for r in rr]
            out[f"{name}_{m}_p50"] = round(q(v, .5), 3)
            out[f"{name}_{m}_p90"] = round(q(v, .9), 3)
            out[f"{name}_{m}_p99"] = round(q(v, .99), 3)
        out[f"{name}_n"] = len(rr)
    return out, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="scr-model")
    ap.add_argument("--conc", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--convs-per-slot", type=int, default=2)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--words", type=int, default=6000)
    ap.add_argument("--sections", type=int, default=24)
    ap.add_argument("--edit-rate", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-thinking", action="store_true", help="send enable_thinking=False")
    ap.add_argument("--json")
    a = ap.parse_args()
    a.extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_thinking else {}
    results = []
    for i, c in enumerate(a.conc):
        res, recs = run_level(a, c, a.seed + 1000 * i)
        print("bench_conc " + json.dumps(res), flush=True)
        results.append({"summary": res, "requests": recs})
    if a.json:
        json.dump(results, open(a.json, "w"))


if __name__ == "__main__":
    main()
