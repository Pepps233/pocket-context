#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Latency micro-benchmark against a running server (SCR on or off).

    python tests/bench.py --port P [--n 20] [--words 9000]

Phase "cold": n sequential single-turn requests with distinct ~words-long documents.
Phase "edit": n sequential two-turn conversations; turn 2 deletes one middle section.
Prints median/mean client latency per phase and turn (max_tokens=8, greedy).
"""
import argparse
import json
import random
import statistics
import time
import urllib.request


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as f:
        out = json.loads(f.read())
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--words", type=int, default=9000)
    ap.add_argument("--model", default="scr-model")
    ap.add_argument("--conc", type=int, default=0)
    ap.add_argument("--conc-n", type=int, default=32)
    a = ap.parse_args()
    rng = random.Random(1)
    W = "amber basalt cobalt delta ember fjord granite harbor indigo juniper kelp lagoon".split()

    def doc(tag, n_sec=20):
        return ["## Section %d (%s)\n" % (i, tag) + " ".join(
            rng.choice(W) + str(rng.randint(0, 999)) for _ in range(a.words // n_sec)) for i in range(n_sec)]

    def ask(secs, hist, q, rid):
        msgs = [{"role": "user", "content": "\n\n".join(secs)}] + hist + [{"role": "user", "content": q}]
        return post(a.port, {"model": a.model, "messages": msgs, "max_tokens": 8, "temperature": 0,
                             "request_id": rid, "chat_template_kwargs": {"enable_thinking": False}})

    res = {"cold": [], "t1": [], "t2": []}
    if a.conc:
        from concurrent.futures import ThreadPoolExecutor
        docs = [doc(f"p{i}") for i in range(a.conc_n)]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(a.conc) as ex:
            lats = list(ex.map(lambda x: ask(x[1], [], "Summarise.", f"bench-par-{x[0]}")[0], enumerate(docs)))
        print(f"bench parallel: n={a.conc_n} conc={a.conc} wall={time.perf_counter() - t0:.2f}s "
              f"median_lat={statistics.median(lats):.3f}s", flush=True)
    for i in range(a.n):
        dt, _ = ask(doc(f"c{i}"), [], "Summarise.", f"bench-cold-{i}")
        res["cold"].append(dt)
    for i in range(a.n):
        secs = doc(f"e{i}")
        dt1, o1 = ask(secs, [], "What is in Section 2?", f"bench-e{i}-t1")
        h = [{"role": "user", "content": "What is in Section 2?"},
             {"role": "assistant", "content": o1["choices"][0]["message"]["content"] or "ok"}]
        dt2, _ = ask(secs[:7] + secs[8:], h, "And Section 15?", f"bench-e{i}-t2")
        res["t1"].append(dt1)
        res["t2"].append(dt2)
    for k, v in res.items():
        print(f"bench {k}: n={len(v)} median={statistics.median(v):.3f}s mean={statistics.mean(v):.3f}s "
              f"min={min(v):.3f}s max={max(v):.3f}s")


if __name__ == "__main__":
    main()
