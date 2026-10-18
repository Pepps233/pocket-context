#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Greedy agreement through a running SCR server (works for any model and any TP size).

    python tests/greedy_server.py --port P --log server.log [--n 8] [--tokens 48]

For each of n conversations: turn 1 = [A][B][C1][X][C2] + question; turn 2 deletes B and X and
asks another question, so C1 and C2 move and SCR relocates them. Turn 2 is sent twice: once
normally (SCR) and once with cache_salt "kvreuse:off-<i>" (SCR off for that request and no
prefix-cache sharing, i.e. a cold prefill). The greedy continuations (--tokens, ignore_eos) are
compared, each answer is checked against the known last term of section C2, and the server log's
`[scr] fin` line confirms that the SCR request relocated tokens and the cold one did not.
SCR is an approximation (relocated entries were computed with B and X still in the context), so
continuations may diverge after a few tokens; identical continuations are typical for dense
models, less so for others. Exit code 1 if a request fails, an SCR turn did not relocate or a
cold turn relocated.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request

W = ("amber basalt cobalt delta ember fjord granite harbor indigo juniper kelp lagoon meadow "
     "nickel onyx prairie quartz river summit tundra umber valley willow xenon").split()


def post(port, model, msgs, n_tok, rid, salt=None, extra=None):
    body = {"model": model, "messages": msgs, "max_tokens": n_tok, "temperature": 0,
            "ignore_eos": True, "request_id": rid, "logprobs": True, "top_logprobs": 1}
    if salt:
        body["cache_salt"] = salt
    body.update(extra or {})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as f:
        out = json.loads(f.read())
    ch = out["choices"][0]
    toks = [t["token"] for t in (ch.get("logprobs") or {}).get("content") or []]
    return ch["message"].get("content") or "", toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--model", default="scr-model")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.no_thinking else {}
    rng = random.Random(0)

    def sec(tag, n):
        return f"## {tag}\n" + " ".join(rng.choice(W) + str(rng.randint(0, 99)) for _ in range(n))

    res = []
    for i in range(a.n):
        A, B, C1, X, C2 = sec("A", 1500), sec("B", 250), sec("C1", 700), sec("X", 80), sec("C2", 600)
        q1 = "Which term is listed first in section C1? Answer briefly."
        q2 = "Which term is listed last in section C2? Answer briefly, then explain."
        m1 = [{"role": "user", "content": "\n\n".join([A, B, C1, X, C2]) + "\n\n" + q1}]
        o1, _ = post(a.port, a.model, m1, 16, f"gs{i}-t1", extra=extra)
        m2 = [{"role": "user", "content": "\n\n".join([A, C1, C2]) + "\n\n" + q2}]
        s_txt, s_tok = post(a.port, a.model, m2, a.tokens, f"gs{i}-t2scr", extra=extra)
        c_txt, c_tok = post(a.port, a.model, m2, a.tokens, f"gs{i}-t2cold", salt=f"kvreuse:off-{i}",
                            extra=extra)
        common = next((k for k, (x, y) in enumerate(zip(s_tok, c_tok)) if x != y), min(len(s_tok), len(c_tok)))
        truth = C2.split()[-1]
        res.append({"conv": i, "identical": s_tok == c_tok, "common_prefix_tokens": common,
                    "len": len(s_tok), "truth": truth, "scr_correct": truth in s_txt,
                    "cold_correct": truth in c_txt, "scr": s_txt[:120], "cold": c_txt[:120]})
    time.sleep(2)
    fin = {}
    for line in open(a.log, errors="ignore"):
        m = re.search(r"\[scr\] fin rid=\S*?(gs\d+-t2\w+)\S* .* reloc=(\d+)", line)
        if m:
            fin[m.group(1)] = int(m.group(2))
    for r in res:
        r["relocated"] = fin.get(f"gs{r['conv']}-t2scr", -1)
        r["cold_relocated"] = fin.get(f"gs{r['conv']}-t2cold", -1)
    rel = [r for r in res if r["relocated"] > 0]
    ok = len(rel) == len(res) and all(r["cold_relocated"] == 0 for r in res)
    summ = {"n": len(res), "relocated": len(rel), "identical_relocated": sum(r["identical"] for r in rel),
            "identical_all": sum(r["identical"] for r in res),
            "min_common_prefix": min((r["common_prefix_tokens"] for r in res), default=0),
            "cold_never_relocated": all(r["cold_relocated"] == 0 for r in res),
            "scr_correct": sum(r["scr_correct"] for r in res), "cold_correct": sum(r["cold_correct"] for r in res),
            "pass": ok}
    for r in res:
        print(json.dumps(r))
    print("greedy_server " + json.dumps(summ))
    if a.json:
        json.dump({"summary": summ, "convs": res}, open(a.json, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
