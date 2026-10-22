#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-request token provenance and prefix-reuse FLOPs from vLLM-SCR server logs.

    python analysis/scr_report.py server.log [more.log ...] [--csv requests.csv] [--json s.json]
        [--n-body-params 7.0e9] [--num-tasks N]

Reads the `[scr] fin ...` line the SCR scheduler writes for every finished request (SCR on or
off). Every prompt token is served one way:

    prompt_tokens = radix_matched + relocated + forwarded

  radix_matched  vLLM prefix-cache hit at first scheduling (request.num_cached_tokens)
  relocated      copied from the conversation's held KV and re-rotated (SCR)
  forwarded      prompt tokens the model ran, counted per scheduled step

The identity is checked per request (it can fail only for preempted requests, which recompute).
Columns match suffix_cache_reuse/analysis/scr_report.py; rid is the client request_id
(vLLM's `chatcmpl-<id>-<8 hex>` engine id with prefix/suffix stripped).
prefix-reuse FLOPs = 2 * N_body * (forwarded + decode).
"""
import argparse
import collections
import csv
import json
import re
import sys

EDIT_TOL = 64
FIN = re.compile(r"\[scr\] fin (.*)$")
KV = re.compile(r"(\w+)=(\S+)")
COLUMNS = ["rid", "log", "session", "prev_prompt_tokens", "class", "prompt_tokens",
           "radix_matched", "relocated", "forwarded", "decode", "blocks", "planned",
           "preempt", "reason", "fallback", "plan_ms", "prefix_reuse_pflops"]


def client_rid(rid):
    rid = re.sub(r"^(chatcmpl|cmpl)-", "", rid)
    return re.sub(r"-[0-9a-f]{8}$", "", rid)


def parse(path):
    rows = []
    with open(path, errors="replace") as fh:
        for ln in fh:
            m = FIN.search(ln)
            if not m:
                continue
            d = dict(KV.findall(m.group(1)))
            rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    ap.add_argument("--n-body-params", type=float, default=6.95e9)  # Qwen3-8B non-embedding
    ap.add_argument("--num-tasks", type=int)
    a = ap.parse_args()
    out, bad = [], 0
    for lg in a.logs:
        for d in parse(lg):
            P, hit, rel, fwd = (int(d[k]) for k in ("prompt", "hit", "reloc", "fwd"))
            dec = int(d["decode"])
            prev = int(d["prev"])
            if int(d.get("preempt", 0)) == 0 and P != hit + rel + fwd:
                bad += 1
            # `edited` by content: the prompt does not extend the conversation's previous prompt
            # (common prefix stops > 64 tokens short). Logs without lcp fall back to the SGLang
            # cache-based rule (hit < prev - 64), which is engine dependent.
            lcp = int(d["lcp"]) if "lcp" in d else hit
            cls = "first" if prev == 0 else ("edited" if lcp < prev - EDIT_TOL else "append")
            out.append({"rid": client_rid(d["rid"]), "log": lg, "session": d["sid"],
                        "prev_prompt_tokens": prev, "class": cls, "prompt_tokens": P,
                        "radix_matched": hit, "relocated": rel, "forwarded": fwd, "decode": dec,
                        "blocks": int(d["blocks"]), "planned": int(d["planned"]),
                        "preempt": int(d.get("preempt", 0)), "reason": d.get("reason"),
                        "fallback": d.get("fallback"), "plan_ms": float(d.get("plan_ms", 0)),
                        "prefix_reuse_pflops": 2 * a.n_body_params * (fwd + dec) / 1e15})
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(out)
    tot = collections.Counter()
    for r in out:
        for k in ("prompt_tokens", "radix_matched", "relocated", "forwarded", "decode"):
            tot[k] += r[k]
            tot[f"{r['class']}_{k}"] += r[k]
        tot[f"n_{r['class']}"] += 1
        tot["n_reloc_req"] += r["relocated"] > 0
    n = len(out)
    pfl = sum(r["prefix_reuse_pflops"] for r in out)
    pfl2 = 2 * a.n_body_params * (tot["forwarded"] + tot["decode"]) / 1e15
    ed = [r for r in out if r["class"] == "edited"]
    summ = {"requests": n, "identity_failures": bad, **dict(tot),
            "edited_requests": len(ed),
            "edited_requests_relocated": sum(r["relocated"] > 0 for r in ed),
            "coverage_edited_relocated_frac": (sum(r["relocated"] > 0 for r in ed) / len(ed)) if ed else None,
            "relocated_share_of_prompt": tot["relocated"] / max(1, tot["prompt_tokens"]),
            "relocated_share_of_nonradix": tot["relocated"] / max(1, tot["prompt_tokens"] - tot["radix_matched"]),
            "prefix_reuse_pflops_total": pfl, "prefix_reuse_pflops_per_request": pfl / max(1, n)}
    if a.num_tasks:
        summ["prefix_reuse_pflops_per_task"] = pfl / a.num_tasks
    for k in ("prompt_tokens", "radix_matched", "relocated", "forwarded"):
        pass
    print(f"requests={n} prompt={tot['prompt_tokens']} radix={tot['radix_matched']} "
          f"relocated={tot['relocated']} forwarded={tot['forwarded']} decode={tot['decode']} "
          f"| classes first={tot['n_first']} append={tot['n_append']} edited={tot['n_edited']} "
          f"| edited relocated {summ['edited_requests_relocated']}/{len(ed)}")
    print(f"prefix-reuse PFLOPs: total {pfl:.4f}, per request {pfl / max(1, n):.6f}")
    ok = bad == 0 and abs(pfl - pfl2) < 1e-6 * max(1, pfl)
    print(f"self-check: {'OK' if ok else 'FAIL'} (identity failures {bad})")
    if a.json:
        json.dump(summ, open(a.json, "w"), indent=1)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
