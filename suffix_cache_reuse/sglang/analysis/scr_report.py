#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-request token provenance and prefix-reuse FLOPs from SGLang server logs.

    python analysis/scr_report.py server.log [more.log ...] \
        [--csv requests.csv] [--json summary.json] \
        [--n-body-params 24.3532e9] [--num-tasks N]

Inputs are server logs written with `--log-requests --log-requests-level 0`
(added by `python -m suffix_cache_reuse.serve`). The script works on logs from
servers with SCR on or off; with SCR off every request has relocated = 0.

Every prompt token of a request is served in exactly one of three ways:

    prompt_tokens = radix_matched + relocated + forwarded

  radix_matched  served by SGLang's radix (prefix) cache
  relocated      copied from SCR's side buffer and re-rotated to a new position
  forwarded      run through the model (prefill)

Sources, per request id (rid):
  SGLang `Finish:` line   prompt_tokens, cached_tokens, completion_tokens
  `v6 splice OK: rid=..`  C'=<rows>, one line per relocated block
  `kv6trace ev=req rid=..` session id and the previous prompt length of the
                           same conversation (KVREUSE_TRACE=1)

On a request with relocated blocks, SGLang's `cached_tokens` includes the
relocated rows twice: once through the prefix length that the scheduler counts,
and once through the overlay's own increment when it splices a block. Hence

    radix_matched = cached_tokens - 2 * relocated
    forwarded     = prompt_tokens - radix_matched - relocated

Prefix-reuse FLOPs count only tokens the model actually runs:

    prefix-reuse FLOPs = 2 * N_body * (forwarded + decode)

where N_body is the model's non-embedding parameter count (default: 24.3532e9,
Qwen3.6-27B) and decode = completion tokens. The summary reports it per request
and, with --num-tasks, per task. The script re-derives the total two ways and
exits with status 2 if any check fails.
"""
import argparse
import collections
import csv
import json
import re
import sys

EDIT_TOL = 64                         # tokens; see `edited` below
QWEN36_27B_N_BODY = 24.3532e9         # non-embedding parameters of Qwen3.6-27B

FINISH_ID = re.compile(r"'id': '([^']+)'")
FINISH_PT = re.compile(r"'prompt_tokens': (\d+)")
FINISH_CT = re.compile(r"'cached_tokens': (\d+)")
FINISH_CP = re.compile(r"'completion_tokens': (\d+)")
SPLICE = re.compile(r"v6 splice OK: rid=(?P<rid>\S+) sid=(?P<sid>\S+) slot=\S+ "
                    r"A\+B'=\d+ C'=(?P<nc>\d+)")
TRACE = re.compile(r"kv6trace ev=req rid=(?P<rid>\S+) sid=(?P<sid>\S+) slot=\S+ "
                   r"n_new=(?P<n_new>\d+) n_old=(?P<n_old>\d+)")
BANNER = re.compile(r"\[kvreuse-site\] .*?max_blocks=(\d+).*?splice_retry=(\d+)")

COLUMNS = ["rid", "log", "session", "prev_prompt_tokens", "class", "prompt_tokens",
           "radix_matched", "relocated", "forwarded", "decode", "blocks",
           "prefix_reuse_pflops"]


def parse_log(path):
    """-> (ordered request rows, banner info) for one server log."""
    fin, order = {}, []
    reloc, blocks = collections.Counter(), collections.Counter()
    trace = {}
    banner = None
    with open(path, errors="replace") as fh:
        for ln in fh:
            if "Finish:" in ln:
                m_id, m_pt = FINISH_ID.search(ln), FINISH_PT.search(ln)
                if not (m_id and m_pt):
                    continue
                rid = m_id.group(1)
                if rid in fin:
                    continue
                m_ct, m_cp = FINISH_CT.search(ln), FINISH_CP.search(ln)
                fin[rid] = {"prompt_tokens": int(m_pt.group(1)),
                            "cached_tokens": int(m_ct.group(1)) if m_ct else 0,
                            "decode": int(m_cp.group(1)) if m_cp else 0}
                order.append(rid)
            elif "v6 splice OK" in ln:
                m = SPLICE.search(ln)
                if m:
                    reloc[m.group("rid")] += int(m.group("nc"))
                    blocks[m.group("rid")] += 1
            elif "kv6trace ev=req " in ln:
                m = TRACE.search(ln)
                if m and m.group("rid") not in trace:
                    trace[m.group("rid")] = (m.group("sid"), int(m.group("n_old")))
            elif banner is None and "[kvreuse-site]" in ln:
                m = BANNER.search(ln)
                if m:
                    banner = {"max_blocks": int(m.group(1)), "splice_retry": int(m.group(2))}
    rows = []
    for rid in order:
        f = fin[rid]
        rel = reloc.get(rid, 0)
        hit = f["cached_tokens"] - 2 * rel
        fwd = f["prompt_tokens"] - hit - rel
        sid, n_old = trace.get(rid, ("", None))
        if n_old is None:
            cls = ""
        elif n_old == 0:
            cls = "first"
        elif hit < n_old - EDIT_TOL:
            cls = "edited"        # the cached prefix stops short of the previous prompt
        else:
            cls = "append"        # the prompt extends the previous prompt
        rows.append({"rid": rid, "log": str(path), "session": sid,
                     "prev_prompt_tokens": "" if n_old is None else n_old, "class": cls,
                     "prompt_tokens": f["prompt_tokens"], "radix_matched": hit,
                     "relocated": rel, "forwarded": fwd, "decode": f["decode"],
                     "blocks": blocks.get(rid, 0)})
    return rows, banner


def summarize(rows, flops_per_token, num_tasks=None):
    n = len(rows)
    tot = collections.Counter()
    for r in rows:
        for k in ("prompt_tokens", "radix_matched", "relocated", "forwarded", "decode", "blocks"):
            tot[k] += r[k]
    new = tot["prompt_tokens"] - tot["radix_matched"]
    pflops = flops_per_token * (tot["forwarded"] + tot["decode"]) / 1e15
    s = {
        "requests": n,
        "requests_with_relocation": sum(1 for r in rows if r["relocated"] > 0),
        "relocated_blocks": tot["blocks"],
        "prompt_tokens": tot["prompt_tokens"],
        "radix_matched": tot["radix_matched"],
        "relocated": tot["relocated"],
        "forwarded": tot["forwarded"],
        "decode": tot["decode"],
        "relocated_share_of_prompt": tot["relocated"] / tot["prompt_tokens"] if tot["prompt_tokens"] else 0.0,
        "relocated_share_of_non_radix": tot["relocated"] / new if new else 0.0,
        "forwarded_share_of_prompt": tot["forwarded"] / tot["prompt_tokens"] if tot["prompt_tokens"] else 0.0,
        "flops_per_token": flops_per_token,
        "prefix_reuse_pflops_total": pflops,
        "prefix_reuse_pflops_per_request": pflops / n if n else 0.0,
    }
    if num_tasks:
        s["tasks"] = num_tasks
        s["prefix_reuse_pflops_per_task"] = pflops / num_tasks
    by_cls = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["class"]:
            c = by_cls[r["class"]]
            c["requests"] += 1
            for k in ("prompt_tokens", "radix_matched", "relocated", "forwarded"):
                c[k] += r[k]
    if by_cls:
        s["by_class"] = {k: dict(v) for k, v in sorted(by_cls.items())}
    return s


def self_check(rows, summary, flops_per_token):
    """Return a list of failed checks (empty when everything holds)."""
    bad = []
    for r in rows:
        if r["radix_matched"] < 0 or r["forwarded"] < 0:
            bad.append(f"rid {r['rid']}: negative radix_matched/forwarded {r}")
        if r["radix_matched"] + r["relocated"] + r["forwarded"] != r["prompt_tokens"]:
            bad.append(f"rid {r['rid']}: radix_matched + relocated + forwarded != prompt_tokens")
    new = summary["prompt_tokens"] - summary["radix_matched"]
    if new != summary["relocated"] + summary["forwarded"]:
        bad.append("sum(prompt - radix_matched) != sum(relocated) + sum(forwarded)")
    n = summary["requests"]
    if n:
        # (a) sum of per-request FLOPs, (b) LIN x (forwarded + decode) / n from the totals
        per_row = sum(r["prefix_reuse_pflops"] for r in rows) / n
        from_totals = flops_per_token * (summary["forwarded"] + summary["decode"]) / 1e15 / n
        got = summary["prefix_reuse_pflops_per_request"]
        for name, want in (("per-request sum", per_row), ("LIN x (forwarded + decode) / n", from_totals)):
            if abs(got - want) > 1e-3 * max(abs(want), 1e-12):
                bad.append(f"prefix_reuse_pflops_per_request {got!r} != {name} {want!r}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+", help="SGLang server log file(s)")
    ap.add_argument("--csv", help="write the per-request table here")
    ap.add_argument("--json", help="write the summary here")
    ap.add_argument("--n-body-params", type=float, default=QWEN36_27B_N_BODY,
                    help="non-embedding parameter count N_body (default: Qwen3.6-27B, 24.3532e9)")
    ap.add_argument("--num-tasks", type=int, help="number of tasks/questions, for per-task FLOPs")
    a = ap.parse_args(argv)

    lin = 2.0 * a.n_body_params
    rows, banners = [], {}
    for p in a.logs:
        r, b = parse_log(p)
        rows += r
        banners[p] = b
    for r in rows:
        r["prefix_reuse_pflops"] = lin * (r["forwarded"] + r["decode"]) / 1e15
    s = summarize(rows, lin, a.num_tasks)
    s["scr"] = {p: (b if b else "off") for p, b in banners.items()}

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(s, fh, indent=2)

    print(f"logs: {len(a.logs)}  requests: {s['requests']}  "
          f"with relocation: {s['requests_with_relocation']}  relocated blocks: {s['relocated_blocks']}")
    for p, b in s["scr"].items():
        print(f"  {p}: SCR {'off' if b == 'off' else 'K=%d, splice_retry=%d' % (b['max_blocks'], b['splice_retry'])}")
    pt = s["prompt_tokens"] or 1
    print(f"prompt tokens    {s['prompt_tokens']:>14,}")
    print(f"  radix matched  {s['radix_matched']:>14,}  {s['radix_matched'] / pt:7.2%}")
    print(f"  relocated      {s['relocated']:>14,}  {s['relocated'] / pt:7.2%}"
          f"   ({s['relocated_share_of_non_radix']:.2%} of non-radix tokens)")
    print(f"  forwarded      {s['forwarded']:>14,}  {s['forwarded'] / pt:7.2%}")
    print(f"decode tokens    {s['decode']:>14,}")
    print(f"prefix-reuse PFLOPs: total {s['prefix_reuse_pflops_total']:.4f}, "
          f"per request {s['prefix_reuse_pflops_per_request']:.6f}"
          + (f", per task {s['prefix_reuse_pflops_per_task']:.4f}" if a.num_tasks else "")
          + f"  (2 x N_body = {lin:.5e} FLOPs/token)")
    for cls, c in (s.get("by_class") or {}).items():
        print(f"  {cls:<7} requests {c['requests']:>6}  prompt {c['prompt_tokens']:>12,}  "
              f"radix {c['radix_matched']:>12,}  relocated {c['relocated']:>10,}  forwarded {c['forwarded']:>12,}")

    bad = self_check(rows, s, lin)
    if bad:
        print("SELF-CHECK FAILED:", file=sys.stderr)
        for b in bad[:20]:
            print("  " + b, file=sys.stderr)
        return 2
    print("self-check: OK (per-request token identity; totals; PFLOPs = 2 x N_body x (forwarded + decode) / n)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
