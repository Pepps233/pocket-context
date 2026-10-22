#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression checks against a running SCR server and its log.

    python tests/regression.py --port 30000 --log server.log

Each check runs a short conversation whose later turns delete sections from the middle
of the first user message, then reads the server log for the request ids:

  salt       a turn sent with a different cache_salt does not reuse the other
             conversation's state; the same turn with the original salt does
  stale_ssm  an appended turn (one prefill chunk) that finishes on its first token drops
             the conversation's saved recurrent state, so the next edited turn is not
             spliced onto it; with a normal middle turn the next edited turn is spliced
  normal     an ordinary conversation drops no saved state

Exits non-zero if a check fails.
"""
import argparse
import re
import sys
import time
import uuid

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from smoke import post, section  # noqa: E402

SYSTEM = ("You are a careful assistant that answers questions about a field survey. "
          "Answer in one short sentence.")


class Conv:
    count = 0

    def __init__(self, a, name, salt=None, n=24):
        self.a, self.name, self.salt = a, f"{name}-{uuid.uuid4().hex[:6]}", salt
        Conv.count += 1
        base = 100 * Conv.count  # distinct text per conversation: no cross-conversation cache hits
        self.secs = [section(base + i) for i in range(n)]
        self.hist = []

    def turn(self, t, kept, question, max_tokens=32, salt=None, keep_history=True):
        doc = "\n\n".join(self.secs[i] for i in kept)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Survey document:\n\n" + doc}] + self.hist + \
               [{"role": "user", "content": question}]
        rid = f"reg-{self.name}-t{t}"
        body = {"model": self.a.model, "messages": msgs, "max_tokens": max_tokens,
                "temperature": 0, "rid": rid, "chat_template_kwargs": {"enable_thinking": False}}
        s = salt if salt is not None else self.salt
        if s is not None:
            body["cache_salt"] = s
        out = post(self.a.port, body)
        text = out["choices"][0]["message"].get("content") or ""
        if keep_history:
            self.hist += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
        return rid


def log_text(path):
    time.sleep(2)
    with open(path, errors="replace") as f:
        return f.read()


def spliced(log, rid):
    return len(re.findall(rf"v6 splice OK: rid={re.escape(rid)} ", log))


def dropped(log, rid):
    return len(re.findall(rf"ssm snapshot dropped: rid={re.escape(rid)} ", log))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="qwen36-27b")
    ap.add_argument("--log", required=True, help="the server's log file")
    a = ap.parse_args(argv)
    n, fails = 24, []

    def check(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)
        if not ok:
            fails.append(name)

    full = list(range(n))
    cut1 = [i for i in full if i != n // 3]
    cut2 = [i for i in cut1 if i not in (n // 6, n // 2)]

    # salt: turn 2 under another salt must not reuse; the same turn under the original salt does
    c = Conv(a, "salt", salt="tenant-a")
    c.turn(1, full, "Which terms does Section 3 list first?")
    r_other = c.turn(2, cut1, "Which terms does Section 20 list first?", salt="tenant-b", keep_history=False)
    r_same = c.turn(3, cut1, "Which terms does Section 20 list first?")
    log = log_text(a.log)
    check("salt/other", spliced(log, r_other) == 0, f"splices={spliced(log, r_other)} (want 0)")
    check("salt/same", spliced(log, r_same) > 0, f"splices={spliced(log, r_same)} (want >0)")

    # stale_ssm: a first-token finish in an appended middle turn blocks the next splice
    for label, mid_tokens, want_splice in (("stale_ssm/first_token", 1, False),
                                           ("stale_ssm/control", 32, True)):
        c = Conv(a, label.replace("/", "-"), salt="tenant-s")
        c.turn(1, full, "Which terms does Section 3 list first?")
        r2 = c.turn(2, full, "Which terms does Section 20 list first?", max_tokens=mid_tokens)
        r3 = c.turn(3, cut1, "Which terms does Section 1 list first?")
        log = log_text(a.log)
        d2, s3 = dropped(log, r2), spliced(log, r3)
        want_drop = 0 if want_splice else 1
        check(label, d2 == want_drop and (s3 > 0) == want_splice,
              f"turn2 dropped={d2} (want {want_drop}), turn3 splices={s3} (want {'>0' if want_splice else 0})")

    # normal: no saved state is dropped in an ordinary conversation
    c = Conv(a, "normal")
    rids = [c.turn(1, full, "Which terms does Section 3 list first?"),
            c.turn(2, cut1, "Which terms does Section 20 list first?"),
            c.turn(3, cut2, "Which terms does Section 1 list first?")]
    log = log_text(a.log)
    nd = sum(dropped(log, r) for r in rids)
    ns = sum(spliced(log, r) for r in rids)
    check("normal", nd == 0 and ns > 0, f"dropped={nd} (want 0), splices={ns} (want >0)")

    print(f"regression: {'all passed' if not fails else 'FAILED ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
