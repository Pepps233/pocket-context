#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end smoke test against a running SGLang server (SCR on or off).

    python tests/smoke.py --port 30000 [--model qwen36-27b] [--rid-prefix smoke]

Runs one conversation of three turns through /v1/chat/completions:
  turn 1  a long document (numbered sections) and a question
  turn 2  the same conversation with one middle section deleted, plus a new question
  turn 3  two more separated sections deleted, plus a new question
Turns 2 and 3 are the shape SCR targets: the sections after each deletion are
unchanged, so with SCR on they are relocated instead of prefilled (turn 3 can
relocate more than one block). Every request carries a rid, so the server log
lines can be matched to the turns:

    python analysis/scr_report.py server.log

Exits non-zero if any request fails or returns an empty completion.
"""
import argparse
import json
import sys
import urllib.request
import uuid

WORDS = ("amber basalt cobalt delta ember fjord granite harbor indigo juniper kelp "
         "lagoon meadow nickel onyx prairie quartz river summit tundra umber valley "
         "willow xenon yarrow zephyr").split()


def section(i, n_words=110):
    body = " ".join(WORDS[(i * 7 + j * 3) % len(WORDS)] + str((i * 31 + j) % 97) for j in range(n_words))
    return f"## Section {i}\nRecord {i} lists the following survey terms: {body}."


def post(port, payload, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="qwen36-27b")
    ap.add_argument("--rid-prefix", default="smoke")
    ap.add_argument("--sections", type=int, default=24)
    a = ap.parse_args(argv)

    system = ("You are a careful assistant that answers questions about a field survey. "
              "Answer in one short sentence.\n\n" + "\n\n".join(section(1000 + i) for i in range(6)))
    secs = [section(i) for i in range(a.sections)]
    conv = uuid.uuid4().hex[:8]

    def ask(turn, kept, question, history):
        doc = "\n\n".join(secs[i] for i in kept)
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": "Survey document:\n\n" + doc}] + history + \
               [{"role": "user", "content": question}]
        rid = f"{a.rid_prefix}-{conv}-turn{turn}"
        out = post(a.port, {"model": a.model, "messages": msgs, "max_tokens": 48,
                            "temperature": 0, "rid": rid,
                            "chat_template_kwargs": {"enable_thinking": False}})
        text = out["choices"][0]["message"].get("content") or ""
        u = out.get("usage", {})
        print(f"{rid}: prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} "
              f"sections={len(kept)} -> {text.strip()[:80]!r}", flush=True)
        if not text.strip():
            raise SystemExit(f"{rid}: empty completion")
        return msgs[1:], text

    n = a.sections
    all_idx = list(range(n))
    q1 = "Which terms does Section 3 list first?"
    _, t1 = ask(1, all_idx, q1, [])
    h = [{"role": "user", "content": q1}, {"role": "assistant", "content": t1}]

    kept2 = [i for i in all_idx if i != n // 3]                       # delete one middle section
    q2 = f"Which terms does Section {n - 2} list first?"
    _, t2 = ask(2, kept2, q2, h)
    h += [{"role": "user", "content": q2}, {"role": "assistant", "content": t2}]

    kept3 = [i for i in kept2 if i not in (n // 6, n // 2, (2 * n) // 3)]  # delete three more, spread out
    q3 = "Which terms does Section 1 list first?"
    ask(3, kept3, q3, h)
    print("smoke: all requests succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
