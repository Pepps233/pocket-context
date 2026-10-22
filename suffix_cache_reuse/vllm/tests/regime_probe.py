#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Strip-regime witness: does the server render prior reasoning sent as `reasoning_content`?

    python tests/regime_probe.py --port P [--model qwen36-27b]

Sends the same tool-loop conversation twice, with and without `reasoning_content` on the
assistant turn (which follows the last user message, so the Qwen3.6 template keeps it), and
prints both prompt_tokens. Exit 1 if the reasoning was not rendered (difference < 50 tokens).
"""
import argparse
import json
import sys
import urllib.request


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as f:
        return json.loads(f.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen36-27b")
    a = ap.parse_args()
    tools = [{"type": "function", "function": {"name": "search", "description": "search",
              "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}}]
    reasoning = "I should look this up. " * 40
    out = {}
    for with_r in (False, True):
        asst = {"role": "assistant", "content": "",
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "search", "arguments": "{\"q\": \"x\"}"}}]}
        if with_r:
            asst["reasoning_content"] = reasoning
        msgs = [{"role": "user", "content": "Find x."}, asst,
                {"role": "tool", "tool_call_id": "call_1", "content": "x is 42."}]
        r = post(a.port, {"model": a.model, "messages": msgs, "tools": tools, "max_tokens": 4,
                          "request_id": f"regime-probe-{int(with_r)}"})
        out[with_r] = r["usage"]["prompt_tokens"]
    d = out[True] - out[False]
    print(f"regime probe: prompt_tokens without reasoning={out[False]} with={out[True]} diff={d} "
          f"-> {'prior reasoning RENDERED (strip regime)' if d >= 50 else 'prior reasoning DROPPED'}")
    return 0 if d >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())
