# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""API compatibility shim, applied in every vLLM process started by vllm_scr.serve (SCR on or off).

vLLM 0.19.1 declares the assistant-message field `reasoning_content` (deprecated) but renders
only `reasoning` into the chat template (entrypoints/chat_utils.py:_parse_chat_message_content).
Many clients (e.g. litellm) send prior reasoning back as `reasoning_content`, which SGLang
renders. Without this shim the Qwen3.6 template never sees prior reasoning, so the server
silently runs a different regime than the template intends (reasoning kept for turns after the
last user message, stripped at the next user message). The shim copies `reasoning_content` into
`reasoning` when only the former is set.
KVREUSE_COMPAT_REASONING_CONTENT=0 disables it.
"""
import os


def apply():
    if os.environ.get("KVREUSE_COMPAT_REASONING_CONTENT", "1") != "1":
        return False
    from vllm.entrypoints import chat_utils as cu
    if getattr(cu, "_vllm_scr_compat", False):
        return True
    orig = cu._parse_chat_message_content

    def _parse(message, *a, **kw):
        if isinstance(message, dict) and message.get("role") == "assistant" \
                and message.get("reasoning") is None and message.get("reasoning_content") is not None:
            message = dict(message, reasoning=message["reasoning_content"])
        return orig(message, *a, **kw)

    cu._parse_chat_message_content = _parse
    cu._vllm_scr_compat = True
    return True
