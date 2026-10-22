# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loaded in every Python process started by vllm_scr.serve (directory put on PYTHONPATH):
applies vllm_scr.compat in the API server and renderer processes. Then runs the next
sitecustomize on sys.path, if any."""
import os
import sys

try:
    from vllm_scr import compat
    if compat.apply():
        print("[scr] compat: assistant reasoning_content -> reasoning", file=sys.stderr, flush=True)
except Exception as e:  # never break start-up
    print(f"[scr] compat not applied: {e}", file=sys.stderr, flush=True)

_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    _d = os.path.abspath(_p or ".")
    _f = os.path.join(_d, "sitecustomize.py")
    if _d != _here and os.path.isfile(_f):
        import importlib.util
        _spec = importlib.util.spec_from_file_location("_scr_next_sitecustomize", _f)
        _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
        break
