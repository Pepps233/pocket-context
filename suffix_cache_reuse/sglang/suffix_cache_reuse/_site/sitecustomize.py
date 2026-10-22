# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Installs Suffix Cache Reuse in every Python process started with this directory
on PYTHONPATH. SGLang starts its scheduler in separate processes, and
sitecustomize is imported at interpreter start-up in each of them, so the patch
is present wherever the scheduler runs.
"""
import os
import sys

from suffix_cache_reuse.config import apply_defaults

apply_defaults()

import suffix_cache_reuse.overlay  # noqa: E402,F401  (patches SGLang on import)

# Run the next sitecustomize on sys.path, if the environment has one.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in sys.path:
    _d = os.path.abspath(_p or ".")
    _f = os.path.join(_d, "sitecustomize.py")
    if _d != _here and os.path.isfile(_f):
        import importlib.util

        _spec = importlib.util.spec_from_file_location("_scr_next_sitecustomize", _f)
        _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
        break
