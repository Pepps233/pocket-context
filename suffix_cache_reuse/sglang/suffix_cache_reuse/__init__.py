# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Suffix Cache Reuse: KV reuse for context-editing agents, as a patch to SGLang.

Nothing is patched when this package is imported. The patch lives in
`suffix_cache_reuse.overlay` and is installed in every SGLang process by
`python -m suffix_cache_reuse.serve`, which puts `suffix_cache_reuse/_site`
(a `sitecustomize.py`) on PYTHONPATH and then starts `sglang.launch_server`.
"""

__version__ = "1.0.0"
SGLANG_VERSION = "0.5.16"
