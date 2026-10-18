# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SCR settings, read from KVREUSE_* environment variables (names mirror the SGLang version).

Defaults are the K=6 setting of the SGLang release (suffix_cache_reuse/config.py) where a
variable has the same meaning in vLLM. A variable already set in the environment wins.
"""
import os

DEFAULTS = {
    "KVREUSE_ENABLED": "1",            # 0: stock vLLM scheduling (accounting log only)
    "KVREUSE_MAX_BLOCKS": "6",         # K: surviving spans relocated per request
    "KVREUSE_V6_MIN_EXTEND": "16",     # last tokens of each span prefilled, not relocated
    "KVREUSE_V6_MIN_TAIL": "64",       # minimum surviving-span length considered
    "KVREUSE_MAX_SESSIONS": "16",      # conversations tracked for matching (LRU)
    "KVREUSE_SIDE_SESSIONS": "12",     # conversations whose previous-turn KV is held
    "KVREUSE_SIDE_TOKENS": "30000",    # max committed tokens held per conversation
    "KVREUSE_BUDGET_FRAC": "0.25",     # max fraction of KV blocks held by the side buffer
    "KVREUSE_MATCH_MIN_TOKENS": "1024",
    "KVREUSE_MATCH_WINDOW": "8192",
    "KVREUSE_MATCH_MIN_RATIO": "0.55",
    "KVREUSE_MATCH_MIN_PREFIX_FRAC": "0.5",
    "KVREUSE_MATCH_MIN_OVERLAP": "0.5",
    "KVREUSE_MATCH_AMBIG_MARGIN": "0.15",
    "KVREUSE_FASTDIFF": "1",
    "KVREUSE_FASTDIFF_MIN": "2048",
    "KVREUSE_CDC_TARGET": "64",
    # 1 (SGLang parity): blocks holding relocated tokens and everything computed after them
    # enter vLLM's prefix cache like any other block (SGLang inserts the finished request,
    # relocated slots included, into its radix tree). 0: never cache a block at or after the
    # first relocated position of a request.
    "KVREUSE_CACHE_RELOCATED": "1",
    "KVREUSE_NO_REROTATE": "0",        # 1: copy keys without re-rotation (ablation only)
    "KVREUSE_SSM_MODE": "fork",        # hybrid models: fork | none
    "KVREUSE_SSM_SAVE": "end",          # hybrid: end = copy the state at the prompt end (SGLang); boundary = zero-copy: hold the cached state at the last block boundary
    "KVREUSE_THINK_IDS": "auto",       # reasoning delimiter ids for the cause label; auto = tokenizer
    "KVREUSE_ALLOW_UNTESTED_VLLM": "0",  # 1: skip the vLLM version pin (the seam check still runs)
    "KVREUSE_AUDIT": "0",              # N>0: block refcount audit every N scheduler steps (and when idle)
    "KVREUSE_AUDIT_STRICT": "0",       # 1: a failed audit raises (tests)
    "KVREUSE_VERIFY": "0",             # 1: worker re-checks every relocation copy (debug, logs per rank)
    "KVREUSE_DEBUG": "0",
    "KVREUSE_TRACE": "1",              # one `[scr] fin` accounting line per request
}


def apply_defaults(environ=None):
    env = os.environ if environ is None else environ
    for k, v in DEFAULTS.items():
        env.setdefault(k, v)
    return {k: env[k] for k in DEFAULTS}


def get(name, cast=str):
    v = os.environ.get(name, DEFAULTS.get(name))
    if cast is bool:
        return str(v) == "1"
    return cast(v)
