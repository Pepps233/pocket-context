# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Default SCR settings (K=6 with slot retry), applied with os.environ.setdefault.

Any variable already set in the environment wins, so for example
`KVREUSE_ENABLED=0` turns SCR off and `KVREUSE_MAX_BLOCKS=3` changes K.
"""
import os

DEFAULTS = {
    "KVREUSE_ENABLED": "1",          # install the patches
    "KVREUSE_MAX_BLOCKS": "6",       # K: surviving blocks relocated per request
    "KVREUSE_SPLICE_RETRY": "8",     # rounds a ready plan waits for the chunked-prefill slot
    "KVREUSE_MB_MIN_GAP": "0",       # no floor on the forwarded stretch between blocks
    "KVREUSE_V6_MIN_EXTEND": "16",   # tokens of each block forwarded instead of relocated
    "KVREUSE_SSM_MODE": "fork",      # recurrent state for linear-attention layers
    "KVREUSE_FORCE": "1",
    "KVREUSE_MIN_EXTRA_TOKENS": "1024",
    "KVREUSE_MIN_EXTRA_FRAC": "0.15",
    "KVREUSE_MAX_SESSIONS": "16",    # conversations tracked for matching
    "KVREUSE_SIDE_SESSIONS": "12",   # conversations held in the side buffer
    "KVREUSE_SIDE_TOKENS": "30000",  # tokens held per conversation
    "KVREUSE_BUDGET_FRAC": "0.25",
    "KVREUSE_TRACE": "1",            # one `kv6trace ev=req` line per request
    "KVREUSE_DEBUG": "1",            # per-request planning lines
}


def apply_defaults(environ=None):
    """Fill in every default that is not already set; return the effective values."""
    env = os.environ if environ is None else environ
    for k, v in DEFAULTS.items():
        env.setdefault(k, v)
    return {k: env[k] for k in DEFAULTS}
