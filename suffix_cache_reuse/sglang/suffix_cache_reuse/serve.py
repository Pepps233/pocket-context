# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch SGLang with Suffix Cache Reuse.

    python -m suffix_cache_reuse.serve --model-path Qwen/Qwen3.6-27B [sglang args...]

All arguments are passed to `python -m sglang.launch_server` unchanged, except
that `--log-requests --log-requests-level 0` is added when `--log-requests` is
absent: the analysis script reads SGLang's per-request `Finish:` lines, and
level 0 logs request metadata only (ids and token counts, no text).

SCR settings come from the environment (see suffix_cache_reuse/config.py);
`KVREUSE_ENABLED=0` starts the same server with SCR off.
"""
import os
import sys

from suffix_cache_reuse import SGLANG_VERSION
from suffix_cache_reuse.config import DEFAULTS, apply_defaults


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--log-requests" not in args:
        args += ["--log-requests", "--log-requests-level", "0"]

    eff = apply_defaults()
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    site_dir = os.path.join(pkg_parent, "suffix_cache_reuse", "_site")
    paths = [site_dir, pkg_parent] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)

    try:
        from importlib.metadata import version
        sgl = version("sglang")
    except Exception:
        sgl = "unknown"
    state = "on" if eff["KVREUSE_ENABLED"] == "1" else "off"
    print(f"[scr] Suffix Cache Reuse {state} | sglang {sgl} (built for {SGLANG_VERSION}) | "
          + " ".join(f"{k}={eff[k]}" for k in DEFAULTS), flush=True)

    cmd = [sys.executable, "-m", "sglang.launch_server"] + args
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
