# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Launch `vllm serve` with Suffix Cache Reuse.

    python -m vllm_scr.serve <model> [vllm serve args...]

KVREUSE_ENABLED=1 (default): SCR scheduler + SCRConnector.
KVREUSE_ENABLED=0: stock vLLM scheduling through the same scheduler subclass, which then only
writes the per-request accounting line (`[scr] fin ...`), so SCR on and off log identically.
"""
import json
import os
import sys

from vllm_scr import VLLM_VERSION
from vllm_scr.config import DEFAULTS, apply_defaults


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    eff = apply_defaults()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sync = "--no-async-scheduling" in args
    if "--scheduler-cls" not in args:
        args += ["--scheduler-cls",
                 "vllm_scr.scheduler." + ("SCRScheduler" if sync else "SCRAsyncScheduler")]
    on = eff["KVREUSE_ENABLED"] == "1"
    if on and "--disable-hybrid-kv-cache-manager" not in args and \
            "--no-disable-hybrid-kv-cache-manager" not in args:
        args += ["--no-disable-hybrid-kv-cache-manager"]   # SCRConnector supports HMA
    if on and "--kv-transfer-config" not in args:
        args += ["--kv-transfer-config", json.dumps({
            "kv_connector": "SCRConnector", "kv_connector_module_path": "vllm_scr.connector",
            "kv_role": "kv_both"})]
    try:
        from importlib.metadata import version
        v = version("vllm")
    except Exception:
        v = "unknown"
    print(f"[scr] Suffix Cache Reuse {'on' if on else 'off'} | vllm {v} (built for {VLLM_VERSION}) | "
          + " ".join(f"{k}={eff[k]}" for k in DEFAULTS), flush=True)
    site = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_site")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [site] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"] + args
    print("[scr] exec: " + " ".join(cmd), flush=True)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
