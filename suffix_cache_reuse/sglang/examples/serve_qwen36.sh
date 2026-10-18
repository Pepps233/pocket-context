#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Serve Qwen3.6-27B on one GPU with SGLang 0.5.16 and Suffix Cache Reuse on.
#
#   bash examples/serve_qwen36.sh                        # SCR on (K=6)
#   KVREUSE_ENABLED=0 bash examples/serve_qwen36.sh      # same server, SCR off
#   KVREUSE_MAX_BLOCKS=3 bash examples/serve_qwen36.sh   # SCR on, K=3
#
# Extra arguments are passed to sglang.launch_server. The server log goes to
# stdout; redirect it to a file for analysis/scr_report.py.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3.6-27B}
PORT=${PORT:-30000}
PYTHON=${PYTHON:-python}

exec "$PYTHON" -m suffix_cache_reuse.serve \
  --model-path "$MODEL" --served-model-name qwen36-27b \
  --host 0.0.0.0 --port "$PORT" --tp-size 1 \
  --context-length 65536 --mem-fraction-static 0.75 --chunked-prefill-size 512 \
  --disable-cuda-graph --tool-call-parser qwen3_coder --trust-remote-code \
  --log-requests --log-requests-level 0 \
  "$@"
