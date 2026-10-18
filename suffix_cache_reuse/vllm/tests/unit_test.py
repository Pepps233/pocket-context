#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only checks of the planner and the key re-rotation (no server, no vLLM engine)."""
import random
import sys

import torch

from vllm_scr import planner
from vllm_scr.connector import rerotate_


def test_plan_deletion():
    rng = random.Random(0)
    A = [rng.randrange(50000) for _ in range(2000)]
    B = [rng.randrange(50000) for _ in range(300)]
    C1 = [rng.randrange(50000) for _ in range(900)]
    X = [rng.randrange(50000) for _ in range(100)]
    C2 = [rng.randrange(50000) for _ in range(700)]
    Q = [rng.randrange(50000) for _ in range(40)]
    old = A + B + C1 + X + C2
    new = A + C1 + C2 + Q          # delete B and X, append Q
    blocks, info = planner.plan_relocation(old, new, len(old), max_blocks=6, min_tail=64, min_extend=16)
    assert len(blocks) == 2, (blocks, info)
    b0, b1 = blocks
    assert b0["bprime_end"] == len(A) and b0["rows"][0] == len(A) + len(B)
    assert len(b0["rows"]) == len(C1) - 16
    assert b1["bprime_end"] == len(A) + len(C1) and len(b1["rows"]) == len(C2) - 16
    for b in blocks:  # relocated rows hold the same tokens as their new positions
        for k, r in enumerate(b["rows"]):
            assert old[r] == new[b["bprime_end"] + k]
    blocks1, _ = planner.plan_relocation(old, new, len(old), max_blocks=1, min_tail=64, min_extend=16)
    assert len(blocks1) == 1 and blocks1[0]["bprime_end"] == len(A)   # K=1 keeps the longest
    _, info = planner.plan_relocation(old, old + Q, len(old))
    assert info["reason"] == "passthrough"
    print("plan_deletion OK", info)


def test_session_match():
    s = {"s1": {"key": None, "token_ids": list(range(3000)), "snap": True},
         "s2": {"key": "salt", "token_ids": list(range(3000)), "snap": True}}
    new = list(range(2500)) + [7] * 100
    assert planner.find_session(s, new, None) == "s1"
    assert planner.find_session(s, new, "salt") == "s2"
    assert planner.find_session(s, list(range(500)) + [9] * 3000, None) is None
    print("session_match OK")


def _rope(x, pos, inv):
    ang = pos.double()[:, None] * inv[None, :]
    cos, sin = torch.cos(ang).float()[:, None, :], torch.sin(ang).float()[:, None, :]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def test_rerotate():
    torch.manual_seed(0)
    D, H, n = 128, 8, 64
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, dtype=torch.float64) / D))
    k = torch.randn(n, H, D)
    old = torch.randint(0, 30000, (n,))
    new = torch.randint(0, 30000, (n,))
    k_old = _rope(k, old, inv)
    k_new = _rope(k, new, inv)
    got = rerotate_(k_old.clone(), new - old, inv, D, True)
    err = (got - k_new).abs().max().item()
    assert err < 1e-3, err
    print(f"rerotate OK (fp32 max abs err {err:.2e})")


if __name__ == "__main__":
    test_plan_deletion()
    test_session_match()
    test_rerotate()
    print("unit tests: all passed")
    sys.exit(0)
