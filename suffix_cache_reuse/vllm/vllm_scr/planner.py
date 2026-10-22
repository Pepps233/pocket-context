# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Session matching, prompt diffing and relocation planning (pure Python, no vLLM imports).

Ported from the SGLang version (suffix_cache_reuse/overlay.py: _cdc_bounds, _fast_opcodes,
_diff_opcodes, _find_session and the v6 tail/multiblock planner in _kvreuse_intercept) with
the same thresholds. Differences:
  * the plan is computed against the token ids whose KV is actually held for the session
    (the snapshot), so row i of the held KV is position i of `snap_ids`;
  * a plan is a list of blocks sorted by position; block = (bprime_end, rows), where the rows
    rows[0]..rows[-1] of the held KV are relocated to positions bprime_end..bprime_end+n-1.
"""
from collections import Counter
from difflib import SequenceMatcher

from vllm_scr import config as C


def _cdc_bounds(ids, target):
    mask = target - 1
    bounds = [0]
    h = 0
    for i, t in enumerate(ids):
        h = ((h << 1) ^ ((t * 0x9E3779B1) & 0xFFFFFFFF)) & 0xFFFFFFFF
        if ((h >> 8) & mask) == 0:
            bounds.append(i + 1)
    if bounds[-1] != len(ids):
        bounds.append(len(ids))
    return bounds


def _chunk_ids(ids, bounds, interner):
    out = []
    for k in range(len(bounds) - 1):
        key = tuple(ids[bounds[k]:bounds[k + 1]])
        v = interner.get(key)
        if v is None:
            v = len(interner)
            interner[key] = v
        out.append(v)
    return out


def _verify_ops(a, b, ops):
    pi = pj = 0
    for t, i1, i2, j1, j2 in ops:
        if i1 != pi or j1 != pj:
            return False
        if t == "equal" and a[i1:i2] != b[j1:j2]:
            return False
        pi, pj = i2, j2
    return pi == len(a) and pj == len(b)


def _fast_opcodes(a, b, target):
    ba = _cdc_bounds(a, target)
    bb = _cdc_bounds(b, target)
    interner = {}
    ca = _chunk_ids(a, ba, interner)
    cb = _chunk_ids(b, bb, interner)
    blocks = []
    for t, i1, i2, j1, j2 in SequenceMatcher(a=ca, b=cb, autojunk=False).get_opcodes():
        if t != "equal":
            continue
        ti1, ti2, tj1, tj2 = ba[i1], ba[i2], bb[j1], bb[j2]
        if ti2 - ti1 != tj2 - tj1:
            return None
        blocks.append([ti1, ti2, tj1, tj2])
    if blocks:
        for n, blk in enumerate(blocks):
            lo_a = blocks[n - 1][1] if n else 0
            lo_b = blocks[n - 1][3] if n else 0
            while blk[0] > lo_a and blk[2] > lo_b and a[blk[0] - 1] == b[blk[2] - 1]:
                blk[0] -= 1
                blk[2] -= 1
        for n in range(len(blocks) - 1, -1, -1):
            blk = blocks[n]
            hi_a = blocks[n + 1][0] if n + 1 < len(blocks) else len(a)
            hi_b = blocks[n + 1][2] if n + 1 < len(blocks) else len(b)
            while blk[1] < hi_a and blk[3] < hi_b and a[blk[1]] == b[blk[3]]:
                blk[1] += 1
                blk[3] += 1
    ops, pi, pj = [], 0, 0
    for ti1, ti2, tj1, tj2 in blocks:
        da, db = ti1 - pi, tj1 - pj
        if da and db:
            ops.append(("replace", pi, ti1, pj, tj1))
        elif da:
            ops.append(("delete", pi, ti1, pj, tj1))
        elif db:
            ops.append(("insert", pi, ti1, pj, tj1))
        ops.append(("equal", ti1, ti2, tj1, tj2))
        pi, pj = ti2, tj2
    da, db = len(a) - pi, len(b) - pj
    if da and db:
        ops.append(("replace", pi, len(a), pj, len(b)))
    elif da:
        ops.append(("delete", pi, len(a), pj, len(b)))
    elif db:
        ops.append(("insert", pi, len(a), pj, len(b)))
    return ops


def diff_opcodes(a, b, stats=None):
    """Token-level opcodes (difflib format) between the old and the new prompt, verified."""
    if C.get("KVREUSE_FASTDIFF", bool) and min(len(a), len(b)) >= C.get("KVREUSE_FASTDIFF_MIN", int):
        try:
            ops = _fast_opcodes(a, b, C.get("KVREUSE_CDC_TARGET", int))
        except Exception:
            ops = None
        if ops is not None and _verify_ops(a, b, ops):
            return ops
        if stats is not None:
            stats["fastdiff_reject"] = stats.get("fastdiff_reject", 0) + 1
    return SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()


def _quick_ratio_shared(a, bcount, lb):
    avail = {}
    matches = 0
    for elt in a:
        numb = avail[elt] if elt in avail else bcount.get(elt, 0)
        avail[elt] = numb - 1
        if numb > 0:
            matches += 1
    la = len(a)
    return 2.0 * matches / (la + lb) if (la + lb) else 1.0


def find_session(sessions, new_ids, key=None, stats=None):
    """Id of the tracked conversation this prompt continues (same rules as SGLang), or None.

    A candidate must have the same cache key (cache_salt), a common prefix of at least
    KVREUSE_MATCH_MIN_TOKENS, and either half its previous prompt as prefix or half overall
    token overlap. Among candidates holding KV ("slotted"), the longest common prefix wins.
    """
    window = C.get("KVREUSE_MATCH_WINDOW", int)
    min_tokens = C.get("KVREUSE_MATCH_MIN_TOKENS", int)
    min_ratio = C.get("KVREUSE_MATCH_MIN_RATIO", float)
    min_pre = C.get("KVREUSE_MATCH_MIN_PREFIX_FRAC", float)
    min_ovl = C.get("KVREUSE_MATCH_MIN_OVERLAP", float)
    margin = C.get("KVREUSE_MATCH_AMBIG_MARGIN", float)
    best_sid, best, best_n, best_unslotted = None, 0.0, 0, 0.0
    head = new_ids[:window]
    bcount, lb = Counter(new_ids), len(new_ids)
    for sid, sess in sessions.items():
        if sess.get("key") != key:
            continue
        old = sess.get("token_ids")
        if not old:
            continue
        n = 0
        for x, y in zip(old[:window], head):
            if x != y:
                break
            n += 1
        if n < min_tokens:
            continue
        r_pre = n / max(1, len(old))
        r_all = _quick_ratio_shared(old, bcount, lb)
        if r_pre < min_pre and r_all < min_ovl:
            continue
        if sess.get("snap") is None:
            best_unslotted = max(best_unslotted, r_all)
            continue
        if n > best_n or (n == best_n and r_all > best):
            best_n, best, best_sid = n, r_all, sid
    if best < min_ratio:
        return None
    if best_unslotted > best + margin:
        if stats is not None:
            stats["match_ambiguous"] = stats.get("match_ambiguous", 0) + 1
        return None
    return best_sid


def plan_relocation(old_ids, new_ids, n_held, max_blocks=None, min_tail=None, min_extend=None):
    """Relocation plan for `new_ids` given the held KV of `old_ids` (rows < n_held valid).

    Returns (blocks, info). blocks is a position-sorted list of dicts
    {"bprime_end": j, "rows": [i0, i0+1, ...], "n_c": span length}; rows are relocated to
    positions j, j+1, ...; the last `min_extend` tokens of every span are prefilled.
    Mirrors the SGLang v6 planner: only spans after the leading equal run (A) qualify, the
    K longest are kept, every span must leave >= min_extend tokens to prefill after it.
    """
    max_blocks = max(1, int(max_blocks if max_blocks is not None else C.get("KVREUSE_MAX_BLOCKS", int)))
    min_tail = int(min_tail if min_tail is not None else C.get("KVREUSE_V6_MIN_TAIL", int))
    kext = max(1, int(min_extend if min_extend is not None else C.get("KVREUSE_V6_MIN_EXTEND", int)))
    info = {"kept": 0, "deleted": 0, "inserted": 0, "cands": 0, "reason": ""}
    if not old_ids or not new_ids:
        info["reason"] = "empty"
        return [], info
    ops = diff_opcodes(old_ids, new_ids, info)
    kept = sum(j2 - j1 for t, i1, i2, j1, j2 in ops if t == "equal")
    deleted = sum(i2 - i1 for t, i1, i2, j1, j2 in ops if t in ("delete", "replace"))
    inserted = sum(j2 - j1 for t, i1, i2, j1, j2 in ops if t in ("insert", "replace"))
    info.update(kept=kept, deleted=deleted, inserted=inserted, n_ops=len(ops))
    # cause of the edit, for the strip-vs-CLM-edit split: does any deleted/replaced old span
    # contain a reasoning delimiter (Qwen3: <think>=151667, </think>=151668)?
    think = {int(x) for x in str(C.get("KVREUSE_THINK_IDS")).split(",") if x.strip().isdigit()}
    strip = other = False
    for t, i1, i2, j1, j2 in ops:
        if t in ("delete", "replace"):
            if think and any(x in think for x in old_ids[i1:i2]):
                strip = True
            else:
                other = True
    info["cause"] = ("strip+edit" if strip and other else "strip" if strip
                     else "edit" if other else "append")
    if inserted == 0 or kept == 0 or deleted == 0:
        info["reason"] = "passthrough"
        return [], info
    if not (ops and ops[0][0] == "equal" and ops[0][1] == 0 and ops[0][3] == 0):
        info["reason"] = "no_leading_equal"
        return [], info
    a_new_end = ops[0][4]
    cands = []
    for k, (t, ci1, ci2, cj1, cj2) in enumerate(ops):
        if k == 0 or t != "equal":
            continue
        n_c = ci2 - ci1
        if not (cj1 >= a_new_end and n_c >= min_tail and ci2 <= n_held):
            continue
        cands.append((n_c, k, ci1, ci2, cj1, cj2))
    cands.sort(reverse=True)
    info["cands"] = len(cands)
    acc = []
    for n_c, k, ci1, ci2, cj1, cj2 in cands:
        if len(acc) >= max_blocks:
            break
        if n_c - kext < min_tail:
            continue
        if old_ids[ci1:ci2 - kext] != new_ids[cj1:cj2 - kext]:
            continue
        if len(new_ids) - (cj2 - kext) < kext:
            continue
        acc.append({"bprime_end": cj1, "rows": list(range(ci1, ci2 - kext)), "n_c": n_c})
    acc.sort(key=lambda b: b["bprime_end"])
    if not acc:
        info["reason"] = "no_tail"
    return acc, info
