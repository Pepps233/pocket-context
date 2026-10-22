#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Scheduler-level tests of SCR on CPU, inside a real vLLM 0.19.1 Scheduler (see sim.py).

    python tests/test_sim.py            # all
    python tests/test_sim.py -k hybrid  # subset (substring match)

Each positive test runs many concurrent multi-turn conversations with random edits and checks,
every step: the attention KV every request attends to is exactly its own (token, position)
sequence (prefix hits and relocated spans included); the recurrent state is never read from a
freed block and is exact or a fork state at the right position; prefix-cached state blocks hold
the state at their own block end; all block reference counts are accounted for (KVREUSE_AUDIT).
At the end the block pool must be whole. Negative controls disable each safeguard and must fail.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim  # noqa: E402

LARGE = dict(num_blocks=8000, env={"KVREUSE_MAX_SESSIONS": "64", "KVREUSE_SIDE_SESSIONS": "48",
                                   "KVREUSE_BUDGET_FRAC": "0.6"})


_ENV_KEYS = ("KVREUSE_MAX_SESSIONS", "KVREUSE_SIDE_SESSIONS", "KVREUSE_BUDGET_FRAC", "KVREUSE_SSM_SAVE")
_orig_run = sim.run


def _run(*a, env=None, **kw):
    for k in _ENV_KEYS:
        os.environ.pop(k, None)
    return _orig_run(*a, env=env, **kw)


sim.run = _run


def _reloc(st):
    return st.get("scr", {}).get("spliced_tokens", 0)


def test_seam():
    import inspect
    from vllm_scr import scheduler as S
    S.check_seam()
    real = inspect.getsource
    try:
        inspect.getsource = lambda f: real(f).replace("self._mamba_block_aligned_split(", "self._x(")
        try:
            S.check_seam()
        except RuntimeError as e:
            assert "no longer calls" in str(e), e
        else:
            raise AssertionError("seam check did not fail on a mismatched schedule()")
    finally:
        inspect.getsource = real
    import vllm
    v = vllm.__version__
    try:
        vllm.__version__ = "0.0.0"
        os.environ["KVREUSE_ALLOW_UNTESTED_VLLM"] = "0"
        try:
            S.check_seam()
        except RuntimeError as e:
            assert "supports vLLM" in str(e)
        else:
            raise AssertionError("version pin did not fail")
        os.environ["KVREUSE_ALLOW_UNTESTED_VLLM"] = "1"
        S.check_seam()
    finally:
        vllm.__version__ = v
        os.environ["KVREUSE_ALLOW_UNTESTED_VLLM"] = "0"
    print("seam OK")


def test_off_is_stock():
    """KVREUSE_ENABLED=0: step-by-step identical scheduling to vLLM's own Scheduler."""
    for hybrid in (False, True):
        a = sim.run(convs=10, turns=4, hybrid=hybrid, scr=False, seed=3, **LARGE)
        b = sim.run(convs=10, turns=4, hybrid=hybrid, scr=False, seed=3, stock=True, **LARGE)
        assert a[2] == b[2], f"hybrid={hybrid}: schedule differs from stock"
        assert {k: v[0] for k, v in a[1].items()} == {k: v[0] for k, v in b[1].items()}
        print(f"off_is_stock hybrid={hybrid} OK ({len(a[2])} steps identical)")


def test_on_appends_only_is_stock():
    """SCR on, no edits (nothing to relocate): same schedule as stock vLLM -- the seam is neutral."""
    for hybrid in (False, True):
        a = sim.run(convs=10, turns=4, hybrid=hybrid, scr=True, seed=4, edit_rate=0.0, **LARGE)
        b = sim.run(convs=10, turns=4, hybrid=hybrid, scr=False, seed=4, stock=True, edit_rate=0.0, **LARGE)
        assert _reloc(a[0]) == 0
        assert a[2] == b[2], f"hybrid={hybrid}: SCR-on append-only schedule differs from stock"
        print(f"on_appends_only_is_stock hybrid={hybrid} OK ({len(a[2])} steps identical)")


def test_dense_stress_equal_outputs():
    """Dense: many conversations, random edits; outputs equal SCR off; pool whole at the end."""
    for seed, asy in ((0, False), (1, False), (2, False), (3, True)):
        on = sim.run(convs=24, turns=6, hybrid=False, scr=True, seed=seed, async_sched=asy, **LARGE)
        off = sim.run(convs=24, turns=6, hybrid=False, scr=False, seed=seed, async_sched=asy, **LARGE)
        assert _reloc(on[0]) > 0, on[0]
        o1 = {k: v[0] for k, v in on[1].items()}
        o2 = {k: v[0] for k, v in off[1].items()}
        relocated = [k for k, v in on[1].items() if v[1].get("reloc", 0) > 0]
        assert o1 == o2, [k for k in o1 if o1[k] != o2.get(k)][:5]
        print(f"dense stress seed={seed} async={asy} OK: {on[0]['requests']} requests, {len(relocated)} relocated "
              f"turns, {_reloc(on[0])} relocated tokens, {on[0]['verified_tokens']} KV positions "
              f"verified, outputs == SCR off")


def test_hybrid_stress():
    """Hybrid (align): safeguards exercised and holding; pool whole at the end."""
    for seed, asy in ((0, False), (1, False), (2, False), (3, True)):
        st, out, _ = sim.run(convs=24, turns=6, hybrid=True, scr=True, seed=seed, async_sched=asy, **LARGE)
        sc = st["scr"]
        assert sc.get("ssm_restore", 0) > 0 and sc.get("ssm_dirty_skipped", 0) > 0, sc
        assert st["hashed_state_checks"] > 0
        print(f"hybrid stress seed={seed} async={asy} OK: {st['requests']} requests, {sc.get('spliced_blocks', 0)} "
              f"spans, restores {sc.get('ssm_restore', 0)}, dirty blocks kept out of the cache "
              f"{sc.get('ssm_dirty_skipped', 0)}, cached state blocks checked {st['hashed_state_checks']}")


def test_hybrid_boundary_save():
    """KVREUSE_SSM_SAVE=boundary: the held state is a reference on the cached boundary state;
    all checks hold and no blocks leak."""
    env = dict(LARGE["env"], KVREUSE_SSM_SAVE="boundary")
    for seed in (0, 1):
        st, _, _ = sim.run(convs=24, turns=6, hybrid=True, scr=True, seed=seed, num_blocks=8000, env=env)
        sc = st["scr"]
        assert sc.get("ssm_boundary_hold", 0) > 0 and sc.get("ssm_restore", 0) > 0 and "ssm_save" not in sc, sc
        print(f"hybrid boundary save seed={seed} OK: holds {sc['ssm_boundary_hold']}, restores "
              f"{sc['ssm_restore']}, spans {sc.get('spliced_blocks', 0)}")


def test_pressure():
    """Tight pool: side-buffer reclaim on allocation failure, LRU release, preemption."""
    for hybrid in (False, True):
        st, _, _ = sim.run(convs=32, turns=5, hybrid=hybrid, scr=True, seed=7, num_blocks=2000,
                           concurrency=24, budget=128, max_out=800,
                           env={"KVREUSE_MAX_SESSIONS": "64", "KVREUSE_SIDE_SESSIONS": "48",
                                "KVREUSE_BUDGET_FRAC": "0.5"})
        sc = st["scr"]
        assert sc.get("side_released", 0) > 0 and sc.get("reclaim_retry", 0) > 0, sc
        assert sc.get("spliced_blocks", 0) > 0, sc
        print(f"pressure hybrid={hybrid} OK: preemptions {st['preemptions']}, side released "
              f"{sc.get('side_released', 0)}, reclaim retries {sc.get('reclaim_retry', 0)}, "
              f"spans {sc.get('spliced_blocks', 0)}, aborts "
              f"{ {k: v for k, v in sc.items() if k.startswith('abort_')} }")


def _force_preempt(s):
    """Fail allocation once for every request that is between its cut and its jump (or right
    after a jump), so vLLM preempts running requests -- mid-plan included."""
    kvm = s.kv_cache_manager
    inner = kvm.allocate_slots
    done = set()

    def alloc(request, *a, **kw):
        st = getattr(request, "_scr", None) or {}
        if (st.get("cut") or st.get("jumped")) and request.request_id not in done:
            done.add(request.request_id)
            return None
        return inner(request, *a, **kw)
    kvm.allocate_slots = alloc


def test_preempt_mid_plan():
    """Preemption between the cut and the jump (and right after it): the plan falls back to
    normal prefill, outputs still equal SCR off (dense), safeguards hold (hybrid), no leaks."""
    for hybrid in (False, True):
        on, out_on, _ = sim.run(convs=16, turns=5, hybrid=hybrid, scr=True, seed=5, patch=_force_preempt,
                                **LARGE)
        sc = on["scr"]
        assert on["preemptions"] > 0 and sc.get("abort_preempted", 0) > 0, (on["preemptions"], sc)
        if not hybrid:
            off = sim.run(convs=16, turns=5, hybrid=False, scr=False, seed=5, **LARGE)
            assert {k: v[0] for k, v in out_on.items()} == {k: v[0] for k, v in off[1].items()}
        print(f"preempt_mid_plan hybrid={hybrid} OK: preemptions {on['preemptions']}, plans aborted "
              f"by preemption {sc.get('abort_preempted', 0)}, spans still done "
              f"{sc.get('spliced_blocks', 0)}" + ("" if hybrid else ", outputs == SCR off"))


def _expect_fail(name, patch, hybrid, needle):
    try:
        sim.run(convs=16, turns=5, hybrid=hybrid, scr=True, seed=0, patch=patch, **LARGE)
    except (sim.SimError, AssertionError) as e:
        assert needle in str(e), f"{name}: failed for another reason: {e}"
        print(f"negative control {name} OK (caught: {str(e)[:90]})")
        return
    raise AssertionError(f"negative control {name}: the bug was not detected")


def test_negative_controls():
    def no_dirty_skip(s):        # dirty recurrent-state blocks would enter the prefix cache
        for g in s._scr_mamba_gids:
            mgr = s.kv_cache_manager.coordinator.single_type_managers[g]
            mgr.cache_blocks = type(mgr).cache_blocks.__get__(mgr)

    def no_deferred_free(s):     # the cut block is freed while preprocess_mamba still reads it
        orig = s._scr_schedule

        def f():
            out = orig()
            if s._scr_deferred:
                s.kv_cache_manager.block_pool.free_blocks(s._scr_deferred)
                s._scr_deferred = []
            return out
        s._scr_schedule = f

    def wrong_delta(s):          # relocated keys land one position off
        orig = s._build_kv_connector_meta

        def f(connector, so):
            meta = orig(connector, so)
            meta.ops = [(g, a, b, d + 1) for g, a, b, d in meta.ops]
            return meta
        s._build_kv_connector_meta = f

    def upstream_keeps_cut_block(s):
        # Simulates an upstream change where align mode no longer frees the previous state
        # block when the request advances (0.19.1 frees it, which is what keeps the cut block
        # out of the prefix cache today). SCR's own dirty-block skip must then hold the line.
        for g in s._scr_mamba_gids:
            mgr = s.kv_cache_manager.coordinator.single_type_managers[g]
            mgr.remove_skipped_blocks = lambda request_id, num_computed_tokens: None

    def both(s):
        upstream_keeps_cut_block(s)
        no_dirty_skip(s)

    st, _, _ = sim.run(convs=16, turns=5, hybrid=True, scr=True, seed=0, patch=upstream_keeps_cut_block,
                       **LARGE)
    print(f"upstream_keeps_cut_block with SCR's skip OK (spans {st['scr'].get('spliced_blocks', 0)}, "
          f"dirty skipped {st['scr'].get('ssm_dirty_skipped', 0)})")
    os.environ["KVREUSE_AUDIT_STRICT"] = "0"
    try:
        _expect_fail("upstream_keeps_cut_block+no_dirty_skip", both, True, "dirty block cached")
        _expect_fail("no_deferred_free", no_deferred_free, True, "freed block")
        _expect_fail("wrong_delta", wrong_delta, False, "KV at position")
    finally:
        os.environ["KVREUSE_AUDIT_STRICT"] = "1"


TESTS = [test_seam, test_off_is_stock, test_on_appends_only_is_stock, test_dense_stress_equal_outputs,
         test_hybrid_stress, test_hybrid_boundary_save, test_pressure, test_preempt_mid_plan, test_negative_controls]

if __name__ == "__main__":
    sel = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    fails = 0
    for t in TESTS:
        if sel and sel not in t.__name__:
            continue
        try:
            t()
        except Exception:
            fails += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print("ALL OK" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
