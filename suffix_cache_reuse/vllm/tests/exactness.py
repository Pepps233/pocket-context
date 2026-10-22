#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Mechanics check of SCR on vLLM, in process (engine core and worker in this process).

    KVREUSE_ENABLED=1 python tests/exactness.py --model <path> [--json out.json]

Conversation: turn 1 = [A][B][C1][X][C2][Q1]; turn 2 = [A][C1][C2][Q2] (B and X deleted,
so C1 and C2 move to earlier positions). Turn 2 is served with SCR; a reference turn 2 with a
different cache_salt is served cold (full prefill). From the held KV of both we compare, for
the relocated rows:

  copy        relocated K (all layers) == re-rotation of turn-1 K, relocated V == turn-1 V
              -> slot mapping and copy are right (error should be at bf16 rounding level)
  layer 0     at layer 0 K depends only on token and position and V only on the token, so
              relocated K/V must equal the cold prefill's layer-0 K/V up to rounding
              -> RoPE re-rotation is right. Control: the same rows without re-rotation.
  deeper      cosine(relocated, cold) per layer: size of the approximation (informational)
  logits      top-5 logprobs of the first turn-2 token, SCR vs cold
Then greedy agreement on a few edited two-turn prompts (48 tokens, SCR vs cold).
"""
import argparse
import json
import os
import random
import sys

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("KVREUSE_ENABLED", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import torch  # noqa: E402

from vllm_scr.config import apply_defaults  # noqa: E402

apply_defaults()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--json")
    ap.add_argument("--n-greedy", type=int, default=8)
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm_scr import planner
    from vllm_scr.connector import rerotate_

    extra = {}
    if a.hybrid:
        extra = dict(language_model_only=True, disable_hybrid_kv_cache_manager=False)
    llm = LLM(model=a.model, enable_prefix_caching=True, gpu_memory_utilization=a.gpu_util,
              max_model_len=32768, async_scheduling=False, **extra,
              scheduler_cls="vllm_scr.scheduler.SCRScheduler",
              kv_transfer_config=KVTransferConfig(kv_connector="SCRConnector",
                                                  kv_connector_module_path="vllm_scr.connector",
                                                  kv_role="kv_both"))
    tok = llm.get_tokenizer()
    core = llm.llm_engine.engine_core.engine_core
    sched = core.scheduler
    conn = get_kv_transfer_group()
    g0 = sched._scr_attn_gids[0]
    views = conn._layers[g0]
    kbs = conn._kbs
    bs = sched._scr_bs[g0]
    rng = random.Random(0)
    words = ("amber basalt cobalt delta ember fjord granite harbor indigo juniper kelp lagoon "
             "meadow nickel onyx prairie quartz river summit tundra umber valley willow xenon").split()

    def text(n_words, tag):
        return f"[{tag}] " + " ".join(rng.choice(words) + str(rng.randint(0, 99)) for _ in range(n_words))

    def ids(s):
        return tok.encode(s, add_special_tokens=False)

    def held(salt, n_old=None):
        """Most recently pinned snapshot for a session with this cache salt."""
        cands = [s for s in sched._scr_sessions.values() if s.get("key") == salt and s.get("snap")]
        return cands[-1]["snap"]

    def read_kv(snap, positions):
        bl = torch.tensor([b.block_id for b in snap["blocks"][g0]])
        p = torch.as_tensor(positions)
        slot = bl[p // bs] * bs + p % bs
        sb, so = slot // kbs, slot % kbs
        dev = views[0][0].device
        sb, so = sb.to(dev), so.to(dev)
        return ([kv[sb, so].float().cpu() for kv, _ in views],
                [vv[sb, so].float().cpu() for _, vv in views])

    sp1 = SamplingParams(max_tokens=1, temperature=0, logprobs=5)
    A, B, C1, X, C2 = ids(text(1500, "A")), ids(text(250, "B")), ids(text(700, "C1")), \
        ids(text(80, "X")), ids(text(600, "C2"))
    Q1, Q2 = ids(" Question one: what is listed first?"), ids(" Question two: what is listed last?")
    old = A + B + C1 + X + C2 + Q1
    new = A + C1 + C2 + Q2
    llm.generate([{"prompt_token_ids": old}], sp1, use_tqdm=False)
    s_old = held(None)
    blocks, info = planner.plan_relocation(s_old["ids"], new, s_old["n_held"])
    # copy the old rows before turn 2 (all candidate rows; the executed subset is taken below)
    all_rows = sorted({r for b in blocks for r in b["rows"]})
    k_all, v_all = read_kv(s_old, all_rows)
    idx_of = {r: i for i, r in enumerate(all_rows)}
    sched._scr_trace.clear()
    o_scr = llm.generate([{"prompt_token_ids": new}], sp1, use_tqdm=False)[0]
    s_new = held(None)
    spans = list(sched._scr_trace)          # what the scheduler actually relocated
    rows = [r0 + k for _, p0, r0, n in spans for k in range(n)]
    newpos = [p0 + k for _, p0, r0, n in spans for k in range(n)]
    sel = torch.tensor([idx_of[r] for r in rows])
    k_old = [x[sel] for x in k_all]
    v_old = [x[sel] for x in v_all]
    assert all(new[p] == s_old["ids"][r] for p, r in zip(newpos, rows)), "token mismatch"
    assert s_new is not s_old, "turn 2 not pinned"
    o_ref = llm.generate([{"prompt_token_ids": new, "cache_salt": "ref"}], sp1, use_tqdm=False)[0]
    s_ref = held("ref")
    k_rel, v_rel = read_kv(s_new, newpos)
    k_ref, v_ref = read_kv(s_ref, newpos)
    delta = torch.tensor(newpos) - torch.tensor(rows)
    res = {"plan_blocks": [(b["bprime_end"], len(b["rows"])) for b in blocks],
           "executed_spans": [(p0, r0, n) for _, p0, r0, n in spans],
           "relocated_rows": len(rows), "conn_ops": conn.n_ops, "conn_tokens": conn.n_tokens,
           "stats": dict(sched._scr_stats)}
    inv, rd, neox = conn._inv.cpu(), conn._rot.rotary_dim, conn._rot.is_neox_style
    copy_k, copy_v = [], []
    for li in range(len(views)):
        exp = rerotate_(k_old[li].to(torch.bfloat16).clone(), delta, inv, rd, neox).float()
        copy_k.append((k_rel[li] - exp).abs().max().item())
        copy_v.append((v_rel[li] - v_old[li]).abs().max().item())
    res["copy_max_abs_err_k_over_layers"] = max(copy_k)
    res["copy_max_abs_err_v_over_layers"] = max(copy_v)
    scale0 = k_ref[0].abs().max().item()
    res["layer0_k_scale"] = scale0
    res["layer0_k_max_abs_err_reloc_vs_cold"] = (k_rel[0] - k_ref[0]).abs().max().item()
    res["layer0_k_mean_abs_err_reloc_vs_cold"] = (k_rel[0] - k_ref[0]).abs().mean().item()
    res["layer0_k_max_abs_err_NO_rerotation_vs_cold"] = (k_old[0] - k_ref[0]).abs().max().item()
    res["layer0_k_mean_abs_err_NO_rerotation_vs_cold"] = (k_old[0] - k_ref[0]).abs().mean().item()
    res["layer0_v_max_abs_err_reloc_vs_cold"] = (v_rel[0] - v_ref[0]).abs().max().item()
    # bf16 reference noise: the same rows of turn 1 vs cold turn 2 re-rotated in fp32
    res["layer0_k_max_abs_err_turn1_rerot_fp32_vs_cold"] = (
        rerotate_(k_old[0].clone(), delta, inv, rd, neox) - k_ref[0]).abs().max().item()

    def cos(x, y):
        return torch.nn.functional.cosine_similarity(x.flatten(1), y.flatten(1), dim=1).mean().item()

    res["cos_k_reloc_vs_cold_per_layer"] = [round(cos(k_rel[i], k_ref[i]), 5) for i in range(len(views))]
    res["cos_v_reloc_vs_cold_per_layer"] = [round(cos(v_rel[i], v_ref[i]), 5) for i in range(len(views))]
    lp_s = o_scr.outputs[0].logprobs[0]
    lp_r = o_ref.outputs[0].logprobs[0]
    res["first_token_scr"] = tok.decode(o_scr.outputs[0].token_ids)
    res["first_token_cold"] = tok.decode(o_ref.outputs[0].token_ids)
    res["top5_scr"] = {tok.decode([t]): round(v.logprob, 4) for t, v in lp_s.items()}
    res["top5_cold"] = {tok.decode([t]): round(v.logprob, 4) for t, v in lp_r.items()}

    # ---- greedy agreement on chat-formatted edited conversations (SCR vs cold)
    sp = SamplingParams(max_tokens=48, temperature=0, logprobs=1)
    agree = []
    for g in range(a.n_greedy):
        secs = [f"## Section {i}\n" + text(110, f"S{g}.{i}") for i in range(24)]
        sysm = "You are a careful assistant. Answer in one short sentence."
        def conv(kept, q, hist):
            return [{"role": "system", "content": sysm},
                    {"role": "user", "content": "Survey document:\n\n" + "\n\n".join(secs[i] for i in kept)}] \
                + hist + [{"role": "user", "content": q}]
        q1 = f"Which terms does Section {3 + g} list first?"
        m1 = conv(range(24), q1, [])
        o1 = llm.chat(m1, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})[0]
        t1 = o1.outputs[0].text
        kept2 = [i for i in range(24) if i not in (5 + g % 4, 15)]
        m2 = conv(kept2, f"Which terms does Section {20 - g % 3} list first?",
                  [{"role": "user", "content": q1}, {"role": "assistant", "content": t1}])
        before = sched._scr_stats["spliced_tokens"]
        o_s = llm.chat(m2, sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})[0]
        moved = sched._scr_stats["spliced_tokens"] - before
        p2 = tok.apply_chat_template(m2, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if not isinstance(p2, list):
            p2 = p2["input_ids"]
        o_c = llm.generate([{"prompt_token_ids": list(p2), "cache_salt": f"kvreuse:off-ref{g}"}], sp,
                           use_tqdm=False)[0]
        ts, tc = list(o_s.outputs[0].token_ids), list(o_c.outputs[0].token_ids)
        n_same = next((i for i, (x, y) in enumerate(zip(ts, tc)) if x != y), min(len(ts), len(tc)))
        lps = [list(d.values())[0].logprob for d in o_s.outputs[0].logprobs[:1]]
        lpc = [list(d.values())[0].logprob for d in o_c.outputs[0].logprobs[:1]]
        agree.append({"conv": g, "relocated": moved, "identical": ts == tc,
                      "common_prefix_tokens": n_same, "len": len(ts),
                      "first_logprob_scr": round(lps[0], 4), "first_logprob_cold": round(lpc[0], 4),
                      "scr": o_s.outputs[0].text[:100], "cold": o_c.outputs[0].text[:100]})
    res["greedy"] = agree
    res["ssm_restore_ops"] = conn.n_restore
    res["ssm_save_ops"] = conn.n_save
    res["note"] = ("hybrid: layer 'index 0' is the first full-attention layer, whose K/V depend on "
                   "earlier linear-attention layers, so the layer-0 rows are NOT a pure RoPE check"
                   if a.hybrid else "dense: layer 0 K/V depend only on token and position")
    res["greedy_identical"] = sum(x["identical"] for x in agree)
    res["greedy_relocated_convs"] = sum(x["relocated"] > 0 for x in agree)
    res["final_stats"] = dict(sched._scr_stats)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
