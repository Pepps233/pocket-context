# Comparison source provenance

These are versioned ports of published inference methods, not generic substitutes using Pocket Context's compressor. No upstream training pipeline or model weights are vendored.

## ReAct / ACM Base

- Paper: https://arxiv.org/html/2607.23809v1 (ACM Base, not ACM-Post-Trained).
- Repository: https://github.com/lixiaochuan2020/agentic-context-management
- Pinned revision: `f06f90e728af8580a4515812425c1620144145a2`.
- Source files: `src/tools.py`, `src/prompts.py`, `src/runner.py`.
- Copied components: native corpus/memory schemas; no-memory and ACM BCP instructions; structured working-memory and recall prompts; memory extraction, thinking stripping and one-retry/final-fallback behavior.
- Code license: MIT; complete upstream notice in `acm.LICENSE`.
- Original ReAct paradigm: https://arxiv.org/abs/2210.03629. The no-memory BCP prompt is the released ACM implementation's ReAct baseline, not the 2022 notebooks.

### Preserved ACM policy

`manage_context()` selects the unsummarized interval, archives its raw provider messages, and returns `[summary_id: N]` followed by the actual summary as its tool result. The compressor emits Knowledge state and Thoughts with concrete next steps, with a 4,096-token output cap and the released parser/retry behavior. Previous compression exchanges are outside the next interval. `query_memory(summary_id, query)` loads exactly one integer-ID archive and uses the released recall prompt; as in the release, recall uses the normal per-turn output allowance, not the compressor's special 4,096-token cap. No verbatim expansion or caller-selected ranges are added.

### Explicit implementation choices / deviations

1. The paper/tool description excludes the issuing assistant from compression; the released evaluation configuration defaults `mc_drop_call_message=True`, including it. This port follows the **paper-described exclusion**, retaining the assistant call with its native tool result. It does not claim bit-for-bit equivalence to that release configuration.
2. A successful mixed memory/research batch advances the next boundary after all sibling results. This prevents future archives from separating a native call from its result; the upstream handler's position assumption is reliable only for a dedicated memory turn.
3. Archives are transactional SQLite records scoped by run and positive integer summary ID, rather than `summary_N.json` files. The model-visible ID and recall behavior are unchanged.
4. Native Chat/Responses transport replaces the original qwen3 XML transport. Auxiliary history rendering preserves visible reasoning, tool names/arguments and results; private billing metadata is excluded.
5. Current-history pressure is checked at 95%, and results are not silently truncated by the memory harness. The configured shared window/tokenizer/budgets replace the authors' serving defaults. Fixed corpus retrieval is shared; published Table 2 scores are not reproduced by this evaluation.
6. The same configured model/endpoint handles policy, compression and recall. This is ACM Base; released post-trained checkpoints are a separate system comparison.

## ReSum (training-free)

- Paper: https://arxiv.org/html/2509.13313v3, Algorithm 1 and Appendices C/D.
- Repository: https://github.com/Alibaba-NLP/DeepResearch
- Pinned revision: `f72f75d8c3eb842f2bbbab096a12206ff66e270f`.
- Source directory: `WebAgent/WebResummer/src`.
- Source files: `prompt.py`, `summary_utils.py`, `react_agent.py`.
- Copied components: initial and recurrent summary prompts, summary-conditioned continuation text, summary normalization, full restart and context-relative trigger recipe.
- Code license: Apache-2.0; complete upstream license in `resum.LICENSE`.

### Preserved ReSum policy

The default threshold is `0.9 * (configured_context_window - 1000)`, matching the released recipe with a scaled window. Summarization sees all turns since the last restart and the previous summary separately. A successful response resets live history to the system instructions plus one user message containing the original question and summary-conditioned continuation. **No latest exchange is retained.** The output cap is 4,096 tokens; original tag normalization/fallback semantics are preserved, including normalization of a nonempty think-only response to an empty summary block. There is no archive retrieval and no ReSum-GRPO policy training.

### Explicit implementation choices / deviations

1. The configured fixed corpus replaces live Google/Jina search and visit. The main prompt is the published no-memory BCP ReAct prompt, not the original WebSailor live-web prompt. The summary and continuation prompts are unchanged.
2. The same model/endpoint is the summarizer in the primary same-policy comparison; the original paper also evaluates separate summarizer models, including ReSumTool-30B. This is not a reproduction of that strongest system configuration.
3. Provider adapters implement transport retry/backoff; an unrecoverable auxiliary error preserves live history and the previous summary, and its billed response remains accounted for. The standalone upstream HTTP client's ten-retry loop is not duplicated.
4. The configured corpus tokenizer and provider schemas replace the original text chat-template tokenizer. The trigger is an estimate, while reported token metrics use provider usage.
5. Original loop budgets/serving defaults are replaced by recorded shared evaluation budgets. Summary calls are counted separately from main policy turns; all inference cost and usage is included.

## Verification and reporting

Offline tests compare copied prompt/schema SHA256 values with the pinned originals and replay native Chat/Responses trajectories through real SQLite archives. They test full resets, repeated boundaries, summary placement, query-only recall, protocol grouping, failure rollback and auxiliary accounting. These tests establish implementation behavior, not benchmark accuracy parity.

Run manifests store upstream revisions, method-specific settings and main-prompt hashes. Source hashes cover the installed Python implementation, including copied prompt constants. All parameter changes require fresh matched runs. Keep simplified variants as separately named ablations if reintroduced; do not label them as these ports.
