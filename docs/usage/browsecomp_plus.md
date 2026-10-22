# BrowseComp-Plus

Run Pocket Context Agent on [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus), a research benchmark with 830 questions and a fixed corpus of 100,195 documents.
This integration uses the official Lucene BM25 index and exports the files expected by the upstream evaluator.
It runs this repository's agent and context tools, so its trajectories differ from the upstream baseline agents.

## Install and prepare

Use Python 3.11 and Java 21 with `java` on your `PATH`.
Set `JAVA_HOME` to your Java 21 installation if your machine has several Java versions.
BM25 inference runs on CPU; running the official Qwen3-32B judge requires a separate Linux GPU environment.

```bash
uv venv --python 3.11
uv pip install --python .venv/bin/python -e '.[browsecomp-plus]'
source .venv/bin/activate
mini-extra browsecomp-plus prepare
```

Preparation streams only question IDs, questions, and reference answers from the pinned dataset revision, decrypts them locally, and downloads the official BM25 index (about 2.2 GB).
The reference answers are for evaluation and are never included in the agent's task or observations.
Hugging Face access is public; `HF_TOKEN` is optional for higher download rate limits.
Model inference requires your provider's credentials, such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY`.
Export credentials in your shell, put them in the repository's ignored `.env`, or use the existing `mini-extra config` global configuration.
Keep credentials out of YAML overrides because run configurations are saved.

Files go under the ignored `data/browsecomp-plus/` directory.
Use `prepare --no-index` to prepare only questions, or `--data-dir PATH` to choose another location.
For a cache inside the ignored data directory, set `HF_HOME="$PWD/data/browsecomp-plus/.cache"` before preparation.
Keep decrypted questions, reference answers, and trajectories private to avoid benchmark contamination.
Add equivalent ignore rules if you choose other data or output directories.

The pinned input revisions are:

| Input | Revision |
| --- | --- |
| `Tevatron/browsecomp-plus` | `144cff8e35b5eaef7e526346aa60774a9deb941f` |
| `Tevatron/browsecomp-plus-indexes` | `b3f37f70c33829eb09d04784a54277a31871fd63` |
| Upstream evaluation code | `046949032b0328319cc9a02663a759ec601d9402` |

## Run

Reserve the exact 150 evaluation questions published with the [implementation paper](https://arxiv.org/pdf/2607.23809):

```bash
mini-extra browsecomp-plus reserve
```

The IDs come from the authors' `bcp_eval_150.json` at revision `f06f90e728af8580a4515812425c1620144145a2` of [agentic-context-management](https://github.com/lixiaochuan2020/agentic-context-management).
This writes ignored `eval150.jsonl` and `development680.jsonl` files.
Both local and Daytona runs select `development680` by default, keeping the 150 evaluation questions out of smoke tests.
Use `--split eval150` explicitly when ready to evaluate.

Start with one question using your configured default model:

```bash
mini-extra browsecomp-plus run --limit 1 \
  --output runs/browsecomp-plus/smoke \
  -c agent.step_limit=10 \
  -c agent.cost_limit=0.25
```

Then run the development set with an explicit provider/model name:

```bash
mini-extra browsecomp-plus run --model PROVIDER/MODEL \
  --workers 4 --output runs/browsecomp-plus/my-model
```

The default limits are 100 model calls, $3, and 1,800 seconds per question.
The cost and time limits are checked between calls and can be exceeded by an in-flight call; they are stopping thresholds rather than hard billing caps.
Four workers can have four requests in flight, and a full benchmark can be expensive.
Override settings with repeatable `-c` options accepting YAML files or `key=value` values.
For example, `-c model.model_kwargs.reasoning_effort='"high"'` passes a provider setting, and `--model-class openrouter` selects another existing model interface.
Use a separate output directory for each model, configuration, dataset, or index.

The corpus interface accepts `search QUERY`, `get_document DOCID`, and `submit FINAL RESPONSE` through the existing bash tool.
These commands do not execute a shell or provide access to the live web, local files, reference answers, or credentials.
Search returns five full documents by default, matching the upstream untruncated BM25 setting.
Use `-c environment.k=10` or `-c environment.snippet_chars=4000` to change retrieval settings, and report those changes when comparing results.
Snippet truncation uses characters; zero keeps full text.
Document lookups return the full document unless a token cap is configured.
The existing `compact_block` and `expand_block` tools remain available for context management.

### OpenRouter and the paper settings

Copy `.env.example` to `.env` and supply `OPENROUTER_API_KEY` and, for cloud workers, `DAYTONA_API_KEY`.
The provided profile selects `qwen/qwen3.5-9b` through OpenRouter with Together as the only allowed provider and provider fallbacks disabled:

```bash
mini-extra browsecomp-plus run --limit 1 \
  -c browsecomp_plus_openrouter \
  --output runs/browsecomp-plus/openrouter
```

This profile uses ten BM25 results, Qwen tokenizer previews of 512 tokens for new documents and 128 for previously retrieved documents, and an 8,192-token document cap.
It sets a 131,072-token context hint, 100 model calls, a $0.50 stopping threshold per task, and a 30-minute task timeout.
The cost threshold is checked between calls and can be exceeded by an in-flight request.
Ten concurrent tasks can therefore consume roughly ten times the per-task threshold.

This is a paper-guided retrieval configuration for this repository's existing agent.
Its `compact_block` and `expand_block` tools differ from the paper's external summarizer and memory querier, and the profile does not reproduce the paper's trained policy.
The model decides whether context tools are useful; a short successful episode can use neither.

### Ten parallel Daytona workers

Daytona is optional for local corpus retrieval.
The cloud runner uses one task per sandbox and shares a snapshot containing Java 21, CPU dependencies, the pinned BM25 index, and the Qwen tokenizer:

```bash
uv pip install --python .venv/bin/python -e '.[browsecomp-plus,daytona]'
mini-extra browsecomp-plus daytona snapshot --name pocket-browsecomp-plus
mini-extra browsecomp-plus daytona create --snapshot pocket-browsecomp-plus --workers 10
mini-extra browsecomp-plus daytona check
mini-extra browsecomp-plus daytona run --workers 10 --limit 20 --success-limit 20 --cleanup
```

`check` verifies real corpus retrieval and tokenizer loading in every worker without model calls.
The launcher sets `JAVA_HOME` before loading Pyjnius, including on snapshots with only a Java runtime.
The application wheel is installed separately from the shared snapshot.
Only the selected task ID, question, and configuration are uploaded for each task; reference answers and the Daytona key stay off the workers.
The OpenRouter key is supplied through sandbox environment variables and is never baked into the snapshot.

`--limit` bounds attempted tasks; `--success-limit` stops after the requested number of submissions without scheduling excess concurrent tasks.
Twenty attempts may produce fewer than twenty submissions if tasks hit limits or fail.
A completed submission does not establish answer correctness; correctness still requires the judge below.
By default, local outputs use a temporary directory and are deleted after the summary is printed.
Remote task workspaces and process sessions are always removed, including on failure.
The reusable corpus snapshot and worker pool remain available, with workers configured to stop and delete after 30 idle minutes.
Use `daytona status` to inspect workers and `daytona delete` to delete the pool explicitly.

To preserve trajectories for an evaluation run, select the held-out split explicitly and opt into artifacts:

```bash
mini-extra browsecomp-plus daytona run --workers 10 --split eval150 --limit 150 \
  --keep-artifacts --output runs/browsecomp-plus/eval150
```

The pool manifest is ignored at `data/browsecomp-plus/daytona-pool.json`.
Create a fresh pool after application changes or after idle deletion.

Use repeated `--query-id ID`, `--offset N`, and `--limit N` to select queries in dataset order.
`--queries FILE` accepts an alternative decrypted JSONL with `query_id` and `query` fields, and `--index DIR` accepts an alternative Lucene index.
Runs automatically skip completed queries, retry incomplete queries, and reject changed settings in an existing output directory.
Use `--redo-existing` to rerun completed queries with the same settings.
A batch exits with status 1 if any query does not finish by submission, while preserving its failure record and trajectory.
Unexpected exceptions preserve the current query's record and propagate to the caller.

Each run contains:

```text
run.yaml
summary.yaml
QUERY_ID.json
trajectories/QUERY_ID/trajectory.json
trajectories/QUERY_ID/context.sqlite
```

The top-level query files contain `query_id`, `status`, `tool_call_counts`, `retrieved_docids`, and `result`, plus model and run metadata.
Their `metrics` include provider-reported peak input tokens, peak input-plus-output tokens, total usage, and semantic tool counts including context tools.
The summary reports the maximum input context across all calls, the mean per-task peak, and average tool calls per attempted task.
Tool counts include `submit`; `avg_research_tool_calls_per_task` excludes it.
When a success target is provided, a second summary reports the same metrics for completed submissions only.
Retrieval IDs include every document returned by search or successfully fetched directly, even if the agent later compacts the observation.
Only submitted answers receive `status: completed`; budget exhaustion remains a failed run for the official evaluator.
Trajectories and context archives are kept in subdirectories so the evaluator sees only query results at the run directory's top level.

## Evaluate

Use the official evaluator rather than a custom answer matcher.
Clone its pinned source into the ignored data directory:

```bash
git clone https://github.com/texttron/BrowseComp-Plus.git data/browsecomp-plus/upstream
git -C data/browsecomp-plus/upstream checkout 046949032b0328319cc9a02663a759ec601d9402
```

On a Linux GPU machine with enough memory for Qwen3-32B, install the judge in a separate environment and evaluate the exported run directory:

```bash
uv venv --python 3.11 data/browsecomp-plus/judge-venv
uv pip install --python data/browsecomp-plus/judge-venv/bin/python 'vllm>=0.9.0' numpy tqdm
data/browsecomp-plus/judge-venv/bin/python \
  data/browsecomp-plus/upstream/scripts_evaluation/evaluate_run.py \
  --input_dir runs/browsecomp-plus/my-model \
  --ground_truth data/browsecomp-plus/queries.jsonl \
  --qrel_evidence data/browsecomp-plus/upstream/topics-qrels/qrel_evidence.txt \
  --eval_dir runs/browsecomp-plus/evals \
  --tensor_parallel_size 1
```

Copy the prepared questions and run outputs to the GPU machine and adjust paths and GPU count as needed.
The judge loads `Qwen/Qwen3-32B`; it does not need a paid model API key.
Upstream requests Qwen3-32B judging for leaderboard submissions.

For an API-based check using the upstream alternate GPT-4.1 judge, export `OPENAI_API_KEY` and run:

```bash
python data/browsecomp-plus/upstream/scripts_evaluation/evaluate_with_openai.py \
  --input_dir runs/browsecomp-plus/my-model \
  --ground_truth data/browsecomp-plus/queries.jsonl \
  --qrel_evidence data/browsecomp-plus/upstream/topics-qrels/qrel_evidence.txt \
  --eval_dir runs/browsecomp-plus/evals-openai
```

This alternate judge incurs API charges and should be identified separately from the official Qwen judge in reported results.

## Validate the integration

```bash
uv pip install --python .venv/bin/python -e '.[dev,browsecomp-plus]'
pytest -q tests/run/test_browsecomp_plus.py
```

These tests create a real Lucene index and run the CLI with the repository's deterministic replay model.
They verify retrieval, token caps, document lookup, submission, context compaction and expansion, usage accounting, held-out partitioning, budget exhaustion, concurrent query isolation, reference-answer isolation, resume behavior, and configuration mismatch rejection without paid API calls.
