# Pocket Context

Pocket Context lets an agent choose which completed exchanges to compress while keeping the original messages available for exact recall.
This repository includes Pocket Context Agent (`pca`), a command-line coding agent, and a Python library for builders who control their own agent loop.

## Quick start: CLI

Requires Python 3.10 or newer and a model provider API key.
Install the package, then run `pca` from the directory you want the agent to work in:

```bash
pip install pocketcontextagent
cd your-project
pca --task "Explain how this project works"
```

The first run guides you through model setup.
Run `pca` without `--task` for an interactive task prompt, or `pca --help` for command options.
See the [full CLI guide](docs/usage/pca.md) and [model setup guide](docs/models/quickstart.md).

## How it works

When the context reaches a configurable usage threshold (50% by default), the agent can call `compact_block` to select a contiguous range of complete exchanges.
A separate compressor writes a short memory, and Pocket Context replaces the selected range with that memory while archiving the originals locally.
The agent can call `expand_block` to retrieve the original messages when it needs a detail the memory omitted.
See the [context management guide](docs/advanced/context_compaction.md) for the full behavior.

## Performance

On 150 BrowseComp-Plus tasks at a 128K context limit, Pocket Context Agent reached **80.67% pass@1** with **61.98K mean peak input tokens** per task.
ACM reached 80.67% with 65.28K tokens, while ReAct reached 68.00% with 71.59K tokens under the same evaluation settings.
On 125 SWE-bench Verified tasks, Pocket Context Agent reached **77.60% pass@1 at 32K**, versus 71.20% for ReAct; at 128K, it reached 80.80% versus 84.00% for ReAct.
These results cover the selected tasks rather than the full benchmarks.
See the [comparison setup](docs/usage/context_comparison.md) for evaluation methods.

## Evaluation tasks

The task selections used in the evaluations are in [`evals/`](evals/):

- [BrowseComp-Plus: 150 tasks](evals/browsecomp-plus-150-task-ids.json)
- [SWE-bench Verified: 125 tasks](evals/swebench-verified-125-task-ids.json)
- [DeepSWE v1.1: 50 tasks](evals/deepswe-v1.1-50-task-ids.json)

## Python library

The `pocket_context` package provides `ContextStore` for recording messages, compacting a selected range with your compression agent, expanding archived messages, and exposing context tools at a chosen usage threshold.
Install `pocketcontextagent` as above, then import `ContextStore` from `pocket_context`.
See the [library quickstart](library/README.md) for a minimal example and prompt customization.

## Contributing and license

See the [contribution guide](docs/contributing.md).
This project is licensed under the [MIT License](LICENSE.md).
