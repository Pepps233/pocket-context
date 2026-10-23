import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pocketcontextagent.agents.default import DefaultAgent
from pocketcontextagent.environments.browsecomp_plus import BrowseCompPlusEnvironment, extract_answer
from pocketcontextagent.exceptions import Submitted
from pocketcontextagent.models.test_models import DeterministicToolcallModel, make_output, make_toolcall_output
from pocketcontextagent.run.benchmarks.browsecomp_plus import app
from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import (
    BM25Retriever,
    decrypt,
    reserve_evaluation,
    select_split,
)
from pocketcontextagent.run.benchmarks.utils.metrics import episode_metrics, summarize_results


@pytest.fixture
def bm25_index(tmp_path: Path) -> Path:
    pytest.importorskip("pyserini")
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "corpus.jsonl").write_text(
        "\n".join(
            json.dumps(document)
            for document in [
                {"id": "city", "contents": "Montréal hosts a jazz festival in Canada. " * 10},
                {"id": "other", "contents": "Bananas grow in tropical climates."},
            ]
        ),
        encoding="utf-8",
    )
    index = tmp_path / "index"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "1",
            "--input",
            str(documents),
            "--index",
            str(index),
            "--storeRaw",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return index


def test_decrypt_upstream_format():
    assert decrypt(
        {"query_id": "42", "query": "d0AVkObqe8BhQoI=", "answer": "bUcSh/wJsch5", "evidence": ["REcf3r/9"], "count": 1}
    ) == {"query_id": "42", "query": "Which city?", "answer": "Montréal", "evidence": ["doc-17"], "count": 1}


def test_real_bm25_environment(bm25_index: Path, tmp_path: Path):
    env = BrowseCompPlusEnvironment(BM25Retriever(bm25_index), k=1, snippet_chars=20)
    assert json.loads(env.execute({"command": "search jazz festival"})["output"])[0] == {
        "docid": "city",
        "score": env.retriever.search("jazz festival", 1)[0]["score"],
        "snippet": "Montréal hosts a jaz",
        "context_status": "new",
    }
    assert "Montréal" in json.loads(env.execute({"command": "get_document city"})["output"])["text"]
    assert "not found" in env.execute({"command": "get_document missing"})["output"]
    assert env.execute({"command": f"touch {tmp_path / 'unwanted'}"})["returncode"] == 1
    assert not (tmp_path / "unwanted").exists()
    assert env.execute({"command": "search"})["returncode"] == 1
    assert env.retrieved_docids == {"city"}
    assert env.tool_call_counts == {"search": 1, "get_document": 2}
    with pytest.raises(Submitted) as submitted:
        env.execute({"command": "submit Explanation: Jazz [city].\nExact Answer: Montréal\nConfidence: 100%"})
    assert submitted.value.messages[0]["extra"]["submission"].endswith("Exact Answer: Montréal\nConfidence: 100%")
    env.retriever.close()


@pytest.mark.parametrize(
    ("step_limit", "status", "memory", "text_answer"),
    [
        (5, "completed", False, False),
        (1, "LimitsExceeded", False, False),
        (10, "completed", True, False),
        (5, "completed", False, True),
    ],
)
def test_benchmark_cli_end_to_end(
    bm25_index: Path, tmp_path: Path, step_limit: int, status: str, memory: bool, text_answer: bool
):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps({"query_id": str(i), "query": "Which city hosts the festival?", "answer": "SECRET_REFERENCE"})
            for i in range(2)
        )
    )
    config = tmp_path / "replay.yaml"
    configuration = {
        "model": {
            "model_name": "deterministic",
            "model_class": "deterministic",
            "outputs": [
                make_output("Look for the festival.", [{"command": "search jazz festival"}]),
                make_output("Read the evidence.", [{"command": "get_document city"}]),
                make_output(
                    "Explanation: Jazz [city].\nExact Answer: Montréal\nConfidence: 100%"
                    if text_answer
                    else "The document identifies the city.",
                    []
                    if text_answer
                    else [{"command": "submit Explanation: Jazz [city].\nExact Answer: Montréal\nConfidence: 100%"}],
                ),
            ],
        },
        "agent": {"step_limit": step_limit, "cost_limit": 0, "compact_min_chars": 0, "compact_min_reduction": 0},
    }
    if memory:
        outputs = configuration["model"]["outputs"]
        outputs[0]["content"] = "Initial corpus search."
        outputs[2:2] = [
            make_output("Check another lead.", [{"command": "search bananas"}]),
            make_output(
                "Archive the evidence.",
                [
                    {
                        "type": "compact_block",
                        "start_quote": "Initial corpus search.",
                        "end_quote": '"text":',
                        "summary": "Montréal hosts a jazz festival [city].",
                    }
                ],
            ),
            make_output("Recover the original evidence.", [{"type": "expand_block", "tag": "ctx-0001"}]),
        ]
        configuration["model"]["model_class"] = "pocketcontextagent.models.test_models.DeterministicToolcallModel"
        configuration["model"]["outputs"] = []
        for index, message in enumerate(outputs):
            action = dict(message["extra"]["actions"][0])
            summary = action.pop("summary", None)
            configuration["model"]["outputs"].append(
                make_toolcall_output(
                    message["content"],
                    [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": action.get("type", "bash"),
                                "arguments": json.dumps({key: value for key, value in action.items() if key != "type"}),
                            },
                        }
                    ],
                    [action | {"tool_call_id": f"call-{index}"}],
                )
            )
            if summary is not None:
                configuration["model"]["outputs"].append(make_output(summary, []))
    config.write_text(yaml.safe_dump(configuration))
    output = tmp_path / "runs"
    command = [
        sys.executable,
        "-m",
        "pocketcontextagent.run.utilities.mini_extra",
        "browsecomp-plus",
        "run",
        "--queries",
        str(queries),
        "--index",
        str(bm25_index),
        "--output",
        str(output),
        "-c",
        str(config),
        "--workers",
        "2",
        "--split",
        "all",
    ]
    environment = os.environ | {"MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config"), "MSWEA_SILENT_STARTUP": "1"}
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert result.returncode == (0 if status == "completed" else 1), result.stdout + result.stderr
    assert {path.name for path in output.glob("*.json")} == {"0.json", "1.json"}
    for query_id in ("0", "1"):
        data = json.loads((output / f"{query_id}.json").read_text())
        assert data["final_answer"] == ("Montréal" if status == "completed" else "")
        assert data["confidence"] == 100
        assert data["query_id"] == query_id
        assert data["status"] == status
        assert data["retrieved_docids"] == (["city", "other"] if memory else ["city"])
        assert data["tool_call_counts"] == {"search": 2 if memory else 1, "get_document": int(status == "completed")}
        if memory:
            assert data["metrics"]["tool_call_counts"] == {
                "search": 2,
                "get_document": 1,
                "compact_block": 1,
                "expand_block": 1,
                "submit": 1,
            }
        assert data["result"] == [
            {
                "type": "output_text",
                "output": "Explanation: Jazz [city].\nExact Answer: Montréal\nConfidence: 100%"
                if status == "completed"
                else "",
            }
        ]
        trajectory = output / "trajectories" / query_id / "trajectory.json"
        assert "SECRET_REFERENCE" not in trajectory.read_text()
        assert (trajectory.parent / "context.sqlite").exists()
        if memory:
            context = json.loads(trajectory.read_text())["context"]
            assert context["revision"] == 1
            assert "ctx-0001" in context["blocks"]
            assert len(context["expansions"]) == 1
    before = (output / "0.json").stat().st_mtime_ns
    resumed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert resumed.returncode == result.returncode, resumed.stdout + resumed.stderr
    assert ((output / "0.json").stat().st_mtime_ns == before) == (status == "completed")
    changed = subprocess.run(
        command + ["-c", "environment.k=2"], capture_output=True, text=True, env=environment, timeout=60
    )
    assert changed.returncode != 0
    assert "avoid mixing results" in changed.stdout + changed.stderr


@pytest.mark.parametrize(
    ("ids", "arguments", "error"),
    [
        (["../escape"], [], "Query IDs must be unique"),
        (["1", "1"], [], "Query IDs must be unique"),
        (["1"], ["--query-id", "missing"], "Unknown query IDs"),
        (["1"], ["--offset", "1"], "No queries selected"),
    ],
)
def test_cli_rejects_invalid_selection(tmp_path: Path, ids: list[str], arguments: list[str], error: str):
    queries = tmp_path / "queries.jsonl"
    queries.write_text("\n".join(json.dumps({"query_id": iid, "query": "Question"}) for iid in ids))
    result = CliRunner().invoke(
        app,
        ["run", "--queries", str(queries), "--model", "deterministic", "--output", str(tmp_path / "runs"), *arguments],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert error in str(result.exception)
    assert not (tmp_path / "runs").exists()


def test_reserved_author_split(tmp_path: Path):
    from pocketcontextagent.config import builtin_config_dir

    manifest = json.loads((builtin_config_dir / "benchmarks/browsecomp_plus_eval150.json").read_text())
    instances = [
        {"query_id": iid, "query": "Synthetic question", "answer": "Synthetic reference"}
        for iid in manifest["query_ids"] + [f"dev-{index}" for index in range(680)]
    ]
    (tmp_path / "queries.jsonl").write_text("\n".join(json.dumps(row) for row in instances))
    assert reserve_evaluation(tmp_path) == {"development680": 680, "eval150": 150}
    evaluation = [json.loads(line) for line in (tmp_path / "eval150.jsonl").read_text().splitlines()]
    development = [json.loads(line) for line in (tmp_path / "development680.jsonl").read_text().splitlines()]
    assert {row["query_id"] for row in evaluation} == set(manifest["query_ids"])
    assert not {row["query_id"] for row in evaluation} & {row["query_id"] for row in development}
    assert select_split(instances, "development680") == development
    with pytest.raises(ValueError, match="Split must be"):
        select_split(instances, "unknown")


def test_token_limited_retrieval(bm25_index: Path, tmp_path: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(
        models.WordLevel(
            {
                word: index
                for index, word in enumerate(
                    ["[UNK]", "Montréal", "hosts", "a", "jazz", "festival", "in", "Canada", "."]
                )
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.save_pretrained(tmp_path / "tokenizer")
    env = BrowseCompPlusEnvironment(
        BM25Retriever(bm25_index),
        k=1,
        tokenizer_name=str(tmp_path / "tokenizer"),
        snippet_tokens=5,
        seen_snippet_tokens=2,
        document_tokens=8,
    )
    assert (
        json.loads(env.execute({"command": "search jazz"})["output"])[0]["snippet"] == "Montréal hosts a jazz festival"
    )
    assert json.loads(env.execute({"command": "search jazz"})["output"])[0]["snippet"] == "Montréal hosts"
    assert json.loads(env.execute({"command": "search jazz"})["output"])[0]["previously_retrieved"] is True
    assert len(env.tokenizer.encode(json.loads(env.execute({"command": "get_document city"})["output"])["text"])) == 8
    assert env.tool_call_counts == {"search": 3, "get_document": 1}
    env.retriever.close()


def test_metrics_count_context_calls_and_deduplicate_responses():
    messages = [
        {
            "extra": {
                "actions": [action | {"tool_call_id": str(index)}],
                "response": {
                    "id": str(index),
                    "usage": {"prompt_tokens": tokens, "completion_tokens": 5, "total_tokens": tokens + 5},
                },
            }
        }
        for index, (action, tokens) in enumerate(
            [
                ({"command": "search jazz"}, 100),
                ({"type": "compact_block"}, 70),
                ({"type": "expand_block"}, 60),
                ({"command": "submit answer"}, 90),
            ]
        )
    ]
    metrics = episode_metrics(
        messages
        + [
            messages[0],
            {
                "object": "response",
                "id": "compaction-agent",
                "output": [],
                "usage": {"input_tokens": 95, "output_tokens": 5, "total_tokens": 100},
            },
        ]
    )
    assert metrics == {
        "peak_input_tokens": 100,
        "peak_total_tokens": 105,
        "total_input_tokens": 415,
        "total_output_tokens": 25,
        "calls_with_usage": 5,
        "tool_call_counts": {"search": 1, "compact_block": 1, "expand_block": 1, "submit": 1},
        "total_tool_calls": 4,
    }
    assert summarize_results([{"metrics": metrics, "status": "completed", "model_stats": {"instance_cost": 0.2}}]) == {
        "tasks": 1,
        "completed": 1,
        "peak_input_tokens": 100,
        "avg_peak_input_tokens": 100,
        "avg_tool_calls_per_task": 4,
        "avg_research_tool_calls_per_task": 3,
        "tool_call_counts": metrics["tool_call_counts"],
        "cost_usd": 0.2,
    }


@pytest.mark.parametrize(("final_action", "valid"), [([], True), ([{"command": "search bananas"}], True), ([], False)])
def test_context_budget_final_generation(bm25_index: Path, tmp_path: Path, final_action: list[dict], valid: bool):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query_id": "budget", "query": "Which city hosts the festival?"}))
    config = tmp_path / "replay.yaml"
    answer = (
        "**Explanation**: Jazz [city].\n**Exact Answer**: Montréal\n**Confidence**: 85%"
        if valid
        else "Insufficient evidence."
    )
    config.write_text(
        yaml.safe_dump(
            {
                "agent": {"context_token_limit": 128000, "cost_limit": 0},
                "model": {
                    "model_class": "deterministic",
                    "model_name": "deterministic",
                    "outputs": [
                        make_output("Investigate. " * 35000, [{"command": "search jazz festival"}]),
                        make_output(answer, final_action),
                        make_output("Should never run", [{"command": "submit WRONG"}]),
                    ],
                },
            }
        )
    )
    output = tmp_path / "runs"
    result = CliRunner().invoke(
        app,
        [
            "run",
            "--queries",
            str(queries),
            "--index",
            str(bm25_index),
            "--output",
            str(output),
            "--split",
            "all",
            "-c",
            str(config),
        ],
    )
    assert result.exit_code == (0 if valid else 1), result.output
    data = json.loads((output / "budget.json").read_text())
    trajectory = json.loads((output / "trajectories/budget/trajectory.json").read_text())
    assert data["status"] == ("completed" if valid else "LimitsExceeded")
    assert data["result"] == [{"type": "output_text", "output": answer}]
    assert data["model_stats"]["api_calls"] == 2
    assert data["tool_call_counts"] == {"search": 1, "get_document": 0}
    assert trajectory["messages"][3]["content"].splitlines()[-1].startswith("[CURRENT CONTEXT TOKEN: ")
    assert "Stop searching" in trajectory["messages"][4]["content"]
    assert trajectory["messages"][-1]["extra"]["context_limit_reached"] is True


@pytest.mark.parametrize(("expand",), [(False,), (True,)])  # noqa: PT006
def test_memory_aware_retrieval(bm25_index: Path, tmp_path: Path, expand: bool):
    turns = [
        ("Initial corpus search.", {"command": "search jazz"}),
        ("Read evidence.", {"command": "get_document city"}),
        ("Try another lead.", {"command": "search bananas"}),
        (
            "Archive evidence.",
            {
                "type": "compact_block",
                "start_quote": "Initial corpus search.",
                "end_quote": '"text":',
                "summary": "Montréal hosts the jazz festival [city].",
            },
        ),
    ]
    turns.append(("Revisit the remaining lead.", {"command": "search bananas"}))
    if expand:
        turns.append(("Recover evidence.", {"type": "expand_block", "tag": "ctx-0001"}))
    turns.extend(
        [
            ("Revisit the city.", {"command": "search jazz"}),
            ("Finish.", {"command": "submit Explanation: Jazz [city].\nExact Answer: Montréal\nConfidence: 90%"}),
        ]
    )
    outputs = []
    for index, (content, arguments) in enumerate(turns):
        action = dict(arguments)
        summary = action.pop("summary", None)
        outputs.append(
            make_toolcall_output(
                content,
                [
                    {
                        "id": str(index),
                        "type": "function",
                        "function": {
                            "name": action.get("type", "bash"),
                            "arguments": json.dumps({key: value for key, value in action.items() if key != "type"}),
                        },
                    }
                ],
                [action | {"tool_call_id": str(index)}],
            )
        )
        if summary is not None:
            outputs.append(make_output(summary, []))
    env = BrowseCompPlusEnvironment(BM25Retriever(bm25_index), k=1)
    agent = DefaultAgent(
        DeterministicToolcallModel(observation_template="{{output.output}}", outputs=outputs),
        env,
        system_template="Research.",
        instance_template="{{task}}",
        cost_limit=0,
        compact_min_chars=0,
        compact_min_reduction=0,
        context_token_limit=128000,
        context_db_path=tmp_path / "context.sqlite",
    )
    assert agent.run("Which city hosts the festival?")["exit_status"] == "Submitted"
    searches = [
        json.loads(message["extra"]["raw_output"])
        for message in agent.messages
        if message.get("role") == "tool" and message.get("extra", {}).get("docids")
    ]
    assert searches[0][0]["context_status"] == "new"
    assert searches[-1][0]["context_status"] == ("in_context" if expand else "compressed")
    assert searches[-2][0]["context_status"] == "in_context"
    assert searches[-1][0]["previously_retrieved"] is True
    assert ("expand_block" in searches[-1][0]["hint"]) != expand
    if not expand:
        assert searches[-1][0]["memory_tags"] == ["ctx-0001"]
    else:
        assert (
            next(
                message
                for message in agent.messages
                if message.get("role") == "tool" and "Expanded [ctx-0001]" in message.get("content", "")
            )["content"]
            .splitlines()[-1]
            .startswith("[CURRENT CONTEXT TOKEN: ")
        )
    env.retriever.close()


@pytest.mark.parametrize(
    ("text", "answer", "confidence"),
    [
        ("Exact Answer: First\nConfidence: 20%\n**Exact Answer:** Montréal**\nConfidence: 85%", "Montréal", 85),
        ("Exact Answer: First\nConfidence: 20%\n**Exact Answer**: Montréal\n**Confidence**: 65%", "Montréal", 65),
        ("**Exact Answer:** **Montréal**\n**Confidence:** **85%**", "Montréal", 85),
        ("Explanation: Evidence [city].\nExact Answer: Montréal", "Montréal", 100),
        ("I will search again.\nConfidence: 10%", "", 10),
        ("Exact Answer:\nConfidence: 10%", "", 10),
    ],
)
def test_answer_extraction(text: str, answer: str, confidence: int):
    assert extract_answer(text) == {"final_answer": answer, "confidence": confidence}
    assert BrowseCompPlusEnvironment.is_final_answer(text) == bool(answer)
