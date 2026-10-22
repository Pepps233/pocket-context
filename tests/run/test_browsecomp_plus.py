import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from pocketcontextagent.environments.browsecomp_plus import BrowseCompPlusEnvironment
from pocketcontextagent.exceptions import Submitted
from pocketcontextagent.models.test_models import make_output
from pocketcontextagent.run.benchmarks.browsecomp_plus import app
from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import BM25Retriever, decrypt


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


@pytest.mark.parametrize(("step_limit", "status"), [(5, "completed"), (1, "LimitsExceeded")])
def test_benchmark_cli_end_to_end(bm25_index: Path, tmp_path: Path, step_limit: int, status: str):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps({"query_id": str(i), "query": "Which city hosts the festival?", "answer": "SECRET_REFERENCE"})
            for i in range(2)
        )
    )
    config = tmp_path / "replay.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "model_name": "deterministic",
                    "model_class": "deterministic",
                    "outputs": [
                        make_output("Look for the festival.", [{"command": "search jazz festival"}]),
                        make_output("Read the evidence.", [{"command": "get_document city"}]),
                        make_output(
                            "The document identifies the city.",
                            [{"command": "submit Explanation: Jazz [city].\nExact Answer: Montréal\nConfidence: 100%"}],
                        ),
                    ],
                },
                "agent": {"step_limit": step_limit, "cost_limit": 0},
            }
        )
    )
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
    ]
    environment = os.environ | {"MSWEA_GLOBAL_CONFIG_DIR": str(tmp_path / "config"), "MSWEA_SILENT_STARTUP": "1"}
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert result.returncode == (0 if status == "completed" else 1), result.stdout + result.stderr
    assert {path.name for path in output.glob("*.json")} == {"0.json", "1.json"}
    for query_id in ("0", "1"):
        data = json.loads((output / f"{query_id}.json").read_text())
        assert data["query_id"] == query_id
        assert data["status"] == status
        assert data["retrieved_docids"] == ["city"]
        assert data["tool_call_counts"] == {"search": 1, "get_document": int(status == "completed")}
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
