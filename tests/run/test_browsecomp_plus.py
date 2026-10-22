import json
import subprocess
import sys
from pathlib import Path

import pytest

from pocketcontextagent.environments.browsecomp_plus import BrowseCompPlusEnvironment
from pocketcontextagent.exceptions import Submitted
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
