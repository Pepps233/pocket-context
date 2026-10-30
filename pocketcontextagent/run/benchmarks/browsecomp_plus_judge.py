"""Run the upstream BrowseComp-Plus API judge with GPT-6 Luna."""

import subprocess
import sys
from pathlib import Path

import typer


def main(
    input_dir: Path = typer.Option(..., exists=True, file_okay=False),
    eval_dir: Path = typer.Option(..., help="Fresh output root for this judge's results."),
    data_dir: Path = typer.Option(Path("data/browsecomp-plus"), exists=True, file_okay=False),
    model: str = typer.Option("gpt-6-luna"),
    max_output_tokens: int = typer.Option(8192, min=1, help="Budget for reasoning and the judge's answer."),
) -> None:
    """Grade saved answers using OPENAI_API_KEY; this makes paid API calls."""
    subprocess.run(
        [
            sys.executable,
            str(data_dir / "upstream/scripts_evaluation/evaluate_with_openai.py"),
            "--input_dir",
            str(input_dir),
            "--ground_truth",
            str(data_dir / "queries.jsonl"),
            "--qrel_evidence",
            str(data_dir / "upstream/topics-qrels/qrel_evidence.txt"),
            "--eval_dir",
            str(eval_dir),
            "--model",
            model,
            "--max_output_tokens",
            str(max_output_tokens),
        ],
        check=True,
    )


if __name__ == "__main__":
    typer.run(main)
