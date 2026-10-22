"""Prepare and run BrowseComp-Plus with Pocket Context Agent."""

from pathlib import Path

import typer

from pocketcontextagent.run.benchmarks.utils.browsecomp_plus import prepare_data

DEFAULT_DATA_DIR = Path("data/browsecomp-plus")
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def benchmark() -> None:
    """Prepare data and evaluate Pocket Context Agent on BrowseComp-Plus."""


@app.command()
def prepare(
    data_dir: Path = typer.Option(DEFAULT_DATA_DIR, help="Local dataset and index directory."),
    download_index: bool = typer.Option(True, "--index/--no-index", help="Download the official BM25 index (2.2 GB)."),
) -> None:
    """Download pinned queries, decrypt locally, and fetch the official BM25 index."""
    prepare_data(data_dir, download_index=download_index)
    typer.echo(f"Prepared BrowseComp-Plus in {data_dir}")


if __name__ == "__main__":
    app()
