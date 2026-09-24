"""Typer CLI for the finance agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from finance_agent.agent.graph import build_agent
from finance_agent.config import get_settings
from finance_agent.db.ingest import generate_synthetic_statement, ingest_csv
from finance_agent.db.schema import init_db
from finance_agent.tools.categoriser import categorise_all

app = typer.Typer(help="Personal Finance AI Agent CLI", no_args_is_help=True)
console = Console()


@app.command("generate-data")
def generate_data(
    output: Path = typer.Option(Path("data/synthetic_statement.csv")),
    n: int = typer.Option(300, help="Number of synthetic transactions"),
    seed: int = 42,
) -> None:
    """Generate a clearly-labelled synthetic bank statement CSV."""
    path = generate_synthetic_statement(output, n_transactions=n, seed=seed)
    console.print(Panel.fit(f"Wrote {n} synthetic rows to {path}"))


@app.command()
def ingest(
    csv_path: Path = typer.Argument(..., exists=True),
    db: Optional[Path] = None,
    with_labels: bool = typer.Option(False, help="Keep CSV category column"),
) -> None:
    """Ingest a bank-statement CSV into SQLite."""
    settings = get_settings()
    db_path = str(db or settings.database_path)
    init_db(db_path)
    n = ingest_csv(db_path, csv_path, use_csv_category=with_labels)
    console.print(f"Inserted {n} rows into {db_path}")


@app.command()
def categorise(db: Optional[Path] = None) -> None:
    """Run the rule-based categoriser on uncategorised rows."""
    settings = get_settings()
    db_path = str(db or settings.database_path)
    result = categorise_all(db_path)
    console.print(result)


@app.command()
def chat(
    message: str = typer.Argument(...),
    db: Optional[Path] = None,
    use_llm: bool = False,
    mock: bool = typer.Option(True, help="Use mock LLM (default True)"),
) -> None:
    """Ask the finance agent a question."""
    settings = get_settings()
    db_path = str(db or settings.database_path)
    agent = build_agent(db_path, mock_llm=mock or not use_llm, settings=settings)
    result = agent.chat(message, use_llm=use_llm)
    console.print(Panel(result["answer"], title=f"route={result.get('route')}"))


@app.command()
def evaluate(
    labelled_csv: Path = typer.Option(Path("data/synthetic_statement.csv")),
    questions: Path = typer.Option(Path("eval/questions.json")),
    with_llm: bool = False,
) -> None:
    """Run categorisation accuracy (always) and optional agent Q&A eval."""
    from finance_agent.evaluate import run_evaluation

    report = run_evaluation(
        labelled_csv=labelled_csv,
        questions_path=questions,
        with_llm=with_llm,
    )
    console.print(Panel.fit(json.dumps(report, indent=2), title="Evaluation"))


@app.command()
def serve(host: Optional[str] = None, port: Optional[int] = None) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "finance_agent.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
    )


if __name__ == "__main__":
    app()
