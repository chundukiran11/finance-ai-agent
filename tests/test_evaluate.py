from pathlib import Path

from finance_agent.db.ingest import generate_synthetic_statement
from finance_agent.evaluate import measure_categorisation_accuracy

ROOT = Path(__file__).resolve().parents[1]


def test_categorisation_accuracy(tmp_path: Path) -> None:
    csv_path = tmp_path / "labelled.csv"
    generate_synthetic_statement(csv_path, n_transactions=120, seed=1)
    report = measure_categorisation_accuracy(csv_path)
    assert report["n"] == 120
    assert report["accuracy"] >= 0.9
