"""Evaluation: rule-based categorisation accuracy + optional agent Q&A."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from finance_agent.db.ingest import ingest_csv
from finance_agent.db.schema import get_connection, init_db
from finance_agent.tools.categoriser import categorise_description


def measure_categorisation_accuracy(labelled_csv: Path) -> dict[str, Any]:
    """Compare rule-based predictions against CSV ground-truth categories.

    Runs WITHOUT an LLM.
    """
    import csv

    total = 0
    correct = 0
    confusion: dict[str, int] = {}
    with labelled_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            truth = (row.get("category") or "").strip()
            if not truth:
                continue
            pred = categorise_description(row["description"])
            total += 1
            if pred == truth:
                correct += 1
            else:
                key = f"{truth}->{pred}"
                confusion[key] = confusion.get(key, 0) + 1
    return {
        "n": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "confusion_top": dict(sorted(confusion.items(), key=lambda x: -x[1])[:10]),
    }


def run_evaluation(
    labelled_csv: Path,
    questions_path: Path,
    *,
    with_llm: bool = False,
) -> dict[str, Any]:
    cat = measure_categorisation_accuracy(labelled_csv)
    report: dict[str, Any] = {
        "categorisation_accuracy": cat["accuracy"],
        "categorisation_correct": cat["correct"],
        "categorisation_n": cat["n"],
        "categorisation_confusion_top": cat["confusion_top"],
    }

    if with_llm:
        from finance_agent.agent.graph import build_agent

        questions = json.loads(questions_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "eval.db")
            init_db(db_path)
            # Ingest without labels so the agent must categorise / query
            ingest_csv(db_path, labelled_csv, use_csv_category=True)
            agent = build_agent(db_path, mock_llm=False)
            hits = 0
            details = []
            for item in questions:
                result = agent.chat(item["question"], use_llm=True)
                expected = (item.get("expected_contains") or "").lower()
                ok = expected in result["answer"].lower() if expected else False
                hits += int(ok)
                details.append({"id": item.get("id"), "ok": ok, "answer": result["answer"]})
            report["agent_answer_accuracy"] = hits / max(len(questions), 1)
            report["agent_answer_hits"] = hits
            report["agent_details"] = details
    else:
        report["agent_answer_accuracy"] = None
        report["note"] = (
            "Agent Q&A accuracy requires a live LLM. "
            "Re-run with --with-llm after configuring an API key."
        )

    out = Path("eval/last_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    compact = {k: v for k, v in report.items() if k != "agent_details"}
    out.write_text(json.dumps(compact, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = run_evaluation(
        Path("data/synthetic_statement.csv"),
        Path("eval/questions.json"),
        with_llm=False,
    )
    print(json.dumps(report, indent=2))
