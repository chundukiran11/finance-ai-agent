def test_spending_summary(agent) -> None:
    result = agent.chat("Give me a spending summary by category")
    assert result["refused"] is False
    assert "category" in result["answer"].lower() or "-" in result["answer"]
    assert result["route"] == "rules"


def test_refuse_out_of_scope(agent) -> None:
    result = agent.chat("Tell me a joke about pirates please")
    assert result["refused"] is True
    assert result["route"] == "guardrail"


def test_refuse_medical(agent) -> None:
    result = agent.chat("Please diagnose my medical symptoms today")
    assert result["refused"] is True


def test_total_income(agent) -> None:
    result = agent.chat("What is my total income?")
    assert "income" in result["answer"].lower()
