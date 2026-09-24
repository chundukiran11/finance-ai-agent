from finance_agent.tools.categoriser import categorise_description


def test_known_merchants() -> None:
    assert categorise_description("[SYNTHETIC] WHOLEFOODS MARKET") == "groceries"
    assert categorise_description("[SYNTHETIC] NETFLIX.COM") == "entertainment"
    assert categorise_description("[SYNTHETIC] ACME CORP PAYROLL") == "income"
    assert categorise_description("[SYNTHETIC] LANDLORD RENT") == "rent"


def test_unknown_falls_back() -> None:
    assert categorise_description("ZZZ UNKNOWN MERCHANT 999") == "uncategorised"
