# Study Guide — Finance AI Agent

## Big picture

An agent is an LLM (or a rule router) that can **call tools**. Here the tools talk to a SQLite ledger of bank transactions. Guardrails stop dangerous SQL and off-topic questions.

---

## File-by-file

### `db/schema.py`
Creates a simple `transactions` table. SQLite keeps the demo zero-ops.

### `db/ingest.py`
- `generate_synthetic_statement`: builds ~300 fake rows; every description starts with `[SYNTHETIC]`.
- `ingest_csv`: loads CSV → SQLite.

**Why synthetic data?** Real bank data is sensitive. Interviewers respect clear labelling of fake data.

### `tools/sql_tool.py`
`validate_select_sql` blocks INSERT/UPDATE/DELETE/DROP/PRAGMA/multi-statements. `run_readonly_sql` wraps the query with a hard LIMIT.

**Why:** LLM-generated SQL is untrusted. Defense in depth beats prompt-only safety.

### `tools/categoriser.py`
Ordered regex rules (payroll → income, Whole Foods → groceries, …). Optional `llm_fallback` callable for unknowns.

**Why rules first?** Fast, free, deterministic, easy to test. LLM as backup for long-tail merchants.

### `tools/aggregates.py`
Pre-built GROUP BY for category / account / month — covers the most common questions without free-form SQL.

### `agent/graph.py`
`FinanceAgent`:
1. Scope guardrail → refuse.
2. Deterministic rule router for common intents (works offline).
3. Optional LLM tool loop with JSON tool calls and one retry on failure.

**Why a rule router at all?** Portfolio demos and CI must work without API keys. The LLM path is there for the “real” agent story.

### `api/app.py` / `cli.py` / `evaluate.py`
Thin interfaces over the same agent. Evaluation always measures categorisation accuracy; agent Q&A is opt-in with `--with-llm`.

---

## Design choices

1. **SELECT-only SQL** rather than an ORM query builder — shows you understand injection / privilege risk.
2. **Rules before LLM** for categorisation — cost, latency, determinism.
3. **Refusal patterns** — production agents need scope control, not just helpfulness.
4. **Retry on tool failure** — fragile tool use is a common agent failure mode.

---

## Interview questions

**1. What is an AI agent?**
A system that plans and uses tools/actions to achieve a goal, not just complete text.

**2. Why validate SQL instead of trusting the model?**
Models can be prompt-injected into emitting `DROP TABLE`. Validation is a hard gate.

**3. How do you stop prompt injection via transaction descriptions?**
Treat DB text as untrusted data, not instructions; strip/sandbox tool outputs; keep system prompt separate.

**4. Rule-based vs LLM categorisation trade-offs?**
Rules: precise, cheap, brittle on novelty. LLM: flexible, expensive, non-deterministic. Hybrid is best.

**5. What is LangGraph for?**
Explicit state machines for multi-step agent workflows (cycles, retries, branching) beyond a single chain.

**6. How would you test an agent?**
Unit-test tools + guardrails; golden-question sets; mock the LLM; track refusal rate and tool-error rate.

**7. How do you handle PII?**
Don't log raw statements; encrypt at rest; redact in traces; synthetic data for demos.

**8. What does read-only mean in practice?**
DB user privileges + statement allow-list + wrapped LIMIT — not just a prompt saying “don't modify”.

**9. Failure modes of tool calling?**
Wrong tool, bad args, timeouts, partial results, hallucinated success. Mitigate with schemas, retries, and grounded answers.

**10. How would you add multi-user support?**
Per-user DB / row-level `user_id` filter forced in every query; authn on `/chat`.

**11. Why SQLite here?**
Zero ops for a portfolio demo; swap to Postgres with the same SQL dialect subset later.

**12. How do you measure categorisation quality?**
Accuracy / F1 on labelled merchants; confusion matrix for systematic misses.

**13. What would you build next?**
Budgets, anomaly detection, streaming bank APIs (Plaid), richer LangGraph graph with explicit nodes, eval harness with LLM-as-judge.

**14. Explain the refusal path.**
Regex/keyword gate → fixed refusal string → no tools invoked. Prevents spending tokens and leaking data on off-topic asks.

**15. Walk through “How much did I spend on groceries?”**
Guardrail pass → rule router extracts category → SELECT SUM(amount) WHERE category='groceries' AND amount<0 → format answer.
