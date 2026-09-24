# Study Guide — Finance AI Agent

## Big picture

An agent is an LLM (or a rule router) that can **call tools**. Here the tools talk to a SQLite ledger of bank transactions. The control flow is a real **LangGraph `StateGraph`** with typed state, nodes, and conditional edges. Guardrails stop dangerous SQL and off-topic questions.

---

## LangGraph topology

```
START → guardrail → (refuse → END)
                  → rules_router → (deterministic answer → END)
                                 → planner ⇄ tools → validate
                                          ↘ final_answer → END
```

| Node | Role |
|------|------|
| `guardrail` | Scope / safety refusal **before** any tool runs |
| `rules_router` | Offline answers for common intents when `use_llm=False` |
| `planner` | `llm.bind_tools(tools).invoke(messages)` — may emit tool calls |
| `tools` | LangGraph `ToolNode` executing the three finance tools |
| `validate` | Inspect tool payloads; on failure, nudge + retry (bounded) |
| `final_answer` | Produce the user-facing string from AI prose or a wrap-up call |

Compiled once in `build_finance_graph()`; `FinanceAgent.chat()` just `invoke`s it.

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

### `tools/aggregates.py`
Pre-built GROUP BY for category / account / month.

### `agent/tools_langchain.py`
Wraps the three tools as LangChain `StructuredTool`s closed over `db_path`. SQL validation errors are returned as `{"ok": false, ...}` so `ToolNode` does not crash the graph.

### `agent/graph.py`
Builds and compiles the `StateGraph`. `FinanceAgent` is a thin façade preserving the public `chat()` API used by CLI / FastAPI.

### `llm/factory.py`
Provider-agnostic factory. `MockChatModel` implements `bind_tools` and an optional `script` of `AIMessage`s (including `tool_calls`) so the graph is testable offline.

### `api/app.py` / `cli.py` / `evaluate.py`
Thin interfaces over the same agent. Evaluation always measures categorisation accuracy; agent Q&A is opt-in with `--with-llm`.

---

## Design choices

1. **Real LangGraph** — explicit nodes/edges are easier to reason about than a hidden `for` loop, and match how production agent frameworks are described.
2. **SELECT-only SQL** inside the tool — hard gate, not just a prompt.
3. **Rules before LLM** when `use_llm=False` — demos and CI stay free of API keys.
4. **Validate + retry node** — tool failure is a first-class edge, not an unhandled exception.
5. **Refusal in its own node** — out-of-scope questions never reach tools.

---

## Interview questions

**1. What is an AI agent?**
A system that plans and uses tools/actions to achieve a goal, not just complete text.

**2. What does LangGraph add over a bare LLM loop?**
Explicit state, named nodes, conditional edges, retries, and a compiled runnable you can test and visualise.

**3. Why validate SQL instead of trusting the model?**
Models can be prompt-injected into emitting `DROP TABLE`. Validation is a hard gate.

**4. How do you stop prompt injection via transaction descriptions?**
Treat DB text as untrusted data, not instructions; strip/sandbox tool outputs; keep system prompt separate.

**5. Rule-based vs LLM categorisation trade-offs?**
Rules: precise, cheap, brittle on novelty. LLM: flexible, expensive, non-deterministic. Hybrid is best.

**6. Walk me through your graph for “How much did I spend?” with `use_llm=False`.**
`guardrail` pass → `rules_router` matches total-spending pattern → `run_readonly_sql` SUM → END with answer. Planner never runs.

**7. Same question with `use_llm=True`?**
`guardrail` → `rules_router` skips (use_llm) → `planner` emits `run_sql` tool call → `tools` → `validate` ok → `final_answer` wrap-up → END.

**8. What happens if the LLM emits `DELETE FROM transactions`?**
`run_sql` validator raises / returns `ok:false` → `validate` increments retry, nudges planner → one more attempt → fallback message if still failing.

**9. How do you test an agent without an API key?**
`MockChatModel` with a `script` of `AIMessage(tool_calls=[...])` then a final prose `AIMessage`; pytest drives `chat(use_llm=True)`.

**10. How would you add multi-user support?**
Per-user DB / forced `user_id` predicate in every query; authn on `/chat`.

**11. Why SQLite here?**
Zero ops for a portfolio demo; swap to Postgres with the same SQL dialect subset later.

**12. What is `bind_tools`?**
Attaches JSON-schema tool definitions to the chat model so it can return structured `tool_calls` instead of free-form text.

**13. What would you build next?**
Budgets, anomaly detection, streaming bank APIs (Plaid), checkpointed graph state, richer eval with LLM-as-judge.

**14. Explain the refusal path.**
`guardrail` node → fixed refusal string → conditional edge to END. No tools invoked.

**15. Difference between `ToolNode` and calling tools yourself?**
`ToolNode` reads `tool_calls` from the last AI message, dispatches by name, and appends `ToolMessage`s — standard LangGraph wiring.
