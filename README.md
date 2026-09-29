# Memory layer

RiskWise can mirror high-risk `/predict` assessments, newly submitted flagged
POs, and recorded mitigation outcomes to Hindsight. Recalled supplier context is used only in the pipeline's LLM-written
flag explanations and the supplier memory endpoint. It never enters model
features, training/evaluation data, risk scores, or fraud scores. Recalled text
is untrusted context, not instructions. Postgres or other application tables
remain the system of record. This checkout does not yet include persistent
outcome storage, so `/outcomes` currently queues only the best-effort Hindsight
mirror.

Reviewer decision and fraud-review handlers, an `actions_log`, external-event
ingestion, supplier inquiries, memos, and audit-report generation are not yet
implemented; their memory hooks remain TODOs until those workflows exist.

Install dependencies with `pip install -r requirements.txt`. Copy
`.env.example` to `.env`, set `GROQ_API_KEY`, then start the API normally. By
default, FastAPI starts and stops an embedded local Hindsight server through
`HindsightServer`; it listens on `127.0.0.1:8888` with MCP enabled, so the
workspace MCP entry can connect while the API is running. Set
`HINDSIGHT_ENABLED=0` to disable memory. `HINDSIGHT_MODE=remote` opts into the
standalone client, using `HINDSIGHT_BASE_URL` and optional `HINDSIGHT_API_KEY`.
`HINDSIGHT_BANK_ID` defaults to `procurement-risk`; `HINDSIGHT_LOCAL_PORT`
defaults to `8888`. Groq powers both the RiskWise explanation pipeline and
Hindsight; `GROQ_MODEL` configures the former and
`HINDSIGHT_API_LLM_MODEL` the latter (default `openai/gpt-oss-20b`). Hindsight
retention requires a Groq tier with sufficient token quota; the Groq free tier
is not enough for its retain calls. The Docker Compose server is an alternative
for remote-client mode; do not run it on the same port as the embedded server.
