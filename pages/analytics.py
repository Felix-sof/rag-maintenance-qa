"""
Streamlit page: the query-log analytics (analytics/*.sql), next to the chat in app.py.

    streamlit run app.py        # then "analytics" in the sidebar

Reads the same database as the query log (DATABASE_URL).
"""

import pandas as pd
import streamlit as st

import analytics
import query_log

st.set_page_config(page_title="Maintenance Copilot - Analytics", page_icon="📊", layout="wide")
st.title("📊 Question analytics")

if not query_log.enabled():
    st.info("Query logging is off. Add `DATABASE_URL` (Postgres) to `.env` for analytics.")
    st.stop()

try:
    log = query_log.get()
    results = {name: pd.DataFrame(analytics.run(log, name)) for name in analytics.queries()}
except Exception as exc:  # noqa: BLE001 - show the problem instead of a blank page
    st.error(f"The database couldn't be read: {exc}")
    st.stop()

if results["daily_volume"].empty:
    st.info("No questions logged yet. Ask some on the chat page, or run `python -m analytics.replay`.")
    st.stop()

st.caption(
    "Source: the `query_logs` table. Rows with source `replay` are the eval questions sent "
    "through the pipeline, not real user traffic."
)

# --- headline numbers -----------------------------------------------------
daily = results["daily_volume"]
refusals = results["refusal_rate"]
latency = results["latency_percentiles"].set_index("outcome")
tiles = st.columns(4)
tiles[0].metric("Questions (last 30 days)", int(daily["questions"].sum()))
if not refusals.empty:  # the last row is ROLLUP's total over every source
    tiles[1].metric("Refusal rate", f"{float(refusals.iloc[-1]['refusal_rate']):.0%}")
if "answered" in latency.index:
    tiles[2].metric("Answer time p50", f"{float(latency.loc['answered', 'p50_ms']) / 1000:.1f} s")
    tiles[3].metric("Answer time p95", f"{float(latency.loc['answered', 'p95_ms']) / 1000:.1f} s")

# --- charts: one measure each, one axis -------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Questions per day")
    st.bar_chart(daily.set_index("day")["questions"], height=260)
with right:
    st.subheader("Most retrieved documents")
    top = results["top_documents"]
    if top.empty:
        st.caption("No question has reached the LLM yet.")
    else:
        st.bar_chart(top.set_index("source")["retrieved"], horizontal=True, height=260)

# --- the full tables ----------------------------------------------------------
LABELS = {
    "daily_volume": "Daily volume and outcomes",
    "refusal_rate": "Refusal rate, by source",
    "latency_percentiles": "Latency percentiles, by outcome (ms)",
    "top_documents": "Documents: retrieved and cited",
    "hard_questions": "Hard questions (weakest matches)",
    "weekly_trend": "Weekly trend",
    "tool_usage": "Copilot tools: use and cost",
}
for name, frame in results.items():
    with st.expander(LABELS.get(name, name), expanded=name in ("hard_questions", "latency_percentiles")):
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(f"SQL: `analytics/{next(p.name for p in analytics.QUERY_DIR.glob(f'*_{name}.sql'))}`")
