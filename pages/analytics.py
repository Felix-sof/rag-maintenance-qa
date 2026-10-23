"""
Streamlit page: the query-log analytics (analytics/*.sql), next to the chat in app.py.

    streamlit run app.py        # then "analytics" in the sidebar

Reads the same database as the query log (DATABASE_URL).
"""

import pandas as pd
import streamlit as st

import analytics
import query_log

st.set_page_config(page_title="Bakım Asistanı - Analitik", page_icon="📊", layout="wide")
st.title("📊 Soru analitiği")

if not query_log.enabled():
    st.info("Sorgu kaydı kapalı. Analitik için `.env` dosyasına `DATABASE_URL` ekleyin (Postgres).")
    st.stop()

try:
    log = query_log.get()
    results = {name: pd.DataFrame(analytics.run(log, name)) for name in analytics.queries()}
except Exception as exc:  # noqa: BLE001 - show the problem instead of a blank page
    st.error(f"Veritabanı okunamadı: {exc}")
    st.stop()

if results["daily_volume"].empty:
    st.info("Henüz kayıtlı soru yok. Sohbet sayfasından soru sorun ya da `python -m analytics.replay` çalıştırın.")
    st.stop()

st.caption(
    "Kaynak: `query_logs` tablosu. `replay` kaynaklı satırlar, değerlendirme sorularının "
    "pipeline'dan geçirilmesiyle üretilmiştir; gerçek kullanıcı trafiği değildir."
)

# --- headline numbers -----------------------------------------------------
daily = results["daily_volume"]
refusals = results["refusal_rate"]
latency = results["latency_percentiles"].set_index("outcome")
tiles = st.columns(4)
tiles[0].metric("Soru (son 30 gün)", int(daily["questions"].sum()))
if not refusals.empty:  # the last row is ROLLUP's total over every source
    tiles[1].metric("Reddetme oranı", f"{float(refusals.iloc[-1]['refusal_rate']):.0%}")
if "answered" in latency.index:
    tiles[2].metric("Cevap süresi p50", f"{float(latency.loc['answered', 'p50_ms']) / 1000:.1f} sn")
    tiles[3].metric("Cevap süresi p95", f"{float(latency.loc['answered', 'p95_ms']) / 1000:.1f} sn")

# --- charts: one measure each, one axis -------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Günlük soru sayısı")
    st.bar_chart(daily.set_index("day")["questions"], height=260)
with right:
    st.subheader("En sık getirilen belgeler")
    top = results["top_documents"]
    if top.empty:
        st.caption("LLM'in okuduğu bir soru henüz yok.")
    else:
        st.bar_chart(top.set_index("source")["retrieved"], horizontal=True, height=260)

# --- the full tables ----------------------------------------------------------
LABELS = {
    "daily_volume": "Günlük hacim ve sonuçlar",
    "refusal_rate": "Reddetme oranı (kaynağa göre)",
    "latency_percentiles": "Gecikme yüzdelikleri (sonuca göre, ms)",
    "top_documents": "Belgeler: getirilme ve atıf oranı",
    "hard_questions": "Zor sorular (en zayıf eşleşmeler)",
    "weekly_trend": "Haftalık eğilim",
}
for name, frame in results.items():
    with st.expander(LABELS.get(name, name), expanded=name in ("hard_questions", "latency_percentiles")):
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(f"SQL: `analytics/{next(p.name for p in analytics.QUERY_DIR.glob(f'*_{name}.sql'))}`")
