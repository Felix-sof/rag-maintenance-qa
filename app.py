"""
Streamlit chat UI for the maintenance knowledge-base assistant.

    streamlit run app.py
"""

from datetime import date

import streamlit as st

import copilot_tools
import query_log
import rag
from chunking import META_SUFFIX
from generator import MODEL
from retriever import get_retriever
from vector_store import UPLOADS_SUBDIR, get_store

DOC_TYPE_LABELS = {"reference": "Official document", "example": "Example / draft"}

st.set_page_config(page_title="Maintenance Copilot", page_icon="🛠️", layout="wide")

st.title("🛠️ Maintenance Copilot")
st.caption(
    "Questions about a milling machine, answered from its maintenance documents, the documented "
    "failure rules and 10,000 recorded processes. Every fact links to its source: [n] for a "
    "document, [T1] for a tool result. Ask in English or Turkish."
)

if "turns" not in st.session_state:
    st.session_state.turns = []  # [{"question", "answer", "result"}]

with st.sidebar:
    st.header("Knowledge base")
    try:
        stats = get_store().stats()
        st.metric("Documents / chunks", f"{stats['documents']} / {stats['chunks']}")
        st.caption(f"Embedding: `{stats['embedding_model']}` · Retrieval: `{get_retriever().mode}`")
        if st.button("🔄 Rebuild the index"):
            with st.spinner("Indexing..."):
                get_store().build(force=True)
            st.rerun()
        for source, error in stats["skipped"].items():
            st.error(f"Couldn't read {source} ({error[:120]})")
    except Exception as exc:  # noqa: BLE001 - show the problem instead of a blank page
        st.error(f"The knowledge base couldn't be loaded: {exc}")

    st.divider()
    st.header("Add a document")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("PDF, Markdown or text", type=["pdf", "md", "txt"], accept_multiple_files=True)
        doc_type = st.selectbox("Type", list(DOC_TYPE_LABELS), format_func=DOC_TYPE_LABELS.get)
        priority = st.number_input("Priority (the higher wins in a conflict)", min_value=1, max_value=10, value=2)
        updated = st.date_input("Revision date", value=date.today())
        submitted = st.form_submit_button("➕ Add and index")
    if submitted and files:
        store = get_store()
        meta = {"type": doc_type, "priority": int(priority), "updated": updated.isoformat()}
        for file in files:
            with st.spinner(f"Indexing {file.name}..."):
                try:
                    source = store.save_upload(file.name, file.getvalue(), meta)
                except ValueError as exc:
                    st.error(str(exc))
                    continue
            chunks = store.chunk_counts().get(source, 0)
            if source in store.skipped():
                st.error(f"Couldn't read {source}: {store.skipped()[source][:150]}")
            elif chunks == 0:
                st.warning(f"{source}: no text could be extracted (a scanned PDF needs OCR first).")
            else:
                st.success(f"{source}: {chunks} chunks indexed.")

    upload_dir = get_store().docs_dir / UPLOADS_SUBDIR
    uploaded = sorted(p for p in upload_dir.glob("*") if p.is_file() and not p.name.endswith(META_SUFFIX))
    if uploaded:
        counts = get_store().chunk_counts()
        st.caption("Uploaded documents")
        for path in uploaded:
            source = f"{UPLOADS_SUBDIR}/{path.name}"
            name_col, delete_col = st.columns([5, 1])
            name_col.caption(f"{path.name} · {counts.get(source, 0)} chunks")
            if delete_col.button("🗑️", key=f"delete-{source}", help="Delete the document and re-index"):
                with st.spinner("Deleting..."):
                    get_store().delete_upload(source)
                st.rerun()

    st.divider()
    top_k = st.slider("Passages to retrieve (top-k)", 1, 8, rag.DEFAULT_TOP_K)
    st.caption(f"LLM: `{MODEL}`")
    if st.button("🗑️ Clear the conversation"):
        st.session_state.turns = []
        st.rerun()


def display_answer(result: dict) -> str:
    """The answer as the page shows it: the pipeline's fixed (Turkish) sentences in English."""
    if result["answer"].strip() == rag.NO_ANSWER:
        return "The knowledge base doesn't answer this question."
    if result.get("error"):
        return "Couldn't generate an answer: " + result["answer"].removeprefix(rag.ERROR_PREFIX).strip()
    return result["answer"]


def render_unsupported_numbers(result: dict) -> None:
    numbers = result.get("unsupported_numbers")
    if numbers:
        st.warning(
            f"These values in the answer are in none of the sources it cites: **{', '.join(numbers)}**. "
            "Check them against the documents before acting on them."
        )


MODE_NAMES = {"HDF": "heat dissipation", "PWF": "power", "OSF": "overstrain", "TWF": "tool wear", "RNF": "random"}


def render_assessment(n: int, assessment: dict) -> None:
    if "error" in assessment:
        st.error(f"[T{n}] The process couldn't be assessed: {assessment['error']}")
        return
    verdict = ", ".join(assessment["triggered"]) or "no rule-based failure"
    st.markdown(f"**[T{n}] Process assessment** — {verdict}")
    rows = []
    for mode, info in assessment["modes"].items():
        if info["triggered"] is None:  # TWF / RNF: a rate, not a verdict
            rate = info.get("dataset_rate", info.get("probability"))
            rows.append({"Mode": f"{mode} ({MODE_NAMES[mode]})", "Status": f"rate {rate * 100:.1f}%", "Conditions": "random"})
            continue
        conditions = "; ".join(
            f"{c['condition']}: {c['value']} ({'met' if c['met'] else 'not met'}, margin {c['margin']})"
            for c in info["conditions"]
        )
        rows.append({"Mode": f"{mode} ({MODE_NAMES[mode]})", "Status": "⚠️ triggered" if info["triggered"] else "no", "Conditions": conditions})
    st.dataframe(rows, hide_index=True, width="stretch")
    similar = assessment.get("similar_recorded_processes")
    if similar:
        modes = ", ".join(f"{m} {c}" for m, c in similar["failed_by_mode"].items() if c)
        st.caption(
            f"Of the {similar['compared']} most similar recorded processes, {similar['failed']} failed"
            + (f" ({modes})" if modes else "") + ". AI4I is a synthetic dataset."
        )
    if assessment["outside_data_range"]:
        st.caption("Outside the range of the recorded data: " + ", ".join(assessment["outside_data_range"]))


def render_tool_calls(result: dict) -> None:
    calls = result.get("tool_calls") or []
    if not calls:
        return
    with st.expander(f"🔧 Tool calls ({len(calls)})", expanded=True):
        for n, call in enumerate(calls, start=1):
            if call["name"] == "assess_process":
                render_assessment(n, call["result"])
            elif "error" in call["result"]:
                st.error(f"[T{n}] {call['name']}: {call['result']['error']}")
            else:
                st.markdown(f"**[T{n}] Dataset query:** `{call['result'].get('query')}`")
                st.dataframe(call["result"].get("rows", []), hide_index=True, width="stretch")


def render_sources(result: dict) -> None:
    sources = result["sources"]
    if not sources:
        return
    cited = [s for s in sources if s["n"] in result["cited"]]
    label = f"📚 Sources ({len(cited)} cited / {len(sources)} retrieved)"
    with st.expander(label):
        if result["search_query"] != result["question"]:
            st.caption(f"Searched for: _{result['search_query']}_")
        for src in sources:
            used = "✅" if src["n"] in result["cited"] else "▫️"
            example = " · example document" if src["doc_type"] == "example" else ""
            ranks = src.get("ranks") or {}
            found_by = " · ".join(
                f"{label} #{ranks[key]}" for key, label in (("vector", "vector"), ("bm25", "keyword")) if ranks.get(key)
            )
            st.markdown(f"{used} **[{src['n']}] {src['source']} › {src['section']}** — similarity {src['score']:.3f}{example}")
            if found_by:
                st.caption(f"Found by: {found_by}" + (f" · reranker score {ranks['rerank']:.2f}" if ranks.get("rerank") is not None else ""))
            st.caption(src["text"][:500] + ("…" if len(src["text"]) > 500 else ""))


for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn["result"].get("conflict"):
            st.warning("The sources disagree on this.")
        st.markdown(display_answer(turn["result"]))
        render_unsupported_numbers(turn["result"])
        render_tool_calls(turn["result"])
        render_sources(turn["result"])

question = st.chat_input(
    "e.g. When does an HDF failure occur? / Type L, 300 K air, 308.4 K process, 1300 rpm, 45 Nm, 120 min wear: any risk?"
)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            history = [{"question": t["question"], "answer": t["answer"]} for t in st.session_state.turns]
            result = rag.answer(question, history=history, top_k=top_k, tools=copilot_tools)
            query_log.record(result, source="ui")
        if result["error"]:
            st.error(display_answer(result))
        else:
            if result["conflict"]:
                st.warning("The sources disagree on this. Check both versions in the answer, and why one was taken to apply.")
            st.markdown(display_answer(result))
            render_unsupported_numbers(result)
            render_tool_calls(result)
        render_sources(result)

    if not result["error"]:  # a failed turn would only confuse the next follow-up
        st.session_state.turns.append({"question": question, "answer": result["answer"], "result": result})
