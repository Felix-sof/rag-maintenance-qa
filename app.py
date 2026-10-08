"""
Streamlit chat UI for the maintenance knowledge-base assistant.

    streamlit run app.py
"""

from datetime import date

import streamlit as st

import rag
from chunking import META_SUFFIX
from generator import MODEL
from retriever import get_retriever
from vector_store import UPLOADS_SUBDIR, get_store

DOC_TYPE_LABELS = {"reference": "Resmi belge", "example": "Örnek / taslak"}

st.set_page_config(page_title="Bakım Asistanı (RAG)", page_icon="🛠️", layout="wide")

st.title("🛠️ Bakım Asistanı")
st.caption(
    "Freze tezgahı bakım bilgi bankası üzerinde soru-cevap. Cevaplar yalnızca "
    "belgelerden üretilir ve her bilgi [n] ile kaynağına bağlanır."
)

if "turns" not in st.session_state:
    st.session_state.turns = []  # [{"question", "answer", "result"}]

with st.sidebar:
    st.header("Bilgi Bankası")
    try:
        stats = get_store().stats()
        st.metric("Belge / parça", f"{stats['documents']} / {stats['chunks']}")
        st.caption(f"Embedding: `{stats['embedding_model']}` · Arama: `{get_retriever().mode}`")
        if st.button("🔄 Yeniden indeksle"):
            with st.spinner("İndeksleniyor..."):
                get_store().build(force=True)
            st.rerun()
        for source, error in stats["skipped"].items():
            st.error(f"Okunamadı: {source} ({error[:120]})")
    except Exception as exc:  # noqa: BLE001 - show the problem instead of a blank page
        st.error(f"Bilgi bankası yüklenemedi: {exc}")

    st.divider()
    st.header("Belge Ekle")
    with st.form("upload", clear_on_submit=True):
        files = st.file_uploader("PDF, Markdown veya metin", type=["pdf", "md", "txt"], accept_multiple_files=True)
        doc_type = st.selectbox("Tür", list(DOC_TYPE_LABELS), format_func=DOC_TYPE_LABELS.get)
        priority = st.number_input("Öncelik (çelişkide yüksek olan geçerli)", min_value=1, max_value=10, value=2)
        updated = st.date_input("Güncelleme tarihi", value=date.today())
        submitted = st.form_submit_button("➕ Ekle ve indeksle")
    if submitted and files:
        store = get_store()
        meta = {"type": doc_type, "priority": int(priority), "updated": updated.isoformat()}
        for file in files:
            with st.spinner(f"{file.name} indeksleniyor..."):
                try:
                    source = store.save_upload(file.name, file.getvalue(), meta)
                except ValueError as exc:
                    st.error(str(exc))
                    continue
            chunks = store.chunk_counts().get(source, 0)
            if source in store.skipped():
                st.error(f"{source} okunamadı: {store.skipped()[source][:150]}")
            elif chunks == 0:
                st.warning(f"{source}: metin çıkarılamadı (taranmış bir PDF olabilir; önce OCR gerekir).")
            else:
                st.success(f"{source}: {chunks} parça indekslendi.")

    upload_dir = get_store().docs_dir / UPLOADS_SUBDIR
    uploaded = sorted(p for p in upload_dir.glob("*") if p.is_file() and not p.name.endswith(META_SUFFIX))
    if uploaded:
        counts = get_store().chunk_counts()
        st.caption("Yüklenen belgeler")
        for path in uploaded:
            source = f"{UPLOADS_SUBDIR}/{path.name}"
            name_col, delete_col = st.columns([5, 1])
            name_col.caption(f"{path.name} · {counts.get(source, 0)} parça")
            if delete_col.button("🗑️", key=f"delete-{source}", help="Belgeyi sil ve yeniden indeksle"):
                with st.spinner("Siliniyor..."):
                    get_store().delete_upload(source)
                st.rerun()

    st.divider()
    top_k = st.slider("Getirilecek parça sayısı (top-k)", 1, 8, rag.DEFAULT_TOP_K)
    st.caption(f"LLM: `{MODEL}`")
    if st.button("🗑️ Konuşmayı temizle"):
        st.session_state.turns = []
        st.rerun()


def render_sources(result: dict) -> None:
    sources = result["sources"]
    if not sources:
        return
    cited = [s for s in sources if s["n"] in result["cited"]]
    label = f"📚 Kaynaklar ({len(cited)} kullanıldı / {len(sources)} getirildi)"
    with st.expander(label):
        if result["search_query"] != result["question"]:
            st.caption(f"Arama sorgusu: _{result['search_query']}_")
        for src in sources:
            used = "✅" if src["n"] in result["cited"] else "▫️"
            example = " · örnek belge" if src["doc_type"] == "example" else ""
            ranks = src.get("ranks") or {}
            found_by = " · ".join(
                f"{label} #{ranks[key]}" for key, label in (("vector", "vektör"), ("bm25", "BM25")) if ranks.get(key)
            )
            st.markdown(f"{used} **[{src['n']}] {src['source']} › {src['section']}** — benzerlik {src['score']:.3f}{example}")
            if found_by:
                st.caption(f"Bulan: {found_by}" + (f" · reranker skoru {ranks['rerank']:.2f}" if ranks.get("rerank") is not None else ""))
            st.caption(src["text"][:500] + ("…" if len(src["text"]) > 500 else ""))


for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn["result"].get("conflict"):
            st.warning("Kaynaklar bu konuda birbiriyle çelişiyor.")
        st.markdown(turn["answer"])
        render_sources(turn["result"])

question = st.chat_input("Örn: HDF arızası hangi koşullarda oluşur? / Takım değişimi nasıl yapılır?")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Belgeler aranıyor..."):
            history = [{"question": t["question"], "answer": t["answer"]} for t in st.session_state.turns]
            result = rag.answer(question, history=history, top_k=top_k)
        if result["error"]:
            st.error(result["answer"])
        else:
            if result["conflict"]:
                st.warning("Kaynaklar bu konuda birbiriyle çelişiyor. Cevaptaki iki versiyonu ve hangisinin neden geçerli sayıldığını kontrol edin.")
            st.markdown(result["answer"])
        render_sources(result)

    if not result["error"]:  # a failed turn would only confuse the next follow-up
        st.session_state.turns.append({"question": question, "answer": result["answer"], "result": result})
