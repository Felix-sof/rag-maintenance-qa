# rag-maintenance-qa

[![CI](https://github.com/Felix-sof/rag-maintenance-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/Felix-sof/rag-maintenance-qa/actions/workflows/ci.yml)

A retrieval-augmented generation (RAG) assistant for industrial maintenance. It
answers questions about failure modes, procedures, troubleshooting and safety
**only from a document knowledge base, citing the source of every fact**.
When the documents don't cover a question, it says so. When two documents
disagree, it shows both versions instead of quietly picking one.

```
$ python rag.py "HDF arızası hangi koşullarda oluşur?"

HDF (Isı Dağıtım Arızası), proses ısısı yeterince dağıtılamadığında oluşur [1].

Tetiklenme koşulu olarak, hava sıcaklığı ile proses sıcaklığı arasındaki farkın 8,6 K'nin
altına düşmesi **ve** aynı anda dönüş hızının 1380 rpm'nin altında olması gerekir [1].
Arızanın gerçekleşmesi için bu iki koşulun birlikte sağlanması zorunludur [1].

 *[1] 0.892  failure_modes.md > Arıza Modları ve Oluşma Koşulları > HDF - Isı Dağıtım Arızası
  [2] 0.877  troubleshooting.md > Sorun Giderme Kılavuzu > Belirti: Proses ve Hava Sıcaklığı Farkı Azalıyor
  ...
```

Turkish-first, and English questions work too. Embeddings and reranking run
locally on CPU; generation uses the Google Gemini free tier.

**Highlights**

- **Hybrid retrieval**: dense vectors and BM25, fused with Reciprocal Rank Fusion
  and re-ranked by a multilingual cross-encoder. On a hard query set, MRR goes
  from **0.786 (vectors only) to 0.942**.
- **Grounded answers**: every claim is cited as `[n]`. Off-topic questions are
  refused by a score gate before any LLM call, and near-domain gaps are refused
  by the prompt.
- **Conflict handling**: documents carry a priority and a revision date. Conflicting
  sources are flagged (`⚠️ Çelişki:`) with both versions cited, then resolved by
  priority and then by recency.
- **PDF, Markdown and text documents**, uploaded from the UI or the API. PDF
  citations point at the page.
- **Follow-up questions**: "Peki adımları neler?" is rewritten into a standalone
  question before retrieval.
- **Measured, not assumed**: a retrieval eval with a CI quality gate, an end-to-end
  answer eval against the live LLM, and 85 offline unit tests.

## Architecture

```
 indexing (automatic, fingerprinted)
 ───────────────────────────────────
 knowledge/**/*.md|txt|pdf ─► chunking.py ─► embeddings.py ─► vector_store.py (ChromaDB)
                              by heading      multilingual-e5   cosine index; rebuilds itself
                              / by PDF page   (local, CPU)      when docs or model change

 answering (rag.py)
 ──────────────────
 question ─► condense ─► retrieve ─────────────────────────► gate ─► generate ─► verify
             follow-ups   retriever.py:                       best     Gemini:    which [n]
             into a       vectors ─┐                          cosine   context    were cited?
             standalone   BM25 ────┴► RRF fusion ─► cross-    < 0.80   only, cite  conflict
             question                               encoder   → "not   [n], flag   flagged?
                                                    rerank    in KB"   conflicts
```

| File | Role |
|---|---|
| `chunking.py` | Markdown is split on headings, with the heading path prepended to each chunk. PDFs are split by page. Metadata comes from a `.meta.json` sidecar, then front matter, then defaults. An unreadable file is skipped and reported; it never breaks the index. |
| `embeddings.py` | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) with E5's `query:`/`passage:` prefixes. |
| `vector_store.py` | Persistent ChromaDB index, fingerprinted on documents + model + chunk size and rebuilt automatically. Handles upload and delete, with path-traversal-safe names and deletion limited to `uploads/`. |
| `retriever.py` | Hybrid search: BM25 with Turkish-aware lowercasing and first-5-character stemming, RRF fusion, and a cross-encoder rerank of the shortlist. |
| `rag.py` | The pipeline: condense → retrieve → gate → generate → verify citations and conflicts. |
| `generator.py` | Gemini client. Retries 429/5xx and dropped TLS connections on fresh clients, and fails fast when the daily quota is exhausted. |
| `app.py` | Streamlit chat UI: cited answers, conflict warnings, source inspector (which retriever found each chunk and its rerank score), document upload and delete. |
| `api.py` | FastAPI: `/ask`, `/search` (with `?mode=` to compare retrievers), `/documents` (GET/POST/DELETE), `/health`. |

## Design decisions, with numbers

### Why hybrid retrieval and a reranker

Embeddings capture meaning ("makine ısınıp yavaş dönüyor" → heat dissipation
failure) but miss exact tokens: failure codes like `OSF`, thresholds like
`8,6 K`. BM25 is the opposite. `evals/retrieval_eval.py` measures both on two
question sets:

- **easy** (20): phrased like the documents.
- **hard** (30): paraphrases with little word overlap, bare codes and numbers,
  acronyms the documents never spell out, and English questions against Turkish
  documents.

| Mode | easy hit@1 | easy MRR | **hard hit@1** | **hard MRR** |
|---|---|---|---|---|
| vector | 95% | 0.975 | 70% | 0.786 |
| bm25 | 70% | 0.825 | 60% | 0.697 |
| hybrid (RRF) | 95% | 0.975 | 77% | 0.822 |
| **hybrid + rerank** (default) | **95%** | **0.975** | **93%** | **0.942** |

On the easy set the modes are tied, because it was already at the ceiling. The
hard set is where retrieval actually differs. The one remaining miss is
"LOTO prosedürü": the acronym never appears in the documents, so no retriever
can match it. Fixing that needs synonym or acronym expansion, not better
ranking. Note that both question sets were written by the author of the
documents, so the absolute numbers are optimistic; the comparison between
modes is the useful part.

BM25 on Turkish needs care. The language is agglutinative ("arıza", "arızası",
"arızaların"), so tokens are truncated to their first 5 characters, a simple
stemmer that has been studied for Turkish IR (Can et al., 2008). Lowercasing
also has to follow Turkish rules (`I` → `ı`, `İ` → `i`).

### Why two layers of refusal

The score gate was calibrated by measuring real cosine similarities, and on its
own it can't do the job. multilingual-e5 squeezes all scores into a narrow band:

| Question | Top cosine |
|---|---|
| Covered by the documents | 0.838 - 0.896 |
| Unrelated ("pizza tarifi", "Türkiye'nin başkenti") | 0.738 - 0.799 |
| Same domain, not covered ("hidrolik pres yağ değişimi", "bu ay kaç arıza oldu") | 0.810 - 0.839 |

The last row overlaps the first, so no threshold separates them. `MIN_SCORE =
0.80` rejects clearly unrelated questions without calling the LLM, and the
prompt rule catches near-domain gaps. The gate always uses the embedding cosine,
whichever retriever ranked the chunks.

### Conflicting sources

Each document has a `priority` (default: `reference` 2, `example` 1) and an
`updated` date, and both appear next to every passage in the prompt. When
passages disagree, the model must start with `⚠️ Çelişki:`, cite every version,
and resolve the conflict by priority, then by recency, or else say it can't be
decided. A silent pick is never acceptable. The answer eval tests both
resolution paths, and every other case fails if it raises a false conflict
alarm.

### Follow-up questions

Searched as-is, *"Peki kontrol adımları neler?"* ranks the wrong procedure
first. With history, the LLM first rewrites it into *"Soğutma sisteminin
kontrol adımları nelerdir?"*, and the right section comes back with a cosine of
0.898.

## Knowledge base

| File | Type | Content |
|---|---|---|
| `failure_modes.md` | reference | Exact triggering conditions of TWF / HDF / PWF / OSF / RNF |
| `sensor_reference.md` | reference | What each sensor measures and how it behaves |
| `maintenance_procedures.md` | example | Tool change, cooling system, spindle/drive checks, periodic plan |
| `troubleshooting.md` | example | Symptom → likely causes → risk → action |
| `safety.md` | example | Lockout/tagout, PPE, hot surfaces and broken tools |

**Provenance.** The `reference` documents are written from the published
description of the AI4I 2020 Predictive Maintenance Dataset (S. Matzka, 2020;
[UCI #601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset),
CC BY 4.0). The `example` documents are **illustrative procedures written for
this project**, not a real manufacturer's manual, and the assistant says so
when it quotes them.

**Adding documents.** You can:

- upload from the UI sidebar,
- `POST /documents`,
- or drop `.md` / `.txt` / `.pdf` files into `knowledge/`.

Markdown can carry front matter:

```markdown
---
title: Hydraulic Press Manual
type: reference        # or: example
priority: 3            # higher wins in a conflict
updated: 2026-09-15
---
```

Uploaded files go to `knowledge/uploads/`, which is git-ignored. Their metadata
is stored in a `.meta.json` sidecar. Scanned PDFs without a text layer are
flagged as needing OCR.

## Quick start

### Docker

```bash
cp .env.example .env            # set GEMINI_API_KEY (free: https://aistudio.google.com/apikey)
docker compose up --build       # UI http://localhost:8501 · API http://localhost:8000/docs
```

The models are baked into the image, so the first start doesn't download anything.

### Local

```bash
python -m venv venv
venv\Scripts\activate                     # macOS/Linux: source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env                    # macOS/Linux: cp; then set GEMINI_API_KEY

streamlit run app.py                      # chat UI
python rag.py "Takım değişimi nasıl yapılır?"
python vector_store.py "HDF"              # retrieval only: no LLM, no API key
uvicorn api:app --reload                  # API, docs at /docs
```

The first query downloads the embedding model and the reranker (~1 GB in
total), which are then cached.

```bash
curl "http://localhost:8000/search?q=OSF&mode=bm25"     # compare retrievers, no LLM
curl -X POST http://localhost:8000/documents -F "file=@manual.pdf" -F "priority=3"
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Peki kontrol adımları neler?",
       "history": [{"question": "Soğutma sistemi ne zaman kontrol edilmeli?", "answer": "..."}]}'
```

`/ask` returns:

- `answer`
- `sources`: every retrieved chunk, with its cosine, the ranks from each
  retriever and the rerank score
- `cited`: the `[n]` the answer actually uses
- `conflict`
- `search_query`: the condensed question

### Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | (required) | |
| `GEMINI_MODEL` | `gemini-3.8-flash` | The free tier currently allows only ~20 requests/day for this model. Another model, e.g. `gemini-3.5-flash`, has its own quota. |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `vector`, `bm25`, `hybrid` or `hybrid_rerank` |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Changing it rebuilds the index automatically. Re-measure `MIN_SCORE` afterwards. |
| `RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | |

## Testing and evaluation

```bash
pip install -r requirements-dev.txt
pytest                                       # 85 tests, ~15 s, offline
python -m evals.retrieval_eval               # retriever comparison (offline, free)
python -m evals.retrieval_eval --mode hybrid_rerank   # per-question detail
python -m evals.answer_eval                  # end to end, against the live Gemini API
```

- **Unit tests** use a deterministic hashing embedder, a fake LLM and PDFs
  generated in code, so they need no network, model download or API key. They
  cover:
  - chunking (Markdown, PDF, sidecars, unreadable files),
  - the index (rebuild, persistence, uploads),
  - hybrid retrieval and fusion,
  - the pipeline (gate, condensing, citations, conflicts),
  - the Gemini retry and quota policy,
  - the API and the eval grader.
- **Retrieval eval**: the table above. CI fails if the default mode's MRR drops
  below 0.90 on either set.
- **Answer eval** (16 cases, graded programmatically). An answer passes only if
  it does all of the following:
  - cites the right section (retrieving it isn't enough),
  - contains the key facts (e.g. both HDF thresholds),
  - refuses 3 out-of-scope questions,
  - flags both conflict cases,
  - raises no false conflict alarms elsewhere.

  It also covers a follow-up and an English question. LLM or network failures
  are recorded as `infra_error`, not as failures (`--retry-infra` re-runs them).

  **Latest full run (2026-10-08, gemini-3.5-flash, hybrid + rerank): 11/11 graded
  cases passed.** The other 5 cases hit the free tier's daily quota mid-run and
  are recorded as `infra_error` in `evals/results/answer_eval.json`. All 5 had
  passed in earlier runs the same day: the two conflict cases on this same
  pipeline, the follow-up and two refusal cases on the earlier vector-only
  retriever. Re-run them with `--retry-infra` once the quota resets.

Keeping the evals separate tells you where a wrong answer came from: either the
right section was never retrieved, or the model didn't use it.

**CI** (`.github/workflows/ci.yml`) runs three jobs:

1. the unit tests,
2. the retrieval quality gate with the real models,
3. a Docker build plus a smoke test that indexes the knowledge base and runs a
   hybrid search inside the container.

## Troubleshooting

- **`SSL: INVALID_SESSION_ID` / slow answers on Windows.** Antivirus HTTPS
  inspection, a VPN or an unstable connection can drop TLS connections at
  random; on the development machine `curl` failed the same way. `generator.py`
  retries up to 12 times on fresh connections, so an answer can take up to
  about a minute.
- **"Gemini quota exhausted … resets in about N h".** The free-tier daily quota
  is used up. Wait for the reset, or set `GEMINI_MODEL` to another model.

## Project structure

```
├── knowledge/              documents (uploads/ is git-ignored)
├── chunking.py             load + split Markdown / text / PDF
├── embeddings.py           local multilingual embeddings
├── vector_store.py         ChromaDB index, auto-rebuild, uploads
├── retriever.py            BM25 + vectors + RRF + cross-encoder
├── rag.py                  the RAG pipeline
├── generator.py            Gemini client with retries
├── app.py · api.py         Streamlit UI · FastAPI
├── evals/                  retrieval_eval.py, answer_eval.py, results/
├── tests/                  85 offline tests
├── Dockerfile · docker-compose.yml
└── .github/workflows/ci.yml
```

## License

The code has no license attached yet. `failure_modes.md` and
`sensor_reference.md` are derived from the AI4I 2020 dataset description
(CC BY 4.0, S. Matzka); keep that attribution if you redistribute them.
