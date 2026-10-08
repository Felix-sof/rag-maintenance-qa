# Maintenance RAG Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about
milling-machine maintenance - failure modes, maintenance procedures,
troubleshooting, safety - **only from a document knowledge base, citing the
source of every fact**. Ask in Turkish (or English):

```
$ python rag.py "HDF arızası hangi koşullarda oluşur?"

<answer citing the passages it used, e.g. "... 8,6 K'nin altına düşer ve dönüş hızı
1380 rpm'nin altındadır [1]." - exact wording depends on the model>

 *[1] 0.892  failure_modes.md > Arıza Modları ve Oluşma Koşulları > HDF - Isı Dağıtım Arızası (...)
  [2] 0.877  troubleshooting.md > Sorun Giderme Kılavuzu > Belirti: Proses ve Hava Sıcaklığı Farkı Azalıyor
  [3] 0.874  sensor_reference.md > Sensör ve Kolon Referansı > Arıza Etiketleri (...)
  [4] 0.852  troubleshooting.md > Sorun Giderme Kılavuzu > Belirti: Dönüş Hızı Düşük
```

(`*` = cited by the answer. The scores and ranking are real retrieval output.)

If the documents don't cover a question, it says so instead of guessing.

Embeddings run locally (no API cost); generation uses the Google Gemini free
tier.

## How it works

```
                       ┌─────────────── indexing (once, auto-refreshed) ───────────────┐
 knowledge/*.md ──► chunking.py ──► embeddings.py ──► vector_store.py (ChromaDB)
                   split by heading   multilingual-e5     cosine index + fingerprint
                       └───────────────────────────────────────────────────────────────┘

 question ─► 1 condense ─► 2 retrieve ─► 3 gate ─► 4 augment ─► 5 generate ─► 6 verify
             (follow-ups)   top-k chunks  score <    numbered    Gemini,        which [n]
                                          MIN_SCORE  context     context-only   were cited
                                          → refuse   [1]..[k]    + cite [n]
```

| File | Role |
|---|---|
| `chunking.py` | Splits documents on Markdown headings so each chunk is one coherent section; prepends the heading path ("Bakım Prosedürleri > Kesici Takım Değişimi") so a chunk says what it's about. Long sections split on sentence boundaries. |
| `embeddings.py` | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) via `sentence-transformers`, on CPU. Uses E5's `query:` / `passage:` prefixes. |
| `vector_store.py` | Persistent ChromaDB index in `data/chroma/`. Stores a fingerprint of the documents + embedding model + chunk size, and rebuilds itself when any of them change - editing a document never leaves stale vectors. |
| `generator.py` | Gemini client with retries for rate limits (429), server errors and flaky TLS connections. |
| `rag.py` | The pipeline (below). |
| `app.py` | Streamlit chat UI: answers, follow-ups, and an expandable source list marking which retrieved chunks the answer actually cited. |
| `api.py` | FastAPI: `POST /ask`, `GET /search` (retrieval only), `GET /health`. |

### The pipeline (`rag.answer`)

1. **Condense.** A follow-up like *"Peki kontrol adımları neler?"* is
   meaningless to a retriever on its own - searched raw, it ranks the wrong
   procedure first. With conversation history, the LLM first rewrites it into
   a standalone question (*"Soğutma sisteminin kontrol adımları nelerdir?"*).
   Standalone questions skip this call.
2. **Retrieve** the top-k chunks (default 4).
3. **Gate.** If even the best chunk scores below `MIN_SCORE`, answer
   *"Bilgi bankasında bu sorunun cevabı yok."* without calling the LLM.
4. **Augment.** Number the chunks `[1]..[k]`, with source, section, and an
   *(örnek belge)* flag for illustrative documents.
5. **Generate.** The system prompt allows only facts from the context, a
   `[n]` citation after each one, and the exact refusal sentence if the
   context doesn't answer the question.
6. **Verify.** Parse which `[n]` the answer actually cites (ignoring
   out-of-range numbers), so the UI can tell *used* sources from merely
   *retrieved* ones.

### Why two layers of refusal

The threshold was calibrated by measuring real scores, and it can't do the
job alone. multilingual-e5 squeezes all similarities into a narrow band:

| Question type | Top score |
|---|---|
| Covered by the docs (10 questions) | 0.838 - 0.896 |
| Clearly unrelated ("pizza tarifi", "Türkiye'nin başkenti", "Python'da liste") | 0.738 - 0.799 |
| Same domain, not in the docs ("hidrolik pres yağ değişimi", "bu ay kaç arıza oldu") | 0.810 - 0.839 |

The last row overlaps the first, so no threshold separates them. `MIN_SCORE =
0.80` cheaply rejects the clearly unrelated ones (no LLM call); near-domain
gaps are left to the prompt's refusal rule. Re-measure if you change the
embedding model or the documents substantially.

## Knowledge base

| File | Type | Content |
|---|---|---|
| `failure_modes.md` | reference | Exact triggering conditions of the five failure modes (TWF, HDF, PWF, OSF, RNF) |
| `sensor_reference.md` | reference | What each sensor measures and how it behaves |
| `maintenance_procedures.md` | example | Tool change, cooling system, spindle/drive checks, periodic plan |
| `troubleshooting.md` | example | Symptom → likely causes → risk → action |
| `safety.md` | example | Lockout/tagout, PPE, hot surfaces and broken tools |

**Provenance.** `reference` documents are written from the published
description of the AI4I 2020 Predictive Maintenance Dataset (S. Matzka, 2020;
[UCI ML Repository #601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset),
CC BY 4.0), so their thresholds are the rules that dataset's synthetic
machine follows. `example` documents are **illustrative procedures written
for this project**, not a real manufacturer's manual; the assistant is told
to say so when it gives steps from them.

**Adding documents.** Drop `.md` or `.txt` files into `knowledge/`. An
optional front matter sets the metadata shown in citations:

```markdown
---
title: Hydraulic Press Maintenance
type: reference
source: Vendor manual, rev. 3
---
```

The index rebuilds itself on the next query (or run `python vector_store.py --rebuild`).

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

copy .env.example .env         # macOS/Linux: cp .env.example .env
# set GEMINI_API_KEY - free key at https://aistudio.google.com/apikey
```

- `sentence-transformers` installs PyTorch. On Linux, install the CPU build
  first to avoid a multi-GB CUDA download:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **Free-tier quota.** At the time of writing, the free tier allows
  `gemini-3.8-flash` only **20 requests per day**, and each question uses 1-2
  requests. When the daily quota runs out, the app stops with a clear message
  instead of retrying for hours. To switch to another model, set `GEMINI_MODEL`
  in `.env`, e.g. `gemini-3.5-flash`, which has a separate quota.
- **Unreliable connections.** On some Windows machines, antivirus HTTPS
  inspection, a VPN or an unstable connection makes TLS connections fail at
  random (`SSL: INVALID_SESSION_ID`). On the development machine `curl` failed
  the same way, so the cause is the network, not Python. `generator.py`
  retries these up to 12 times on fresh connections, so a single answer can
  take up to about a minute.
- The embedding model (~470 MB) downloads on first use and is cached under
  `~/.cache/huggingface`. Build the index ahead of time with
  `python vector_store.py --rebuild`.

## Usage

```bash
streamlit run app.py                        # chat UI at http://localhost:8501
python rag.py "Takım değişimi nasıl yapılır?" # one question from the CLI
python vector_store.py "HDF"                # retrieval only - no LLM, no API key needed
uvicorn api:app --reload                    # HTTP API, docs at http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Peki kontrol adımları neler?",
       "history": [{"question": "Soğutma sistemi ne zaman kontrol edilmeli?",
                    "answer": "HDF arızaları arttığında [1]."}]}'
```

The response contains `answer`, `search_query` (the condensed question that
was actually searched), `sources` (every retrieved chunk, numbered as in the
prompt, with similarity `score`), and `cited` (the numbers the answer uses).

## Testing and evaluation

```bash
pip install -r requirements-dev.txt
pytest                              # 52 tests, ~8 s, offline
python -m evals.retrieval_eval      # retriever only: hit@k / MRR (offline, free)
python -m evals.answer_eval         # full pipeline against the live Gemini API
```

- **Tests** use a deterministic hashing embedder and a fake LLM, so they need
  no network, model download or API key. They cover chunking, index
  build/rebuild/persistence, the pipeline (gate, condensing, citation
  parsing, error handling), the Gemini retry policy, the API, and the eval grader.
- **Retrieval eval** (20 labeled questions): is the section that answers the
  question in the top k? Current result with multilingual-e5-small:
  **hit@4 100%, hit@1 95%, MRR 0.975**. Treat this as optimistic, because the
  questions were written alongside the documents.
- **Answer eval** (14 cases, graded programmatically): the answer must cite
  the right section (retrieving it isn't enough), must contain the key facts
  (e.g. both HDF thresholds), and must refuse three out-of-scope questions:
  near-domain, unrelated, and live data the documents don't contain. It also
  covers one follow-up and one English question. LLM failures are reported as
  `infra_error`, not as failures. Results go to `evals/results/answer_eval.json`
  (re-run just those with `--retry-infra`; a partial re-run is merged into the
  previous results).

  **Result (2026-10-08): 14/14 passed.** 8 cases ran on `gemini-3.8-flash`. The
  other 6 ran on `gemini-3.5-flash`: 5 of them hit network or quota errors on
  3.8, and 1 (`lockout_tagout`) was a grader bug. The answer correctly said
  *"kilidinizi"*, but the grader looked for *"kilit"*, so Turkish consonant
  mutation made it miss. Each row in the results file records the model it ran
  on. The two near-domain refusals (`hidrolik pres` at 0.832, `bu ay kaç arıza`
  at 0.840) passed the score gate, as the calibration predicted, and the prompt
  rule caught them.

Keeping the two evals separate tells you where a wrong answer came from:
either the right section was never retrieved, or it was retrieved and the
model didn't use it.

## Project structure

```
├── knowledge/            # the documents (Markdown / text)
├── chunking.py           # load + split documents
├── embeddings.py         # local multilingual embeddings
├── vector_store.py       # ChromaDB index, auto-rebuild, search
├── generator.py          # Gemini client with retries
├── rag.py                # the RAG pipeline
├── app.py                # Streamlit UI
├── api.py                # FastAPI
├── evals/                # retrieval_eval.py, answer_eval.py
└── tests/                # pytest suite (offline)
```

## License

The code has no license attached yet. `failure_modes.md` and
`sensor_reference.md` are derived from the AI4I 2020 dataset description
(CC BY 4.0, S. Matzka); keep that attribution if you redistribute them.
