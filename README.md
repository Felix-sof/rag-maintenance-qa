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
- **ChromaDB or PostgreSQL + pgvector**: the index runs locally in ChromaDB or in
  Postgres (HNSW vector index, full-text search, SQL migrations), with
  identical retrieval scores on both.
- **SQL analytics**: every question is logged to Postgres. Six commented SQL
  queries (percentiles, window functions, `ROLLUP`) report volume, refusal
  rate, latency, the most used documents and the hardest questions, through
  `/stats` and a Streamlit page.
- **Grounded answers**: every claim is cited as `[n]`. Off-topic questions are
  refused by a score gate before any LLM call, and near-domain gaps are refused
  by the prompt.
- **Numbers are checked, not trusted**: every number in an answer must appear in
  a source it cites. A threshold the model made up, or took from a passage it
  didn't cite, is flagged without a second LLM call.
- **Conflict handling**: documents carry a priority and a revision date. Conflicting
  sources are flagged (`⚠️ Çelişki:`) with both versions cited, then resolved by
  priority and then by recency.
- **PDF, Markdown and text documents**, uploaded from the UI or the API. PDF
  citations point at the page.
- **Follow-up questions**: "Peki adımları neler?" is rewritten into a standalone
  question before retrieval.
- **Measured, not assumed**: a retrieval eval and an off-topic gate eval, both
  with a CI quality gate, an end-to-end answer eval against the live LLM, and
  154 unit tests.

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
             into a       vectors ─┐                          rerank   context    were cited?
             standalone   BM25 ────┴► RRF fusion ─► cross-    score    only, cite  conflict
             question                               encoder   < -5.0   [n], flag   flagged?
                                                    rerank    → "not   conflicts   numbers in
                                                              in KB"               the sources?
```

| File | Role |
|---|---|
| `chunking.py` | Markdown is split on headings, with the heading path prepended to each chunk. PDFs are split by page. Metadata comes from a `.meta.json` sidecar, then front matter, then defaults. An unreadable file is skipped and reported; it never breaks the index. |
| `embeddings.py` | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) with E5's `query:`/`passage:` prefixes. |
| `vector_store.py` | The index, fingerprinted on documents + model + chunk size and rebuilt automatically. ChromaDB by default (`VECTOR_BACKEND=chroma`). Handles upload and delete, with path-traversal-safe names and deletion limited to `uploads/`. |
| `pg_store.py` | The same index in PostgreSQL + pgvector (`VECTOR_BACKEND=pgvector`): HNSW vector search, Postgres full-text search, transactional rebuilds. |
| `db/` | Plain-SQL migrations and their runner, and a local Postgres for development without Docker. |
| `retriever.py` | Hybrid search: BM25 (or Postgres full-text search) with Turkish-aware lowercasing and first-5-character stemming, RRF fusion, and a cross-encoder rerank of the shortlist. |
| `rag.py` | The pipeline: condense → retrieve → gate → generate → verify citations, conflicts and numbers. |
| `grounding.py` | The number check: compares every number in the answer, by value (`8,6` = `8.6`, `13.000` = `13,000`), with the passages it cites. |
| `generator.py` | Gemini client. Retries 429/5xx and dropped TLS connections on fresh clients, and fails fast when the daily quota is exhausted. |
| `app.py` | Streamlit chat UI: cited answers, conflict and unsupported-number warnings, source inspector (which retriever found each chunk and its rerank score), document upload and delete. |
| `api.py` | FastAPI: `/ask`, `/search` (with `?mode=` to compare retrievers), `/documents` (GET/POST/DELETE), `/stats`, `/health`. |
| `rate_limit.py` | Per-client and global daily limits on `/ask`, the endpoint that calls Gemini. |
| `deploy/Dockerfile` | The cloud image: API only, multi-stage, models baked in, read-only, models loaded before traffic. |
| `query_log.py` · `analytics/` | Every question logged to Postgres, and the SQL that reads the log: volume, refusal rate, latency percentiles, top documents, hard questions, weekly trend. |

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

### ChromaDB or PostgreSQL + pgvector

The index can live in a local ChromaDB folder (the default, no server needed)
or in PostgreSQL with the pgvector extension. Postgres is what a deployment
needs: the index survives a container restart, several processes can share
it, and a rebuild replaces it in one transaction.

The schema (`db/migrations/001_pgvector_index.sql`) has three tables:

- `documents`: title, type, priority, date, plus the raw front matter as `jsonb`;
- `chunks`: the text, a `vector(384)` embedding with an HNSW index on cosine
  distance, and two `tsvector` columns with GIN indexes;
- `index_state`: the fingerprint the rows were built from.

At 31 chunks the HNSW index changes nothing measurable, since a full scan is
just as fast. It is there because the same query has to stay fast at much
larger sizes.

**Keyword search: keep BM25 or use Postgres full-text search?** Postgres can
do the keyword half of hybrid search itself, so it needn't load the corpus into
the app's memory. Two ways were measured against the in-memory BM25:

- `pg_fts_turkish`: Postgres's built-in Snowball stemmer for Turkish;
- `pg_fts_stems`: the BM25 retriever's own 5-character stems, matched by Postgres.

| Store | Keyword search | Mode | easy MRR | hard MRR |
|---|---|---|---|---|
| ChromaDB | BM25 (in memory) | vector only | 0.975 | 0.786 |
| pgvector | BM25 (in memory) | vector only | 0.975 | 0.786 |
| pgvector | BM25 (in memory) | keyword only | 0.825 | 0.697 |
| pgvector | `pg_fts_turkish` | keyword only | 0.908 | 0.683 |
| pgvector | `pg_fts_stems` | keyword only | 0.858 | 0.750 |
| pgvector | BM25 (in memory) | hybrid | 0.975 | 0.822 |
| pgvector | `pg_fts_turkish` | hybrid | 0.950 | 0.761 |
| pgvector | `pg_fts_stems` | hybrid | 1.000 | 0.767 |
| **either** | **any of the three** | **hybrid + rerank** | **0.975** | **0.942** |

- pgvector returns the same neighbours as ChromaDB, so every number is
  identical between the two stores.
- On their own, both full-text variants beat BM25 on the easy set. Fused with
  the vectors, BM25 is ahead on the hard set (0.822 against about 0.76).
  Postgres's `ts_rank` has no inverse document frequency: "arıza", which is in
  22 of the 31 chunks, weighs as much as "OSF".
- After the reranker all three end at the same MRR, so the choice doesn't change
  the default pipeline's answers. BM25 stays the default. `pg_fts_stems` is the
  one to switch to when the corpus no longer fits in memory.
- `ts_rank`'s length normalisation (flag 1) made both full-text variants
  slightly worse, so it is off.
- With 20 and 30 questions, one question moves hit@1 by 5 and 3.3 points.
  Differences of one or two questions are noise.

Two Turkish details matter in Postgres:

- **Lowercasing:** Postgres lowercases with the database's locale. Under the C
  locale `lower('ISI İŞLEM')` returns `isi İŞlem`, and the Turkish stemmer then
  indexes "ISI" as `is`. Text is therefore lowercased in Python
  (`turkish_lower`) before it reaches Postgres. `tests/test_pg_store.py` checks
  that a document's "ISI" is found by the query "ısı".
- **Matching:** Query terms are OR-ed (`a | b | c`) as in BM25. Postgres's
  `plainto_tsquery` would require every word of the question to be in the
  passage.

### Why two layers of refusal

A score gate refuses a question before any LLM call when even the best retrieved
chunk is a weak match. Two signals could carry it: the embedding cosine, or the
cross-encoder's rerank score. `evals/gate_eval.py` measures both on 130
questions, and on its own neither can do the job:

| Question | Best cosine | Best rerank score |
|---|---|---|
| Covered by the documents (75) | 0.764 to 0.913 | -4.58 to 10.42 |
| Same domain, not covered ("hidrolik pres yağ değişimi", "bu ay kaç arıza oldu") (32) | 0.764 to 0.865 | -9.63 to -0.08 |
| Unrelated ("pizza tarifi", "Türkiye'nin başkenti") (23) | 0.693 to 0.808 | -7.90 to -2.86 |

The rows overlap on both signals, so no threshold separates them. What a gate
can do is stop the clearest cases without refusing a question the documents
answer. Questions refused before the LLM:

| Gate | Covered (wrongly refused) | Same domain, not covered | Unrelated |
|---|---|---|---|
| cosine < 0.80 | 4 / 75 | 5 / 32 | 20 / 23 |
| **rerank < -5.0** (default) | **0 / 75** | 11 / 32 | 15 / 23 |

The cosine gate, which this project used first, refuses four covered questions
phrased unlike the documents: "KKD listesi", "Rulmandan ses geliyor" and two
English ones. multilingual-e5 squeezes every score into a narrow band, and a
paraphrase or a cross-lingual question lands in the same band as "kahve nasıl
demlenir". The reranker reads the question and the chunk together; its gate
refuses no covered question and stops about as many of the others overall (26
of 55 against 25), more of the near-domain ones and fewer of the unrelated.
Whatever passes the gate goes to the LLM, where the prompt rule refuses it. That
costs a call, not a wrong answer.

The threshold was chosen on 85 of the questions and then checked on 45 written
afterwards: 0 of their 25 covered questions were refused. The margin is thin
all the same (the lowest covered question scores -4.58), and all the questions
come from one author. CI fails if the gate refuses any covered question. Modes
without a reranker fall back to the cosine gate.

### Checking the numbers in an answer

Citations show where an answer says it comes from, not that the source says the
same thing. For a maintenance answer the costly case is a number: a limit or a
threshold that reads plausibly, carries a `[n]`, and is in no passage. Numbers
are also the one kind of claim that can be compared exactly, so `grounding.py`
checks each one against the passages the answer cites, with no second LLM call:

- by value, not spelling: `8,6` = `8.6`, and `13.000` = `13,000` = `13 000`;
- against the cited passages only: a number from a passage that was retrieved
  but not cited counts as unsupported;
- numbers from the question itself are allowed, as are citation markers and
  list numbering.

Unsupported numbers are returned by `/ask`, shown as a warning in the UI, and
fail the case in the answer eval. Replayed over the 10 recorded live answers
that state a source, the check raises no false alarm; with a threshold in those
answers changed by hand (1380 to 1500 rpm, 8,6 to 8,2 K, and four more) it
catches every one. It can't judge a claim without a number, and it flags a
count the model made itself ("şu 4 neden"), because no passage states it.

### Query analytics in SQL

When `DATABASE_URL` is set, every question the API or the UI handles is logged
to Postgres (`query_log.py`, schema in `db/migrations/002_query_logs.sql`):

- **`query_logs`**: one row per question. It records the question and its
  condensed form, an `outcome`, the best cosine and rerank scores, latency
  (total, retrieval, LLM), LLM calls and tokens, the model and the retrieval
  mode.
- **`query_log_hits`**: one row per retrieved chunk, with its rank, scores and
  whether the answer cited it.

Design choices:

- **`outcome` is one column with a `CHECK` constraint**: `answered`,
  `refused_gate`, `refused_llm` or `error`. Separate `refused`/`error` flags
  could hold impossible combinations.
- **Hits are rows, not an array**: "most retrieved documents" becomes a
  `GROUP BY`. They carry no foreign key to `chunks` and keep a copy of the
  source and section, because a rebuild replaces the chunks and the log must
  outlive it.
- **The row is written before the response goes out, not in a background
  task**: Cloud Run only gives an instance CPU while it handles a request.
- **Logging can't break an answer**: a failure is logged as a warning and
  swallowed. After a failure, logging pauses for a minute, so an unreachable
  database doesn't make every request wait out a connection timeout.

The queries are plain SQL files in `analytics/`, each commented with what it
measures:

| File | Measures | SQL worth a look |
|---|---|---|
| `01_daily_volume.sql` | questions per day and how they ended | `count(*) FILTER (WHERE ...)`, days cut in Turkish time |
| `02_refusal_rate.sql` | refusal rate, and the share refused by the gate without an LLM call | `GROUP BY ROLLUP` for a total row |
| `03_latency_percentiles.sql` | p50 / p95 latency per outcome, retrieval vs LLM time | `percentile_cont(...) WITHIN GROUP` |
| `04_top_documents.sql` | documents retrieved most, and how often they're then cited | join of hits to logs, `cite_rate` |
| `05_hard_questions.sql` | the weakest matches, LLM refusals, unsupported numbers | `percent_rank()`, `DISTINCT ON` |
| `06_weekly_trend.sql` | week-over-week change, 4-week moving average, running total | `generate_series` to fill empty weeks, `lag()`, window frames |

`tests/test_query_log.py` seeds rows whose statistics can be worked out by hand
and checks each query's output. For example, latencies of 100, 300 and 500 ms
must give p50 = 300 and an interpolated p95 = 480.

`GET /stats` returns the first four queries as JSON. The Streamlit app has an
**analytics** page with all six.

There are no real users yet, so `python -m analytics.replay` produces the data:
it sends the 130 gate-eval questions through the real pipeline and logs them
as `source = 'replay'`, which keeps them apart from real `api` and `ui`
traffic in every query. It resumes where it stopped, and it stops by itself
when the Gemini free tier's daily quota runs out.

### A public endpoint on a zero budget

The API is meant to run publicly on a free tier, where abuse costs money or takes
the demo down. These are the layers of defence, from cheapest to hardest:

- **Input limits.** A question is capped at 500 characters and the history at 4
  turns of 2,000. Anything longer is rejected before it reaches the embedder or
  the reranker.
- **Per-client limits.** `/ask` allows 5 requests a minute per client. `/search`,
  `/stats` and `/documents` allow 60. The limit is checked before the input, so
  a stream of invalid requests is limited too.
- **A global daily cap.** `/ask` allows 20 requests per 24 hours from everyone
  together, because a per-client limit is beaten by switching IP addresses.
  In-memory counters reset whenever a scale-to-zero instance restarts, so the
  cap is also read from the query log in Postgres: no more than 20 questions a
  day reach the LLM, restarts or not.
- **No database work in `/health`.** The platform's probes call it without a
  rate limit, so its index stats are cached for 30 s.
- **Read-only.** Uploads and deletes are off, and a serving instance never
  rebuilds the index.

What the app can't do on its own: every request, even one answered with a 429,
still reaches the container. The bounds on cost are the platform's: a single
instance at most, a budget alert, and a Gemini key from a project without
billing, whose free quota simply runs out instead of being charged.

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
- `unsupported_numbers`: numbers in the answer that none of its cited sources
  contain (empty when everything checks out)
- `search_query`: the condensed question
- `outcome`: `answered`, `refused_gate`, `refused_llm` or `error`
- `timings` (total, retrieval and LLM ms) and `usage` (LLM calls and tokens)

### PostgreSQL + pgvector

```bash
pip install pgserver                        # Postgres 16 + pgvector, no Docker needed
python -m db.local_postgres                 # starts it, prints DATABASE_URL for .env
# in .env: VECTOR_BACKEND=pgvector and DATABASE_URL=...
python vector_store.py --rebuild            # migrates the schema, indexes knowledge/
python -m evals.retrieval_eval --compare-backends
```

Any Postgres with the `vector` extension works, e.g. a free Neon database
(`DATABASE_URL=postgresql://...?sslmode=require`), or
`docker compose --profile pgvector up`. Migrations run automatically on the
first connection, or explicitly with `python -m db.migrate`.

### Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | (required) | |
| `GEMINI_MODEL` | `gemini-3.8-flash` | The free tier currently allows only ~20 requests/day for this model. Another model, e.g. `gemini-3.5-flash`, has its own quota. |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `vector`, `bm25`, `hybrid` or `hybrid_rerank` |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | Changing it rebuilds the index automatically. Re-measure `MIN_SCORE` afterwards. |
| `RERANK_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `MIN_RERANK_SCORE` is on this model's score scale. Re-measure it with `python -m evals.gate_eval` if you change the model. |
| `VECTOR_BACKEND` | `chroma` | `chroma` or `pgvector` |
| `DATABASE_URL` | | Postgres connection string, for `pgvector` |
| `KEYWORD_BACKEND` | `bm25` | `bm25`, or with `pgvector` also `pg_fts_stems` / `pg_fts_turkish` |
| `QUERY_LOG` | `on` | Questions are logged to `DATABASE_URL` when it is set; `off` turns that off. |
| `READ_ONLY` | `0` | `1` in a public deployment: uploads and deletes are refused, and the pgvector index is served as built, never rebuilt by a serving instance. |
| `WARM_UP` | `0` | `1` loads the models at startup, before the API accepts requests. |
| `RATE_LIMIT_PER_MINUTE` | `5` | `/ask` requests per client per minute |
| `ASK_LIMIT_PER_DAY` | `20` | `/ask` requests per 24 hours from everyone together, and LLM answers per 24 hours in the query log (which a restart doesn't reset); caps the Gemini spend |

## Testing and evaluation

```bash
pip install -r requirements-dev.txt
pytest                                       # 154 tests, ~20 s, offline (29 need TEST_DATABASE_URL)
python -m evals.retrieval_eval               # retriever comparison (offline, free)
python -m evals.retrieval_eval --mode hybrid_rerank   # per-question detail
python -m evals.gate_eval                    # off-topic gate comparison (offline, free)
python -m evals.retrieval_eval --compare-backends   # ChromaDB vs pgvector (needs DATABASE_URL)
python -m evals.answer_eval                  # end to end, against the live Gemini API
python -m analytics.replay --limit 30        # log eval questions through the pipeline (Gemini + DATABASE_URL)
python -m analytics                          # run the analytics queries and print the results
```

- **Unit tests** use a deterministic hashing embedder, a fake LLM and PDFs
  generated in code, so they need no network, model download or API key. The
  pgvector tests also need a Postgres (`TEST_DATABASE_URL`). They cover:
  - chunking (Markdown, PDF, sidecars, unreadable files),
  - the index (rebuild, persistence, uploads),
  - hybrid retrieval and fusion,
  - the pipeline (gate, condensing, citations, conflicts),
  - the number check,
  - the pgvector backend: migrations, rebuilds, vector and full-text search,
    Turkish lowercasing,
  - the query log and every analytics query, on hand-checkable data,
  - the Gemini retry and quota policy,
  - the API and the eval grader.
- **Retrieval eval**: the table above. CI fails if the default mode's MRR drops
  below 0.90 on either set.
- **Gate eval**: the refusal table above. CI fails if the gate refuses a
  question the documents cover.
- **Answer eval** (16 cases, graded programmatically). An answer passes only if
  it does all of the following:
  - cites the right section (retrieving it isn't enough),
  - contains the key facts (e.g. both HDF thresholds),
  - refuses 3 out-of-scope questions,
  - flags both conflict cases,
  - raises no false conflict alarms elsewhere,
  - states no number that its cited sources don't contain.

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

1. the unit tests, with a pgvector service container for the Postgres tests,
2. the retrieval quality gate and the off-topic gate check with the real models,
   on both ChromaDB and pgvector,
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
├── vector_store.py         the index (ChromaDB), auto-rebuild, uploads
├── pg_store.py             the index in PostgreSQL + pgvector
├── db/                     SQL migrations, migration runner, local Postgres
├── retriever.py            BM25 + vectors + RRF + cross-encoder
├── rag.py                  the RAG pipeline
├── grounding.py            number check on the answer
├── generator.py            Gemini client with retries
├── app.py · api.py         Streamlit UI · FastAPI
├── query_log.py            logs every question to Postgres
├── analytics/              the analytics SQL, the report and the replay
├── pages/analytics.py      Streamlit analytics page
├── evals/                  retrieval_eval.py, gate_eval.py, answer_eval.py, results/
├── tests/                  154 tests
├── Dockerfile · docker-compose.yml   local images (UI + API)
├── deploy/Dockerfile       the cloud image (API only)
└── .github/workflows/ci.yml
```

## License

The code has no license attached yet. `failure_modes.md` and
`sensor_reference.md` are derived from the AI4I 2020 dataset description
(CC BY 4.0, S. Matzka); keep that attribution if you redistribute them.
