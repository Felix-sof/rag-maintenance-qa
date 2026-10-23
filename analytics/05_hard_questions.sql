-- The hardest questions: the ones the retriever matched least confidently.
--
-- percent_rank() places each question among all others by its best rerank
-- score: 0 is the weakest match, 1 the strongest. The weakest 10% that still
-- passed the gate, plus everything the LLM refused or answered with a number
-- its sources don't contain, are where a missing document, a synonym ("LOTO")
-- or a better chunk would help most. Gate refusals are left out: they were
-- judged off-topic, not hard. A question asked several times is listed once,
-- with its latest outcome (DISTINCT ON keeps the first row per question in
-- the ORDER BY).

WITH scored AS (
    SELECT
        id, created_at, question, outcome, top_rerank, top_cosine, unsupported_numbers,
        percent_rank() OVER (ORDER BY top_rerank) AS match_percentile
    FROM query_logs
    WHERE outcome IN ('answered', 'refused_llm') AND top_rerank IS NOT NULL
),
latest AS (
    SELECT DISTINCT ON (lower(question)) *
    FROM scored
    ORDER BY lower(question), created_at DESC
)
SELECT
    question,
    outcome,
    round(top_rerank::numeric, 2)        AS top_rerank,
    round(match_percentile::numeric, 2)  AS match_percentile,
    unsupported_numbers
FROM latest
WHERE match_percentile <= 0.10
   OR outcome = 'refused_llm'
   OR cardinality(unsupported_numbers) > 0
ORDER BY top_rerank
LIMIT 20;
