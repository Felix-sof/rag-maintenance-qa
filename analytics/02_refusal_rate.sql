-- How often the assistant answers "not in the knowledge base", and which layer decided it.
--
-- refusal_rate: refused / questions that got a verdict. Errors are left out
--   of the denominator - a failed LLM call is neither an answer nor a refusal.
-- gate_share: the share of refusals made by the score gate, i.e. without
--   paying for an LLM call. The rest were refused by the LLM itself.
-- ROLLUP adds a total row (source = 'all') under the per-source rows.

SELECT
    coalesce(source, 'all')                                           AS source,
    count(*)                                                          AS questions,
    round(avg((outcome <> 'answered')::int), 3)                       AS refusal_rate,
    count(*) FILTER (WHERE outcome = 'refused_gate')                  AS refused_by_gate,
    count(*) FILTER (WHERE outcome = 'refused_llm')                   AS refused_by_llm,
    round(
        count(*) FILTER (WHERE outcome = 'refused_gate')::numeric
        / nullif(count(*) FILTER (WHERE outcome <> 'answered'), 0),
        3
    )                                                                 AS gate_share
FROM query_logs
WHERE outcome <> 'error'
GROUP BY ROLLUP (source)
ORDER BY grouping(source), source;
