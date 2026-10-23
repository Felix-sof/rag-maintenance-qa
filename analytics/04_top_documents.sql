-- Which documents the retriever brings back, and how often the answer then uses them.
--
-- retrieved:    times one of the document's chunks was among the passages shown to the LLM
-- ranked_first: times it was passage [1]
-- cite_rate:    cited / retrieved. A document retrieved often but seldom cited
--               takes a context slot without helping - a candidate for better
--               chunking, or for a lower rank.
-- Only questions the LLM actually read count; a gate refusal's chunks were
-- never shown to it.

SELECT
    h.source,
    count(*)                                AS retrieved,
    count(DISTINCT h.query_log_id)          AS questions,
    count(*) FILTER (WHERE h.rank = 1)      AS ranked_first,
    count(*) FILTER (WHERE h.cited)         AS cited,
    round(avg(h.cited::int), 3)             AS cite_rate
FROM query_log_hits h
JOIN query_logs q ON q.id = h.query_log_id
WHERE q.outcome IN ('answered', 'refused_llm')
GROUP BY h.source
ORDER BY retrieved DESC, h.source
LIMIT 10;
