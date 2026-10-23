-- Which copilot tools the questions used, and what it cost.
--
-- tools is an array per question; unnest() turns it into one row per tool
-- used, so a question that called two tools counts once under each. A
-- question that called none is counted under '(none)': the plain
-- knowledge-base answers, for comparison. Tool questions take an extra LLM
-- call, which the latency and token columns should show.

SELECT
    tool,
    count(*)                                                                    AS questions,
    count(*) FILTER (WHERE outcome = 'answered')                                AS answered,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)::numeric)    AS p50_ms,
    round(avg(llm_calls), 2)                                                    AS avg_llm_calls,
    round(avg(prompt_tokens + output_tokens))                                   AS avg_tokens
FROM query_logs
CROSS JOIN LATERAL unnest(CASE WHEN cardinality(tools) = 0 THEN ARRAY['(none)'] ELSE tools END) AS tool
WHERE outcome <> 'refused_gate'
GROUP BY tool
ORDER BY questions DESC;
