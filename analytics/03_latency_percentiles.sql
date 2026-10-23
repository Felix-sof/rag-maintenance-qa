-- p50 and p95 latency per outcome over the last 30 days, and where the time goes.
--
-- Percentiles, not the mean: a handful of slow LLM retries would pull a mean
-- up while most people got a fast answer; p95 shows that tail on its own.
-- Split by outcome because a gate refusal never calls the LLM: mixed in, it
-- would make answered questions look faster than they are.
-- percentile_cont interpolates between the two nearest values, so p50 of
-- (100, 200) is 150.

SELECT
    outcome,
    count(*)                                                                    AS questions,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms)::numeric)    AS p50_ms,
    round(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric)    AS p95_ms,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY retrieval_ms)::numeric)  AS retrieval_p50_ms,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY llm_ms)::numeric)        AS llm_p50_ms,
    round(avg(prompt_tokens + output_tokens))                                   AS avg_tokens
FROM query_logs
WHERE created_at >= now() - interval '30 days'
GROUP BY outcome
ORDER BY questions DESC;
