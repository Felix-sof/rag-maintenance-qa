-- Daily question volume over the last 30 days, and how each day's questions ended.
--
-- Measures the load on the system and, per day, the mix of outcomes. FILTER
-- counts each subset in the same single pass over the rows, instead of one
-- subquery per outcome. Days are cut at midnight in Turkey, not in UTC (the
-- database's clock): a question at 01:30 local time belongs to that day.
-- Turkey has been on UTC+3 all year since 2016, so a fixed offset is exact,
-- and it works on servers built without the time zone database.

SELECT
    (created_at AT TIME ZONE INTERVAL '+03:00')::date                    AS day,
    count(*)                                                             AS questions,
    count(*) FILTER (WHERE outcome = 'answered')                         AS answered,
    count(*) FILTER (WHERE outcome IN ('refused_gate', 'refused_llm'))   AS refused,
    count(*) FILTER (WHERE outcome = 'error')                            AS errors
FROM query_logs
WHERE created_at >= now() - interval '30 days'
GROUP BY day
ORDER BY day;
