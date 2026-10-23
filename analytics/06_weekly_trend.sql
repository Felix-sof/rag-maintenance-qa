-- Week-by-week trend, with window functions.
--
-- generate_series fills in weeks with no questions, so that lag() compares a
-- week with the one right before it, not with the last week that had traffic.
-- change:          questions minus the previous week's (lag)
-- moving_avg_4w:   average over this week and the three before it (a frame of 4 rows)
-- running_total:   every question so far
-- refusal_rate and p95_ms are per week, as in 02 and 03. Weeks start on
-- Monday at midnight in Turkey (UTC+3, see 01).

WITH weekly AS (
    SELECT
        date_trunc('week', created_at AT TIME ZONE INTERVAL '+03:00')::date              AS week,
        count(*)                                                                          AS questions,
        avg((outcome <> 'answered')::int) FILTER (WHERE outcome <> 'error')               AS refusal_rate,
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)                         AS p95_ms
    FROM query_logs
    GROUP BY week
),
weeks AS (
    SELECT generate_series(min(week), max(week), interval '1 week')::date AS week
    FROM weekly
)
SELECT
    w.week,
    coalesce(k.questions, 0)                                                   AS questions,
    coalesce(k.questions, 0) - lag(coalesce(k.questions, 0)) OVER by_week      AS change,
    round(avg(coalesce(k.questions, 0)) OVER (by_week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW), 1)
                                                                               AS moving_avg_4w,
    sum(coalesce(k.questions, 0)) OVER (by_week ROWS UNBOUNDED PRECEDING)      AS running_total,
    round(k.refusal_rate::numeric, 3)                                          AS refusal_rate,
    round(k.p95_ms::numeric)                                                   AS p95_ms
FROM weeks w
LEFT JOIN weekly k USING (week)
WINDOW by_week AS (ORDER BY w.week)
ORDER BY w.week;
