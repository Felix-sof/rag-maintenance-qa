-- 004: which copilot tools each question used (empty for a plain knowledge-base answer).

ALTER TABLE query_logs ADD COLUMN tools text[] NOT NULL DEFAULT '{}';
