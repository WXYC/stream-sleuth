-- The flowsheet export evaluation/corpus.py reads (plan §5.3), checked in verbatim.
-- Run in psql against a read-only session (PGOPTIONS='-c default_transaction_read_only=on')
-- from a dated export directory; never overwrite an earlier export.
-- 1. Every flowsheet row since 2022, all entry types (talk-heavy hours are chosen by
--    talkset/message counts), ordered the way Backend's time-window read orders them.
\copy (SELECT id, show_id, play_order, legacy_entry_id, entry_type, add_time, artist_name, track_title, album_title, rotation_id, album_id FROM wxyc_schema.flowsheet WHERE add_time >= '2022-01-01' ORDER BY add_time, id) TO 'flowsheet.csv' WITH (FORMAT csv, HEADER true)

-- 2. The flowsheet-etl watermark that fixes ETL_STOP (§4).
\copy (SELECT job_name, last_run FROM wxyc_schema.cronjob_runs WHERE job_name = 'flowsheet-etl') TO 'cronjob_runs.csv' WITH (FORMAT csv, HEADER true)
