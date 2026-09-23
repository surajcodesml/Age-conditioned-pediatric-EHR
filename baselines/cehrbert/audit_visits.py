import duckdb

def audit_visit_construction_duckdb(parquet_path):
    print("Connecting to DuckDB...")
    con = duckdb.connect()
    
    # Total events
    total_events = con.execute(f"SELECT COUNT(*) FROM '{parquet_path}'").fetchone()[0]
    
    # Events with direct encounter mapping
    events_with_hadm = con.execute(f"SELECT COUNT(*) FROM '{parquet_path}' WHERE hadm_id IS NOT NULL").fetchone()[0]
    
    # Compute admission intervals and recover missing hadm_ids
    # We join events with missing hadm_id to admissions of the same subject_id
    query_recovered = f"""
    WITH admissions AS (
        SELECT subject_id, hadm_id, MIN(timestamp_days) as min_t, MAX(timestamp_days) as max_t
        FROM '{parquet_path}'
        WHERE hadm_id IS NOT NULL
        GROUP BY subject_id, hadm_id
    ),
    missing AS (
        SELECT row_number() OVER () as id, subject_id, timestamp_days
        FROM '{parquet_path}'
        WHERE hadm_id IS NULL
    ),
    recovered AS (
        SELECT m.id, m.subject_id, a.hadm_id
        FROM missing m
        JOIN admissions a ON m.subject_id = a.subject_id
        WHERE m.timestamp_days >= a.min_t - 1.0 AND m.timestamp_days <= a.max_t + 1.0
    )
    SELECT COUNT(DISTINCT id) FROM recovered
    """
    print("Computing recovered events...")
    events_recovered = con.execute(query_recovered).fetchone()[0]
    
    events_using_other = 0
    events_remaining = total_events - events_with_hadm - events_recovered
    
    pct_pseudo = (events_remaining / total_events) * 100 if total_events > 0 else 0
    
    print("Computing visits...")
    # Real visits
    real_visits = con.execute(f"SELECT COUNT(DISTINCT subject_id || '_' || hadm_id) FROM '{parquet_path}' WHERE hadm_id IS NOT NULL").fetchone()[0]
    
    # Pseudo visits
    query_pseudo = f"""
    WITH admissions AS (
        SELECT subject_id, hadm_id, MIN(timestamp_days) as min_t, MAX(timestamp_days) as max_t
        FROM '{parquet_path}'
        WHERE hadm_id IS NOT NULL
        GROUP BY subject_id, hadm_id
    ),
    missing AS (
        SELECT row_number() OVER () as id, subject_id, timestamp_days, floor(timestamp_days) as day
        FROM '{parquet_path}'
        WHERE hadm_id IS NULL
    ),
    recovered AS (
        SELECT m.id
        FROM missing m
        JOIN admissions a ON m.subject_id = a.subject_id
        WHERE m.timestamp_days >= a.min_t - 1.0 AND m.timestamp_days <= a.max_t + 1.0
    ),
    remaining AS (
        SELECT subject_id, day
        FROM missing
        WHERE id NOT IN (SELECT id FROM recovered)
    )
    SELECT COUNT(DISTINCT subject_id || '_' || day) FROM remaining
    """
    pseudo_visits = con.execute(query_pseudo).fetchone()[0]
    
    # Mean/Median events per visit
    # We can get counts per visit and compute stats
    query_counts = f"""
    WITH admissions AS (
        SELECT subject_id, hadm_id, MIN(timestamp_days) as min_t, MAX(timestamp_days) as max_t
        FROM '{parquet_path}'
        WHERE hadm_id IS NOT NULL
        GROUP BY subject_id, hadm_id
    ),
    missing AS (
        SELECT row_number() OVER () as id, subject_id, timestamp_days, floor(timestamp_days) as day
        FROM '{parquet_path}'
        WHERE hadm_id IS NULL
    ),
    recovered AS (
        SELECT m.id, m.subject_id, a.hadm_id, row_number() OVER (PARTITION BY m.id ORDER BY a.hadm_id) as rn
        FROM missing m
        JOIN admissions a ON m.subject_id = a.subject_id
        WHERE m.timestamp_days >= a.min_t - 1.0 AND m.timestamp_days <= a.max_t + 1.0
    ),
    dedup_recovered AS (
        SELECT id, subject_id, hadm_id FROM recovered WHERE rn = 1
    ),
    remaining AS (
        SELECT subject_id, day
        FROM missing
        WHERE id NOT IN (SELECT id FROM dedup_recovered)
    ),
    visit_counts AS (
        SELECT subject_id || '_' || hadm_id as visit_id, COUNT(*) as cnt
        FROM (
            SELECT subject_id, CAST(hadm_id AS VARCHAR) as hadm_id FROM '{parquet_path}' WHERE hadm_id IS NOT NULL
            UNION ALL
            SELECT subject_id, CAST(hadm_id AS VARCHAR) as hadm_id FROM dedup_recovered
        )
        GROUP BY subject_id, hadm_id
        
        UNION ALL
        
        SELECT subject_id || '_day_' || CAST(day AS VARCHAR) as visit_id, COUNT(*) as cnt
        FROM remaining
        GROUP BY subject_id, day
    )
    SELECT AVG(cnt), median(cnt) FROM visit_counts
    """
    print("Computing mean/median events...")
    mean_events, median_events = con.execute(query_counts).fetchone()
    if mean_events is None:
        mean_events = 0
    if median_events is None:
        median_events = 0
        
    report = f"""total events: {total_events}
events with direct encounter mapping: {events_with_hadm}
events recovered by timestamp-to-admission mapping: {events_recovered}
events using other encounter IDs: {events_using_other}
events remaining in pseudo-visits: {events_remaining}
percentage of all tokens in pseudo-visits: {pct_pseudo:.2f}%
number of real visits: {real_visits}
number of pseudo-visits: {pseudo_visits}
median/mean events per visit: {median_events:.1f} / {mean_events:.1f}
"""
    print(report)

if __name__ == "__main__":
    import sys
    path = "data/processed/test_events.parquet"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    audit_visit_construction_duckdb(path)
