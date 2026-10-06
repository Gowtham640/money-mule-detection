"""SQLite persistence (WAL mode).

The detection pipeline thread is the only writer. API handlers read through
their own thread-local connections, which WAL lets run concurrently with the
writer.
"""

import json
import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger("mule.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    account_id            TEXT PRIMARY KEY,
    created_at            TEXT NOT NULL,
    created_epoch         REAL NOT NULL,
    device_id             TEXT NOT NULL,
    ip_address            TEXT,
    home_channel          TEXT,
    balance               REAL NOT NULL DEFAULT 0,
    status                TEXT NOT NULL DEFAULT 'normal',
    early_risk            REAL NOT NULL DEFAULT 0,
    current_risk          REAL NOT NULL DEFAULT 0,
    rf_score              REAL,
    gnn_score             REAL,
    rule_score            INTEGER,
    mule_role             TEXT NOT NULL DEFAULT 'Unclassified',
    is_flagged            INTEGER NOT NULL DEFAULT 0,
    flagged_at            TEXT,
    explanation           TEXT,
    is_banned             INTEGER NOT NULL DEFAULT 0,
    banned_at             TEXT,
    is_attack_participant INTEGER NOT NULL DEFAULT 0,
    origin                TEXT NOT NULL DEFAULT 'seed',
    last_activity_at      TEXT,
    updated_at            TEXT
);

CREATE TABLE IF NOT EXISTS attack_runs (
    attack_run_id            TEXT PRIMARY KEY,
    attack_type              TEXT NOT NULL,
    attack_name              TEXT NOT NULL,
    status                   TEXT NOT NULL,
    started_at               TEXT NOT NULL,
    started_epoch            REAL NOT NULL,
    steps_finished_at        TEXT,
    finished_at              TEXT,
    planned_steps            INTEGER NOT NULL,
    transaction_count        INTEGER NOT NULL DEFAULT 0,
    participants             TEXT NOT NULL,
    labeled_participants     TEXT NOT NULL,
    warned_before_start      TEXT,
    first_warning_at         TEXT,
    first_warning_latency_ms INTEGER,
    detected_at              TEXT,
    detection_latency_ms     INTEGER,
    detected_accounts        TEXT,
    missed_accounts          TEXT,
    false_positive_accounts  TEXT,
    max_risk                 REAL,
    precision                REAL,
    recall                   REAL,
    threshold_before         REAL,
    threshold_after          REAL,
    pattern_similarity       REAL,
    pattern_match_run_id     TEXT,
    role_counts              TEXT,
    error                    TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
    seq            INTEGER PRIMARY KEY,
    transaction_id TEXT NOT NULL UNIQUE,
    timestamp      TEXT NOT NULL,
    ts_epoch       REAL NOT NULL,
    sender         TEXT NOT NULL REFERENCES accounts(account_id),
    receiver       TEXT NOT NULL REFERENCES accounts(account_id),
    amount         REAL NOT NULL CHECK (amount > 0),
    channel        TEXT NOT NULL,
    device_id      TEXT,
    ip_address     TEXT,
    source         TEXT NOT NULL,
    is_attack      INTEGER NOT NULL DEFAULT 0,
    attack_run_id  TEXT REFERENCES attack_runs(attack_run_id),
    attack_type    TEXT,
    attack_step    INTEGER,
    processing_ms  REAL,
    CHECK (sender <> receiver)
);
CREATE INDEX IF NOT EXISTS idx_tx_ts ON transactions(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_tx_sender ON transactions(sender, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_tx_receiver ON transactions(receiver, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_tx_attack ON transactions(attack_run_id);

CREATE TABLE IF NOT EXISTS detection_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    account_id     TEXT NOT NULL REFERENCES accounts(account_id),
    timestamp      TEXT NOT NULL,
    ts_epoch       REAL NOT NULL,
    rule_score     INTEGER NOT NULL,
    rule_flags     TEXT NOT NULL,
    rf_score       REAL NOT NULL,
    gnn_score      REAL,
    fused_score    REAL NOT NULL,
    early_risk     REAL NOT NULL,
    threshold      REAL NOT NULL,
    status         TEXT NOT NULL,
    detected       INTEGER NOT NULL,
    mule_role      TEXT,
    drift_score    REAL
);
CREATE INDEX IF NOT EXISTS idx_det_account ON detection_results(account_id, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_det_tx ON detection_results(transaction_id);

CREATE VIEW IF NOT EXISTS risk_history AS
    SELECT account_id, timestamp, ts_epoch, early_risk, fused_score,
           rf_score, gnn_score, rule_score, status
    FROM detection_results;

CREATE TABLE IF NOT EXISTS alerts (
    alert_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    ts_epoch       REAL NOT NULL,
    account_id     TEXT REFERENCES accounts(account_id),
    transaction_id TEXT,
    alert_type     TEXT NOT NULL,
    severity       TEXT NOT NULL,
    risk_score     REAL,
    message        TEXT NOT NULL,
    details        TEXT,
    attack_run_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts_epoch);
CREATE INDEX IF NOT EXISTS idx_alerts_account ON alerts(account_id);

CREATE TABLE IF NOT EXISTS pattern_memory (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    attack_run_id  TEXT REFERENCES attack_runs(attack_run_id),
    created_at     TEXT NOT NULL,
    attack_type    TEXT,
    signature      TEXT NOT NULL,
    similarity     REAL,
    matched_run_id TEXT
);

CREATE TABLE IF NOT EXISTS system_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Tables cleared by a full reset, children first.
RESET_ORDER = [
    "detection_results",
    "alerts",
    "pattern_memory",
    "transactions",
    "attack_runs",
    "accounts",
    "system_state",
]


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._local = threading.local()

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.conn = conn
        return conn

    def init_schema(self):
        conn = self.connect()
        conn.executescript(SCHEMA)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        logger.info("database ready path=%s journal_mode=%s", self.path, mode)
        return mode

    # -- generic read helpers (safe from any thread) -------------------------
    def query(self, sql, params=()):
        return [dict(row) for row in self.connect().execute(sql, params).fetchall()]

    def query_one(self, sql, params=()):
        row = self.connect().execute(sql, params).fetchone()
        return dict(row) if row else None

    def scalar(self, sql, params=()):
        row = self.connect().execute(sql, params).fetchone()
        return row[0] if row else None

    def ping(self):
        try:
            self.connect().execute("SELECT 1").fetchone()
            return "ok"
        except Exception as exc:  # pragma: no cover - surfaced in /api/health
            return f"error: {exc}"

    def table_counts(self):
        names = ["accounts", "transactions", "detection_results", "alerts", "attack_runs", "pattern_memory"]
        counts = {name: self.scalar(f"SELECT COUNT(*) FROM {name}") for name in names}
        counts["risk_history"] = self.scalar("SELECT COUNT(*) FROM risk_history")
        return counts

    def get_state(self, key, default=None):
        row = self.query_one("SELECT value FROM system_state WHERE key = ?", (key,))
        return json.loads(row["value"]) if row else default


def dumps(value):
    return json.dumps(value, separators=(",", ":"), default=str)


def loads(value, default=None):
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
