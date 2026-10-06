"""Runtime configuration for the live money-mule detection service.

Every value can be overridden with an environment variable of the same name
prefixed with ``MM_`` (e.g. ``MM_ATTACK_STEP_INTERVAL_MS=500``).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name, default, cast=str):
    raw = os.environ.get(f"MM_{name}")
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


DEMO_MODE = _env("DEMO_MODE", True, bool)

HOST = _env("HOST", "127.0.0.1")
PORT = _env("PORT", 8088, int)

# Split dashboard (python -m frontend.dev_server). Backend stays on PORT.
FRONTEND_HOST = _env("FRONTEND_HOST", "127.0.0.1")
FRONTEND_PORT = _env("FRONTEND_PORT", 3000, int)


def _origin(host, port):
    return f"http://{host}:{port}"


API_ORIGIN = _env("API_ORIGIN", _origin(HOST, PORT))
FRONTEND_ORIGIN = _env("FRONTEND_ORIGIN", _origin(FRONTEND_HOST, FRONTEND_PORT))
# Browser Origin values allowed to call /api and open /ws/live.
CORS_ORIGINS = [
    origin.strip()
    for origin in _env(
        "CORS_ORIGINS",
        ",".join(
            [
                FRONTEND_ORIGIN,
                _origin("localhost", FRONTEND_PORT),
                API_ORIGIN,
                _origin("localhost", PORT),
            ]
        ),
    ).split(",")
    if origin.strip()
]

DB_PATH = Path(_env("DB_PATH", str(BASE_DIR / "data" / "money_mule.db")))
RF_MODEL_PATH = Path(_env("RF_MODEL_PATH", str(BASE_DIR / "fraud_model.pkl")))
GNN_MODEL_PATH = Path(_env("GNN_MODEL_PATH", str(BASE_DIR / "gnn_model.pth")))
MODEL_META_PATH = Path(_env("MODEL_META_PATH", str(BASE_DIR / "model_meta.json")))

NUM_ACCOUNTS = _env("NUM_ACCOUNTS", 500, int)
SEED = _env("SEED", 0, int)  # 0 = nondeterministic

# Normal traffic cadence. Demo mode keeps the stream readable on screen.
NORMAL_TX_MIN_INTERVAL_MS = _env("NORMAL_TX_MIN_INTERVAL_MS", 250 if DEMO_MODE else 60, int)
NORMAL_TX_MAX_INTERVAL_MS = _env("NORMAL_TX_MAX_INTERVAL_MS", 700 if DEMO_MODE else 200, int)
ATTACK_STEP_INTERVAL_MS = _env("ATTACK_STEP_INTERVAL_MS", 700 if DEMO_MODE else 250, int)
NEW_ACCOUNT_INTERVAL_SEC = _env("NEW_ACCOUNT_INTERVAL_SEC", 45, float)
SPAWNER_AUTOSTART = _env("SPAWNER_AUTOSTART", True, bool)

# Sliding window used for graph features (must match the window the models
# were trained on; the trained value is recorded in model_meta.json).
FEATURE_WINDOW_SEC = _env("FEATURE_WINDOW_SEC", 120.0, float)
# Minimum spacing between GNN passes over the window graph (0 = every tx).
GNN_MIN_INTERVAL_MS = _env("GNN_MIN_INTERVAL_MS", 0, int)

# Fusion threshold the adaptive loop starts from (moves after each attack run).
# Default: the rule calibrated by train_live (model_meta.json "decision").
DETECTION_THRESHOLD_INITIAL = _env("DETECTION_THRESHOLD_INITIAL", None, float)
# The adaptive loop may lower the threshold at most this far below the
# calibrated value (beyond that the false-positive rate grows sharply).
THRESHOLD_BAND_BELOW = _env("THRESHOLD_BAND_BELOW", 0.04, float)
THRESHOLD_MAX = _env("THRESHOLD_MAX", 0.80, float)
# Grace period after an attack's last step before it is evaluated.
ATTACK_EVAL_GRACE_SEC = _env("ATTACK_EVAL_GRACE_SEC", 3.0, float)

DECAY_TICK_SEC = _env("DECAY_TICK_SEC", 2.0, float)
METRICS_INTERVAL_SEC = _env("METRICS_INTERVAL_SEC", 0.5, float)
HEARTBEAT_SEC = _env("HEARTBEAT_SEC", 15.0, float)
VOLUME_BUCKET_SEC = _env("VOLUME_BUCKET_SEC", 10, int)
VOLUME_BUCKETS = _env("VOLUME_BUCKETS", 60, int)
BASELINE_REFRESH_SEC = _env("BASELINE_REFRESH_SEC", 90.0, float)

# Memory bounds for live structures (the database keeps everything).
RECENT_TX_MEMORY = _env("RECENT_TX_MEMORY", 300, int)
WS_CLIENT_QUEUE = _env("WS_CLIENT_QUEUE", 2000, int)

LOG_LEVEL = _env("LOG_LEVEL", "INFO")
