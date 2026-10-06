"""FastAPI application: REST for snapshots, history and commands; WebSocket
``/ws/live`` for incremental events. Optionally serves ``frontend/`` as well.

API/WS bind ``MM_HOST:MM_PORT`` (8088). The split dashboard is
``python -m frontend.dev_server`` on ``MM_FRONTEND_PORT`` (3000).
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import settings
from backend.attacks import ATTACK_CATALOG, attack_types
from backend.db import loads
from backend.pipeline import iso, serialize_run
from backend.runtime import Runtime

logger = logging.getLogger("mule.api")
FRONTEND_DIR = settings.BASE_DIR / "frontend"
LIB_DIR = settings.BASE_DIR / "lib"

runtime = Runtime()


def configure_logging():
    root = logging.getLogger()
    if not any(getattr(h, "_mule", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler._mule = True
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel(settings.LOG_LEVEL)
    logging.getLogger("mule").setLevel(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(_app):
    configure_logging()
    logger.info("backend starting (demo_mode=%s, db=%s)", settings.DEMO_MODE, settings.DB_PATH)
    await runtime.start()
    yield
    await runtime.stop()


app = FastAPI(title="Money Mule Detection", lifespan=lifespan)
# Dashboard is served on FRONTEND_PORT; REST + WS stay on PORT (8088).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def revalidate_static(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        # Dashboard assets revalidate via ETag so an updated build is never stale.
        response.headers["Cache-Control"] = "no-cache"
    return response


class SpawnAttack(BaseModel):
    attack_type: str


class BanRequest(BaseModel):
    account_ids: list[str] = Field(min_length=1)


class ManualTransaction(BaseModel):
    sender: str
    receiver: str
    amount: float
    channel: str = "UPI"


class SpawnerControl(BaseModel):
    running: bool


def _require_ready():
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="backend is starting")


# ------------------------------------------------------------------ health/state
@app.get("/api/health")
def health():
    return runtime.health()


@app.get("/api/live/state")
async def live_state():
    """Initial snapshot. ``seq`` is the last event folded into it; clients
    ignore WebSocket events with ``seq <= snapshot.seq``."""
    _require_ready()
    snapshot = await runtime.call(runtime.pipeline.snapshot)
    snapshot["metrics"]["connected_clients"] = runtime.ws.count
    snapshot["health"] = {
        "spawner": "running" if runtime.pipeline.spawner_running else "paused",
        "rf_model": runtime.models.status["rf"],
        "gnn_model": runtime.models.status["gnn"],
        "shap": runtime.models.status["shap"],
        "model_meta": runtime.models.status["meta"],
    }
    return snapshot


@app.get("/api/metrics")
async def metrics():
    _require_ready()
    data = await runtime.call(runtime.pipeline.metrics)
    data["connected_clients"] = runtime.ws.count
    return data


# ---------------------------------------------------------------------- accounts
@app.get("/api/accounts")
def list_accounts(
    status: str | None = Query(None, pattern="^(normal|early|fraud|banned)$"),
    q: str | None = None,
    sort: str = Query("risk", pattern="^(risk|early|id|activity)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    order = {
        "risk": "current_risk DESC",
        "early": "early_risk DESC",
        "id": "account_id",
        "activity": "last_activity_at DESC",
    }[sort]
    where, params = [], []
    if status:
        where.append("status = ?")
        params.append(status)
    if q:
        where.append("account_id LIKE ?")
        params.append(f"%{q.upper()}%")
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = runtime.db.query(
        f"SELECT account_id, status, early_risk, current_risk, rf_score, gnn_score, rule_score, mule_role, "
        f"is_flagged, flagged_at, is_banned, device_id, home_channel, created_at, origin, last_activity_at, balance "
        f"FROM accounts {clause} ORDER BY {order} LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )
    total = runtime.db.scalar(f"SELECT COUNT(*) FROM accounts {clause}", params)
    return {"total": total, "limit": limit, "offset": offset, "accounts": rows}


@app.get("/api/accounts/{account_id}")
async def account_detail(account_id: str):
    _require_ready()
    detail = await runtime.call(runtime.pipeline.account_detail, account_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"account {account_id} not found")
    return detail


@app.post("/api/accounts/ban")
async def ban_accounts(body: BanRequest):
    _require_ready()
    try:
        banned, events = await runtime.call(runtime.pipeline.ban, body.account_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    runtime.publish(events)
    return {"status": "ok", "banned": banned}


# ------------------------------------------------------------------ transactions
@app.get("/api/transactions")
def list_transactions(
    limit: int = Query(50, ge=1, le=500),
    before_seq: int | None = Query(None, ge=1),
    account: str | None = None,
    attack_run_id: str | None = None,
    source: str | None = Query(None, pattern="^(normal|attack|manual)$"),
):
    where, params = [], []
    if before_seq:
        where.append("seq < ?")
        params.append(before_seq)
    if account:
        where.append("(sender = ? OR receiver = ?)")
        params.extend([account, account])
    if attack_run_id:
        where.append("attack_run_id = ?")
        params.append(attack_run_id)
    if source:
        where.append("source = ?")
        params.append(source)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = runtime.db.query(f"SELECT * FROM transactions {clause} ORDER BY seq DESC LIMIT ?", (*params, limit))
    for row in rows:
        row["is_attack"] = bool(row["is_attack"])
    next_before = rows[-1]["seq"] if len(rows) == limit else None
    return {"transactions": rows, "next_before_seq": next_before}


@app.get("/api/transactions/{transaction_id}")
def transaction_detail(transaction_id: str):
    tx = runtime.db.query_one("SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,))
    if tx is None:
        raise HTTPException(status_code=404, detail=f"transaction {transaction_id} not found")
    tx["is_attack"] = bool(tx["is_attack"])
    detections = runtime.db.query("SELECT * FROM detection_results WHERE transaction_id = ?", (transaction_id,))
    for row in detections:
        row["rule_flags"] = loads(row["rule_flags"], {})
    return {"transaction": tx, "detections": detections}


@app.post("/api/transactions", status_code=201)
async def submit_transaction(body: ManualTransaction):
    """Inject a transaction by hand; it takes the same path as generated ones."""
    _require_ready()
    try:
        events = await runtime.call(runtime.pipeline.ingest, {**body.model_dump(), "source": "manual"})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    runtime.publish(events)
    tx = next(e["data"] for e in events if e["type"] == "transaction_created")
    scores = [e["data"] for e in events if e["type"] == "account_updated" and e["data"].get("transaction_id")]
    return {"transaction": tx, "accounts": scores}


# ------------------------------------------------------------------------ alerts
@app.get("/api/alerts")
def list_alerts(
    limit: int = Query(50, ge=1, le=500),
    before_id: int | None = Query(None, ge=1),
    alert_type: str | None = None,
):
    where, params = [], []
    if before_id:
        where.append("alert_id < ?")
        params.append(before_id)
    if alert_type:
        where.append("alert_type = ?")
        params.append(alert_type)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = runtime.db.query(f"SELECT * FROM alerts {clause} ORDER BY alert_id DESC LIMIT ?", (*params, limit))
    for row in rows:
        row["details"] = loads(row["details"], {})
    return {"alerts": rows}


# ----------------------------------------------------------------------- attacks
@app.get("/api/attacks/types")
def list_attack_types():
    return {"types": attack_types()}


@app.get("/api/attacks")
def list_attacks(limit: int = Query(50, ge=1, le=500)):
    rows = runtime.db.query("SELECT * FROM attack_runs ORDER BY started_epoch DESC LIMIT ?", (limit,))
    return {"attack_runs": [serialize_run(r) for r in rows]}


@app.get("/api/attacks/{attack_run_id}")
def attack_detail(attack_run_id: str):
    row = runtime.db.query_one("SELECT * FROM attack_runs WHERE attack_run_id = ?", (attack_run_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"attack run {attack_run_id} not found")
    txs = runtime.db.query("SELECT * FROM transactions WHERE attack_run_id = ? ORDER BY seq", (attack_run_id,))
    for tx in txs:
        tx["is_attack"] = bool(tx["is_attack"])
    run = serialize_run(row)
    participants = run["participants"]
    detections = []
    if participants:
        marks = ",".join("?" * len(participants))
        detections = runtime.db.query(
            f"SELECT d.account_id, d.timestamp, d.fused_score, d.rf_score, d.gnn_score, d.rule_score, d.early_risk, "
            f"d.status, d.transaction_id FROM detection_results d WHERE d.account_id IN ({marks}) "
            f"AND d.ts_epoch >= ? ORDER BY d.id",
            (*participants, row["started_epoch"] - 1),
        )
    alerts = runtime.db.query("SELECT * FROM alerts WHERE attack_run_id = ? ORDER BY alert_id", (attack_run_id,))
    for alert in alerts:
        alert["details"] = loads(alert["details"], {})
    return {"attack_run": run, "transactions": txs, "detections": detections, "alerts": alerts}


@app.post("/api/attacks/spawn", status_code=202)
async def spawn_attack(body: SpawnAttack):
    """Starts an attack run and returns immediately; steps stream over WS."""
    _require_ready()
    if body.attack_type not in ATTACK_CATALOG:
        raise HTTPException(status_code=422, detail=f"unknown attack_type {body.attack_type!r}; "
                                                    f"choose one of {sorted(ATTACK_CATALOG)}")
    try:
        run = await runtime.spawn_attack(body.attack_type)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started", "attack_run": run}


# ------------------------------------------------------------------------ charts
@app.get("/api/charts/volume")
def volume_chart(bucket_sec: int = Query(60, ge=5, le=3600), minutes: int = Query(60, ge=1, le=1440)):
    since = time.time() - minutes * 60
    rows = runtime.db.query(
        "SELECT CAST(ts_epoch / ? AS INTEGER) * ? AS bucket, COUNT(*) AS total, SUM(is_attack) AS attack, "
        "ROUND(SUM(amount), 2) AS amount FROM transactions WHERE ts_epoch >= ? GROUP BY bucket ORDER BY bucket",
        (bucket_sec, bucket_sec, since),
    )
    for row in rows:
        row["bucket_start"] = iso(row["bucket"])
    return {"bucket_sec": bucket_sec, "series": rows}


@app.get("/api/charts/channels")
def channel_chart():
    return {"channels": runtime.db.query(
        "SELECT channel, COUNT(*) AS total, SUM(is_attack) AS attack FROM transactions GROUP BY channel ORDER BY channel"
    )}


# ----------------------------------------------------------------------- control
@app.post("/api/spawner")
async def control_spawner(body: SpawnerControl):
    _require_ready()
    runtime.publish(await runtime.call(runtime.pipeline.set_spawner, body.running))
    return {"spawner_running": body.running}


@app.post("/api/admin/reset")
async def reset(confirm: bool = Body(False, embed=True)):
    if not confirm:
        raise HTTPException(status_code=400, detail="send {\"confirm\": true} to wipe the database")
    _require_ready()
    try:
        events = await runtime.call(runtime.pipeline.reset)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    runtime.publish(events)
    return {"status": "reset"}


# --------------------------------------------------------------------- websocket
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    hello = {
        "type": "hello",
        "server_time": iso(time.time()),
        "seq": runtime.pipeline.seq,
        "spawner_running": runtime.pipeline.spawner_running,
    }
    await runtime.ws.serve(websocket, hello)


@app.get("/config.js", include_in_schema=False)
def frontend_config():
    # Same-origin dashboard: empty origin keeps fetch/WS on this host.
    body = (
        'window.MM_API_ORIGIN = "";\n'
        f"window.MM_FRONTEND_ORIGIN = {settings.FRONTEND_ORIGIN!r};\n"
    )
    return Response(content=body, media_type="application/javascript")


# ------------------------------------------------------------------------ static
if LIB_DIR.exists():
    app.mount("/lib", StaticFiles(directory=str(LIB_DIR)), name="lib")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
