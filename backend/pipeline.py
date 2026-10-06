"""The canonical ingestion and detection pipeline.

Every transaction (normal traffic, attack steps, manual API submissions) goes
through ``LivePipeline.ingest``:

    validate -> balances -> sliding-window graph -> early-warning signals
    -> features -> rules + Random Forest + GNN -> fusion -> detection
    -> roles / drift / SHAP -> SQLite (one DB transaction) -> events

All methods run on a single dedicated thread (see ``runtime.Runtime.call``),
so in-memory state, database writes and event sequence numbers never race.
Methods return the list of events they produced; the runtime broadcasts them.
"""

import logging
import math
import random
import statistics
import time
from collections import Counter, deque
from datetime import datetime, timezone

from backend import settings
from backend.attacks import AttackContext, attack_types, build_plan
from backend.db import RESET_ORDER, dumps, loads
from backend.detection import (
    MAX_RULE_SCORE,
    adaptive_threshold_update,
    behavioral_drift_score,
    classify_role,
    evaluate_rules,
    fuse_scores,
)
from backend.early_warning import EarlyWarningEngine
from backend.feature_store import RF_FEATURE_COLUMNS, FeatureStore
from backend.generator import CHANNELS, generate_accounts
from backend.risk_memory import best_match, signature_from_window
from backend.simulator import TrafficSimulator

logger = logging.getLogger("mule.pipeline")

DRIFT_ALERT_SCORE = 3.0
WARNING_ALERT_COOLDOWN_SEC = 120.0


def iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def r4(value):
    return None if value is None else round(float(value), 4)


class LivePipeline:
    def __init__(self, db, models):
        self.db = db
        self.models = models
        self.rng = random.Random(settings.SEED or None)
        self.spawner_running = settings.SPAWNER_AUTOSTART
        self._reset_memory()

    # ------------------------------------------------------------------ state
    def _reset_memory(self):
        self.accounts = {}
        self.features = FeatureStore(settings.FEATURE_WINDOW_SEC)
        self.early = EarlyWarningEngine()
        self.sim = TrafficSimulator(self.rng)
        self.gnn_scores = {}
        self.last_gnn_at = 0.0
        self.detection_threshold = 0.5
        self.calibrated_threshold = 0.5
        self.threshold_floor = 0.3
        self.min_early_risk = 0.0
        self.last_eval_iso = None
        self.seq = 0
        self.next_tx_seq = 1
        self.next_account_num = 1
        self.next_run_num = 1
        self.last_ts = 0.0
        self.recent_tx = deque(maxlen=settings.RECENT_TX_MEMORY)
        self.recent_alerts = deque(maxlen=100)
        self.active_runs = {}
        self.latencies = deque(maxlen=500)
        self.tx_times = deque()
        self.volume = {}
        self.channel_counts = Counter()
        self.tx_by_source = Counter()
        self.alerts_by_type = Counter()
        self.run_totals = {}
        self.attack_type_stats = {}
        self.baseline = {}
        self.baseline_std = {}
        self.baseline_at = 0.0
        self.warning_alerted_at = {}
        self.pattern_memory = []
        self.pattern_run_ids = []
        self.last_pattern = None
        self._events = []
        self.started_at = time.time()

    def _event(self, event_type, data):
        self.seq += 1
        self._events.append({"type": event_type, "seq": self.seq, "timestamp": iso(time.time()), "data": data})

    def _take_events(self):
        events, self._events = self._events, []
        return events

    def active_ids(self):
        return [a for a, info in self.accounts.items() if not info["is_banned"]]

    def flagged_ids(self):
        return {a for a, info in self.accounts.items() if info["is_flagged"]}

    def status_of(self, acc):
        info = self.accounts[acc]
        if info["is_banned"]:
            return "banned"
        if info["is_flagged"]:
            return "fraud"
        if acc in self.early.warning_ids:
            return "early"
        return "normal"

    def account_payload(self, acc):
        info = self.accounts[acc]
        state = self.early.state_of(acc)
        return {
            "account_id": acc,
            "status": self.status_of(acc),
            "early_risk": state["risk"],
            "signals": state["signals"],
            "reasons": state["reasons"] if state["risk"] >= 0.2 or info["is_flagged"] else [],
            "fused_score": r4(info["fused"]),
            "rf_score": r4(info["rf"]),
            "gnn_score": r4(info["gnn"]),
            "rule_score": info["rule_score"],
            "rules_fired": [k for k, fired in (info["rule_flags"] or {}).items() if fired],
            "role": info["role"],
            "drift_score": r4(info["drift"]),
            "drift_top": info["drift_top"],
            "flagged_at": info["flagged_at"],
            "device_id": info["device_id"],
            "home_channel": info["home_channel"],
            "created_at": info["created_at"],
            "origin": info["origin"],
            "last_activity": iso(info["last_activity"]) if info["last_activity"] else None,
        }

    # ---------------------------------------------------------------- startup
    def bootstrap(self):
        conn = self.db.connect()
        if not self.db.scalar("SELECT COUNT(*) FROM accounts"):
            self._seed_accounts()
        for row in self.db.query("SELECT * FROM accounts"):
            self._load_account(row)
        self.next_account_num = max((self._account_num(a) for a in self.accounts), default=0) + 1
        self.next_tx_seq = (self.db.scalar("SELECT MAX(seq) FROM transactions") or 0) + 1
        self.next_run_num = (self.db.scalar("SELECT COUNT(*) FROM attack_runs") or 0) + 1
        now = time.time()
        interrupted = conn.execute(
            "UPDATE attack_runs SET status='interrupted', finished_at=?, error='service restarted mid-run' "
            "WHERE status IN ('running', 'evaluating')",
            (iso(now),),
        ).rowcount
        if interrupted:
            logger.warning("marked %d unfinished attack run(s) as interrupted", interrupted)
        decision = self.models.meta.get("decision", {})
        self.calibrated_threshold = float(settings.DETECTION_THRESHOLD_INITIAL or decision.get("threshold") or 0.5)
        self.min_early_risk = float(decision.get("min_early_risk", 0.0))
        self.threshold_floor = max(0.3, self.calibrated_threshold - settings.THRESHOLD_BAND_BELOW)
        self.detection_threshold = self.calibrated_threshold
        last = self.db.query_one(
            "SELECT threshold_after, finished_at FROM attack_runs WHERE threshold_after IS NOT NULL "
            "ORDER BY started_epoch DESC LIMIT 1"
        )
        if last is not None:
            self.detection_threshold = self._clamp_threshold(float(last["threshold_after"]))
            self.last_eval_iso = last["finished_at"]
        for row in self.db.query("SELECT attack_run_id, signature FROM pattern_memory ORDER BY id"):
            self.pattern_memory.append(loads(row["signature"]))
            self.pattern_run_ids.append(row["attack_run_id"])
        self._replay_recent(now)
        self._load_counters(now)
        self.sim.set_accounts(self.accounts, self.active_ids())
        self.early.set_excluded(self.flagged_ids() | {a for a in self.accounts if self.accounts[a]["is_banned"]})
        self.early.select_warnings(self.active_ids())
        self._run_gnn(now, force=True)
        self._refresh_baseline(now)
        logger.info(
            "state restored accounts=%d transactions=%d window_tx=%d flagged=%d threshold=%.3f",
            len(self.accounts), self.tx_by_source.total(), len(self.features.window),
            len(self.flagged_ids()), self.detection_threshold,
        )

    @staticmethod
    def _account_num(acc):
        digits = "".join(ch for ch in acc if ch.isdigit())
        return int(digits) if digits else 0

    def _seed_accounts(self):
        frame = generate_accounts(settings.NUM_ACCOUNTS, rng=self.rng)
        now = time.time()
        rows = []
        for row in frame.itertuples(index=False):
            created = row.creation_time.timestamp()
            rows.append((row.account_id, iso(created), created, row.device_id, row.ip_address, row.channel,
                         float(row.balance), "seed", iso(now)))
        conn = self.db.connect()
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO accounts (account_id, created_at, created_epoch, device_id, ip_address, home_channel, "
            "balance, origin, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.execute("COMMIT")
        logger.info("seeded %d accounts", len(rows))

    def _load_account(self, row):
        acc = row["account_id"]
        self.accounts[acc] = {
            "account_id": acc,
            "created_epoch": row["created_epoch"],
            "created_at": row["created_at"],
            "device_id": row["device_id"],
            "ip_address": row["ip_address"],
            "home_channel": row["home_channel"],
            "balance": row["balance"],
            "origin": row["origin"],
            "is_banned": bool(row["is_banned"]),
            "is_flagged": bool(row["is_flagged"]),
            "flagged_at": row["flagged_at"],
            "role": row["mule_role"],
            "fused": row["current_risk"],
            "rf": row["rf_score"],
            "gnn": row["gnn_score"],
            "rule_score": row["rule_score"],
            "rule_flags": {},
            "drift": None,
            "drift_top": [],
            "explanation": loads(row["explanation"]),
            "last_activity": None,
        }
        self.features.register_account(acc, row["created_epoch"], row["device_id"])
        self.early.register_account(acc, row["created_epoch"], time.time())
        self._sync_device_sizes([row["device_id"]])

    def _sync_device_sizes(self, devices):
        for device in devices:
            members = self.features.device_cluster(device)
            for member in members:
                self.early.device_size[member] = len(members)

    def _replay_recent(self, now):
        """Rebuilds the window graph and warning state from persisted rows."""
        horizon = now - max(settings.FEATURE_WINDOW_SEC, EarlyWarningEngine.HISTORY_SEC)
        rows = self.db.query(
            "SELECT * FROM transactions WHERE ts_epoch >= ? ORDER BY seq", (horizon,)
        )
        for tx in rows:
            ts = tx["ts_epoch"]
            if tx["sender"] not in self.accounts or tx["receiver"] not in self.accounts:
                continue
            if ts >= now - settings.FEATURE_WINDOW_SEC:
                self.features.add(ts, tx["sender"], tx["receiver"], tx["amount"], tx["channel"], bool(tx["is_attack"]))
            self.early.apply_transaction(tx["sender"], tx["receiver"], ts, tx["amount"], tx["channel"])
            for acc in (tx["sender"], tx["receiver"]):
                self.accounts[acc]["last_activity"] = ts
            self.last_ts = max(self.last_ts, ts)
        self.features.expire(now)
        self.early.decay_hot(now)
        for tx in self.db.query(
            "SELECT * FROM transactions ORDER BY seq DESC LIMIT ?", (min(150, settings.RECENT_TX_MEMORY),)
        )[::-1]:
            self.recent_tx.append(self._tx_payload_from_row(tx))
        for alert in self.db.query("SELECT * FROM alerts ORDER BY alert_id DESC LIMIT 60")[::-1]:
            self.recent_alerts.append(self._alert_payload(alert))

    def _load_counters(self, now):
        for row in self.db.query("SELECT source, COUNT(*) AS n FROM transactions GROUP BY source"):
            self.tx_by_source[row["source"]] = row["n"]
        for row in self.db.query("SELECT channel, COUNT(*) AS n FROM transactions GROUP BY channel"):
            self.channel_counts[row["channel"]] = row["n"]
        for row in self.db.query("SELECT alert_type, COUNT(*) AS n FROM alerts GROUP BY alert_type"):
            self.alerts_by_type[row["alert_type"]] = row["n"]
        bucket = settings.VOLUME_BUCKET_SEC
        since = now - bucket * settings.VOLUME_BUCKETS
        for row in self.db.query(
            "SELECT CAST(ts_epoch / ? AS INTEGER) AS b, SUM(is_attack = 0) AS normal, SUM(is_attack = 1) AS attack "
            "FROM transactions WHERE ts_epoch >= ? GROUP BY b",
            (bucket, since),
        ):
            self.volume[row["b"]] = [row["normal"], row["attack"]]
        for row in self.db.query("SELECT * FROM attack_runs"):
            self._account_run_totals(row)

    def _account_run_totals(self, run):
        """Folds a persisted attack run into aggregate statistics."""
        stats = self.attack_type_stats.setdefault(run["attack_type"], {"runs": 0, "detected": 0, "evaluated": 0})
        stats["runs"] += 1
        totals = self.run_totals.setdefault("all", {"runs": 0, "completed": 0, "detected": 0, "tp": 0, "fp": 0,
                                                     "fn": 0, "latencies": [], "warning_latencies": []})
        totals["runs"] += 1
        if run["status"] == "completed":
            totals["completed"] += 1
            stats["evaluated"] += 1
            detected = loads(run["detected_accounts"], [])
            totals["tp"] += len(detected)
            totals["fp"] += len(loads(run["false_positive_accounts"], []))
            totals["fn"] += len(loads(run["missed_accounts"], []))
        if run["detected_at"]:
            totals["detected"] += 1
            stats["detected"] += 1
            totals["latencies"].append(run["detection_latency_ms"])
        if run["first_warning_latency_ms"] is not None:
            totals["warning_latencies"].append(run["first_warning_latency_ms"])

    # ---------------------------------------------------------------- ingest
    def spawn_normal(self):
        if not self.spawner_running:
            return []
        spec = self.sim.next_transaction()
        if spec is None:
            return []
        spec["source"] = "normal"
        return self.ingest(spec)

    def _validate(self, spec):
        sender, receiver = str(spec.get("sender", "")), str(spec.get("receiver", ""))
        if sender not in self.accounts:
            raise ValueError(f"unknown sender account {sender!r}")
        if receiver not in self.accounts:
            raise ValueError(f"unknown receiver account {receiver!r}")
        if sender == receiver:
            raise ValueError("sender and receiver must differ")
        for acc in (sender, receiver):
            if self.accounts[acc]["is_banned"]:
                raise ValueError(f"account {acc} is banned")
        try:
            amount = float(spec.get("amount"))
        except (TypeError, ValueError):
            raise ValueError("amount must be a number") from None
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("amount must be a positive finite number")
        channel = str(spec.get("channel") or "")
        if channel not in CHANNELS:
            raise ValueError(f"channel must be one of {CHANNELS}")
        return sender, receiver, round(amount, 2), channel

    def ingest(self, spec):
        started = time.perf_counter()
        sender, receiver, amount, channel = self._validate(spec)
        source = spec.get("source", "manual")
        run_id = spec.get("attack_run_id")
        is_attack = bool(run_id)
        run = self.active_runs.get(run_id) if run_id else None
        if is_attack and run is None:
            raise ValueError(f"attack run {run_id} is not active")

        now = max(time.time(), self.last_ts + 0.001)
        self.last_ts = now
        tx_seq = self.next_tx_seq
        tx_id = f"TX{tx_seq:09d}"
        s_info, r_info = self.accounts[sender], self.accounts[receiver]
        if s_info["balance"] < amount:
            if not is_attack:
                raise ValueError(f"insufficient balance in {sender}")
            s_info["balance"] += amount  # mule operator cash deposit funding the step
        s_info["balance"] -= amount
        r_info["balance"] += amount
        device = spec.get("device_id") or s_info["device_id"]

        # Sliding-window graph and early-warning signals
        changed_edges, removed_edges, _ = self.features.expire(now)
        edge = self.features.add(now, sender, receiver, amount, channel, is_attack)
        self.early.apply_transaction(sender, receiver, now, amount, channel)
        self._run_gnn(now)

        # Score the affected accounts
        affected = [sender, receiver]
        feats = {acc: self.features.features(acc, now) for acc in affected}
        rf_scores = self.models.rf_predict([feats[acc] for acc in affected])
        scored = {}
        for acc, rf in zip(affected, rf_scores):
            rule_score, rule_flags, rule_reasons = evaluate_rules(feats[acc])
            gnn = float(self.gnn_scores[acc]) if self.models.gnn_ready and acc in self.gnn_scores else None
            fused = fuse_scores(rf, rule_score, gnn)
            drift, drift_top = self._drift(acc, feats[acc])
            scored[acc] = {"rf": rf, "gnn": gnn, "fused": fused, "rule_score": rule_score,
                           "rule_flags": rule_flags, "rule_reasons": rule_reasons, "drift": drift, "drift_top": drift_top}

        prev_status = {acc: self.status_of(acc) for acc in affected}
        prev_roles = {acc: self.accounts[acc]["role"] for acc in affected}
        newly_flagged = []
        for acc in affected:
            info, s = self.accounts[acc], scored[acc]
            info.update(rf=s["rf"], gnn=s["gnn"], fused=s["fused"], rule_score=s["rule_score"],
                        rule_flags=s["rule_flags"], drift=s["drift"], drift_top=s["drift_top"], last_activity=now)
            if (not info["is_flagged"] and s["fused"] >= self.detection_threshold
                    and self.early.state_of(acc)["risk"] >= self.min_early_risk):
                info["is_flagged"] = True
                info["flagged_at"] = iso(now)
                newly_flagged.append(acc)
            if info["is_flagged"]:
                role = classify_role(feats[acc])
                if role != "Unclassified" or info["role"] == "Unclassified":
                    info["role"] = role
            if run is not None and acc in run["labeled"]:
                run["max_risk"] = max(run["max_risk"], s["fused"])

        if newly_flagged:
            self.early.set_excluded(self.flagged_ids() | {a for a in self.accounts if self.accounts[a]["is_banned"]})
        entered, left = self.early.select_warnings(self.active_ids())

        # ---- events (transaction first so the UI has the edge before colours)
        tx_payload = {
            "transaction_id": tx_id,
            "seq": tx_seq,
            "timestamp": iso(now),
            "ts": round(now, 3),
            "sender": sender,
            "receiver": receiver,
            "amount": amount,
            "channel": channel,
            "device_id": device,
            "source": source,
            "is_attack": is_attack,
            "attack_run_id": run_id,
            "attack_type": spec.get("attack_type"),
            "attack_step": spec.get("attack_step"),
            "sender_score": r4(scored[sender]["fused"]),
            "receiver_score": r4(scored[receiver]["fused"]),
            "sender_status": self.status_of(sender),
            "receiver_status": self.status_of(receiver),
        }
        self._event("transaction_created", tx_payload)
        self._event("graph_edge_upserted", edge)
        if changed_edges or removed_edges:
            self._event("graph_edges_expired", {"changed": changed_edges, "removed": removed_edges})

        status_changed = set(entered) | set(left)
        for acc in affected:
            self._event("account_updated", {**self.account_payload(acc), "prev_status": prev_status[acc],
                                            "transaction_id": tx_id})
        for acc in status_changed - set(affected):
            self._event("account_updated", {**self.account_payload(acc), "prev_status": "early" if acc in left else "normal"})

        alerts = []
        db_account_updates = set(affected) | status_changed
        for acc in entered:
            state = self.early.state_of(acc)
            self._event("early_warning", {"account_id": acc, "warning_score": state["risk"],
                                          "signals": list(state["signals"]), "reasons": state["reasons"],
                                          "threshold": self.early.last_context["threshold"]})
            last = self.warning_alerted_at.get(acc, 0)
            if now - last >= WARNING_ALERT_COOLDOWN_SEC:
                self.warning_alerted_at[acc] = now
                alerts.append(self._make_alert(now, acc, tx_id, "early_warning", "medium", state["risk"],
                                               f"{acc} entered early warning: {', '.join(state['reasons'][:3]) or 'multiple signals'}",
                                               {"signals": state["signals"], "threshold": self.early.last_context["threshold"]}))
            self._note_run_warning(acc, now)
        for acc in left:
            self._event("early_warning_cleared", {"account_id": acc, "warning_score": self.early.state_of(acc)["risk"]})

        for acc in newly_flagged:
            info, s = self.accounts[acc], scored[acc]
            try:
                info["explanation"] = self.models.explain(feats[acc])
            except Exception as exc:  # explanation must never block detection
                logger.error("SHAP explanation failed for %s: %s", acc, exc)
                info["explanation"] = {"available": False, "reason": str(exc)}
            severity = "critical" if s["fused"] >= 0.8 else "high"
            reasons = s["rule_reasons"][:3] + self.early.state_of(acc)["reasons"][:2]
            detail = {"fused_score": r4(s["fused"]), "rf_score": r4(s["rf"]), "gnn_score": r4(s["gnn"]),
                      "rule_score": s["rule_score"], "threshold": self.detection_threshold, "role": info["role"],
                      "features": feats[acc], "explanation": info["explanation"]}
            self._event("fraud_detected", {"account_id": acc, "transaction_id": tx_id, **detail})
            alerts.append(self._make_alert(
                now, acc, tx_id, "fraud_detected", severity, s["fused"],
                f"{acc} flagged as money mule ({info['role']}): score {s['fused']:.2f} >= {self.detection_threshold:.2f}"
                + (f" — {', '.join(reasons)}" if reasons else ""),
                detail,
            ))
            logger.info("fraud detected account=%s fused=%.3f rf=%.3f gnn=%s rule=%d role=%s",
                        acc, s["fused"], s["rf"], None if s["gnn"] is None else round(s["gnn"], 3),
                        s["rule_score"], info["role"])
            self._note_run_detection(acc, now)

        for acc in affected:
            if self.accounts[acc]["role"] != prev_roles[acc]:
                self._event("role_updated", {"account_id": acc, "old_role": prev_roles[acc],
                                             "new_role": self.accounts[acc]["role"]})
            if (scored[acc]["drift"] or 0) >= DRIFT_ALERT_SCORE and now - self.warning_alerted_at.get(("drift", acc), 0) >= WARNING_ALERT_COOLDOWN_SEC:
                self.warning_alerted_at[("drift", acc)] = now
                alerts.append(self._make_alert(
                    now, acc, tx_id, "behavioral_drift", "low", scored[acc]["drift"],
                    f"{acc} behavioural drift {scored[acc]['drift']:.1f}σ ({', '.join(scored[acc]['drift_top']) or 'mixed'})",
                    {"drift_score": scored[acc]["drift"], "top_changes": scored[acc]["drift_top"]},
                ))

        if run is not None:
            run["transaction_count"] += 1
            self._event("attack_step", {
                "attack_run_id": run_id, "attack_type": run["attack_type"], "step": spec.get("attack_step"),
                "total_steps": run["planned_steps"], "transaction_id": tx_id, "sender": sender,
                "receiver": receiver, "amount": amount, "channel": channel,
            })

        processing_ms = (time.perf_counter() - started) * 1000
        tx_payload["processing_ms"] = round(processing_ms, 2)
        self._persist_ingest(now, tx_seq, tx_id, sender, receiver, amount, channel, device, s_info, source,
                             is_attack, run_id, spec, processing_ms, scored, db_account_updates, alerts, run)

        # in-memory stats (after the commit succeeded)
        self.next_tx_seq += 1
        self.recent_tx.append(tx_payload)
        self.latencies.append(processing_ms)
        self.tx_times.append(now)
        self.tx_by_source[source] += 1
        self.channel_counts[channel] += 1
        bucket = int(now // settings.VOLUME_BUCKET_SEC)
        counts = self.volume.setdefault(bucket, [0, 0])
        counts[1 if is_attack else 0] += 1
        for alert in alerts:
            self.recent_alerts.append(alert)
            self.alerts_by_type[alert["alert_type"]] += 1
            self._event("alert_created", alert)
        return self._take_events()

    def _persist_ingest(self, now, tx_seq, tx_id, sender, receiver, amount, channel, device, s_info, source,
                        is_attack, run_id, spec, processing_ms, scored, account_ids, alerts, run):
        conn = self.db.connect()
        conn.execute("BEGIN")
        try:
            conn.execute(
                "INSERT INTO transactions (seq, transaction_id, timestamp, ts_epoch, sender, receiver, amount, "
                "channel, device_id, ip_address, source, is_attack, attack_run_id, attack_type, attack_step, "
                "processing_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (tx_seq, tx_id, iso(now), now, sender, receiver, amount, channel, device, s_info.get("ip_address"),
                 source, int(is_attack), run_id, spec.get("attack_type"), spec.get("attack_step"),
                 round(processing_ms, 3)),
            )
            conn.executemany(
                "INSERT INTO detection_results (transaction_id, account_id, timestamp, ts_epoch, rule_score, "
                "rule_flags, rf_score, gnn_score, fused_score, early_risk, threshold, status, detected, mule_role, "
                "drift_score) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (tx_id, acc, iso(now), now, s["rule_score"], dumps(s["rule_flags"]), s["rf"], s["gnn"],
                     s["fused"], self.early.state_of(acc)["risk"], self.detection_threshold, self.status_of(acc),
                     int(self.accounts[acc]["is_flagged"]), self.accounts[acc]["role"], s["drift"])
                    for acc, s in scored.items()
                ],
            )
            self._persist_accounts(conn, account_ids, now)
            for alert in alerts:
                alert["alert_id"] = conn.execute(
                    "INSERT INTO alerts (timestamp, ts_epoch, account_id, transaction_id, alert_type, severity, "
                    "risk_score, message, details, attack_run_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (alert["timestamp"], now, alert["account_id"], alert["transaction_id"], alert["alert_type"],
                     alert["severity"], alert["risk_score"], alert["message"], dumps(alert.pop("full_details")),
                     alert.get("attack_run_id")),
                ).lastrowid
            if run is not None:
                self._persist_run(conn, run)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def _persist_accounts(self, conn, account_ids, now):
        rows = []
        for acc in account_ids:
            info = self.accounts[acc]
            rows.append((
                info["balance"], self.status_of(acc), self.early.state_of(acc)["risk"], info["fused"] or 0.0,
                info["rf"], info["gnn"], info["rule_score"], info["role"], int(info["is_flagged"]), info["flagged_at"],
                dumps(info["explanation"]) if info["explanation"] else None, int(info["is_banned"]),
                iso(info["last_activity"]) if info["last_activity"] else None, info["device_id"], iso(now), acc,
            ))
        conn.executemany(
            "UPDATE accounts SET balance=?, status=?, early_risk=?, current_risk=?, rf_score=?, gnn_score=?, "
            "rule_score=?, mule_role=?, is_flagged=?, flagged_at=?, explanation=?, is_banned=?, last_activity_at=?, "
            "device_id=?, updated_at=? WHERE account_id=?",
            rows,
        )

    def _make_alert(self, now, acc, tx_id, alert_type, severity, score, message, details):
        run_id = next((rid for rid, run in self.active_runs.items() if acc in run["participants"]), None)
        return {
            "alert_id": None,
            "full_details": details,
            "timestamp": iso(now),
            "account_id": acc,
            "transaction_id": tx_id,
            "alert_type": alert_type,
            "severity": severity,
            "risk_score": r4(score),
            "message": message,
            "details": compact_details(details),
            "attack_run_id": run_id,
        }

    @staticmethod
    def _alert_payload(row):
        return {**row, "details": compact_details(loads(row.get("details"), {}))}

    def _tx_payload_from_row(self, row):
        return {
            "transaction_id": row["transaction_id"],
            "seq": row["seq"],
            "timestamp": row["timestamp"],
            "ts": row["ts_epoch"],
            "sender": row["sender"],
            "receiver": row["receiver"],
            "amount": row["amount"],
            "channel": row["channel"],
            "device_id": row["device_id"],
            "source": row["source"],
            "is_attack": bool(row["is_attack"]),
            "attack_run_id": row["attack_run_id"],
            "attack_type": row["attack_type"],
            "attack_step": row["attack_step"],
            "processing_ms": row["processing_ms"],
        }

    # ------------------------------------------------------------ GNN / drift
    def _run_gnn(self, now, force=False):
        if not self.models.gnn_ready:
            return
        if not force and (now - self.last_gnn_at) * 1000 < settings.GNN_MIN_INTERVAL_MS:
            return
        nodes, x, edge_index = self.features.graph_tensors(now)
        if x is None:
            return
        scores = self.models.gnn_predict(x, edge_index)
        self.gnn_scores = dict(zip(nodes, (float(s) for s in scores)))
        self.last_gnn_at = now

    def _refresh_baseline(self, now):
        snapshot = {acc: self.features.features(acc, now) for acc in self.accounts}
        std = {}
        for column in RF_FEATURE_COLUMNS:
            values = [f[column] for f in snapshot.values()]
            std[column] = statistics.pstdev(values) if len(values) > 1 else 0.0
        self.baseline, self.baseline_std, self.baseline_at = snapshot, std, now

    def _drift(self, acc, current):
        base = self.baseline.get(acc)
        if base is None:
            return None, []
        return behavioral_drift_score(base, current, self.baseline_std, RF_FEATURE_COLUMNS)

    # ------------------------------------------------------------- periodic
    def tick(self):
        """Window expiry, risk decay, warning refresh and baseline rotation."""
        now = time.time()
        changed, removed, _ = self.features.expire(now)
        if changed or removed:
            self._event("graph_edges_expired", {"changed": changed, "removed": removed})
        before = {acc: self.early.states[acc]["risk"] for acc in self.early.hot if acc in self.early.states}
        self.early.decay_hot(now)
        entered, left = self.early.select_warnings(self.active_ids())
        decayed = []
        for acc, old in before.items():
            new = self.early.states[acc]["risk"]
            if abs(new - old) >= 0.01 or acc in entered or acc in left:
                decayed.append({"account_id": acc, "early_risk": round(new, 4), "status": self.status_of(acc)})
        for acc in set(entered) | set(left):
            if acc not in before:
                decayed.append({"account_id": acc, "early_risk": self.early.state_of(acc)["risk"],
                                "status": self.status_of(acc)})
        if decayed:
            self._event("accounts_decayed", {"accounts": decayed})
        for acc in entered:
            state = self.early.state_of(acc)
            self._event("early_warning", {"account_id": acc, "warning_score": state["risk"],
                                          "signals": list(state["signals"]), "reasons": state["reasons"],
                                          "threshold": self.early.last_context["threshold"]})
            self._note_run_warning(acc, now)
        for acc in left:
            self._event("early_warning_cleared", {"account_id": acc, "warning_score": self.early.state_of(acc)["risk"]})
        if entered or left:
            conn = self.db.connect()
            conn.execute("BEGIN")
            self._persist_accounts(conn, set(entered) | set(left), now)
            for run in self.active_runs.values():
                self._persist_run(conn, run)
            conn.execute("COMMIT")
        if now - self.baseline_at >= settings.BASELINE_REFRESH_SEC:
            self._refresh_baseline(now)
        cutoff = int(now // settings.VOLUME_BUCKET_SEC) - settings.VOLUME_BUCKETS
        for bucket in [b for b in self.volume if b < cutoff]:
            del self.volume[bucket]
        return self._take_events()

    # --------------------------------------------------------------- accounts
    def _new_account_id(self):
        acc = f"A{self.next_account_num:04d}"
        self.next_account_num += 1
        return acc

    def _create_account(self, info, origin):
        now = time.time()
        acc = info["account_id"]
        created = info.get("created_epoch", now)
        row = {
            "account_id": acc,
            "created_at": iso(created),
            "created_epoch": created,
            "device_id": info["device_id"],
            "ip_address": info.get("ip_address"),
            "home_channel": info.get("home_channel"),
            "balance": float(info.get("balance", 0)),
            "origin": origin,
            "is_banned": 0,
            "is_flagged": 0,
            "flagged_at": None,
            "mule_role": "Unclassified",
            "current_risk": 0.0,
            "rf_score": None,
            "gnn_score": None,
            "rule_score": None,
            "explanation": None,
        }
        self.db.connect().execute(
            "INSERT INTO accounts (account_id, created_at, created_epoch, device_id, ip_address, home_channel, "
            "balance, origin, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (acc, row["created_at"], created, row["device_id"], row["ip_address"], row["home_channel"],
             row["balance"], origin, iso(now)),
        )
        self._load_account(row)
        self.sim.set_accounts(self.accounts, self.active_ids())
        self._event("account_created", self.account_payload(acc))
        logger.info("account created id=%s origin=%s device=%s", acc, origin, row["device_id"])
        return acc

    def onboard_account(self):
        acc = self._new_account_id()
        info = {
            "account_id": acc,
            "device_id": f"DV-{acc}",
            "ip_address": f"192.168.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}",
            "home_channel": self.rng.choice(CHANNELS),
            "balance": self.rng.randint(5000, 50000),
        }
        self._create_account(info, "onboarded")
        return self._take_events()

    def ban(self, account_ids):
        now = time.time()
        banned = []
        for acc in dict.fromkeys(str(a) for a in account_ids):
            info = self.accounts.get(acc)
            if info is None:
                raise ValueError(f"unknown account {acc!r}")
            if info["is_banned"]:
                continue
            info["is_banned"] = True
            info["banned_at"] = iso(now)
            banned.append(acc)
        if not banned:
            return [], []
        conn = self.db.connect()
        conn.execute("BEGIN")
        conn.executemany("UPDATE accounts SET is_banned=1, banned_at=?, status='banned', updated_at=? WHERE account_id=?",
                         [(iso(now), iso(now), acc) for acc in banned])
        conn.execute("COMMIT")
        self.sim.set_accounts(self.accounts, self.active_ids())
        self.early.set_excluded(self.flagged_ids() | {a for a in self.accounts if self.accounts[a]["is_banned"]})
        self.early.select_warnings(self.active_ids())
        for acc in banned:
            self._event("account_updated", self.account_payload(acc))
        self._event("accounts_banned", {"account_ids": banned})
        logger.info("banned %d account(s): %s", len(banned), ", ".join(banned))
        return banned, self._take_events()

    # ----------------------------------------------------------- attack runs
    def create_attack_run(self, attack_type):
        now = time.time()
        active = self.active_ids()
        window_active = set(self.features.active_accounts())
        ctx = AttackContext(
            accounts={a: self.accounts[a] for a in active},
            active_ids=active,
            now=now,
            rng=self.rng,
            idle_ids=set(active) - window_active,
            new_account_id=self._new_account_id,
        )
        plan = build_plan(attack_type, ctx)
        run_id = f"ATK-{self.next_run_num:04d}"
        self.next_run_num += 1
        for info in plan.new_accounts:
            self._create_account(info, "attack")
        if plan.device_updates:
            touched_devices = set()
            for acc, device in plan.device_updates.items():
                touched_devices.update(self.features.set_device(acc, device))
                self.accounts[acc]["device_id"] = device
            self._sync_device_sizes(touched_devices)
            conn = self.db.connect()
            conn.execute("BEGIN")
            self._persist_accounts(conn, plan.device_updates.keys(), now)
            conn.execute("COMMIT")
            for acc in plan.device_updates:
                self._event("account_updated", {**self.account_payload(acc), "device_changed": True})
        labeled = list(plan.labeled)
        run = {
            "attack_run_id": run_id,
            "attack_type": plan.attack_type,
            "attack_name": plan.attack_name,
            "status": "running",
            "started_at": iso(now),
            "started_epoch": now,
            "planned_steps": len(plan.steps),
            "transaction_count": 0,
            "participants": plan.participants,
            "labeled": set(labeled),
            "labeled_list": labeled,
            "warned_before_start": [a for a in labeled if a in self.early.warning_ids],
            "already_flagged": [a for a in labeled if self.accounts[a]["is_flagged"]],
            "first_warning_at": None,
            "first_warning_epoch": None,
            "detected_at": None,
            "detected_epoch": None,
            "max_risk": 0.0,
        }
        conn = self.db.connect()
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO attack_runs (attack_run_id, attack_type, attack_name, status, started_at, started_epoch, "
            "planned_steps, participants, labeled_participants, warned_before_start, threshold_before) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, plan.attack_type, plan.attack_name, "running", run["started_at"], now, len(plan.steps),
             dumps(plan.participants), dumps(labeled), dumps(run["warned_before_start"]), self.detection_threshold),
        )
        conn.executemany("UPDATE accounts SET is_attack_participant=1 WHERE account_id=?", [(a,) for a in labeled])
        conn.execute("COMMIT")
        self.active_runs[run_id] = run
        stats = self.attack_type_stats.setdefault(plan.attack_type, {"runs": 0, "detected": 0, "evaluated": 0})
        stats["runs"] += 1
        self.run_totals.setdefault("all", {"runs": 0, "completed": 0, "detected": 0, "tp": 0, "fp": 0, "fn": 0,
                                           "latencies": [], "warning_latencies": []})["runs"] += 1
        self._event("attack_started", self.run_payload(run))
        logger.info("attack started run=%s type=%s steps=%d participants=%s",
                    run_id, plan.attack_type, len(plan.steps), ",".join(plan.participants))
        steps = [
            {"sender": s.sender, "receiver": s.receiver, "amount": float(s.amount), "channel": s.channel,
             "device_id": s.device_id, "source": "attack", "attack_run_id": run_id,
             "attack_type": plan.attack_type, "attack_step": i + 1}
            for i, s in enumerate(plan.steps)
        ]
        return self.run_payload(run), steps, self._take_events()

    def run_payload(self, run):
        return {
            "attack_run_id": run["attack_run_id"],
            "attack_type": run["attack_type"],
            "attack_name": run["attack_name"],
            "status": run["status"],
            "started_at": run["started_at"],
            "planned_steps": run["planned_steps"],
            "transaction_count": run["transaction_count"],
            "participants": run["participants"],
            "labeled_participants": run["labeled_list"],
            "first_warning_at": run["first_warning_at"],
            "first_warning_latency_ms": self._latency(run, "first_warning_epoch"),
            "detected_at": run["detected_at"],
            "detection_latency_ms": self._latency(run, "detected_epoch"),
            "max_risk": r4(run["max_risk"]),
        }

    @staticmethod
    def _latency(run, key):
        if run.get(key) is None:
            return None
        return int((run[key] - run["started_epoch"]) * 1000)

    def _clamp_threshold(self, value):
        return round(min(max(value, self.threshold_floor), settings.THRESHOLD_MAX), 3)

    def _note_run_warning(self, acc, now):
        for run in self.active_runs.values():
            if acc in run["labeled"] and run["first_warning_epoch"] is None:
                run["first_warning_epoch"] = now
                run["first_warning_at"] = iso(now)
                self._event("attack_updated", self.run_payload(run))

    def _note_run_detection(self, acc, now):
        for run in self.active_runs.values():
            if acc in run["labeled"] and run["detected_epoch"] is None:
                run["detected_epoch"] = now
                run["detected_at"] = iso(now)
                if run["first_warning_epoch"] is None:
                    run["first_warning_epoch"] = now
                    run["first_warning_at"] = iso(now)
                self._event("attack_detected", {**self.run_payload(run), "detected_by": acc})
                logger.info("attack detected run=%s via=%s latency_ms=%d", run["attack_run_id"], acc,
                            self._latency(run, "detected_epoch"))

    def _persist_run(self, conn, run, extra=None):
        fields = {
            "status": run["status"],
            "transaction_count": run["transaction_count"],
            "first_warning_at": run["first_warning_at"],
            "first_warning_latency_ms": self._latency(run, "first_warning_epoch"),
            "detected_at": run["detected_at"],
            "detection_latency_ms": self._latency(run, "detected_epoch"),
            "max_risk": run["max_risk"],
            **(extra or {}),
        }
        assignments = ", ".join(f"{k}=?" for k in fields)
        conn.execute(f"UPDATE attack_runs SET {assignments} WHERE attack_run_id=?",
                     (*fields.values(), run["attack_run_id"]))

    def finish_attack_steps(self, run_id):
        run = self.active_runs.get(run_id)
        if run is None:
            return []
        now = time.time()
        run["status"] = "evaluating"
        conn = self.db.connect()
        conn.execute("BEGIN")
        self._persist_run(conn, run, {"steps_finished_at": iso(now)})
        conn.execute("COMMIT")
        self._event("attack_updated", self.run_payload(run))
        return self._take_events()

    def fail_attack(self, run_id, error):
        run = self.active_runs.pop(run_id, None)
        if run is None:
            return []
        run["status"] = "failed"
        conn = self.db.connect()
        conn.execute("BEGIN")
        self._persist_run(conn, run, {"finished_at": iso(time.time()), "error": str(error)[:500]})
        conn.execute("COMMIT")
        self._event("attack_completed", {**self.run_payload(run), "error": str(error)})
        return self._take_events()

    def evaluate_attack(self, run_id):
        run = self.active_runs.pop(run_id, None)
        if run is None:
            return []
        now = time.time()
        labeled = run["labeled"]
        detected = sorted(a for a in labeled if self.accounts[a]["is_flagged"])
        missed = sorted(labeled - set(detected))
        # Operational precision: every account flagged since the previous
        # evaluation that is not an injected mule of a run active in that
        # period is a false positive. (Counting only the few seconds of this
        # run would make precision ~1.0 by construction.)
        since_iso = self.last_eval_iso or iso(self.started_at)
        since_epoch = datetime.fromisoformat(since_iso.replace("Z", "+00:00")).timestamp()
        concurrent = set()
        for other in self.active_runs.values():
            concurrent |= other["labeled"]
        recent_mules = {
            acc for row in self.db.query(
                "SELECT labeled_participants FROM attack_runs WHERE started_epoch >= ?",
                (since_epoch - settings.FEATURE_WINDOW_SEC,),
            ) for acc in loads(row["labeled_participants"], [])
        }
        false_positives = sorted(
            acc for acc, info in self.accounts.items()
            if info["is_flagged"] and info["flagged_at"] and info["flagged_at"] > since_iso
            and acc not in labeled and acc not in concurrent and acc not in recent_mules
        )
        precision = len(detected) / (len(detected) + len(false_positives)) if (detected or false_positives) else None
        recall = len(detected) / len(labeled) if labeled else None
        threshold_before = self.detection_threshold
        if precision is not None and recall is not None:
            self.detection_threshold = self._clamp_threshold(adaptive_threshold_update(
                threshold_before, {"1": {"precision": precision, "recall": recall}}
            ))
        self.last_eval_iso = iso(now)
        signature = signature_from_window(self.features, detected, now) if detected else None
        similarity, match_index = best_match(signature, self.pattern_memory) if signature else (0.0, None)
        matched_run = self.pattern_run_ids[match_index] if match_index is not None else None
        role_counts = dict(Counter(self.accounts[a]["role"] for a in detected))
        run["status"] = "completed"
        extra = {
            "finished_at": iso(now),
            "detected_accounts": dumps(detected),
            "missed_accounts": dumps(missed),
            "false_positive_accounts": dumps(false_positives),
            "precision": precision,
            "recall": recall,
            "threshold_before": threshold_before,
            "threshold_after": self.detection_threshold,
            "pattern_similarity": similarity if signature else None,
            "pattern_match_run_id": matched_run,
            "role_counts": dumps(role_counts),
        }
        conn = self.db.connect()
        conn.execute("BEGIN")
        self._persist_run(conn, run, extra)
        if signature:
            conn.execute(
                "INSERT INTO pattern_memory (attack_run_id, created_at, attack_type, signature, similarity, matched_run_id) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, iso(now), run["attack_type"], dumps(signature), similarity, matched_run),
            )
        conn.execute("COMMIT")
        if signature:
            self.pattern_memory.append(signature)
            self.pattern_run_ids.append(run_id)
            self.last_pattern = {"attack_run_id": run_id, "attack_type": run["attack_type"], "signature": signature,
                                 "similarity": round(similarity, 4), "matched_run_id": matched_run}
        totals = self.run_totals["all"]
        totals["completed"] += 1
        totals["tp"] += len(detected)
        totals["fp"] += len(false_positives)
        totals["fn"] += len(missed)
        stats = self.attack_type_stats[run["attack_type"]]
        stats["evaluated"] += 1
        if run["detected_epoch"] is not None:
            totals["detected"] += 1
            totals["latencies"].append(self._latency(run, "detected_epoch"))
            stats["detected"] += 1
        if run["first_warning_epoch"] is not None:
            totals["warning_latencies"].append(self._latency(run, "first_warning_epoch"))
        outcome = {
            **self.run_payload(run),
            "finished_at": iso(now),
            "detected_accounts": detected,
            "missed_accounts": missed,
            "false_positive_accounts": false_positives,
            "warned_before_start": run["warned_before_start"],
            "precision": r4(precision),
            "recall": r4(recall),
            "threshold_before": threshold_before,
            "threshold_after": self.detection_threshold,
            "pattern_similarity": r4(similarity) if signature else None,
            "pattern_match_run_id": matched_run,
            "role_counts": role_counts,
        }
        self._event("attack_completed", outcome)
        if self.detection_threshold != threshold_before:
            self._event("threshold_updated", {"old": threshold_before, "new": self.detection_threshold,
                                              "precision": r4(precision), "recall": r4(recall),
                                              "attack_run_id": run_id})
        if signature:
            self._event("pattern_memory_updated", {**self.last_pattern, "stored": len(self.pattern_memory)})
        logger.info("attack completed run=%s detected=%d/%d fp=%d precision=%s recall=%s threshold %.3f->%.3f",
                    run_id, len(detected), len(labeled), len(false_positives), r4(precision), r4(recall),
                    threshold_before, self.detection_threshold)
        return self._take_events()

    # -------------------------------------------------------------- reading
    def metrics(self):
        now = time.time()
        while self.tx_times and now - self.tx_times[0] > 5.0:
            self.tx_times.popleft()
        active = self.active_ids()
        flagged = [a for a in active if self.accounts[a]["is_flagged"]]
        totals = self.run_totals.get("all", {"runs": 0, "completed": 0, "detected": 0, "tp": 0, "fp": 0, "fn": 0,
                                             "latencies": [], "warning_latencies": []})
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
        bins = [0] * 10
        fused_bins = [0] * 10
        rule_high = rule_medium = rule_scored = drift_alerts = 0
        for acc in active:
            info = self.accounts[acc]
            bins[min(int(self.early.states[acc]["risk"] * 10), 9)] += 1
            fused_bins[min(int((info["fused"] or 0.0) * 10), 9)] += 1
            if info["rule_score"] is not None:
                rule_scored += 1
                rule_high += info["rule_score"] >= 7
                rule_medium += 4 <= info["rule_score"] < 7
            drift_alerts += (info["drift"] or 0.0) >= DRIFT_ALERT_SCORE
        bucket_now = int(now // settings.VOLUME_BUCKET_SEC)
        first = bucket_now - settings.VOLUME_BUCKETS + 1
        series = [self.volume.get(b, [0, 0]) for b in range(first, bucket_now + 1)]
        latencies = sorted(self.latencies)
        return {
            "server_time": iso(now),
            "seq": self.seq,
            "tx_total": self.tx_by_source.total(),
            "tx_by_source": dict(self.tx_by_source),
            "tps": round(len(self.tx_times) / 5.0, 2),
            "accounts_total": len(self.accounts),
            "accounts_active": len(active),
            "accounts_in_window": len(self.features.tx_count),
            "early_warning_count": len(self.early.warning_ids),
            "flagged_count": len(flagged),
            "banned_count": len(self.accounts) - len(active),
            "alerts_total": sum(self.alerts_by_type.values()),
            "alerts_by_type": dict(self.alerts_by_type),
            "attacks_total": totals["runs"],
            "attacks_running": len(self.active_runs),
            "attacks_completed": totals["completed"],
            "attacks_detected": totals["detected"],
            "detection_rate": round(totals["detected"] / totals["completed"], 4) if totals["completed"] else None,
            "avg_detection_latency_ms": round(statistics.mean(totals["latencies"])) if totals["latencies"] else None,
            "avg_first_warning_latency_ms": round(statistics.mean(totals["warning_latencies"])) if totals["warning_latencies"] else None,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": r4(precision),
            "recall": r4(recall),
            "f1": r4(f1),
            "detection_threshold": self.detection_threshold,
            "calibrated_threshold": self.calibrated_threshold,
            "threshold_band": [self.threshold_floor, settings.THRESHOLD_MAX],
            "min_early_risk": self.min_early_risk,
            "warning_threshold": self.early.last_context["threshold"],
            "warning_pct": self.early.last_context["warning_pct"],
            "risk_distribution": {"edges": [i / 10 for i in range(11)], "early": bins, "fused": fused_bins},
            "volume_series": {"bucket_sec": settings.VOLUME_BUCKET_SEC, "start": first * settings.VOLUME_BUCKET_SEC,
                              "normal": [s[0] for s in series], "attack": [s[1] for s in series]},
            "channel_counts": dict(self.channel_counts),
            "role_counts": dict(Counter(self.accounts[a]["role"] for a in flagged)),
            "rule_high_count": rule_high,
            "rule_medium_count": rule_medium,
            "rule_scored_count": rule_scored,
            "drift_alert_count": drift_alerts,
            "attack_types": self.attack_type_stats,
            "pipeline_ms_p50": round(latencies[len(latencies) // 2], 2) if latencies else None,
            "pipeline_ms_p95": round(latencies[int(len(latencies) * 0.95) - 1], 2) if len(latencies) >= 20 else None,
            "rf_ms": round(self.models.rf_latency_ms, 2),
            "gnn_ms": round(self.models.gnn_latency_ms, 2),
            "spawner_running": self.spawner_running,
            "edges_in_window": len(self.features.pairs),
        }

    def drift_table(self, limit=15):
        rows = [
            {"account_id": a, "drift_score": r4(i["drift"]), "top_changes": i["drift_top"], "status": self.status_of(a)}
            for a, i in self.accounts.items() if i["drift"] is not None and i["drift"] >= 1.0
        ]
        rows.sort(key=lambda r: r["drift_score"], reverse=True)
        return rows[:limit]

    def history(self):
        runs = self.db.query(
            "SELECT attack_run_id, attack_type, attack_name, status, started_at, detection_latency_ms, "
            "first_warning_latency_ms, precision, recall, threshold_before, threshold_after, role_counts, "
            "pattern_similarity FROM attack_runs WHERE status='completed' ORDER BY started_epoch DESC LIMIT 40"
        )[::-1]
        for run in runs:
            run["role_counts"] = loads(run["role_counts"], {})
        role_totals = Counter()
        for run in runs:
            role_totals.update(run["role_counts"])
        return {"runs": runs, "role_totals": dict(role_totals)}

    def attack_runs(self, limit=25):
        rows = self.db.query("SELECT * FROM attack_runs ORDER BY started_epoch DESC LIMIT ?", (limit,))
        out = []
        for row in rows:
            live = self.active_runs.get(row["attack_run_id"])
            out.append(self.run_payload(live) if live else serialize_run(row))
        return out

    def snapshot(self):
        accounts = [self.account_payload(acc) for acc in self.accounts]
        return {
            "seq": self.seq,
            "server_time": iso(time.time()),
            "config": {
                "feature_window_sec": settings.FEATURE_WINDOW_SEC,
                "normal_tx_interval_ms": [settings.NORMAL_TX_MIN_INTERVAL_MS, settings.NORMAL_TX_MAX_INTERVAL_MS],
                "attack_step_interval_ms": settings.ATTACK_STEP_INTERVAL_MS,
                "demo_mode": settings.DEMO_MODE,
                "max_rule_score": MAX_RULE_SCORE,
            },
            "accounts": accounts,
            "graph": {"edges": self.features.edges()},
            "recent_transactions": list(self.recent_tx)[-150:],
            "alerts": list(self.recent_alerts)[-60:],
            "attack_runs": self.attack_runs(),
            "attack_types": attack_types(),
            "metrics": self.metrics(),
            "drift": self.drift_table(),
            "pattern_memory": {"stored": len(self.pattern_memory), "last": self.last_pattern},
            "history": self.history(),
            "models": self.models.health(),
        }

    def account_detail(self, acc):
        if acc not in self.accounts:
            return None
        now = time.time()
        info = self.accounts[acc]
        feats = self.features.features(acc, now)
        rule_score, rule_flags, rule_reasons = evaluate_rules(feats)
        try:
            live_explanation = self.models.explain(feats) if self.features.tx_count.get(acc) else None
        except Exception as exc:
            live_explanation = {"available": False, "reason": str(exc)}
        neighbors = []
        for cp, (count, amount) in self.features.out_edges.get(acc, {}).items():
            neighbors.append({"account_id": cp, "direction": "out", "count": count, "amount": round(amount, 2),
                              "status": self.status_of(cp)})
        for cp, (count, amount) in self.features.in_edges.get(acc, {}).items():
            neighbors.append({"account_id": cp, "direction": "in", "count": count, "amount": round(amount, 2),
                              "status": self.status_of(cp)})
        device_members = sorted(self.features.device_cluster(info["device_id"]))
        history = self.db.query(
            "SELECT transaction_id, timestamp, ts_epoch, early_risk, fused_score, rf_score, gnn_score, rule_score, "
            "status FROM detection_results WHERE account_id=? ORDER BY id DESC LIMIT 200", (acc,)
        )[::-1]
        txs = self.db.query(
            "SELECT * FROM transactions WHERE sender=? OR receiver=? ORDER BY seq DESC LIMIT 50", (acc, acc)
        )
        counterparties = self.db.query(
            "SELECT cp, SUM(n) AS n, SUM(amount) AS amount FROM ("
            " SELECT receiver AS cp, COUNT(*) AS n, SUM(amount) AS amount FROM transactions WHERE sender=? GROUP BY receiver"
            " UNION ALL"
            " SELECT sender AS cp, COUNT(*) AS n, SUM(amount) AS amount FROM transactions WHERE receiver=? GROUP BY sender"
            ") GROUP BY cp ORDER BY n DESC, amount DESC LIMIT 10", (acc, acc)
        )
        alerts = self.db.query("SELECT * FROM alerts WHERE account_id=? ORDER BY alert_id DESC LIMIT 25", (acc,))
        runs = self.db.query(
            "SELECT attack_run_id, attack_type, attack_name, status, started_at, labeled_participants "
            "FROM attack_runs WHERE participants LIKE ? ORDER BY started_epoch DESC", (f'%"{acc}"%',)
        )
        totals = self.db.query_one(
            "SELECT COUNT(*) AS n, COALESCE(SUM(CASE WHEN sender=? THEN amount END),0) AS sent, "
            "COALESCE(SUM(CASE WHEN receiver=? THEN amount END),0) AS received "
            "FROM transactions WHERE sender=? OR receiver=?", (acc, acc, acc, acc)
        )
        return {
            "account": {**self.account_payload(acc), "balance": round(info["balance"], 2),
                        "ip_address": info["ip_address"], "device_members": device_members},
            "features": feats,
            "rules": {"score": rule_score, "max_score": MAX_RULE_SCORE, "flags": rule_flags, "reasons": rule_reasons},
            "scores": {"rf": r4(info["rf"]), "gnn": r4(info["gnn"]), "fused": r4(info["fused"]),
                       "threshold": self.detection_threshold, "early_risk": self.early.state_of(acc)["risk"],
                       "warning_threshold": self.early.last_context["threshold"]},
            "early_warning": self.early.state_of(acc),
            "drift": {"score": r4(info["drift"]), "top_changes": info["drift_top"],
                      "baseline": self.baseline.get(acc), "baseline_age_sec": round(now - self.baseline_at, 1)},
            "explanation_at_detection": info["explanation"],
            "explanation_now": live_explanation,
            "risk_history": history,
            "transactions": [self._tx_payload_from_row(t) for t in txs],
            "totals": totals,
            "window_neighbors": neighbors,
            "top_counterparties": counterparties,
            "alerts": [self._alert_payload(a) for a in alerts],
            "simulation_ground_truth": {
                "note": "From the attack simulator, not the detector.",
                "attack_runs": [
                    {**{k: v for k, v in r.items() if k != "labeled_participants"},
                     "labeled_mule": acc in loads(r["labeled_participants"], [])}
                    for r in runs
                ],
            },
        }

    # -------------------------------------------------------------- control
    def set_spawner(self, running):
        self.spawner_running = bool(running)
        self._event("system_status", {"spawner_running": self.spawner_running})
        logger.info("normal traffic spawner %s", "resumed" if running else "paused")
        return self._take_events()

    def reset(self):
        if self.active_runs:
            raise ValueError("cannot reset while an attack is running")
        conn = self.db.connect()
        conn.execute("BEGIN")
        for table in RESET_ORDER:
            conn.execute(f"DELETE FROM {table}")
        conn.execute("COMMIT")
        spawner = self.spawner_running
        seq = self.seq
        self._reset_memory()
        self.seq = seq
        self.spawner_running = spawner
        self.bootstrap()
        self._event("system_reset", {"reason": "operator reset"})
        logger.warning("database reset and accounts re-seeded")
        return self._take_events()


def compact_details(details):
    """Alert details without the bulky SHAP/feature blocks (kept in the DB)."""
    return {k: v for k, v in (details or {}).items() if k not in ("features", "explanation")}


def serialize_run(row):
    out = dict(row)
    for key in ("participants", "labeled_participants", "warned_before_start", "detected_accounts",
                "missed_accounts", "false_positive_accounts"):
        out[key] = loads(row.get(key), [])
    out["role_counts"] = loads(row.get("role_counts"), {})
    return out
