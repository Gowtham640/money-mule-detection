"""Async orchestration: background tasks, the attack runner and WebSockets.

``Runtime.call`` executes a pipeline method on the single pipeline thread and
returns its result, so the event loop stays free for HTTP and WebSocket I/O
while all state mutation stays serialised.
"""

import asyncio
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import WebSocket, WebSocketDisconnect

from backend import settings
from backend.db import Database
from backend.model import ModelRuntime
from backend.pipeline import LivePipeline, iso

logger = logging.getLogger("mule.runtime")


def encode(message):
    return json.dumps(message, separators=(",", ":"), default=str, allow_nan=False)


class _Client:
    def __init__(self, websocket):
        self.websocket = websocket
        self.queue = asyncio.Queue(maxsize=settings.WS_CLIENT_QUEUE)
        self.sender = None
        self.peer = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "?"


class ConnectionManager:
    def __init__(self):
        self.clients = set()
        self.messages_sent = 0

    @property
    def count(self):
        return len(self.clients)

    async def serve(self, websocket: WebSocket, hello):
        await websocket.accept()
        client = _Client(websocket)
        self.clients.add(client)
        client.sender = asyncio.create_task(self._sender(client))
        logger.info("websocket connected peer=%s clients=%d", client.peer, self.count)
        self._enqueue(client, encode(hello))
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = json.loads(raw)
                except ValueError:
                    logger.debug("ignoring malformed websocket frame from %s", client.peer)
                    continue
                if isinstance(message, dict) and message.get("type") == "ping":
                    self._enqueue(client, encode({"type": "pong", "server_time": iso(time.time())}))
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.warning("websocket receive error peer=%s: %s", client.peer, exc)
        finally:
            await self._drop(client)

    async def _sender(self, client):
        try:
            while True:
                text = await client.queue.get()
                await client.websocket.send_text(text)
                self.messages_sent += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info("websocket send failed peer=%s: %s", client.peer, exc)
            await self._drop(client)

    def _enqueue(self, client, text):
        try:
            client.queue.put_nowait(text)
        except asyncio.QueueFull:
            # A client that cannot keep up is disconnected; it reconnects and
            # resynchronises from a fresh snapshot instead of lagging forever.
            logger.warning("websocket client %s too slow, disconnecting", client.peer)
            asyncio.create_task(self._drop(client, close=True))

    async def _drop(self, client, close=False):
        if client not in self.clients:
            return
        self.clients.discard(client)
        if client.sender and client.sender is not asyncio.current_task():
            client.sender.cancel()
        if close:
            try:
                await client.websocket.close(code=1013)
            except Exception:
                pass
        logger.info("websocket disconnected peer=%s clients=%d", client.peer, self.count)

    def broadcast(self, message):
        if not self.clients:
            return
        text = encode(message)
        for client in list(self.clients):
            self._enqueue(client, text)

    def broadcast_events(self, events):
        if not events:
            return
        if len(events) == 1:
            self.broadcast(events[0])
        else:
            self.broadcast({"type": "batch", "events": events})


class Runtime:
    def __init__(self):
        self.db = Database(settings.DB_PATH)
        self.models = ModelRuntime()
        self.pipeline = LivePipeline(self.db, self.models)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self.ws = ConnectionManager()
        self.tasks = []
        self.attack_tasks = {}
        self.rng = random.Random()
        self.started_at = None
        self.ready = False
        self.errors = 0
        self.last_error = None

    async def call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    def publish(self, events):
        self.ws.broadcast_events(events)

    # ---------------------------------------------------------------- lifecycle
    async def start(self):
        self.started_at = time.time()
        await self.call(self.db.init_schema)
        await self.call(self.models.load)
        await self.call(self.pipeline.bootstrap)
        self.tasks = [
            asyncio.create_task(self._loop("normal-spawner", self._spawn_normal_once, self._spawn_delay)),
            asyncio.create_task(self._loop("tick", self._tick_once, lambda: settings.DECAY_TICK_SEC)),
            asyncio.create_task(self._loop("metrics", self._metrics_once, lambda: settings.METRICS_INTERVAL_SEC)),
            asyncio.create_task(self._loop("heartbeat", self._heartbeat_once, lambda: settings.HEARTBEAT_SEC)),
            asyncio.create_task(self._loop("onboarding", self._onboard_once, lambda: settings.NEW_ACCOUNT_INTERVAL_SEC)),
        ]
        self.ready = True
        logger.info(
            "runtime started spawner=%s interval=%d-%dms attack_step=%dms window=%.0fs",
            "running" if self.pipeline.spawner_running else "paused",
            settings.NORMAL_TX_MIN_INTERVAL_MS, settings.NORMAL_TX_MAX_INTERVAL_MS,
            settings.ATTACK_STEP_INTERVAL_MS, settings.FEATURE_WINDOW_SEC,
        )

    async def stop(self):
        self.ready = False
        for task in [*self.tasks, *self.attack_tasks.values()]:
            task.cancel()
        await asyncio.gather(*self.tasks, *self.attack_tasks.values(), return_exceptions=True)
        self.executor.shutdown(wait=True)
        logger.info("runtime stopped")

    async def _loop(self, name, body, delay):
        logger.info("background task started: %s", name)
        while True:
            await asyncio.sleep(delay())
            try:
                await body()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.errors += 1
                self.last_error = f"{name}: {exc}"
                logger.exception("background task %s failed: %s", name, exc)

    def _spawn_delay(self):
        return self.rng.uniform(settings.NORMAL_TX_MIN_INTERVAL_MS, settings.NORMAL_TX_MAX_INTERVAL_MS) / 1000.0

    async def _spawn_normal_once(self):
        self.publish(await self.call(self.pipeline.spawn_normal))

    async def _tick_once(self):
        self.publish(await self.call(self.pipeline.tick))

    async def _metrics_once(self):
        if not self.ws.count:
            return
        metrics = await self.call(self.pipeline.metrics)
        metrics["connected_clients"] = self.ws.count
        self.ws.broadcast({"type": "metrics_updated", "seq": None, "timestamp": metrics["server_time"], "data": metrics})

    async def _heartbeat_once(self):
        self.ws.broadcast({"type": "heartbeat", "server_time": iso(time.time()), "seq": self.pipeline.seq})

    async def _onboard_once(self):
        if self.pipeline.spawner_running:
            self.publish(await self.call(self.pipeline.onboard_account))

    # ------------------------------------------------------------------ attacks
    async def spawn_attack(self, attack_type):
        run, steps, events = await self.call(self.pipeline.create_attack_run, attack_type)
        self.publish(events)
        run_id = run["attack_run_id"]
        task = asyncio.create_task(self._run_attack(run_id, steps))
        self.attack_tasks[run_id] = task
        task.add_done_callback(lambda _t: self.attack_tasks.pop(run_id, None))
        return run

    async def _run_attack(self, run_id, steps):
        skipped = 0
        try:
            for index, step in enumerate(steps):
                if index:
                    await asyncio.sleep(settings.ATTACK_STEP_INTERVAL_MS / 1000.0)
                try:
                    self.publish(await self.call(self.pipeline.ingest, step))
                except ValueError as exc:  # e.g. a participant was banned mid-attack
                    skipped += 1
                    logger.warning("attack %s step %d skipped: %s", run_id, index + 1, exc)
            self.publish(await self.call(self.pipeline.finish_attack_steps, run_id))
            await asyncio.sleep(settings.ATTACK_EVAL_GRACE_SEC)
            self.publish(await self.call(self.pipeline.evaluate_attack, run_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("attack %s failed: %s", run_id, exc)
            self.publish(await self.call(self.pipeline.fail_attack, run_id, exc))

    # ------------------------------------------------------------------ health
    def health(self):
        counts = self.db.table_counts()
        spawner = "running" if self.pipeline.spawner_running else "paused"
        if not self.tasks or self.tasks[0].done():
            spawner = "stopped"
        return {
            "backend": "ok" if self.ready else "starting",
            "database": self.db.ping(),
            "database_path": str(self.db.path),
            "journal_mode": self.db.scalar("PRAGMA journal_mode"),
            "spawner": spawner,
            "websocket": "ready" if self.ready else "starting",
            "connected_clients": self.ws.count,
            "ws_messages_sent": self.ws.messages_sent,
            **self.models.health(),
            "transaction_count": counts["transactions"],
            "table_counts": counts,
            "active_attacks": list(self.attack_tasks),
            "background_errors": self.errors,
            "last_error": self.last_error,
            "uptime_sec": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "event_seq": self.pipeline.seq,
        }
