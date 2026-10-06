"""Early-warning risk engine (refactored from the former ``RealTimeEngine``).

Every transaction updates behavioural signals for the sender and receiver;
signals stack into a per-account warning score that decays over time. The
warning set is the top-ranked accounts above an adaptive percentile threshold.

Changes from the original engine:
  * no ground truth: the ``attack`` signal (read from ``is_attack``) and the
    ``is_fraud`` override that forced risk to 0.97 are gone, so warnings come
    only from observable behaviour;
  * no random jitter on scores;
  * timestamps are epoch seconds supplied by the caller (live or simulated);
  * transaction generation and DataFrame state moved out (simulator/pipeline).
"""

import math
from collections import Counter, deque


class EarlyWarningEngine:
    STRONG_RISK = 0.92
    DECAY = 0.92
    SIGNAL_DECAY = 0.88
    DECAY_PERIOD = 12.0
    WARNING_PERCENTILE = 0.90
    MIN_WARNING_PCT = 0.01
    MAX_WARNING_PCT = 0.06
    MIN_WARNING_THRESHOLD = 0.40
    MAX_WARNING_THRESHOLD = 0.68
    WARNING_ENTER_SIGNALS = 2
    WARNING_RETAIN_SIGNALS = 1
    WARNING_RETAIN_FACTOR = 0.78
    WARNING_RETAIN_FLOOR = 0.18
    WARNING_TX_PER_STEP = 20.0
    WARNING_MAX_CREDIT = 8.0
    BURST_SEC = 45.0
    SHORT_GAP_SEC = 8.0
    DORMANT_SEC = 120.0
    HISTORY_SEC = 600.0
    BASELINE_AMOUNT = 7500.0
    SIGNAL_LABELS = {
        "new": "New account with unusually fast activity",
        "velocity": "High transaction velocity",
        "high_value": "High-value transaction",
        "trend": "Increasing transaction trend",
        "short_gap": "Short interval transactions",
        "repeat": "Repeated amounts",
        "target_bias": "Single target bias",
        "fan_in": "Early fan-in pattern",
        "fan_out": "Early fan-out pattern",
        "dormant": "Dormant account activation",
        "neighbor": "Connected to a suspicious neighbor",
        "device": "Shared device cluster",
    }

    def __init__(self):
        self.states = {}
        self.created_at = {}
        self.device_size = {}  # account -> accounts on same device (kept by caller)
        self.excluded = set()  # flagged / banned accounts never enter the warning ranking
        self.hot = set()
        self.amount_ema = self.BASELINE_AMOUNT
        self.threshold_ema = 0.0
        self.warning_ids = set()
        self.adjust_credit = 10.0
        self.last_context = {"threshold": 0.0, "warning_pct": self.MIN_WARNING_PCT, "max_count": 0}

    # -- accounts ------------------------------------------------------------
    def register_account(self, account_id, created_epoch, now):
        self.created_at[account_id] = float(created_epoch)
        if account_id not in self.states:
            self.states[account_id] = {
                "risk": 0.0,
                "signals": {key: 0.0 for key in self.SIGNAL_LABELS},
                "recent": deque(maxlen=48),
                "last": None,
                "decay_at": now,
            }

    def set_excluded(self, account_ids):
        self.excluded = set(account_ids)
        self.warning_ids -= self.excluded

    # -- scoring ---------------------------------------------------------------
    @staticmethod
    def _clamp(value):
        return max(0.0, min(1.0, float(value)))

    def _recent(self, state, now, seconds, direction=None):
        recent = state["recent"]
        while recent and now - recent[0]["ts"] > self.HISTORY_SEC:
            recent.popleft()
        return [
            item for item in recent
            if (direction is None or item["dir"] == direction) and now - item["ts"] <= seconds
        ]

    @staticmethod
    def _bucket(amount):
        amount = float(amount)
        if amount >= 10000:
            return round(amount / 5000.0) * 5000
        if amount >= 1000:
            return round(amount / 500.0) * 500
        return round(amount / 100.0) * 100

    def _signals(self, aid, cp, now, amount, direction):
        state = self.states[aid]
        recent_short = self._recent(state, now, self.BURST_SEC)
        recent_out = self._recent(state, now, 180.0, "out")
        age_sec = max(now - self.created_at.get(aid, now), 0.0)
        gap = max(now - state["last"], 0.0) if state["last"] is not None else None
        short_avg = (sum(i["amount"] for i in recent_short) / len(recent_short)) if recent_short else self.BASELINE_AMOUNT
        baseline = max(self.BASELINE_AMOUNT, self.amount_ema, short_avg)
        bucket = self._bucket(amount)
        repeated = sum(1 for item in state["recent"] if self._bucket(item["amount"]) == bucket)
        trend_window = [item["amount"] for item in list(state["recent"])[-6:]] + [amount]
        device_size = int(self.device_size.get(aid, 1))
        signals = {}

        if age_sec <= 7 * 24 * 3600 and (len(recent_short) >= 1 or amount >= baseline * 1.5):
            newness = 1.0 - min(age_sec / (7 * 24 * 3600), 1.0)
            signals["new"] = min(0.26, 0.12 + 0.10 * newness + 0.02 * len(recent_short))
        if gap is not None and gap >= self.DORMANT_SEC and amount >= baseline * 1.25:
            signals["dormant"] = min(0.22, 0.10 + min(gap / 600.0, 1.0) * 0.12)
        if len(recent_short) + 1 >= 4:
            signals["velocity"] = min(0.24, 0.08 + 0.03 * max((len(recent_short) + 1) - 3, 0))
        if gap is not None and gap <= self.SHORT_GAP_SEC:
            signals["short_gap"] = min(0.18, 0.07 + ((self.SHORT_GAP_SEC - gap) / self.SHORT_GAP_SEC) * 0.11)
        if amount >= baseline * 2.1:
            signals["high_value"] = min(0.28, 0.10 + min((amount / max(baseline, 1.0)) - 2.1, 3.0) * 0.06)
        if len(trend_window) >= 6:
            prev_avg = sum(trend_window[:-3]) / max(len(trend_window[:-3]), 1)
            last_avg = sum(trend_window[-3:]) / 3.0
            if prev_avg > 0 and last_avg >= prev_avg * 1.45:
                signals["trend"] = min(0.18, 0.07 + min((last_avg / prev_avg) - 1.45, 2.0) * 0.05)
        if repeated >= 2:
            signals["repeat"] = min(0.16, 0.06 + repeated * 0.03)
        if direction == "out":
            targets = [item["cp"] for item in recent_out] + [cp]
            if len(targets) >= 4:
                ratio = max(Counter(targets).values()) / len(targets)
                if ratio >= 0.7:
                    signals["target_bias"] = min(0.18, 0.07 + (ratio - 0.7) * 0.35)
            burst = self._recent(state, now, 70.0, "out")
            burst_targets = {item["cp"] for item in burst} | {cp}
            if len(burst_targets) >= 3 and len(burst) + 1 >= 4:
                signals["fan_out"] = min(0.26, 0.09 + len(burst_targets) * 0.03)
        if direction == "in":
            burst = self._recent(state, now, 70.0, "in")
            burst_sources = {item["cp"] for item in burst} | {cp}
            if len(burst_sources) >= 3 and len(burst) + 1 >= 4:
                signals["fan_in"] = min(0.26, 0.09 + len(burst_sources) * 0.03)
        cp_state = self.states.get(cp)
        if cp_state and (cp_state["risk"] >= 0.32 or cp in self.warning_ids or cp in self.excluded):
            signals["neighbor"] = min(0.22, 0.05 + min(cp_state["risk"], 1.0) * 0.18)
        if device_size >= 3 and (len(recent_short) >= 1 or amount >= baseline * 1.2):
            signals["device"] = min(0.16, 0.04 + min(device_size, 8) * 0.015)
        strong_keys = ("velocity", "high_value", "fan_in", "fan_out", "target_bias", "neighbor")
        strong = sum(1 for k in strong_keys if signals.get(k, 0.0) >= 0.10) >= 3 and sum(signals.values()) >= 0.42
        return signals, strong

    def _decay_one(self, aid, now):
        state = self.states.get(aid)
        if not state:
            return
        elapsed = max(now - state["decay_at"], 0.0)
        if elapsed <= 0:
            return
        risk_decay = math.pow(self.DECAY, elapsed / self.DECAY_PERIOD)
        signal_decay = math.pow(self.SIGNAL_DECAY, elapsed / self.DECAY_PERIOD)
        state["risk"] = self._clamp(state["risk"] * risk_decay)
        for key, value in state["signals"].items():
            value *= signal_decay
            state["signals"][key] = 0.0 if value < 0.015 else value
        state["decay_at"] = now
        if state["risk"] < 0.02 and not any(v >= 0.04 for v in state["signals"].values()):
            self.hot.discard(aid)

    def _apply_side(self, aid, cp, now, amount, direction, channel):
        state = self.states.get(aid)
        if state is None:
            return
        self._decay_one(aid, now)
        signals, strong = self._signals(aid, cp, now, amount, direction)
        state["recent"].append({"ts": now, "amount": float(amount), "cp": cp, "dir": direction, "channel": channel})
        delta = 0.0
        for key, value in signals.items():
            state["signals"][key] = min(1.4, state["signals"].get(key, 0.0) + value)
            delta += value
        if delta > 0:
            state["risk"] = self._clamp(state["risk"] + delta * (1.12 if strong else 1.0))
            self.hot.add(aid)
        if strong:
            state["risk"] = self._clamp(max(state["risk"], self.STRONG_RISK))
        state["last"] = now

    def apply_transaction(self, sender, receiver, now, amount, channel):
        self.amount_ema = self.amount_ema * 0.94 + float(amount) * 0.06
        self._apply_side(sender, receiver, now, amount, "out", channel)
        self._apply_side(receiver, sender, now, amount, "in", channel)
        self.adjust_credit = min(self.WARNING_MAX_CREDIT, self.adjust_credit + 1.0 / self.WARNING_TX_PER_STEP)

    def decay_hot(self, now):
        for aid in list(self.hot):
            self._decay_one(aid, now)

    def state_of(self, aid):
        state = self.states.get(aid)
        if state is None:
            return {"risk": 0.0, "signals": {}, "reasons": [], "signal_count": 0}
        breakdown = {k: round(v, 4) for k, v in state["signals"].items() if v >= 0.04}
        ranked = sorted(((k, v) for k, v in state["signals"].items() if v >= 0.06), key=lambda kv: kv[1], reverse=True)
        return {
            "risk": round(self._clamp(state["risk"]), 4),
            "signals": breakdown,
            "reasons": [self.SIGNAL_LABELS[k] for k, _ in ranked[:4]],
            "signal_count": len(breakdown),
            "last": state["last"],
        }

    # -- adaptive warning set --------------------------------------------------
    def select_warnings(self, active_ids):
        """Recomputes the warning set; returns (entered, left)."""
        ranked = []
        for aid in active_ids:
            if aid in self.excluded:
                continue
            state = self.states.get(aid)
            if state is None:
                continue
            signal_count = sum(1 for v in state["signals"].values() if v >= 0.04)
            ranked.append((aid, self._clamp(state["risk"]), signal_count))
        previous = set(self.warning_ids)
        if not ranked:
            self.warning_ids = set()
            self.last_context = {"threshold": 0.0, "warning_pct": self.MIN_WARNING_PCT, "max_count": 0}
            return set(), previous

        risks = sorted(r for _, r, _ in ranked)
        percentile = _quantile(risks, self.WARNING_PERCENTILE)
        percentile = max(self.MIN_WARNING_THRESHOLD, min(self.MAX_WARNING_THRESHOLD, percentile))
        self.threshold_ema = percentile if self.threshold_ema <= 0 else self.threshold_ema * 0.80 + percentile * 0.20
        threshold = max(self.MIN_WARNING_THRESHOLD, min(self.MAX_WARNING_THRESHOLD, self.threshold_ema))
        median = _quantile(risks, 0.50)
        dispersion = max(threshold - median, 0.0)
        warning_pct = min(self.MAX_WARNING_PCT, max(self.MIN_WARNING_PCT, self.MIN_WARNING_PCT + dispersion * 0.10))
        max_count = max(1, math.ceil(len(ranked) * self.MAX_WARNING_PCT))
        target_count = max(1, math.ceil(len(ranked) * warning_pct))
        retain_threshold = max(self.WARNING_RETAIN_FLOOR, threshold * self.WARNING_RETAIN_FACTOR)

        by_risk = sorted(ranked, key=lambda row: row[1], reverse=True)
        retained = [r for r in by_risk if r[0] in previous and r[2] >= self.WARNING_RETAIN_SIGNALS and r[1] >= retain_threshold]
        entering = [r for r in by_risk if r[2] >= self.WARNING_ENTER_SIGNALS and r[1] >= threshold]
        if len(entering) < target_count:
            seen = {r[0] for r in entering}
            for r in by_risk:
                if len(entering) >= target_count:
                    break
                if r[0] not in seen and r[2] >= self.WARNING_ENTER_SIGNALS and r[1] >= max(retain_threshold, median + 0.02):
                    entering.append(r)
                    seen.add(r[0])

        prev_count = len(previous)
        if target_count > prev_count:
            next_count = min(target_count, prev_count + max(1, int(self.adjust_credit)))
        elif target_count < prev_count:
            next_count = max(target_count, prev_count - max(1, int(self.adjust_credit * 0.5)))
        else:
            next_count = target_count
        next_count = min(next_count, max_count)
        consumed = abs(next_count - prev_count)
        if consumed:
            self.adjust_credit = max(0.0, self.adjust_credit - consumed)

        selected = []
        chosen = set()
        for pool in (retained, entering):
            for row in pool:
                if len(selected) >= next_count:
                    break
                if row[0] not in chosen:
                    selected.append(row)
                    chosen.add(row[0])
        self.warning_ids = chosen
        self.last_context = {
            "threshold": round(threshold, 4),
            "warning_pct": round(warning_pct, 4),
            "max_count": max_count,
        }
        return chosen - previous, previous - chosen


def _quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))
