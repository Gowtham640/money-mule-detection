"""Stateful generator of legitimate (normal) transactions.

Extends ``generator.generate_normal_transactions`` (80% of payments inside a
community cluster, 500-8000 amounts, random channel, sender's device, skipped
when the sender lacks funds) with a few benign behaviours real traffic has,
so the detectors face genuine false-positive pressure:

  * merchants: ~3% of accounts receive a share of payments from anyone
  * payment sessions: an account occasionally pays several bills in a row
  * occasional legitimate high-value transfers (rent, savings moves)
"""

import zlib

from backend.generator import CHANNELS


class TrafficSimulator:
    INTRA_CLUSTER_P = 0.80
    MERCHANT_SHARE = 0.03
    MERCHANT_P = 0.08
    SESSION_START_P = 0.03
    HIGH_VALUE_P = 0.015

    def __init__(self, rng):
        self.rng = rng
        self.accounts = {}
        self.active = []
        self.clusters = []
        self.merchants = []
        self.cluster_of = {}
        self.session = None

    def set_accounts(self, accounts, active_ids):
        """``accounts`` is the live account dict (balances read directly)."""
        self.accounts = accounts
        self.active = sorted(active_ids)
        n = len(self.active)
        size = max(10, n // 20)
        self.clusters = [self.active[i:i + size] for i in range(0, n, size)]
        self.cluster_of = {acc: idx for idx, cluster in enumerate(self.clusters) for acc in cluster}
        # Merchant role is a stable property of the account id.
        merchant_count = max(1, int(n * self.MERCHANT_SHARE))
        self.merchants = sorted(self.active, key=lambda a: zlib.crc32(f"merchant:{a}".encode()))[:merchant_count]
        if self.session and self.session["account"] not in self.cluster_of:
            self.session = None

    def _amount(self):
        if self.rng.random() < self.HIGH_VALUE_P:
            return self.rng.randint(15000, 60000)
        return self.rng.randint(500, 8000)

    def _pick_pair(self):
        if self.session:
            sender = self.session["account"]
            self.session["remaining"] -= 1
            if self.session["remaining"] <= 0:
                self.session = None
            cluster = self.clusters[self.cluster_of[sender]]
            return sender, self.rng.choice(self.merchants + cluster)
        cluster = self.rng.choice(self.clusters)
        if self.rng.random() < self.INTRA_CLUSTER_P and len(cluster) >= 2:
            sender, receiver = self.rng.choice(cluster), self.rng.choice(cluster)
        else:
            sender, receiver = self.rng.choice(self.active), self.rng.choice(self.active)
        if self.rng.random() < self.MERCHANT_P:
            receiver = self.rng.choice(self.merchants)
        if self.rng.random() < self.SESSION_START_P:
            self.session = {"account": sender, "remaining": self.rng.randint(2, 4)}
        return sender, receiver

    def next_transaction(self):
        if len(self.active) < 2:
            return None
        for _ in range(8):
            sender, receiver = self._pick_pair()
            if sender == receiver:
                continue
            info = self.accounts.get(sender)
            if not info or info.get("is_banned") or self.accounts.get(receiver, {}).get("is_banned"):
                continue
            amount = self._amount()
            if info["balance"] < amount:
                continue
            return {
                "sender": sender,
                "receiver": receiver,
                "amount": float(amount),
                "channel": self.rng.choice(CHANNELS),
                "device_id": info["device_id"],
                "ip_address": info.get("ip_address"),
            }
        return None
