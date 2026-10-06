"""Incremental sliding-window transaction graph.

Holds the last ``window_sec`` seconds of transactions and maintains per-account
aggregates so the nine graph features used by the rules / Random Forest (plus
``account_age_days`` for the GNN) are O(degree) to read after every
transaction. The same class is used by the live pipeline and by
``backend.train_live`` so models are trained on exactly the features they see
at inference time.

Feature semantics (same names as ``features.extract_node_features``):
  in_degree / out_degree  distinct counterparties sending to / receiving from
  total_in/out_amount     sum of amounts in the window
  retention_ratio         max(0, (in - out) / in), 0 when nothing came in
  unique_neighbors        distinct counterparties in either direction
  unique_channels         distinct channels used
  device_cluster_size     accounts registered on the same device
  transaction_count       transactions touching the account
  account_age_days        whole days since account creation
"""

from collections import Counter, defaultdict, deque

import numpy as np

RF_FEATURE_COLUMNS = [
    "in_degree",
    "out_degree",
    "total_in_amount",
    "total_out_amount",
    "retention_ratio",
    "unique_neighbors",
    "unique_channels",
    "device_cluster_size",
    "transaction_count",
]
GNN_FEATURE_COLUMNS = RF_FEATURE_COLUMNS + ["account_age_days"]


def edge_id(sender, receiver):
    return f"{sender}>{receiver}"


class FeatureStore:
    def __init__(self, window_sec: float):
        self.window_sec = float(window_sec)
        self.window = deque()  # (ts, sender, receiver, amount, channel, is_attack)
        self.out_edges = defaultdict(dict)  # acc -> {cp: [count, amount]}
        self.in_edges = defaultdict(dict)
        self.in_amount = defaultdict(float)
        self.out_amount = defaultdict(float)
        self.tx_count = Counter()
        self.channels = defaultdict(Counter)
        self.pairs = {}  # (s, r) -> {"count", "amount", "attack", "last_ts"}
        self.device_of = {}
        self.device_members = defaultdict(set)
        self.created_at = {}

    # -- accounts ------------------------------------------------------------
    def register_account(self, account_id, created_epoch, device_id):
        self.created_at[account_id] = float(created_epoch)
        self.set_device(account_id, device_id)

    def set_device(self, account_id, device_id):
        """Returns the device ids whose cluster size changed."""
        previous = self.device_of.get(account_id)
        if previous == device_id:
            return []
        if previous is not None:
            self.device_members[previous].discard(account_id)
            if not self.device_members[previous]:
                del self.device_members[previous]
        self.device_of[account_id] = device_id
        self.device_members[device_id].add(account_id)
        return [d for d in (previous, device_id) if d is not None]

    def device_cluster(self, device_id):
        return set(self.device_members.get(device_id, ()))

    # -- window maintenance ----------------------------------------------------
    def add(self, ts, sender, receiver, amount, channel, is_attack=False):
        """Adds a transaction; returns the updated edge payload."""
        amount = float(amount)
        self.window.append((ts, sender, receiver, amount, channel, bool(is_attack)))
        self._bump(sender, receiver, amount, channel, +1)
        pair = self.pairs.get((sender, receiver))
        if pair is None:
            pair = {"count": 0, "amount": 0.0, "attack": 0, "last_ts": ts}
            self.pairs[(sender, receiver)] = pair
        pair["count"] += 1
        pair["amount"] += amount
        pair["attack"] += 1 if is_attack else 0
        pair["last_ts"] = ts
        return self.edge_payload(sender, receiver)

    def expire(self, now):
        """Drops transactions older than the window.

        Returns (changed_edges, removed_edge_ids, touched_accounts).
        """
        cutoff = now - self.window_sec
        touched_pairs = set()
        touched_accounts = set()
        while self.window and self.window[0][0] < cutoff:
            ts, sender, receiver, amount, channel, is_attack = self.window.popleft()
            self._bump(sender, receiver, amount, channel, -1)
            pair = self.pairs.get((sender, receiver))
            if pair is not None:
                pair["count"] -= 1
                pair["amount"] -= amount
                pair["attack"] -= 1 if is_attack else 0
                if pair["count"] <= 0:
                    del self.pairs[(sender, receiver)]
            touched_pairs.add((sender, receiver))
            touched_accounts.update((sender, receiver))
        changed, removed = [], []
        for sender, receiver in touched_pairs:
            if (sender, receiver) in self.pairs:
                changed.append(self.edge_payload(sender, receiver))
            else:
                removed.append(edge_id(sender, receiver))
        return changed, removed, touched_accounts

    def _bump(self, sender, receiver, amount, channel, sign):
        for acc, cp, edges, totals in (
            (sender, receiver, self.out_edges, self.out_amount),
            (receiver, sender, self.in_edges, self.in_amount),
        ):
            stats = edges[acc].get(cp)
            if stats is None:
                stats = [0, 0.0]
                edges[acc][cp] = stats
            stats[0] += sign
            stats[1] += sign * amount
            if stats[0] <= 0:
                del edges[acc][cp]
                if not edges[acc]:
                    del edges[acc]
            totals[acc] += sign * amount
            if abs(totals[acc]) < 1e-6:
                totals.pop(acc, None)
            self.tx_count[acc] += sign
            if self.tx_count[acc] <= 0:
                del self.tx_count[acc]
            self.channels[acc][channel] += sign
            if self.channels[acc][channel] <= 0:
                del self.channels[acc][channel]
                if not self.channels[acc]:
                    del self.channels[acc]

    # -- reads -------------------------------------------------------------
    def edge_payload(self, sender, receiver):
        pair = self.pairs[(sender, receiver)]
        return {
            "id": edge_id(sender, receiver),
            "source": sender,
            "target": receiver,
            "count": pair["count"],
            "amount": round(pair["amount"], 2),
            "attack_count": pair["attack"],
            "last_ts": pair["last_ts"],
        }

    def edges(self):
        return [self.edge_payload(s, r) for (s, r) in self.pairs]

    def active_accounts(self):
        return list(self.tx_count.keys())

    def neighbors(self, account_id):
        return set(self.in_edges.get(account_id, {})) | set(self.out_edges.get(account_id, {}))

    def features(self, account_id, now):
        in_edges = self.in_edges.get(account_id, {})
        out_edges = self.out_edges.get(account_id, {})
        total_in = max(self.in_amount.get(account_id, 0.0), 0.0)
        total_out = max(self.out_amount.get(account_id, 0.0), 0.0)
        retention = max(0.0, (total_in - total_out) / total_in) if total_in > 0 else 0.0
        neighbors = set(in_edges) | set(out_edges)
        device = self.device_of.get(account_id)
        created = self.created_at.get(account_id, now)
        return {
            "in_degree": len(in_edges),
            "out_degree": len(out_edges),
            "total_in_amount": round(total_in, 2),
            "total_out_amount": round(total_out, 2),
            "retention_ratio": round(retention, 6),
            "unique_neighbors": len(neighbors),
            "unique_channels": len(self.channels.get(account_id, {})),
            "device_cluster_size": len(self.device_members.get(device, ())) if device else 1,
            "transaction_count": self.tx_count.get(account_id, 0),
            "account_age_days": int(max(now - created, 0.0) // 86400),
        }

    def graph_tensors(self, now):
        """Window graph in the exact form ``gnn.graph_to_pyg`` produces.

        Nodes are accounts with at least one transaction in the window;
        features are min-max normalised per column across the snapshot and
        edges are made bidirectional.
        """
        nodes = self.active_accounts()
        if not nodes:
            return nodes, None, None
        index = {acc: i for i, acc in enumerate(nodes)}
        raw = np.array(
            [[float(f[c]) for c in GNN_FEATURE_COLUMNS] for f in (self.features(a, now) for a in nodes)],
            dtype=np.float32,
        )
        col_min = raw.min(axis=0)
        col_range = raw.max(axis=0) - col_min
        col_range[col_range == 0] = 1.0
        x = (raw - col_min) / col_range
        src, dst = [], []
        for sender, receiver in self.pairs:
            i, j = index.get(sender), index.get(receiver)
            if i is None or j is None:
                continue
            src.extend((i, j))
            dst.extend((j, i))
        edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
        return nodes, x, edge_index
