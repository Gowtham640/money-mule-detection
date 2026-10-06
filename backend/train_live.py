"""Train the Random Forest and GAT on the live feature distribution.

The original models were trained on graphs built only from attack
transactions, so at inference over a live sliding window they flagged ~97% of
ordinary active accounts (RF) or nothing at all (GNN). This script replays a
simulated stream through the exact components the live service uses:

  TrafficSimulator  -> normal transactions (same generator logic)
  attacks.build_plan -> injected attacks (same plans the Spawn button runs)
  FeatureStore      -> sliding-window graph features (same window length)

A sample is taken for the sender and receiver after every transaction, which
is precisely when the live pipeline scores accounts. An account is labelled 1
while it has a labelled attack transaction inside the window.

Usage:  python -m backend.train_live [--attacks 300] [--seed 7]
"""

import argparse
import heapq
import json
import random
import time
from collections import defaultdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from backend import settings
from backend.attacks import ATTACK_CATALOG, AttackContext, build_plan
from backend.detection import evaluate_rules, fuse_scores
from backend.early_warning import EarlyWarningEngine
from backend.feature_store import GNN_FEATURE_COLUMNS, RF_FEATURE_COLUMNS, FeatureStore
from backend.generator import CHANNELS, generate_accounts
from backend.model import file_sha256
from backend.simulator import TrafficSimulator

SIM_START = 1_700_000_000.0


def simulate(num_attacks, seed, window_sec, gnn_normal_snapshot_p=0.025, gnn_scorer=None):
    """Replays a simulated stream. With ``gnn_scorer`` (calibration mode) every
    sample also carries the live GNN score and early-warning score, exactly
    as the pipeline computes them, and no GNN training snapshots are kept."""
    rng = random.Random(seed)
    np.random.seed(seed)
    frame = generate_accounts(settings.NUM_ACCOUNTS, rng=rng)
    real_now = datetime.now()
    accounts = {}
    for row in frame.itertuples(index=False):
        accounts[row.account_id] = {
            "created_epoch": SIM_START - (real_now - row.creation_time).total_seconds(),
            "device_id": row.device_id,
            "ip_address": row.ip_address,
            "home_channel": row.channel,
            "balance": float(row.balance),
            "is_banned": False,
        }
    store = FeatureStore(window_sec)
    early = EarlyWarningEngine() if gnn_scorer else None
    for acc, info in accounts.items():
        store.register_account(acc, info["created_epoch"], info["device_id"])

    def sync_early(acc, now):
        if early is None:
            return
        early.register_account(acc, accounts[acc]["created_epoch"], now)
        members = store.device_cluster(accounts[acc]["device_id"])
        for member in members:
            early.device_size[member] = len(members)

    for acc in accounts:
        sync_early(acc, SIM_START)
    sim = TrafficSimulator(rng)
    sim.set_accounts(accounts, list(accounts))
    next_id = [len(accounts)]

    def new_account_id():
        next_id[0] += 1
        return f"A{next_id[0]:04d}"

    def add_account(info, acc_id):
        accounts[acc_id] = info
        store.register_account(acc_id, info["created_epoch"], info["device_id"])
        sim.set_accounts(accounts, list(accounts))
        sync_early(acc_id, info["created_epoch"])

    samples = []
    snapshots = []
    runs = []
    last_attack_tx = {}
    label_run = {}
    attack_types = list(ATTACK_CATALOG)
    lo, hi = settings.NORMAL_TX_MIN_INTERVAL_MS / 1000, settings.NORMAL_TX_MAX_INTERVAL_MS / 1000
    step_gap = settings.ATTACK_STEP_INTERVAL_MS / 1000

    t = SIM_START
    next_normal = t + rng.uniform(lo, hi)
    next_attack = t + window_sec + rng.uniform(5, 20)  # let the window fill first
    next_onboard = t + settings.NEW_ACCOUNT_INTERVAL_SEC
    pending = []  # heap of (time, order, run_index, step)
    order = 0

    def label(acc, now):
        last = last_attack_tx.get(acc)
        return 1 if last is not None and now - last <= window_sec else 0

    def process(now, sender, receiver, amount, channel, run_index=None):
        store.expire(now)
        is_attack = run_index is not None
        store.add(now, sender, receiver, amount, channel, is_attack)
        accounts[sender]["balance"] -= amount
        accounts[receiver]["balance"] += amount
        if is_attack:
            labeled = runs[run_index]["labeled"]
            for acc in (sender, receiver):
                if acc in labeled:
                    last_attack_tx[acc] = now
                    label_run[acc] = run_index
        gnn_scores = None
        if early is not None:
            early.apply_transaction(sender, receiver, now, amount, channel)
            nodes, x, edge_index = store.graph_tensors(now)
            gnn_scores = dict(zip(nodes, gnn_scorer(x, edge_index)))
        for acc in (sender, receiver):
            feats = store.features(acc, now)
            lab = label(acc, now)
            sample = {
                **feats,
                "label": lab,
                "t": now,
                "account_id": acc,
                "attack_type": runs[run_index]["type"] if is_attack else None,
                "run": run_index,
                "label_run": label_run.get(acc) if lab else None,
            }
            if early is not None:
                sample["gnn"] = float(gnn_scores[acc])
                sample["early_risk"] = early.state_of(acc)["risk"]
            samples.append(sample)
        if early is None and (is_attack or rng.random() < gnn_normal_snapshot_p):
            nodes, x, edge_index = store.graph_tensors(now)
            if x is not None:
                y = np.array([label(acc, now) for acc in nodes], dtype=np.float32)
                snapshots.append({"t": now, "x": x, "edge_index": edge_index, "y": y})

    while len(runs) < num_attacks or pending:
        candidates = [next_normal, next_onboard]
        if len(runs) < num_attacks:
            candidates.append(next_attack)
        if pending:
            candidates.append(pending[0][0])
        t = min(candidates)
        if pending and t == pending[0][0]:
            _, _, run_index, step = heapq.heappop(pending)
            if accounts[step.sender]["balance"] < step.amount:
                accounts[step.sender]["balance"] += step.amount  # cash deposit
            process(t, step.sender, step.receiver, float(step.amount), step.channel, run_index)
        elif t == next_attack and len(runs) < num_attacks:
            idle = set(accounts) - set(store.active_accounts())
            ctx = AttackContext(accounts, list(accounts), t, rng, idle, new_account_id)
            plan = build_plan(attack_types[len(runs) % len(attack_types)], ctx)
            for info in plan.new_accounts:
                add_account({**info, "is_banned": False}, info["account_id"])
            for acc, device in plan.device_updates.items():
                accounts[acc]["device_id"] = device
                store.set_device(acc, device)
                sync_early(acc, t)
            runs.append({"type": plan.attack_type, "start": t, "labeled": set(plan.labeled)})
            for i, step in enumerate(plan.steps):
                order += 1
                heapq.heappush(pending, (t + i * step_gap, order, len(runs) - 1, step))
            next_attack = t + rng.uniform(25, 70)
        elif t == next_onboard:
            acc_id = new_account_id()
            add_account({
                "created_epoch": t,
                "device_id": f"DV-{acc_id}",
                "ip_address": f"192.168.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
                "home_channel": rng.choice(CHANNELS),
                "balance": float(rng.randint(5000, 50000)),
                "is_banned": False,
            }, acc_id)
            next_onboard = t + settings.NEW_ACCOUNT_INTERVAL_SEC
        else:
            spec = sim.next_transaction()
            if spec:
                process(t, spec["sender"], spec["receiver"], spec["amount"], spec["channel"])
            next_normal = t + rng.uniform(lo, hi)
    return pd.DataFrame(samples), snapshots, runs


def evaluate_rf(model, test, runs, threshold=0.5):
    scores = model.predict_proba(test[RF_FEATURE_COLUMNS])[:, 1]
    pred = (scores >= threshold).astype(int)
    y = test["label"].values
    result = {
        "samples": int(len(test)),
        "positives": int(y.sum()),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4) if 0 < y.sum() < len(y) else None,
        "normal_flag_rate": round(float(pred[y == 0].mean()), 4) if (y == 0).any() else None,
    }
    # Per attack type: share of labelled mules whose score crossed the
    # threshold at least once while they were labelled.
    test = test.assign(score=scores)
    per_type = defaultdict(lambda: [0, 0])
    positives = test[test["label"] == 1]
    for run_index in sorted(set(positives["run"].dropna().astype(int))):
        run = runs[run_index]
        for acc in run["labeled"]:
            rows = test[(test["account_id"] == acc) & (test["t"] >= run["start"]) & (test["label"] == 1)]
            if rows.empty:
                continue
            per_type[run["type"]][1] += 1
            per_type[run["type"]][0] += int(rows["score"].max() >= threshold)
    result["mule_detection_by_type"] = {
        k: round(v[0] / v[1], 3) for k, v in sorted(per_type.items()) if v[1]
    }
    return result


def train_rf(samples, runs, split_t):
    train = samples[samples["t"] < split_t]
    test = samples[samples["t"] >= split_t]
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(train[RF_FEATURE_COLUMNS], train["label"])
    model.n_jobs = None  # single-row live inference is faster without joblib dispatch
    model.feature_list = RF_FEATURE_COLUMNS
    evaluation = evaluate_rf(model, test, runs)
    evaluation["train_samples"] = int(len(train))
    evaluation["train_positives"] = int(train["label"].sum())
    return model, evaluation


def train_gnn(snapshots, split_t, epochs=60, seed=7):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader

    from backend.gnn import FraudGAT
    from backend.train_gnn import focal_loss

    torch.manual_seed(seed)
    to_data = lambda s: Data(  # noqa: E731
        x=torch.from_numpy(s["x"]), edge_index=torch.from_numpy(s["edge_index"]), y=torch.from_numpy(s["y"])
    )
    train = [to_data(s) for s in snapshots if s["t"] < split_t]
    test = [to_data(s) for s in snapshots if s["t"] >= split_t]
    model = FraudGAT(in_channels=len(GNN_FEATURE_COLUMNS))
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=32, shuffle=True)
    best_state, best_loss = None, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimiser.zero_grad()
            loss = focal_loss(model(batch.x, batch.edge_index), batch.y)
            loss.backward()
            optimiser.step()
            total += loss.item()
        avg = total / max(len(loader), 1)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  GNN epoch {epoch:>3}/{epochs} focal loss {avg:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data in test:
            preds.append(model(data.x, data.edge_index).numpy())
            labels.append(data.y.numpy())
    p = np.concatenate(preds)
    y = np.concatenate(labels)
    pred = (p >= 0.5).astype(int)
    evaluation = {
        "test_graphs": len(test),
        "test_nodes": int(len(y)),
        "positives": int(y.sum()),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None,
        "train_graphs": len(train),
    }
    return best_state, evaluation


FP_BUDGET_PER_HOUR = 6.0


def calibrate(rf, gnn_state, seed, num_attacks, window):
    """Chooses the live decision rule on an unseen simulated stream.

    Replays the stream through the live scoring path (RF + GNN + rules fused
    with the pipeline's weights, plus the early-warning engine) and, for each
    candidate rule ``fused >= T and early_risk >= E``, measures what the live
    service would do: accounts flagged (flags are sticky, like live) while not
    being an injected mule, per hour of traffic, versus the share of injected
    mules flagged while they were active. Picks the rule with the best mule
    recall whose false-positive rate fits the budget.
    """
    import torch

    from backend.gnn import FraudGAT

    gnn = None
    if gnn_state is not None:
        gnn = FraudGAT(in_channels=len(GNN_FEATURE_COLUMNS))
        gnn.load_state_dict(gnn_state)
        gnn.eval()

    def scorer(x, edge_index):
        if gnn is None:
            return [None] * len(x)
        with torch.inference_mode():
            return gnn(torch.from_numpy(x), torch.from_numpy(edge_index)).numpy()

    samples, _, runs = simulate(num_attacks, seed, window, gnn_scorer=scorer)
    samples = samples.sort_values("t", kind="stable").reset_index(drop=True)
    samples["rf"] = rf.predict_proba(samples[RF_FEATURE_COLUMNS])[:, 1]
    samples["rule"] = [evaluate_rules(row)[0] for row in samples[RF_FEATURE_COLUMNS].to_dict("records")]
    samples["fused"] = [
        fuse_scores(rf_, rule, None if gnn is None else g)
        for rf_, rule, g in zip(samples["rf"], samples["rule"], samples["gnn"])
    ]
    hours = (samples["t"].max() - samples["t"].min()) / 3600.0
    labeled_pairs = {(i, acc) for i, run in enumerate(runs) for acc in run["labeled"]}
    grid = []
    for threshold in np.round(np.arange(0.50, 0.86, 0.02), 2):
        for min_early in (0.0, 0.1, 0.2, 0.3, 0.4):
            hits = samples[(samples["fused"] >= threshold) & (samples["early_risk"] >= min_early)]
            first = hits.drop_duplicates("account_id")
            fps = int((first["label"] == 0).sum())
            caught = {(int(r), a) for r, a in zip(first.loc[first["label"] == 1, "label_run"], first.loc[first["label"] == 1, "account_id"])}
            detected_runs = {r for r, _ in caught}
            grid.append({
                "threshold": float(threshold),
                "min_early_risk": min_early,
                "fp_per_hour": round(float(fps / hours), 2),
                "mule_recall": round(len(caught) / max(len(labeled_pairs), 1), 4),
                "attack_detection_rate": round(len(detected_runs) / max(len(runs), 1), 4),
            })
    feasible = [g for g in grid if g["fp_per_hour"] <= FP_BUDGET_PER_HOUR] or sorted(grid, key=lambda g: g["fp_per_hour"])[:1]
    best = max(feasible, key=lambda g: (g["mule_recall"], g["attack_detection_rate"], -g["fp_per_hour"]))
    hits = samples[(samples["fused"] >= best["threshold"]) & (samples["early_risk"] >= best["min_early_risk"])]
    first = hits.drop_duplicates("account_id")
    caught = first[first["label"] == 1]
    by_type = {}
    for i, run in enumerate(runs):
        flagged = set(caught.loc[caught["label_run"] == i, "account_id"])
        entry = by_type.setdefault(run["type"], {"runs": 0, "detected_runs": 0, "mules": 0, "mules_flagged": 0})
        entry["runs"] += 1
        entry["detected_runs"] += bool(flagged)
        entry["mules"] += len(run["labeled"])
        entry["mules_flagged"] += len(flagged & run["labeled"])
    return {
        **best,
        "by_attack_type": by_type,
        "fp_budget_per_hour": FP_BUDGET_PER_HOUR,
        "stream_hours": round(hours, 2),
        "attacks": len(runs),
        "seed": seed,
        "grid": grid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attacks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--calibration-attacks", type=int, default=80)
    parser.add_argument("--calibrate-only", action="store_true", help="recalibrate the saved models only")
    args = parser.parse_args()

    window = settings.FEATURE_WINDOW_SEC
    started = time.time()
    if args.calibrate_only:
        import torch

        meta = json.loads(settings.MODEL_META_PATH.read_text())
        rf = joblib.load(settings.RF_MODEL_PATH)
        state = torch.load(settings.GNN_MODEL_PATH, weights_only=True) if settings.GNN_MODEL_PATH.exists() else None
        print(f"Calibrating saved models on an unseen stream ({args.calibration_attacks} attacks)...")
        meta["decision"] = calibrate(rf, state, meta.get("seed", args.seed) + 1000, args.calibration_attacks, window)
        print("  chosen:", json.dumps({k: v for k, v in meta["decision"].items() if k != "grid"}))
        settings.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
        return
    print(f"Simulating stream: {args.attacks} attacks, window={window}s, seed={args.seed}")
    samples, snapshots, runs = simulate(args.attacks, args.seed, window)
    print(f"  {len(samples)} account samples ({int(samples['label'].sum())} positive), {len(snapshots)} graph snapshots")
    split_t = samples["t"].quantile(0.8)

    print("Training Random Forest...")
    rf, rf_eval = train_rf(samples, runs, split_t)
    joblib.dump(rf, settings.RF_MODEL_PATH)
    print("  RF held-out:", json.dumps({k: v for k, v in rf_eval.items() if k != "mule_detection_by_type"}))
    print("  RF mule detection by attack type:", rf_eval["mule_detection_by_type"])

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trainer": "backend.train_live",
        "feature_window_sec": window,
        "rf_features": RF_FEATURE_COLUMNS,
        "gnn_features": GNN_FEATURE_COLUMNS,
        "seed": args.seed,
        "attacks_simulated": args.attacks,
        "normal_tx_interval_ms": [settings.NORMAL_TX_MIN_INTERVAL_MS, settings.NORMAL_TX_MAX_INTERVAL_MS],
        "attack_step_interval_ms": settings.ATTACK_STEP_INTERVAL_MS,
        "rf_eval": rf_eval,
        "rf_sha256": file_sha256(settings.RF_MODEL_PATH),
    }
    state = None
    if not args.skip_gnn:
        import torch

        print("Training GNN (GAT)...")
        state, gnn_eval = train_gnn(snapshots, split_t, epochs=args.epochs, seed=args.seed)
        torch.save(state, settings.GNN_MODEL_PATH)
        print("  GNN held-out:", json.dumps(gnn_eval))
        meta["gnn_eval"] = gnn_eval
        meta["gnn_sha256"] = file_sha256(settings.GNN_MODEL_PATH)
    print(f"Calibrating decision rule on an unseen stream ({args.calibration_attacks} attacks)...")
    decision = calibrate(rf, state, args.seed + 1000, args.calibration_attacks, window)
    print("  chosen:", json.dumps({k: v for k, v in decision.items() if k != "grid"}))
    for row in decision["grid"]:
        if row["min_early_risk"] in (0.0, decision["min_early_risk"]) and row["threshold"] in (0.5, 0.56, 0.6, 0.66, 0.7, 0.76, 0.8):
            print("   ", row)
    meta["decision"] = decision
    settings.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved models + {settings.MODEL_META_PATH.name} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
