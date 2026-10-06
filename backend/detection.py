import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import shap
import numpy as np
import joblib

def load_model(model_path="fraud_model.pkl"):
    return joblib.load(model_path)

# -----------------------------
# Rule-Based Fraud Detection
# -----------------------------

# Each rule: (key, reason label, points, predicate on a feature row).
# The batch function below and the live pipeline both evaluate this list.
RULES = [
    ("fan_in", "High In-Degree", 2, lambda r: r["in_degree"] >= 2),
    ("fan_out", "High Out-Degree", 2, lambda r: r["out_degree"] >= 2),
    ("pass_through", "Low Retention Ratio", 2,
     lambda r: r["retention_ratio"] < 0.2 and r["total_in_amount"] > 0),
    ("shared_device", "Shared Device Cluster", 3, lambda r: r["device_cluster_size"] > 2),
    ("channel_burst", "High Channel Diversity", 2, lambda r: r["unique_channels"] > 2),
    ("high_velocity", "High Transaction Count", 1, lambda r: r["transaction_count"] >= 3),
]
MAX_RULE_SCORE = sum(points for _, _, points, _ in RULES)


def evaluate_rules(row):
    """Returns (score, {rule_key: fired}, [reasons]) for one feature row."""
    score = 0
    flags = {}
    reasons = []
    for key, label, points, predicate in RULES:
        fired = bool(predicate(row))
        flags[key] = fired
        if fired:
            score += points
            reasons.append(label)
    return score, flags, reasons


def rule_based_detection(features_df):

    risk_results = []

    for _, row in features_df.iterrows():

        score, _, reasons = evaluate_rules(row)

        risk_results.append({
            "account_id": row["account_id"],
            "risk_score": score,
            "reasons": reasons,
            "is_fraud": row["is_fraud"]
        })

    return pd.DataFrame(risk_results)

# -----------------------------
# ML-Based Fraud Detection
# -----------------------------

def ml_detection(features_df):

    feature_columns = [
        "in_degree",
        "out_degree",
        "total_in_amount",
        "total_out_amount",
        "retention_ratio",
        "unique_neighbors",
        "unique_channels",
        "device_cluster_size",
        "transaction_count"
    ]

    X = features_df[feature_columns]
    y = features_df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    model.feature_list = feature_columns

    # Evaluation
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    feature_importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    # SHAP Explainer (create only once)
    explainer = shap.TreeExplainer(model)

    return model, cm, report, feature_importances, explainer

# -----------------------------
# ML Prediction (Inference)
# -----------------------------

def ml_predict(model, features_df, risk_df, threshold=0.5):

    feature_columns = model.feature_list
    X = features_df[feature_columns]

    probabilities = model.predict_proba(X)[:, 1]

    results = features_df[["account_id", "is_fraud"]].copy()
    results["ml_score"] = probabilities

    # Carry gnn_score forward if present in features_df
    if "gnn_score" in features_df.columns:
        results["gnn_score"] = features_df["gnn_score"].values

    merged = results.merge(risk_df, on="account_id", how="left")

    max_risk = merged["risk_score"].max() + 1e-6
    merged["rule_score_norm"] = merged["risk_score"] / max_risk

    if "gnn_score" in merged.columns:
        merged["final_score"] = (
            0.5 * merged["ml_score"] +
            0.3 * merged["rule_score_norm"] +
            0.2 * merged["gnn_score"]
        )
    else:
        merged["final_score"] = (
            0.6 * merged["ml_score"] +
            0.4 * merged["rule_score_norm"]
        )

    merged["predicted_label"] = (merged["final_score"] > threshold).astype(int)

    return merged

def fuse_scores(rf_score, rule_score, gnn_score=None):
    """Live per-account version of the hybrid score in ``ml_predict``.

    Same weights (0.5 ML + 0.3 rules + 0.2 GNN, or 0.6/0.4 without the GNN).
    ``ml_predict`` normalises rule points by the batch maximum; a single
    account has no batch, so points are normalised by the maximum attainable
    rule score instead.
    """
    rule_norm = float(rule_score) / MAX_RULE_SCORE
    if gnn_score is None:
        return 0.6 * float(rf_score) + 0.4 * rule_norm
    return 0.5 * float(rf_score) + 0.3 * rule_norm + 0.2 * float(gnn_score)


def explain_risk_categories(shap_values, feature_columns):

    category_map = {
        "Velocity Risk": [
            "transaction_count",
            "out_degree"
        ],
        "Shared Device Risk": [
            "device_cluster_size"
        ],
        "Ring Participation Risk": [
            "unique_neighbors",
            "in_degree",
            "out_degree"
        ],
        "Channel Risk": [
            "unique_channels"
        ],
        "Retention Risk": [
            "retention_ratio"
        ]
    }

    category_scores = {}

    for category, features in category_map.items():

        total_impact = 0

        for feature in features:
            if feature in feature_columns:
                idx = feature_columns.index(feature)

                value = shap_values[idx]

                # Force scalar (handle array cases safely)
                if isinstance(value, (list, np.ndarray)):
                    value = float(np.array(value).flatten()[0])

                total_impact += abs(float(value))

        category_scores[category] = round(float(np.array(total_impact).flatten()[0]), 4)

    return category_scores

# -----------------------------
# Fraud Role Classification
# -----------------------------
def classify_role(row):

    # 1️⃣ Ring Coordinator (strongest structural pattern)
    if (
        row["in_degree"] >= 2 and
        row["out_degree"] >= 2 and
        row["unique_neighbors"] >= 3
    ):
        return "Ring Coordinator"

    # 2️⃣ Distributor Mule
    if row["out_degree"] > row["in_degree"] and row["retention_ratio"] < 0.3:
        return "Distributor Mule"

    # 3️⃣ Collector Mule
    if row["in_degree"] > row["out_degree"] and row["retention_ratio"] < 0.3:
        return "Collector Mule"

    # Pass-Through Mule: balanced in/out that keeps almost nothing (the
    # in == out case the two rules above do not cover, e.g. ring members)
    if row["in_degree"] >= 1 and row["out_degree"] >= 1 and row["retention_ratio"] < 0.3:
        return "Pass-Through Mule"

    # 4️⃣ Entry Node
    if row["in_degree"] == 0 and row["out_degree"] > 0:
        return "Entry Node"

    # 5️⃣ Exit Node
    if row["in_degree"] > 0 and row["retention_ratio"] > 0.6:
        return "Exit Node"

    return "Unclassified"


def classify_fraud_roles(features_df):

    roles = []

    for _, row in features_df.iterrows():
        roles.append({
            "account_id": row["account_id"],
            "role": classify_role(row)
        })

    return pd.DataFrame(roles)

def early_stage_detection(features_df, top_pct=0.12, min_signals=2):
    """
    Flags only the TOP 12% most anomalous accounts as early warning.
 
    Uses an anomaly score computed from 6 behavioral signals.
    Each signal is RELATIVE to the current account population —
    so it works correctly with any dataset size or age distribution.
 
    Signals:
      1. New account        (age < 30th percentile of all ages)
      2. Shared device      (device_cluster_size >= 3)
      3. Pass-through       (retention_ratio < 0.2 AND has incoming tx)
      4. High velocity      (transaction_count > mean + 1.5×std)
      5. High fan-out       (out_degree >= 3)
      6. Multi-channel      (unique_channels >= 3)
 
    Scoring:
      Each signal contributes a weighted score.
      Only accounts with 2+ signals AND in top 12% by score are flagged.
      This ensures early detection is SELECTIVE and MEANINGFUL.
 
    Why this approach is better:
      - Absolute thresholds (account_age <= 5 days) fail with synthetic data
      - Relative thresholds adapt to whatever accounts currently exist
      - Top-% cutoff ensures the number of warnings stays proportional
      - Min signals prevents single-anomaly false positives
    """
    import numpy as np
 
    if features_df.empty:
        return pd.DataFrame(
            columns=["account_id", "early_risk_score", "anomaly_score", "signals"]
        )
 
    # ── Population statistics ──────────────────────────────────────
    age_p30        = features_df["account_age_days"].quantile(0.30)
    mean_tx        = features_df["transaction_count"].mean()
    std_tx         = features_df["transaction_count"].std()
    std_tx         = std_tx if std_tx > 0 else 1.0
    mean_in        = features_df["total_in_amount"].mean()
 
    results = []
 
    for _, row in features_df.iterrows():
        signals      = []
        anomaly_score = 0.0
 
        # Signal 1: Relatively new account
        if row["account_age_days"] < age_p30:
            weight = 1.0 - (row["account_age_days"] / max(age_p30, 1))
            anomaly_score += 0.25 * weight
            signals.append("New Account")
 
        # Signal 2: Shared device cluster
        if row["device_cluster_size"] >= 3:
            anomaly_score += 0.20 * min(row["device_cluster_size"] / 10.0, 1.0)
            signals.append("Shared Device")
 
        # Signal 3: Pass-through behavior
        if row["retention_ratio"] < 0.20 and row["total_in_amount"] > 0:
            pass_through_intensity = 1.0 - row["retention_ratio"] / 0.20
            anomaly_score += 0.20 * pass_through_intensity
            signals.append("Pass-Through")
 
        # Signal 4: High velocity (z-score based)
        z_tx = (row["transaction_count"] - mean_tx) / std_tx
        if z_tx > 1.5:
            anomaly_score += min(z_tx * 0.08, 0.25)
            signals.append("High Velocity")
 
        # Signal 5: High fan-out
        if row["out_degree"] >= 3:
            anomaly_score += 0.15 * min(row["out_degree"] / 5.0, 1.0)
            signals.append("High Fan-Out")
 
        # Signal 6: Multi-channel
        if row["unique_channels"] >= 3:
            anomaly_score += 0.15
            signals.append("Multi-Channel")
 
        # Dampen single-signal accounts
        if len(signals) < min_signals:
            anomaly_score *= 0.2
 
        results.append({
            "account_id":    row["account_id"],
            "anomaly_score": round(anomaly_score, 4),
            "signal_count":  len(signals),
            "signals":       ", ".join(signals) if signals else "None",
        })
 
    if not results:
        return pd.DataFrame()
 
    results_df = pd.DataFrame(results).sort_values(
        "anomaly_score", ascending=False
    )
 
    # Apply top-% cutoff AND minimum signal requirement
    cutoff_idx = max(1, int(len(results_df) * top_pct))
    top_df     = results_df.head(cutoff_idx)
    flagged    = top_df[
        (top_df["signal_count"] >= min_signals) &
        (top_df["anomaly_score"] >= 0.15)   # meaningful threshold
    ].copy()
 
    flagged["early_risk_score"] = flagged["signal_count"]
 
    return flagged.reset_index(drop=True)

def behavioral_drift_detection(
    baseline_features,
    current_features,
    feature_columns,
    threshold=0.4
):
    """
    Detects accounts that significantly CHANGED behavior between
    baseline (pre-attack) and current (post-attack) snapshots.
 
    This is a KEY detection layer because:
    - Some fraud accounts look individually normal (low absolute scores)
    - But their CHANGE in behavior from baseline is dramatic
    - Example: account dormant for weeks suddenly makes 15 transactions
 
    Method:
      1. Normalize both vectors using combined mean/std
      2. Compute Euclidean distance between normalized vectors
      3. Flag accounts with drift > threshold
 
    Features compared:
      transaction_count, in_degree, out_degree, retention_ratio,
      unique_neighbors, unique_channels, device_cluster_size
 
    Why drift matters for fraud detection:
      Money mule accounts are ACTIVATED — they go from quiet to busy.
      This activation is captured as behavioral drift even when the
      absolute feature values don't exceed any rule threshold.
    """
    import numpy as np
 
    if baseline_features is None or baseline_features.empty:
        return pd.DataFrame()
    if current_features is None or current_features.empty:
        return pd.DataFrame()
 
    drift_results = []
 
    baseline_lookup = baseline_features.set_index("account_id")
    current_lookup  = current_features.set_index("account_id")
 
    # Only compare accounts present in both snapshots
    common_accounts = set(baseline_lookup.index).intersection(
        set(current_lookup.index)
    )
 
    for acc in common_accounts:
        try:
            # Get only the feature columns that exist
            valid_cols = [c for c in feature_columns
                          if c in baseline_lookup.columns
                          and c in current_lookup.columns]
 
            if not valid_cols:
                continue
 
            baseline_vec = baseline_lookup.loc[acc][valid_cols].values.astype(float)
            current_vec  = current_lookup.loc[acc][valid_cols].values.astype(float)
 
            # Replace NaN
            baseline_vec = np.nan_to_num(baseline_vec, 0)
            current_vec  = np.nan_to_num(current_vec, 0)
 
            combined = np.vstack([baseline_vec, current_vec])
            mean     = combined.mean(axis=0)
            std      = combined.std(axis=0)
            std[std == 0] = 1e-6
 
            baseline_scaled = (baseline_vec - mean) / std
            current_scaled  = (current_vec  - mean) / std
 
            drift_score = np.linalg.norm(current_scaled - baseline_scaled)
 
            if drift_score > threshold:
                # Identify which features changed most
                changes     = np.abs(current_scaled - baseline_scaled)
                top_idx     = np.argsort(changes)[::-1][:3]
                top_changed = [valid_cols[i] for i in top_idx if changes[i] > 0.3]
 
                drift_results.append({
                    "account_id":     acc,
                    "drift_score":    round(float(drift_score), 4),
                    "top_changes":    ", ".join(top_changed) if top_changed else "mixed",
                })
 
        except Exception:
            continue
 
    if not drift_results:
        return pd.DataFrame()
 
    return pd.DataFrame(drift_results).sort_values(
        "drift_score", ascending=False
    ).reset_index(drop=True)

DRIFT_SCALE_FLOOR = {
    "in_degree": 1.0,
    "out_degree": 1.0,
    "total_in_amount": 5000.0,
    "total_out_amount": 5000.0,
    "retention_ratio": 0.25,
    "unique_neighbors": 1.0,
    "unique_channels": 1.0,
    "device_cluster_size": 1.0,
    "transaction_count": 1.0,
}


def behavioral_drift_score(baseline_row, current_row, population_std, feature_columns):
    """Live drift for one account against its own earlier window.

    ``behavioral_drift_detection`` standardises each feature using only the
    two vectors being compared, so any change maps to +/-1 and the score just
    counts changed features. Here each feature difference is scaled by the
    population spread at baseline time (with a floor), giving an RMS z-score.
    Returns (score, [top changed features]).
    """
    z = []
    for column in feature_columns:
        scale = max(float(population_std.get(column, 0.0)), DRIFT_SCALE_FLOOR.get(column, 1.0))
        z.append((float(current_row.get(column, 0.0)) - float(baseline_row.get(column, 0.0))) / scale)
    z = np.array(z)
    score = float(np.sqrt(np.mean(z ** 2))) if len(z) else 0.0
    order = np.argsort(np.abs(z))[::-1]
    top = [feature_columns[i] for i in order[:3] if abs(z[i]) >= 1.0]
    return round(score, 4), top


def adaptive_threshold_update(current_threshold, report):

    if "1" not in report:
        return current_threshold

    fraud_recall = report["1"]["recall"]
    fraud_precision = report["1"]["precision"]

    # Missing mules (recall < precision) lowers the threshold; false
    # positives (precision < recall) raise it. The original sign was inverted.
    adjustment = 0.1 * (fraud_recall - fraud_precision)

    # ✅ Smooth update (momentum-based)
    target_threshold = current_threshold + adjustment
    new_threshold = 0.8 * current_threshold + 0.2 * target_threshold

    # Clamp
    new_threshold = max(0.3, min(0.8, new_threshold))

    return round(new_threshold, 3)

def load_explainer(model):
    """
    Recreate SHAP explainer for a trained model.
    This is used in the frontend after loading the saved model.
    """
    explainer = shap.TreeExplainer(model)
    return explainer