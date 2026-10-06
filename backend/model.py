"""Loads and serves the trained Random Forest, its SHAP explainer and the GAT.

The previous version required CUDA and refused to start without an NVIDIA
GPU. Inference here runs on CPU by default (``MM_TORCH_DEVICE`` overrides);
the window graph is a few hundred nodes, which a GAT handles in milliseconds.
Load failures are recorded in ``status`` and surfaced by ``/api/health``;
scores are never substituted when a model is missing.
"""

import hashlib
import json
import logging
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from backend import settings
from backend.detection import explain_risk_categories
from backend.feature_store import GNN_FEATURE_COLUMNS, RF_FEATURE_COLUMNS

logger = logging.getLogger("mule.models")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_explanations(categories):
    explanations = []
    if categories.get("Velocity Risk", 0) > 0.1:
        explanations.append("Velocity - Rapid fund movement")
    if categories.get("Shared Device Risk", 0) > 0.05:
        explanations.append("Shared Device - Multi-account control")
    if categories.get("Ring Participation Risk", 0) > 0.1:
        explanations.append("Ring - Embedded in fraud ring")
    if categories.get("Retention Risk", 0) > 0.05:
        explanations.append("Retention - Pass-through mule behavior")
    if categories.get("Channel Risk", 0) > 0.05:
        explanations.append("Channel - Multi-channel burst activity")
    if not explanations:
        explanations.append("Flagged by structural position in the transaction graph.")
    return explanations


class ModelRuntime:
    def __init__(self):
        self.rf_model = None
        self.explainer = None
        self.gnn_model = None
        self.torch = None
        self.device = os.environ.get("MM_TORCH_DEVICE", "cpu")
        self.meta = {}
        self.status = {"rf": "not_loaded", "gnn": "not_loaded", "shap": "not_loaded", "meta": "not_loaded"}
        self.rf_latency_ms = 0.0
        self.gnn_latency_ms = 0.0

    @property
    def rf_ready(self):
        return self.rf_model is not None

    @property
    def gnn_ready(self):
        return self.gnn_model is not None

    def load(self):
        self._load_meta()
        self._load_rf()
        self._load_gnn()

    def _load_meta(self):
        path = settings.MODEL_META_PATH
        if not path.exists():
            self.status["meta"] = "missing: run `python -m backend.train_live` to train live-window models"
            logger.warning("model metadata missing at %s", path)
            return
        self.meta = json.loads(path.read_text())
        trained_window = self.meta.get("feature_window_sec")
        if trained_window is not None and abs(float(trained_window) - settings.FEATURE_WINDOW_SEC) > 1e-6:
            self.status["meta"] = (
                f"window mismatch: models trained on {trained_window}s, runtime uses {settings.FEATURE_WINDOW_SEC}s"
            )
            logger.warning(self.status["meta"])
        else:
            self.status["meta"] = "ok"

    def _load_rf(self):
        path = settings.RF_MODEL_PATH
        try:
            model = joblib.load(path)
            features = list(getattr(model, "feature_list", RF_FEATURE_COLUMNS))
            if features != RF_FEATURE_COLUMNS:
                raise ValueError(f"feature order {features} != expected {RF_FEATURE_COLUMNS}")
            if getattr(model, "n_features_in_", len(RF_FEATURE_COLUMNS)) != len(RF_FEATURE_COLUMNS):
                raise ValueError(f"model expects {model.n_features_in_} features")
            expected_hash = self.meta.get("rf_sha256")
            if expected_hash and expected_hash != file_sha256(path):
                logger.warning("fraud_model.pkl does not match model_meta.json (retrained outside train_live?)")
            model.feature_list = features
            self.rf_model = model
            self.status["rf"] = "loaded"
            logger.info("random forest loaded path=%s trees=%s", path, len(model.estimators_))
        except Exception as exc:
            self.rf_model = None
            self.status["rf"] = f"error: {exc}"
            logger.error("random forest failed to load: %s", exc)
            return
        try:
            import shap

            self.explainer = shap.TreeExplainer(self.rf_model)
            self.status["shap"] = "ready"
        except Exception as exc:
            self.explainer = None
            self.status["shap"] = f"error: {exc}"
            logger.error("SHAP explainer unavailable: %s", exc)

    def _load_gnn(self):
        path = settings.GNN_MODEL_PATH
        if not path.exists():
            self.status["gnn"] = f"unavailable: {path.name} not found"
            logger.warning("GNN model missing at %s; fusion falls back to RF + rules", path)
            return
        try:
            import torch

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)  # torch.jit notice from torch_geometric
                from backend.gnn import FraudGAT

            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
            model = FraudGAT(in_channels=len(GNN_FEATURE_COLUMNS)).to(self.device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state = torch.load(path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
                state = state["state_dict"]
            model.load_state_dict(state)
            model.eval()
            self.torch = torch
            self.gnn_model = model
            self.status["gnn"] = "loaded"
            logger.info("GNN (GAT) loaded path=%s device=%s", path, self.device)
        except Exception as exc:
            self.gnn_model = None
            self.status["gnn"] = f"error: {exc}"
            logger.error("GNN failed to load; fusion falls back to RF + rules: %s", exc)

    # -- inference -----------------------------------------------------------
    def rf_predict(self, rows):
        if self.rf_model is None:
            raise RuntimeError(f"random forest not available ({self.status['rf']})")
        started = time.perf_counter()
        frame = pd.DataFrame([[row[c] for c in RF_FEATURE_COLUMNS] for row in rows], columns=RF_FEATURE_COLUMNS)
        scores = self.rf_model.predict_proba(frame)[:, 1]
        self.rf_latency_ms = (time.perf_counter() - started) * 1000
        return [float(s) for s in scores]

    def gnn_predict(self, x, edge_index):
        if self.gnn_model is None:
            return None
        started = time.perf_counter()
        torch = self.torch
        with torch.inference_mode():
            scores = self.gnn_model(
                torch.from_numpy(x).to(self.device),
                torch.from_numpy(edge_index).to(self.device),
            )
        self.gnn_latency_ms = (time.perf_counter() - started) * 1000
        return scores.float().cpu().numpy()

    def explain(self, row):
        """SHAP attribution of the RF score for one feature row."""
        if self.explainer is None:
            return {"available": False, "reason": self.status["shap"]}
        frame = pd.DataFrame([[row[c] for c in RF_FEATURE_COLUMNS]], columns=RF_FEATURE_COLUMNS)
        values = self.explainer.shap_values(frame)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[0][:, 1]
        elif values.ndim == 2:
            values = values[0]
        categories = explain_risk_categories(list(values), RF_FEATURE_COLUMNS)
        return {
            "available": True,
            "features": [
                {"feature": c, "value": row[c], "shap": round(float(v), 4)}
                for c, v in sorted(zip(RF_FEATURE_COLUMNS, values), key=lambda kv: abs(kv[1]), reverse=True)
            ],
            "categories": [
                {"category": k, "contribution": round(float(v) * 100, 2)}
                for k, v in sorted(categories.items(), key=lambda kv: kv[1], reverse=True)
            ],
            "explanations": build_explanations(categories),
        }

    def health(self):
        return {
            "rf_model": self.status["rf"],
            "gnn_model": self.status["gnn"],
            "shap": self.status["shap"],
            "model_meta": self.status["meta"],
            "model_device": self.device,
            "trained_at": self.meta.get("trained_at"),
            "trained_window_sec": self.meta.get("feature_window_sec"),
            "rf_eval": self.meta.get("rf_eval"),
            "gnn_eval": self.meta.get("gnn_eval"),
            "rf_latency_ms": round(self.rf_latency_ms, 2),
            "gnn_latency_ms": round(self.gnn_latency_ms, 2),
        }
