from pathlib import Path
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
FEATURE_DIR = ROOT / "data" / "processed" / "features"
TABLE_DIR = ROOT / "outputs" / "tables"
METRICS_DIR = ROOT / "outputs" / "metrics"
MODEL_DIR = ROOT / "outputs" / "models"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEED = 42
TARGET = "risk_stage"
CLASS_ORDER = ["critical", "warning", "normal"]

def load_feature_columns():
    path = TABLE_DIR / "model_feature_columns.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature column file: {path}")
    cols = path.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in cols if c.strip()]
    if not cols:
        raise ValueError("Feature column list is empty.")
    return cols

def build_models():
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                multi_class="auto"
            ))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            loss_function="MultiClass",
            random_seed=SEED,
            verbose=False
        )
    }
    return models

def safe_predict_proba(model, X_test, n_classes):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] == n_classes:
            return proba
    return None

def evaluate_classification(y_true, y_pred, y_proba=None):
    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    result = {
        "Accuracy": round(float(acc), 6),
        "Precision_Macro": round(float(precision_macro), 6),
        "Recall_Macro": round(float(recall_macro), 6),
        "F1_Macro": round(float(f1_macro), 6),
        "Precision_Weighted": round(float(precision_weighted), 6),
        "Recall_Weighted": round(float(recall_weighted), 6),
        "F1_Weighted": round(float(f1_weighted), 6),
    }

    if y_proba is not None:
        try:
            y_onehot = np.eye(y_proba.shape[1])[y_true]
            result["ROC_AUC_OVR"] = round(float(roc_auc_score(y_onehot, y_proba, average="macro", multi_class="ovr")), 6)
            result["PR_AUC_Macro"] = round(float(average_precision_score(y_onehot, y_proba, average="macro")), 6)
        except Exception:
            result["ROC_AUC_OVR"] = np.nan
            result["PR_AUC_Macro"] = np.nan
    else:
        result["ROC_AUC_OVR"] = np.nan
        result["PR_AUC_Macro"] = np.nan

    return result

def train_eval_subset(subset, feature_cols):
    train_path = FEATURE_DIR / f"{subset}_train_features.csv"
    test_path = FEATURE_DIR / f"{subset}_test_features.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train feature file: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test feature file: {test_path}")

    print(f"\n[{subset}] Loading feature files")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df[feature_cols].astype("float32")
    X_test = test_df[feature_cols].astype("float32")

    encoder = LabelEncoder()
    encoder.fit(CLASS_ORDER)

    y_train = encoder.transform(train_df[TARGET].astype(str))
    y_test = encoder.transform(test_df[TARGET].astype(str))

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Classes    : {list(encoder.classes_)}")

    results = []
    models = build_models()

    for model_name, model in models.items():
        print(f"  Training {model_name}...")

        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train

        start_pred = time.perf_counter()
        y_pred = model.predict(X_test)
        y_pred = np.asarray(y_pred).reshape(-1).astype(int)
        y_proba = safe_predict_proba(model, X_test, len(encoder.classes_))
        infer_time = time.perf_counter() - start_pred

        metrics = evaluate_classification(y_test, y_pred, y_proba)

        row = {
            "subset": subset,
            "target": TARGET,
            "model": model_name,
            **metrics,
            "train_time_sec": round(float(train_time), 6),
            "inference_time_sec": round(float(infer_time), 6),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "num_features": int(len(feature_cols)),
        }

        results.append(row)

        model_path = MODEL_DIR / f"{subset}_{model_name}_risk_stage.joblib"
        joblib.dump({
            "model": model,
            "label_encoder_classes": list(encoder.classes_),
            "feature_cols": feature_cols,
        }, model_path)

        pred_out = pd.DataFrame({
            "subset": subset,
            "unit_id": test_df["unit_id"].values,
            "cycle": test_df["cycle"].values,
            "y_true": encoder.inverse_transform(y_test),
            "y_pred": encoder.inverse_transform(y_pred),
            "model": model_name,
        })

        if y_proba is not None:
            for idx, cls_name in enumerate(encoder.classes_):
                pred_out[f"proba_{cls_name}"] = y_proba[:, idx]

        pred_path = METRICS_DIR / f"{subset}_{model_name}_risk_predictions.csv"
        pred_out.to_csv(pred_path, index=False)

        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(encoder.classes_)))
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in encoder.classes_],
            columns=[f"pred_{c}" for c in encoder.classes_]
        )
        cm_path = METRICS_DIR / f"{subset}_{model_name}_confusion_matrix.csv"
        cm_df.to_csv(cm_path)

        print(
            f"    Acc={row['Accuracy']:.4f}, Macro-F1={row['F1_Macro']:.4f}, "
            f"Macro-Recall={row['Recall_Macro']:.4f}, ROC-AUC={row['ROC_AUC_OVR']}, "
            f"train={row['train_time_sec']:.2f}s"
        )

    return results

def main():
    print("=" * 80)
    print("TRAIN ML BASELINES - RISK-STAGE CLASSIFICATION")
    print("=" * 80)

    feature_cols = load_feature_columns()
    print(f"Loaded model features: {len(feature_cols)}")

    all_results = []

    for subset in SUBSETS:
        subset_results = train_eval_subset(subset, feature_cols)
        all_results.extend(subset_results)

        partial_df = pd.DataFrame(all_results)
        partial_df.to_csv(TABLE_DIR / "ml_risk_classification_results_partial.csv", index=False)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["subset", "F1_Macro", "Accuracy"], ascending=[True, False, False]).reset_index(drop=True)

    out_csv = TABLE_DIR / "ml_risk_classification_results.csv"
    out_json = TABLE_DIR / "ml_risk_classification_results.json"

    results_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n[Saved]")
    print(out_csv)
    print(out_json)

    print("\n[Best model per subset by Macro-F1]")
    best = results_df.loc[results_df.groupby("subset")["F1_Macro"].idxmax()]
    print(best[["subset", "model", "Accuracy", "Precision_Macro", "Recall_Macro", "F1_Macro", "ROC_AUC_OVR", "PR_AUC_Macro"]].to_string(index=False))

    print("\n[Final status]")
    print("STATUS: RISK_CLASSIFICATION_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
