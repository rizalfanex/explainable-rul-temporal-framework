from pathlib import Path
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
FEATURE_DIR = ROOT / "data" / "processed" / "features"
TABLE_DIR = ROOT / "outputs" / "tables"
METRICS_DIR = ROOT / "outputs" / "metrics"
MODEL_DIR = ROOT / "outputs" / "models"
LOG_DIR = ROOT / "outputs" / "logs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEED = 42
TARGET = "RUL_capped"

def nasa_score(y_true, y_pred):
    """
    NASA scoring function.
    Late prediction and early prediction are penalized asymmetrically.
    d = predicted - true
    d < 0 means early prediction.
    d >= 0 means late prediction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true

    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0
    )
    return float(np.sum(score))

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
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=SEED))
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=SEED,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1
        ),
        "CatBoost": CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            random_seed=SEED,
            verbose=False
        )
    }
    return models

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, 125)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    score = nasa_score(y_true, y_pred)

    return {
        "MAE": round(float(mae), 6),
        "RMSE": round(float(rmse), 6),
        "R2": round(float(r2), 6),
        "NASA_Score": round(float(score), 6),
    }

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

    missing_train = sorted(set(feature_cols) - set(train_df.columns))
    missing_test = sorted(set(feature_cols) - set(test_df.columns))
    if missing_train or missing_test:
        raise ValueError(f"Missing feature columns. train={missing_train[:5]}, test={missing_test[:5]}")

    X_train = train_df[feature_cols].astype("float32")
    y_train = train_df[TARGET].astype("float32")
    X_test = test_df[feature_cols].astype("float32")
    y_test = test_df[TARGET].astype("float32")

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    results = []
    models = build_models()

    for model_name, model in models.items():
        print(f"  Training {model_name}...")

        start_train = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_train

        start_pred = time.perf_counter()
        pred = model.predict(X_test)
        infer_time = time.perf_counter() - start_pred

        metrics = evaluate(y_test, pred)

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

        model_path = MODEL_DIR / f"{subset}_{model_name}_rul_capped.joblib"
        joblib.dump(model, model_path)

        pred_out = pd.DataFrame({
            "subset": subset,
            "unit_id": test_df["unit_id"].values,
            "cycle": test_df["cycle"].values,
            "y_true": y_test.values,
            "y_pred": np.clip(pred, 0, 125),
            "model": model_name,
        })
        pred_path = METRICS_DIR / f"{subset}_{model_name}_predictions.csv"
        pred_out.to_csv(pred_path, index=False)

        print(
            f"    MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, "
            f"R2={row['R2']:.4f}, NASA={row['NASA_Score']:.2f}, "
            f"train={row['train_time_sec']:.2f}s, infer={row['inference_time_sec']:.2f}s"
        )

    return results

def main():
    print("=" * 80)
    print("TRAIN ML BASELINES - RUL REGRESSION")
    print("=" * 80)

    feature_cols = load_feature_columns()
    print(f"Loaded model features: {len(feature_cols)}")

    all_results = []

    for subset in SUBSETS:
        subset_results = train_eval_subset(subset, feature_cols)
        all_results.extend(subset_results)

        partial_df = pd.DataFrame(all_results)
        partial_df.to_csv(TABLE_DIR / "ml_baseline_rul_results_partial.csv", index=False)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["subset", "RMSE", "MAE"]).reset_index(drop=True)

    out_csv = TABLE_DIR / "ml_baseline_rul_results.csv"
    out_json = TABLE_DIR / "ml_baseline_rul_results.json"

    results_df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n[Saved]")
    print(out_csv)
    print(out_json)

    print("\n[Best model per subset by RMSE]")
    best = results_df.loc[results_df.groupby("subset")["RMSE"].idxmin()]
    print(best[["subset", "model", "MAE", "RMSE", "R2", "NASA_Score"]].to_string(index=False))

    print("\n[Final status]")
    print("STATUS: ML_BASELINES_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
