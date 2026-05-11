from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd

ROOT = Path.cwd()
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_DIR = ROOT / "data" / "processed" / "features"
TABLE_DIR = ROOT / "outputs" / "tables"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = [f"op_setting_{i}" for i in range(1, 4)]
WINDOWS = [5, 10, 20]

ID_COLS = ["subset", "split", "unit_id", "cycle"]
TARGET_COLS = ["RUL", "RUL_capped", "risk_stage"]
LEAKAGE_COLS = ["max_cycle", "observed_max_cycle", "true_rul_after_last_cycle"]

def add_temporal_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    feature_dict = {}

    # Keep identifiers and targets.
    for col in ID_COLS + TARGET_COLS:
        feature_dict[col] = df[col].values

    # Observable base features.
    for col in OP_COLS + SENSOR_COLS:
        feature_dict[col] = df[col].astype("float32").values

    # Cycle-only observable features.
    feature_dict["cycle_log1p"] = np.log1p(df["cycle"].values).astype("float32")
    feature_dict["cycle_sqrt"] = np.sqrt(df["cycle"].values).astype("float32")

    grouped = df.groupby("unit_id", group_keys=False)

    # Lag/delta/pct-change.
    for col in SENSOR_COLS:
        lag1 = grouped[col].shift(1)
        delta1 = df[col] - lag1
        pct1 = delta1 / (lag1.abs() + 1e-8)

        feature_dict[f"{col}_lag1"] = lag1.fillna(0.0).astype("float32").values
        feature_dict[f"{col}_delta1"] = delta1.fillna(0.0).astype("float32").values
        feature_dict[f"{col}_pct_change1"] = pct1.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32").values

    # Rolling statistics: fast and scientifically useful.
    # Slope is intentionally excluded here because rolling polyfit is expensive.
    for window in WINDOWS:
        print(f"  Rolling window = {window}")

        for col in SENSOR_COLS:
            roll = grouped[col].rolling(window=window, min_periods=1)

            mean_s = roll.mean().reset_index(level=0, drop=True).astype("float32")
            std_s = roll.std().reset_index(level=0, drop=True).fillna(0.0).astype("float32")
            min_s = roll.min().reset_index(level=0, drop=True).astype("float32")
            max_s = roll.max().reset_index(level=0, drop=True).astype("float32")
            range_s = (max_s - min_s).astype("float32")

            feature_dict[f"{col}_roll{window}_mean"] = mean_s.values
            feature_dict[f"{col}_roll{window}_std"] = std_s.values
            feature_dict[f"{col}_roll{window}_min"] = min_s.values
            feature_dict[f"{col}_roll{window}_max"] = max_s.values
            feature_dict[f"{col}_roll{window}_range"] = range_s.values

    out = pd.DataFrame(feature_dict)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out

def get_model_feature_columns(df: pd.DataFrame) -> list:
    excluded = set(ID_COLS + TARGET_COLS + LEAKAGE_COLS)
    return [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]

def summarize(df: pd.DataFrame, subset: str, split: str, feature_cols: list) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    return {
        "subset": subset,
        "split": split,
        "rows": int(len(df)),
        "units": int(df["unit_id"].nunique()),
        "num_total_columns": int(df.shape[1]),
        "num_model_features": int(len(feature_cols)),
        "missing_values": int(df.isna().sum().sum()),
        "infinite_values": int(np.isinf(numeric_df).sum().sum()),
        "rul_min": float(df["RUL"].min()),
        "rul_max": float(df["RUL"].max()),
        "rul_capped_min": float(df["RUL_capped"].min()),
        "rul_capped_max": float(df["RUL_capped"].max()),
        "critical_count": int((df["risk_stage"] == "critical").sum()),
        "warning_count": int((df["risk_stage"] == "warning").sum()),
        "normal_count": int((df["risk_stage"] == "normal").sum()),
    }

def process_one(subset: str, split: str):
    start = time.time()
    in_path = PROCESSED_DIR / f"{subset}_{split}_with_rul.csv"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    print(f"\n[{subset} {split}] Reading {in_path}")
    df = pd.read_csv(in_path)

    required = set(ID_COLS + OP_COLS + SENSOR_COLS + TARGET_COLS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {in_path.name}: {missing}")

    feat_df = add_temporal_features_fast(df)
    feature_cols = get_model_feature_columns(feat_df)

    out_path = FEATURE_DIR / f"{subset}_{split}_features.csv"
    feat_df.to_csv(out_path, index=False)

    elapsed = time.time() - start
    summary = summarize(feat_df, subset, split, feature_cols)

    print(f"Saved: {out_path}")
    print(f"Rows: {summary['rows']}")
    print(f"Total columns: {summary['num_total_columns']}")
    print(f"Model features: {summary['num_model_features']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Infinite values: {summary['infinite_values']}")
    print(f"Elapsed seconds: {elapsed:.2f}")

    return feat_df, summary, feature_cols

def write_feature_metadata(feature_cols: list):
    rows = []
    for col in feature_cols:
        if col in OP_COLS:
            category = "operational_setting"
        elif col in SENSOR_COLS:
            category = "raw_sensor"
        elif col.startswith("cycle_"):
            category = "cycle_observable"
        elif "_lag1" in col:
            category = "lag"
        elif "_delta1" in col:
            category = "delta"
        elif "_pct_change1" in col:
            category = "percentage_change"
        elif "_roll" in col and "_mean" in col:
            category = "rolling_mean"
        elif "_roll" in col and "_std" in col:
            category = "rolling_std"
        elif "_roll" in col and "_min" in col:
            category = "rolling_min"
        elif "_roll" in col and "_max" in col:
            category = "rolling_max"
        elif "_roll" in col and "_range" in col:
            category = "rolling_range"
        else:
            category = "other"

        rows.append({"feature": col, "category": category})

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(TABLE_DIR / "feature_metadata.csv", index=False)

    with open(TABLE_DIR / "model_feature_columns.txt", "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(col + "\n")

def main():
    print("=" * 80)
    print("TEMPORAL FEATURE ENGINEERING - NASA C-MAPSS - FAST VERSION")
    print("=" * 80)

    all_train = []
    all_test = []
    summaries = []
    reference_cols = None

    for subset in SUBSETS:
        train_df, train_summary, train_cols = process_one(subset, "train")
        test_df, test_summary, test_cols = process_one(subset, "test")

        if train_cols != test_cols:
            raise ValueError(f"Train/test feature mismatch in {subset}")

        if reference_cols is None:
            reference_cols = train_cols
        elif reference_cols != train_cols:
            raise ValueError(f"Cross-subset feature mismatch at {subset}")

        all_train.append(train_df)
        all_test.append(test_df)
        summaries.append(train_summary)
        summaries.append(test_summary)

    print("\n[Combining all subsets]")
    combined_train = pd.concat(all_train, axis=0, ignore_index=True)
    combined_test = pd.concat(all_test, axis=0, ignore_index=True)

    combined_train_out = FEATURE_DIR / "cmapss_all_train_features.csv"
    combined_test_out = FEATURE_DIR / "cmapss_all_test_features.csv"

    combined_train.to_csv(combined_train_out, index=False)
    combined_test.to_csv(combined_test_out, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(TABLE_DIR / "feature_engineering_summary.csv", index=False)

    with open(TABLE_DIR / "feature_engineering_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    write_feature_metadata(reference_cols)

    print("\n[Saved combined files]")
    print(combined_train_out)
    print(combined_test_out)

    print("\n[Saved metadata]")
    print(TABLE_DIR / "feature_engineering_summary.csv")
    print(TABLE_DIR / "feature_metadata.csv")
    print(TABLE_DIR / "model_feature_columns.txt")

    print("\n[Final status]")
    print("STATUS: TEMPORAL_FEATURES_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
