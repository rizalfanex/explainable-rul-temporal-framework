from pathlib import Path
import json
import sys
import pandas as pd
import numpy as np

ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw" / "cmapss"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"
LOG_DIR = ROOT / "outputs" / "logs"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
RUL_CAP = 125

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def read_cmapss_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    if df.shape[1] > len(COLUMNS):
        df = df.iloc[:, :len(COLUMNS)]

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Unexpected column count in {path.name}: got {df.shape[1]}, expected {len(COLUMNS)}"
        )

    df.columns = COLUMNS
    return df

def read_rul_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :1]
    df.columns = ["true_rul_after_last_cycle"]
    df["unit_id"] = np.arange(1, len(df) + 1)
    return df[["unit_id", "true_rul_after_last_cycle"]]

def assign_risk_stage(rul: float) -> str:
    if rul <= 50:
        return "critical"
    if rul <= 125:
        return "warning"
    return "normal"

def build_train_labels(train_df: pd.DataFrame, subset: str) -> pd.DataFrame:
    df = train_df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.merge(max_cycle, on="unit_id", how="left")

    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL_capped"] = df["RUL"].clip(upper=RUL_CAP)
    df["risk_stage"] = df["RUL"].apply(assign_risk_stage)

    df["subset"] = subset
    df["split"] = "train"

    return df

def build_test_labels(test_df: pd.DataFrame, rul_df: pd.DataFrame, subset: str) -> pd.DataFrame:
    df = test_df.copy()

    observed_max_cycle = df.groupby("unit_id")["cycle"].max().rename("observed_max_cycle")
    df = df.merge(observed_max_cycle, on="unit_id", how="left")
    df = df.merge(rul_df, on="unit_id", how="left")

    if df["true_rul_after_last_cycle"].isna().any():
        missing_units = df.loc[df["true_rul_after_last_cycle"].isna(), "unit_id"].unique().tolist()
        raise ValueError(f"Missing test RUL for units in {subset}: {missing_units[:10]}")

    df["max_cycle"] = df["observed_max_cycle"] + df["true_rul_after_last_cycle"]
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL_capped"] = df["RUL"].clip(upper=RUL_CAP)
    df["risk_stage"] = df["RUL"].apply(assign_risk_stage)

    df["subset"] = subset
    df["split"] = "test"

    return df

def summarize_labels(df: pd.DataFrame, subset: str, split: str) -> dict:
    stage_counts = df["risk_stage"].value_counts().to_dict()

    summary = {
        "subset": subset,
        "split": split,
        "rows": int(len(df)),
        "units": int(df["unit_id"].nunique()),
        "rul_min": float(df["RUL"].min()),
        "rul_max": float(df["RUL"].max()),
        "rul_mean": round(float(df["RUL"].mean()), 4),
        "rul_capped_min": float(df["RUL_capped"].min()),
        "rul_capped_max": float(df["RUL_capped"].max()),
        "rul_capped_mean": round(float(df["RUL_capped"].mean()), 4),
        "normal_count": int(stage_counts.get("normal", 0)),
        "warning_count": int(stage_counts.get("warning", 0)),
        "critical_count": int(stage_counts.get("critical", 0)),
    }

    return summary

def main():
    print("=" * 80)
    print("BUILD RUL LABELS - NASA C-MAPSS")
    print("=" * 80)

    all_train = []
    all_test = []
    summaries = []

    for subset in SUBSETS:
        print(f"\n[{subset}] Processing")

        train_path = RAW_DIR / f"train_{subset}.txt"
        test_path = RAW_DIR / f"test_{subset}.txt"
        rul_path = RAW_DIR / f"RUL_{subset}.txt"

        train_raw = read_cmapss_file(train_path)
        test_raw = read_cmapss_file(test_path)
        rul_df = read_rul_file(rul_path)

        train_labeled = build_train_labels(train_raw, subset)
        test_labeled = build_test_labels(test_raw, rul_df, subset)

        train_out = PROCESSED_DIR / f"{subset}_train_with_rul.csv"
        test_out = PROCESSED_DIR / f"{subset}_test_with_rul.csv"

        train_labeled.to_csv(train_out, index=False)
        test_labeled.to_csv(test_out, index=False)

        all_train.append(train_labeled)
        all_test.append(test_labeled)

        train_summary = summarize_labels(train_labeled, subset, "train")
        test_summary = summarize_labels(test_labeled, subset, "test")
        summaries.extend([train_summary, test_summary])

        print(f"Saved train: {train_out}")
        print(f"Saved test : {test_out}")
        print(f"Train RUL range: {train_summary['rul_min']} to {train_summary['rul_max']}")
        print(f"Test  RUL range: {test_summary['rul_min']} to {test_summary['rul_max']}")
        print(
            "Train risk-stage counts: "
            f"normal={train_summary['normal_count']}, "
            f"warning={train_summary['warning_count']}, "
            f"critical={train_summary['critical_count']}"
        )
        print(
            "Test risk-stage counts : "
            f"normal={test_summary['normal_count']}, "
            f"warning={test_summary['warning_count']}, "
            f"critical={test_summary['critical_count']}"
        )

    combined_train = pd.concat(all_train, axis=0, ignore_index=True)
    combined_test = pd.concat(all_test, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    combined_train_out = PROCESSED_DIR / "cmapss_all_train_with_rul.csv"
    combined_test_out = PROCESSED_DIR / "cmapss_all_test_with_rul.csv"
    summary_out = TABLE_DIR / "rul_label_summary.csv"
    summary_json_out = TABLE_DIR / "rul_label_summary.json"

    combined_train.to_csv(combined_train_out, index=False)
    combined_test.to_csv(combined_test_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    with open(summary_json_out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\n[Combined files]")
    print(combined_train_out)
    print(combined_test_out)

    print("\n[Summary files]")
    print(summary_out)
    print(summary_json_out)

    print("\n[Final status]")
    print("STATUS: RUL_LABELS_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
