from pathlib import Path
import pandas as pd
import json
import sys

ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "raw" / "cmapss"
OUT_DIR = ROOT / "outputs" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def read_cmapss_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python"
    )

    if df.shape[1] > len(COLUMNS):
        df = df.iloc[:, :len(COLUMNS)]

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Unexpected column count in {path.name}: "
            f"got {df.shape[1]}, expected {len(COLUMNS)}"
        )

    df.columns = COLUMNS
    return df

def read_rul_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :1]
    df.columns = ["true_rul"]
    df["unit_id"] = range(1, len(df) + 1)
    return df[["unit_id", "true_rul"]]

def audit_subset(subset: str) -> dict:
    train_path = RAW_DIR / f"train_{subset}.txt"
    test_path = RAW_DIR / f"test_{subset}.txt"
    rul_path = RAW_DIR / f"RUL_{subset}.txt"

    train_df = read_cmapss_file(train_path)
    test_df = read_cmapss_file(test_path)
    rul_df = read_rul_file(rul_path)

    train_max_cycle = train_df.groupby("unit_id")["cycle"].max()
    test_max_cycle = test_df.groupby("unit_id")["cycle"].max()

    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    op_cols = [c for c in train_df.columns if c.startswith("op_setting_")]

    constant_sensors = []
    low_variance_sensors = []

    for c in sensor_cols:
        nunique = train_df[c].nunique(dropna=False)
        std = float(train_df[c].std())
        if nunique <= 1:
            constant_sensors.append(c)
        if std < 1e-8:
            low_variance_sensors.append(c)

    audit = {
        "subset": subset,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "rul_rows": int(len(rul_df)),
        "train_units": int(train_df["unit_id"].nunique()),
        "test_units": int(test_df["unit_id"].nunique()),
        "train_cycle_min": int(train_max_cycle.min()),
        "train_cycle_max": int(train_max_cycle.max()),
        "train_cycle_mean": round(float(train_max_cycle.mean()), 4),
        "test_cycle_min": int(test_max_cycle.min()),
        "test_cycle_max": int(test_max_cycle.max()),
        "test_cycle_mean": round(float(test_max_cycle.mean()), 4),
        "missing_values_train": int(train_df.isna().sum().sum()),
        "missing_values_test": int(test_df.isna().sum().sum()),
        "duplicate_rows_train": int(train_df.duplicated().sum()),
        "duplicate_rows_test": int(test_df.duplicated().sum()),
        "num_operational_settings": len(op_cols),
        "num_sensors": len(sensor_cols),
        "constant_sensors_train": ",".join(constant_sensors) if constant_sensors else "None",
        "low_variance_sensors_train": ",".join(low_variance_sensors) if low_variance_sensors else "None",
    }

    return audit

def main():
    print("=" * 80)
    print("NASA C-MAPSS DATASET AUDIT")
    print("=" * 80)
    print(f"Project root : {ROOT}")
    print(f"Raw data dir : {RAW_DIR}")

    existing_files = sorted([p.name for p in RAW_DIR.glob("*")])
    print("\n[Files found]")
    if existing_files:
        for name in existing_files:
            print(f"- {name}")
    else:
        print("No files found in data/raw/cmapss")

    all_audits = []
    missing_any = False

    for subset in SUBSETS:
        required = [
            RAW_DIR / f"train_{subset}.txt",
            RAW_DIR / f"test_{subset}.txt",
            RAW_DIR / f"RUL_{subset}.txt",
        ]

        missing = [str(p) for p in required if not p.exists()]
        if missing:
            missing_any = True
            print(f"\n[{subset}] MISSING")
            for m in missing:
                print(f"- {m}")
            continue

        print(f"\n[{subset}] Auditing...")
        audit = audit_subset(subset)
        all_audits.append(audit)

        for k, v in audit.items():
            print(f"{k}: {v}")

    out_json = OUT_DIR / "cmapss_audit_summary.json"
    out_csv = OUT_DIR / "cmapss_audit_summary.csv"

    if all_audits:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_audits, f, indent=2)

        pd.DataFrame(all_audits).to_csv(out_csv, index=False)
        print("\n[Saved]")
        print(out_json)
        print(out_csv)

    print("\n[Final status]")
    if missing_any:
        print("STATUS: DATASET_FILES_MISSING")
        print("Action: place train_FD001.txt, test_FD001.txt, RUL_FD001.txt, etc. into data/raw/cmapss")
    else:
        print("STATUS: DATASET_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
