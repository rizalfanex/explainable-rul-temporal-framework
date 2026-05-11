from pathlib import Path
import sys
import time
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
FEATURE_DIR = ROOT / "data" / "processed" / "features"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEQ_COLS = [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
TARGET = "RUL_capped"

SEED = 42
WINDOW = 30
STRIDE_TRAIN = 3
STRIDE_TEST = 3
BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
HIDDEN = 64
TAB_HIDDEN = 128

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def nasa_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))

def evaluate(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, 125)
    y_true = np.asarray(y_true, dtype=float)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 6),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 6),
        "R2": round(float(r2_score(y_true, y_pred)), 6),
        "NASA_Score": round(float(nasa_score(y_true, y_pred)), 6),
    }

def load_feature_columns():
    path = TABLE_DIR / "model_feature_columns.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature list: {path}")
    cols = path.read_text(encoding="utf-8").splitlines()
    return [c.strip() for c in cols if c.strip()]

def build_hybrid_windows(df, seq_cols, tab_cols, target_col, window, stride):
    Xseq, Xtab, ylist = [], [], []
    for _, g in df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        seq_arr = g[seq_cols].values.astype("float32")
        tab_arr = g[tab_cols].values.astype("float32")
        y = g[target_col].values.astype("float32")
        if len(g) < window:
            continue
        for end in range(window - 1, len(g), stride):
            start = end - window + 1
            Xseq.append(seq_arr[start:end + 1])
            Xtab.append(tab_arr[end])
            ylist.append(y[end])
    return (
        np.asarray(Xseq, dtype=np.float32),
        np.asarray(Xtab, dtype=np.float32),
        np.asarray(ylist, dtype=np.float32),
    )

class HybridGRUFeatureRegressor(nn.Module):
    def __init__(self, seq_dim, tab_dim, hidden=64, tab_hidden=128):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, batch_first=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.tab = nn.Sequential(
            nn.Linear(tab_dim, tab_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(tab_hidden),
            nn.Dropout(0.15),
            nn.Linear(tab_hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 1),
        )

    def forward(self, xseq, xtab):
        h, _ = self.gru(xseq)
        weights = torch.softmax(self.attn(h).squeeze(-1), dim=1)
        seq_emb = torch.sum(h * weights.unsqueeze(-1), dim=1)
        tab_emb = self.tab(xtab)
        z = torch.cat([seq_emb, tab_emb], dim=1)
        return self.head(z).squeeze(-1)

def train_model(model, Xseq_train, Xtab_train, y_train, Xseq_test, Xtab_test, device):
    ds = TensorDataset(
        torch.tensor(Xseq_train),
        torch.tensor(Xtab_train),
        torch.tensor(y_train),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb_seq, xb_tab, yb in loader:
            xb_seq = xb_seq.to(device)
            xb_tab = xb_tab.to(device)
            yb = yb.to(device)

            pred = model(xb_seq, xb_tab)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        if epoch in [1, 5, 10, 15, EPOCHS]:
            print(f"      epoch={epoch:02d}, train_mse={np.mean(losses):.4f}")

    model.eval()
    preds = []
    test_ds = TensorDataset(torch.tensor(Xseq_test), torch.tensor(Xtab_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for xb_seq, xb_tab in test_loader:
            xb_seq = xb_seq.to(device)
            xb_tab = xb_tab.to(device)
            pred = model(xb_seq, xb_tab).detach().cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds)
    return model, preds

def process_subset(subset, feature_cols, device):
    train_path = FEATURE_DIR / f"{subset}_train_features.csv"
    test_path = FEATURE_DIR / f"{subset}_test_features.csv"

    print(f"\n[{subset}] Loading feature files")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    seq_scaler = StandardScaler()
    seq_scaler.fit(train_df[SEQ_COLS].astype("float32"))
    train_df[SEQ_COLS] = seq_scaler.transform(train_df[SEQ_COLS].astype("float32")).astype("float32")
    test_df[SEQ_COLS] = seq_scaler.transform(test_df[SEQ_COLS].astype("float32")).astype("float32")

    tab_scaler = StandardScaler()
    tab_scaler.fit(train_df[feature_cols].astype("float32"))
    train_df[feature_cols] = tab_scaler.transform(train_df[feature_cols].astype("float32")).astype("float32")
    test_df[feature_cols] = tab_scaler.transform(test_df[feature_cols].astype("float32")).astype("float32")

    Xseq_train, Xtab_train, y_train = build_hybrid_windows(
        train_df, SEQ_COLS, feature_cols, TARGET, WINDOW, STRIDE_TRAIN
    )
    Xseq_test, Xtab_test, y_test = build_hybrid_windows(
        test_df, SEQ_COLS, feature_cols, TARGET, WINDOW, STRIDE_TEST
    )

    print(f"  Train seq windows: {Xseq_train.shape}")
    print(f"  Train tab windows: {Xtab_train.shape}")
    print(f"  Test seq windows : {Xseq_test.shape}")
    print(f"  Test tab windows : {Xtab_test.shape}")

    model = HybridGRUFeatureRegressor(
        seq_dim=len(SEQ_COLS),
        tab_dim=len(feature_cols),
        hidden=HIDDEN,
        tab_hidden=TAB_HIDDEN,
    )

    start = time.perf_counter()
    trained, pred = train_model(model, Xseq_train, Xtab_train, y_train, Xseq_test, Xtab_test, device)
    train_time = time.perf_counter() - start

    metrics = evaluate(y_test, pred)

    row = {
        "subset": subset,
        "target": TARGET,
        "model": "HybridGRUFeatureAttention",
        "window": WINDOW,
        "stride_train": STRIDE_TRAIN,
        "stride_test": STRIDE_TEST,
        "num_sequence_features": len(SEQ_COLS),
        "num_tabular_features": len(feature_cols),
        **metrics,
        "train_time_sec": round(float(train_time), 6),
        "train_windows": int(len(Xseq_train)),
        "test_windows": int(len(Xseq_test)),
    }

    model_path = MODEL_DIR / f"{subset}_HybridGRUFeatureAttention.pt"
    torch.save(
        {
            "model_state_dict": trained.state_dict(),
            "seq_cols": SEQ_COLS,
            "tabular_cols": feature_cols,
            "window": WINDOW,
            "metrics": row,
        },
        model_path,
    )

    pred_path = METRICS_DIR / f"{subset}_HybridGRUFeatureAttention_predictions.csv"
    pd.DataFrame({
        "subset": subset,
        "model": "HybridGRUFeatureAttention",
        "y_true": y_test,
        "y_pred": np.clip(pred, 0, 125),
    }).to_csv(pred_path, index=False)

    print(
        f"    MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, "
        f"R2={row['R2']:.4f}, NASA={row['NASA_Score']:.2f}, "
        f"train={row['train_time_sec']:.2f}s"
    )

    return row

def generate_figures(df):
    ax = df.set_index("subset")["RMSE"].plot(kind="bar", figsize=(8, 5))
    ax.set_title("Hybrid GRU-Feature Attention Model RMSE")
    ax.set_xlabel("Subset")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = FIG_DIR / "fig_hybrid_model_rmse_by_subset.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

def main():
    print("=" * 80)
    print("HYBRID FEATURE-SEQUENCE MODEL - RUL REGRESSION")
    print("=" * 80)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feature_cols = load_feature_columns()
    print(f"Tabular features: {len(feature_cols)}")

    rows = []
    for subset in SUBSETS:
        row = process_subset(subset, feature_cols, device)
        rows.append(row)
        pd.DataFrame(rows).to_csv(TABLE_DIR / "hybrid_rul_results_partial.csv", index=False)

    df = pd.DataFrame(rows).sort_values("subset")
    out = TABLE_DIR / "hybrid_rul_results.csv"
    df.to_csv(out, index=False)
    generate_figures(df)

    print("\n[Hybrid results]")
    print(df[["subset", "model", "MAE", "RMSE", "R2", "NASA_Score"]].to_string(index=False))
    print("\n[Final status]")
    print("STATUS: HYBRID_MODEL_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
