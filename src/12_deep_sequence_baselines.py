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
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
MODEL_DIR = ROOT / "outputs" / "models"
LOG_DIR = ROOT / "outputs" / "logs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
SEQ_COLS = [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
TARGET = "RUL_capped"

SEED = 42
WINDOW = 30
STRIDE_TRAIN = 3
STRIDE_TEST = 3
BATCH_SIZE = 256
EPOCHS = 18
LR = 1e-3
HIDDEN = 64

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

def build_windows(df, seq_cols, target_col, window, stride):
    X_list, y_list = [], []
    for _, g in df.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        arr = g[seq_cols].values.astype("float32")
        y = g[target_col].values.astype("float32")
        if len(g) < window:
            continue
        for end in range(window - 1, len(g), stride):
            start = end - window + 1
            X_list.append(arr[start:end + 1])
            y_list.append(y[end])
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y

class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class TCNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.net(x).squeeze(-1)
        return self.head(z).squeeze(-1)

def train_model(model, X_train, y_train, X_test, device):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))

        if epoch in [1, 5, 10, EPOCHS]:
            print(f"      epoch={epoch:02d}, train_mse={np.mean(losses):.4f}")

    model.eval()
    preds = []
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test)), batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds)
    return model, preds

def process_subset(subset, device):
    train_path = PROCESSED_DIR / f"{subset}_train_with_rul.csv"
    test_path = PROCESSED_DIR / f"{subset}_test_with_rul.csv"

    print(f"\n[{subset}] Loading labeled data")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    scaler = StandardScaler()
    scaler.fit(train_df[SEQ_COLS].astype("float32"))

    train_df[SEQ_COLS] = scaler.transform(train_df[SEQ_COLS].astype("float32")).astype("float32")
    test_df[SEQ_COLS] = scaler.transform(test_df[SEQ_COLS].astype("float32")).astype("float32")

    X_train, y_train = build_windows(train_df, SEQ_COLS, TARGET, WINDOW, STRIDE_TRAIN)
    X_test, y_test = build_windows(test_df, SEQ_COLS, TARGET, WINDOW, STRIDE_TEST)

    print(f"  Train windows: {X_train.shape}")
    print(f"  Test windows : {X_test.shape}")

    models = {
        "GRU": GRURegressor(input_dim=len(SEQ_COLS), hidden_dim=HIDDEN),
        "LSTM": LSTMRegressor(input_dim=len(SEQ_COLS), hidden_dim=HIDDEN),
        "TCN": TCNRegressor(input_dim=len(SEQ_COLS), hidden_dim=HIDDEN),
    }

    rows = []
    for name, model in models.items():
        print(f"  Training {name}...")
        start_train = time.perf_counter()
        trained, pred = train_model(model, X_train, y_train, X_test, device)
        train_time = time.perf_counter() - start_train

        metrics = evaluate(y_test, pred)
        row = {
            "subset": subset,
            "target": TARGET,
            "model": name,
            "window": WINDOW,
            "stride_train": STRIDE_TRAIN,
            "stride_test": STRIDE_TEST,
            "num_sequence_features": len(SEQ_COLS),
            **metrics,
            "train_time_sec": round(float(train_time), 6),
            "train_windows": int(len(X_train)),
            "test_windows": int(len(X_test)),
        }
        rows.append(row)

        model_path = MODEL_DIR / f"{subset}_{name}_deep_sequence.pt"
        torch.save(
            {
                "model_state_dict": trained.state_dict(),
                "model_name": name,
                "seq_cols": SEQ_COLS,
                "window": WINDOW,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "metrics": row,
            },
            model_path,
        )

        pred_path = ROOT / "outputs" / "metrics" / f"{subset}_{name}_deep_sequence_predictions.csv"
        pd.DataFrame({
            "subset": subset,
            "model": name,
            "y_true": y_test,
            "y_pred": np.clip(pred, 0, 125),
        }).to_csv(pred_path, index=False)

        print(
            f"    MAE={row['MAE']:.4f}, RMSE={row['RMSE']:.4f}, "
            f"R2={row['R2']:.4f}, NASA={row['NASA_Score']:.2f}, "
            f"train={row['train_time_sec']:.2f}s"
        )

    return rows

def generate_figures(df):
    pivot = df.pivot(index="model", columns="subset", values="RMSE")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Deep Sequence Baselines for RUL Prediction")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = FIG_DIR / "fig_deep_sequence_rmse_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved: {out}")

def main():
    print("=" * 80)
    print("DEEP SEQUENCE BASELINES - RUL REGRESSION")
    print("=" * 80)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_rows = []
    for subset in SUBSETS:
        rows = process_subset(subset, device)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(TABLE_DIR / "deep_sequence_rul_results_partial.csv", index=False)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["subset", "RMSE", "MAE"]).reset_index(drop=True)
    out = TABLE_DIR / "deep_sequence_rul_results.csv"
    df.to_csv(out, index=False)

    best = df.loc[df.groupby("subset")["RMSE"].idxmin()].copy().sort_values("subset")
    best_out = TABLE_DIR / "deep_sequence_best_by_subset.csv"
    best.to_csv(best_out, index=False)

    generate_figures(df)

    print("\n[Best deep sequence model per subset]")
    print(best[["subset", "model", "MAE", "RMSE", "R2", "NASA_Score"]].to_string(index=False))
    print("\n[Final status]")
    print("STATUS: DEEP_SEQUENCE_BASELINES_READY")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
