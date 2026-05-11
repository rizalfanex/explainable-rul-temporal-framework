import sys
import platform
import importlib
from pathlib import Path

PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "torch",
]

OPTIONAL_PACKAGES = [
    "xgboost",
    "lightgbm",
    "catboost",
    "shap",
    "yaml",
    "tqdm",
    "joblib",
]

def check_import(package_name: str) -> bool:
    try:
        importlib.import_module(package_name)
        return True
    except Exception:
        return False

def main():
    print("=" * 80)
    print("ENVIRONMENT CHECK - PREDICTIVE MAINTENANCE RUL IEEE PROJECT")
    print("=" * 80)

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version}")
    print(f"Platform          : {platform.platform()}")

    print("\n[Core packages]")
    core_ok = True
    for pkg in PACKAGES:
        ok = check_import(pkg)
        print(f"{pkg:<15}: {'OK' if ok else 'MISSING'}")
        core_ok = core_ok and ok

    print("\n[Optional research packages]")
    for pkg in OPTIONAL_PACKAGES:
        ok = check_import(pkg)
        print(f"{pkg:<15}: {'OK' if ok else 'MISSING'}")

    print("\n[Torch CUDA]")
    try:
        import torch
        print(f"torch version     : {torch.__version__}")
        print(f"CUDA available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count : {torch.cuda.device_count()}")
            print(f"CUDA device name  : {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Torch check failed: {e}")

    print("\n[Project directories]")
    required_dirs = [
        "data/raw/cmapss",
        "data/interim",
        "data/processed",
        "outputs/figures",
        "outputs/tables",
        "outputs/metrics",
        "outputs/models",
        "outputs/logs",
        "src",
        "configs",
        "paper/figures",
        "paper/tables",
        "paper/notes",
        "notebooks",
    ]

    dirs_ok = True
    root = Path.cwd()
    for d in required_dirs:
        path = root / d
        exists = path.exists()
        print(f"{d:<25}: {'OK' if exists else 'MISSING'}")
        dirs_ok = dirs_ok and exists

    print("\n[Final status]")
    if core_ok and dirs_ok:
        print("STATUS: READY")
    else:
        print("STATUS: NOT READY")

if __name__ == "__main__":
    main()
