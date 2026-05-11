from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path.cwd()
FIG_DIR = ROOT / "outputs" / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

def add_box(ax, xy, width, height, text, fontsize=9):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True
    )

def add_arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.5)
    )

def framework_diagram():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    add_box(ax, (0.4, 4.9), 2.2, 1.0, "NASA C-MAPSS\nFD001-FD004")
    add_box(ax, (0.4, 3.2), 2.2, 1.0, "Raw Multivariate\nSensor Time-Series")
    add_box(ax, (3.2, 4.9), 2.4, 1.0, "RUL Label\nConstruction")
    add_box(ax, (3.2, 3.2), 2.4, 1.0, "Temporal Feature\nEngineering")
    add_box(ax, (6.2, 5.4), 2.4, 0.9, "RUL Regression\nML Baselines")
    add_box(ax, (6.2, 4.1), 2.4, 0.9, "Deep Sequence\nBaselines")
    add_box(ax, (6.2, 2.8), 2.4, 0.9, "Risk-Stage\nClassification")
    add_box(ax, (9.2, 4.8), 2.3, 1.0, "Ablation Study\nFeature Groups")
    add_box(ax, (9.2, 3.2), 2.3, 1.0, "Bootstrap CI\nValidation")
    add_box(ax, (9.2, 1.6), 2.3, 1.0, "SHAP\nExplainability")
    add_box(ax, (12.0, 4.0), 1.7, 1.2, "Predictive\nMaintenance\nDecision Support")

    add_arrow(ax, (2.6, 5.4), (3.2, 5.4))
    add_arrow(ax, (2.6, 3.7), (3.2, 3.7))
    add_arrow(ax, (5.6, 3.7), (6.2, 5.85))
    add_arrow(ax, (5.6, 3.7), (6.2, 4.55))
    add_arrow(ax, (5.6, 3.7), (6.2, 3.25))
    add_arrow(ax, (8.6, 5.85), (9.2, 5.3))
    add_arrow(ax, (8.6, 5.85), (9.2, 3.7))
    add_arrow(ax, (8.6, 5.85), (9.2, 2.1))
    add_arrow(ax, (11.5, 5.3), (12.0, 4.7))
    add_arrow(ax, (11.5, 3.7), (12.0, 4.5))
    add_arrow(ax, (11.5, 2.1), (12.0, 4.3))

    ax.set_title(
        "Overall Framework for Explainable RUL Prediction and Risk Assessment",
        fontsize=14,
        pad=20
    )

    out1 = FIG_DIR / "fig_framework_architecture.png"
    out2 = PAPER_FIG_DIR / "fig_framework_architecture.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

def feature_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    add_box(ax, (0.4, 2.5), 2.0, 1.0, "Engine Unit\nTime-Series")
    add_box(ax, (3.0, 4.2), 2.2, 0.9, "Operational\nSettings")
    add_box(ax, (3.0, 3.0), 2.2, 0.9, "Raw Sensor\nReadings")
    add_box(ax, (3.0, 1.8), 2.2, 0.9, "Cycle-Derived\nVariables")
    add_box(ax, (5.9, 4.2), 2.2, 0.9, "Lag Features")
    add_box(ax, (5.9, 3.0), 2.2, 0.9, "Delta / Percentage\nChange")
    add_box(ax, (5.9, 1.8), 2.2, 0.9, "Rolling Statistics\nMean/Std/Min/Max/Range")
    add_box(ax, (8.8, 3.0), 2.1, 1.0, "Temporal Degradation\nFeature Vector")
    add_box(ax, (11.5, 4.0), 2.0, 0.9, "RUL Regression")
    add_box(ax, (11.5, 2.6), 2.0, 0.9, "Risk-Stage\nClassification")
    add_box(ax, (11.5, 1.2), 2.0, 0.9, "SHAP Feature\nInterpretation")

    add_arrow(ax, (2.4, 3.0), (3.0, 4.65))
    add_arrow(ax, (2.4, 3.0), (3.0, 3.45))
    add_arrow(ax, (2.4, 3.0), (3.0, 2.25))
    add_arrow(ax, (5.2, 3.45), (5.9, 4.65))
    add_arrow(ax, (5.2, 3.45), (5.9, 3.45))
    add_arrow(ax, (5.2, 3.45), (5.9, 2.25))
    add_arrow(ax, (8.1, 4.65), (8.8, 3.6))
    add_arrow(ax, (8.1, 3.45), (8.8, 3.5))
    add_arrow(ax, (8.1, 2.25), (8.8, 3.4))
    add_arrow(ax, (10.9, 3.5), (11.5, 4.45))
    add_arrow(ax, (10.9, 3.5), (11.5, 3.05))
    add_arrow(ax, (10.9, 3.5), (11.5, 1.65))

    ax.set_title(
        "Temporal Degradation Feature Engineering Pipeline",
        fontsize=14,
        pad=20
    )

    out1 = FIG_DIR / "fig_temporal_feature_pipeline.png"
    out2 = PAPER_FIG_DIR / "fig_temporal_feature_pipeline.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"Saved: {out1}")
    print(f"Saved: {out2}")

def main():
    print("=" * 80)
    print("GENERATE ARCHITECTURE AND FEATURE PIPELINE DIAGRAMS")
    print("=" * 80)

    framework_diagram()
    feature_pipeline_diagram()

    print("\n[Final status]")
    print("STATUS: DIAGRAMS_READY")

if __name__ == "__main__":
    main()
