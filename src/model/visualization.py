import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from src.model.config import CONFIDENCE_LEVEL

sns.set_theme(style="whitegrid", palette="muted")


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Modèle",
) -> plt.Figure:

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Analyse des résidus — {model_name}", fontsize=14)

    # Résidus vs valeurs prédites
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=5, color="steelblue")
    axes[0].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Valeurs prédites (secondes)")
    axes[0].set_ylabel("Résidus (secondes)")
    axes[0].set_title("Résidus vs Prédictions")

    # Distribution des résidus
    axes[1].hist(residuals, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Résidus (secondes)")
    axes[1].set_ylabel("Fréquence")
    axes[1].set_title("Distribution des résidus")

    plt.tight_layout()
    return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Modèle",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.2, s=5, color="steelblue")

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Prédiction parfaite")

    ax.set_xlabel("Temps réel (secondes)")
    ax.set_ylabel("Temps prédit (secondes)")
    ax.set_title(f"Prédictions vs Réalité — {model_name}")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confidence_intervals(
    ci_results: dict[str, dict],
    top_n: int = 30,
    model_name: str = "Modèle",
    confidence_level: float = CONFIDENCE_LEVEL,
) -> plt.Figure:

    sorted_items = sorted(
        ci_results.items(),
        key=lambda x: abs(x[1]["coef"]),
        reverse=True,
    )[:top_n]

    names = [item[0] for item in sorted_items]
    coefs = [item[1]["coef"] for item in sorted_items]
    lowers = [item[1]["coef"] - item[1]["lower"] for item in sorted_items]
    uppers = [item[1]["upper"] - item[1]["coef"] for item in sorted_items]
    is_significant = [
        not (item[1]["lower"] <= 0 <= item[1]["upper"])
        for item in sorted_items
    ]

    colors = ["tomato" if sig else "steelblue" for sig in is_significant]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    y_pos = np.arange(len(names))

    ax.barh(y_pos, coefs, xerr=[lowers, uppers], color=colors,
            align="center", alpha=0.8, ecolor="gray", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Coefficient (standardisé)")
    ax.set_title(
        f"Coefficients avec IC à {int(confidence_level * 100)}% — {model_name}\n"
        f"(Rouge = significativement ≠ 0)"
    )
    plt.tight_layout()
    return fig


def plot_metrics_comparison(metrics: dict[str, dict]) -> plt.Figure:
    model_names = list(metrics.keys())
    rmse_vals = [m["rmse"] for m in metrics.values()]
    mae_vals = [m["mae"] for m in metrics.values()]
    r2_vals = [m["r2"] for m in metrics.values()]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Comparaison des modèles — jeu de test", fontsize=14)

    for ax, values, title, unit in zip(
        axes,
        [rmse_vals, mae_vals, r2_vals],
        ["RMSE", "MAE", "R²"],
        ["secondes", "secondes", ""],
    ):
        bars = ax.bar(model_names, values, color=["steelblue", "darkorange"], alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel(unit)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.1f}" if unit == "secondes" else f"{val:.4f}",
                ha="center", va="bottom", fontsize=10,
            )

    plt.tight_layout()
    return fig
