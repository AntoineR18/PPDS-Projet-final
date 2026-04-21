import numpy as np
from sklearn.pipeline import Pipeline

from config import CONFIDENCE_LEVEL, N_BOOTSTRAP, RANDOM_STATE


def bootstrap_confidence_intervals(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict]: 
    rng = np.random.default_rng(random_state)
    n_samples = X_train.shape[0]
    alpha = 1.0 - confidence_level
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    # Récupération des paramètres du pipeline original pour recréer des clones
    model_params = pipeline.get_params()

    boot_coefs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        X_boot, y_boot = X_train[idx], y_train[idx]

        # Clone du pipeline pour éviter de modifier l'original
        boot_pipeline = _clone_pipeline(pipeline, model_params)
        boot_pipeline.fit(X_boot, y_boot)

        coefs = boot_pipeline.named_steps["model"].coef_
        boot_coefs.append(coefs)

    boot_coefs = np.array(boot_coefs)  # shape (n_bootstrap, n_features)

    # Coefficients du modèle original (sur l'ensemble d'entraînement complet)
    original_coefs = pipeline.named_steps["model"].coef_

    results = {}
    for i, name in enumerate(feature_names):
        results[name] = {
            "coef": float(original_coefs[i]),
            "lower": float(np.percentile(boot_coefs[:, i], lower_pct)),
            "upper": float(np.percentile(boot_coefs[:, i], upper_pct)),
        }

    return results


def _clone_pipeline(pipeline: Pipeline, params: dict) -> Pipeline:
    from sklearn.base import clone
    return clone(pipeline)


def significant_features(
    ci_results: dict[str, dict],
) -> dict[str, dict]:
    return {
        name: stats
        for name, stats in ci_results.items()
        if not (stats["lower"] <= 0 <= stats["upper"])
    }
