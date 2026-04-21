import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.config import (
    CV_FOLDS,
    ELASTICNET_ALPHAS,
    ELASTICNET_L1_RATIOS,
    RANDOM_STATE,
    RIDGE_ALPHAS,
    TEST_SIZE,
)


TrainTestSplit = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def prepare_arrays(
    X: pl.DataFrame,
    y: pl.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> TrainTestSplit:
    X_np = X.to_numpy().astype(np.float64)
    y_np = y.to_numpy().astype(np.float64)
    return train_test_split(
        X_np, y_np,
        test_size=test_size,
        random_state=random_state,
    )

def _make_ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])


def _make_elasticnet_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=10_000)),
    ])


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: list[float] = RIDGE_ALPHAS,
    cv: int = CV_FOLDS,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=_make_ridge_pipeline(),
        param_grid={"model__alpha": alphas},
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: list[float] = ELASTICNET_ALPHAS,
    l1_ratios: list[float] = ELASTICNET_L1_RATIOS,
    cv: int = CV_FOLDS,
) -> GridSearchCV:
    grid = GridSearchCV(
        estimator=_make_elasticnet_pipeline(),
        param_grid={
            "model__alpha": alphas,
            "model__l1_ratio": l1_ratios,
        },
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


# ── Évaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(
    grid: GridSearchCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = grid.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "best_params": grid.best_params_,
    }
