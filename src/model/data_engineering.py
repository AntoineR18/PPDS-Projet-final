import polars as pl

from src.model.config import (
    COLUMNS_TO_DROP,
    DATA_PATH,
    N_SPLITS,
    NATIONALITY_MIN_COUNT,
    SPLIT_COLUMNS_TO_DROP_SUFFIXES,
    TARGET,
)


def _split_columns(n_splits: int) -> list[str]:
    base_fields = [
        "realTime", "officialTime", "pace", "speed", "distance",
        "rankGeneral", "rankSex", "rankCategory", "location", "position",
    ]
    return [
        f"split_{i}_{field}"
        for i in range(1, n_splits + 1)
        for field in base_fields
    ]


def _unused_split_columns(n_splits: int, total_splits: int = 10) -> list[str]:
    base_fields = [
        "realTime", "officialTime", "pace", "speed", "distance",
        "rankGeneral", "rankSex", "rankCategory", "location", "position",
    ]
    return [
        f"split_{i}_{field}"
        for i in range(n_splits + 1, total_splits + 1)
        for field in base_fields
    ]


def _split_columns_to_drop(n_splits: int) -> list[str]:
    return [
        f"split_{i}_{suffix}"
        for i in range(1, n_splits + 1)
        for suffix in SPLIT_COLUMNS_TO_DROP_SUFFIXES
    ]



def load_data(path=DATA_PATH) -> pl.DataFrame:
    return pl.read_parquet(path)


def remove_dnf(df: pl.DataFrame, n_splits: int = N_SPLITS) -> pl.DataFrame:
    
    required_cols = _split_columns(n_splits) + [TARGET]
    existing = [c for c in required_cols if c in df.columns]
    mask = pl.all_horizontal([pl.col(c).is_not_null() for c in existing])
    return df.filter(mask)


def drop_leakage_columns(df: pl.DataFrame, n_splits: int = N_SPLITS) -> pl.DataFrame:
    to_drop = (
        COLUMNS_TO_DROP
        + _unused_split_columns(n_splits)
        + _split_columns_to_drop(n_splits)
    )
    existing = [c for c in to_drop if c in df.columns]
    return df.drop(existing)


def group_rare_nationalities(
    df: pl.DataFrame,
    min_count: int = NATIONALITY_MIN_COUNT,
) -> pl.DataFrame:
    kept = (
        df.group_by("nationality")
        .len()
        .filter(pl.col("len") >= min_count)
        ["nationality"]
        .to_list()
    )

    return df.with_columns(
        pl.when(pl.col("nationality").is_in(kept))
        .then(pl.col("nationality"))
        .otherwise(pl.lit("OTHER"))
        .alias("nationality")
    )


def one_hot_encode(df: pl.DataFrame) -> pl.DataFrame:

    categorical_cols = ["sex", "category", "nationality"]
    existing = [c for c in categorical_cols if c in df.columns]
    return df.to_dummies(columns=existing)



def build_dataset(
    n_splits: int = N_SPLITS,
    nationality_min_count: int = NATIONALITY_MIN_COUNT,
    path=DATA_PATH,
) -> pl.DataFrame:
    df = load_data(path)
    df = remove_dnf(df, n_splits)
    df = drop_leakage_columns(df, n_splits)
    df = group_rare_nationalities(df, nationality_min_count)
    df = one_hot_encode(df)
    a = df.null_count().sum_horizontal().sum() == 0
    print(a)
    '''
    assert df.null_count().sum_horizontal().sum() == 0, (
        "Des valeurs manquantes subsistent après le pipeline."
    )
    '''

    return df


def split_features_target(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    y = df[TARGET]
    X = df.drop(TARGET)
    return X, y