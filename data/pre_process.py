import polars as pl


INPUT_PATH = "data/test_splits_clean.csv"
OUTPUT_PATH = "data/donnees_finales.parquet"


def parse_hhmmss_seconds(col_name: str) -> pl.Expr:
    """Convertit une colonne "HH:MM:SS" en Int64."""
    split = pl.col(col_name).str.split(":")
    hours = split.list.get(0).cast(pl.Int64)
    minutes = split.list.get(1).cast(pl.Int64)
    seconds = split.list.get(2).cast(pl.Int64)
    return (hours * 3600 + minutes * 60 + seconds).alias(col_name)


def parse_mmss_seconds(col_name: str) -> pl.Expr:
    """Convertit une colonne "M:SS" en Int64."""
    split = pl.col(col_name).str.split(":")
    minutes = split.list.get(0).cast(pl.Int64)
    seconds = split.list.get(1).cast(pl.Int64)
    return (minutes * 60 + seconds).alias(col_name)


def pre_process_splits(df=None):
    if df is None:
        df = pl.read_csv(INPUT_PATH)


    cols_to_drop = [
        "id",
        "raceId",
        "photoLink",
        "videoLink",
        "registrationCode",
        "pace",
        "addDistance",
        "generalRanking",
        "sexRanking",
        "categoryRanking"
    ]

    cols_to_drop_split_suffix = [
        "position",
        "location",
        "officialTime",
        "pace",
        "rankGeneral",
        "rankSex",
        "rankCategory"
    ]

    cols_to_drop = cols_to_drop + [
        f"split_{n}_" + suff for n in range(1, 11) for suff in cols_to_drop_split_suffix
        ]

    cols_existing = [c for c in cols_to_drop if c in df.columns]

    df_clean = df.filter(
        [pl.col(f"split_{n}_position") == n+1 for n in range(1, 11)]
    )

    df_clean = df_clean.drop(cols_existing)

    # Colonnes Time
    time_cols = [c for c in df_clean.columns if c.endswith("Time")]

    # Colonnes pace
    pace_cols = [c for c in df_clean.columns if c.endswith("pace")]

    df_clean = df_clean.with_columns(
        [parse_hhmmss_seconds(c) for c in time_cols] +
        [parse_mmss_seconds(c) for c in pace_cols]
    )

    df_clean.write_parquet(OUTPUT_PATH)


if __name__ == "__main__":
    pre_process_splits()
