import polars as pl

INPUT_PATH = "data/test_splits_clean.csv"
OUTPUT_PATH = "data/test_parquet.parquet"


def parse_hhmmss(col_name: str) -> pl.Expr:
    """Convertit une colonne HH:MM:SS en Duration (nanosecondes)"""
    split = pl.col(col_name).str.split(":")
    hours = split.list.get(0).cast(pl.Int64)
    minutes = split.list.get(1).cast(pl.Int64)
    seconds = split.list.get(2).cast(pl.Int64)
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return (total_seconds * 1_000_000_000).cast(pl.Duration("ns")).alias(col_name)


def parse_mmss(col_name: str) -> pl.Expr:
    """Convertit une colonne M:SS en Duration (nanosecondes)"""
    split = pl.col(col_name).str.split(":")
    minutes = split.list.get(0).cast(pl.Int64)
    seconds = split.list.get(1).cast(pl.Int64)
    total_seconds = minutes * 60 + seconds
    return (total_seconds * 1_000_000_000).cast(pl.Duration("ns")).alias(col_name)


def main():
    df = pl.read_csv(INPUT_PATH)

    cols_to_drop = [
        "id",
        "raceId",
        "registrationCode",
        "photoLink",
        "videoLink",
        "addDistance",
        "team",
    ]

    cols_existing = [c for c in cols_to_drop if c in df.columns]

    df_clean = df.drop(cols_existing)

    # Colonnes Time
    time_cols = [c for c in df_clean.columns if c.endswith("Time")]

    # Colonne pace
    pace_cols = [c for c in df_clean.columns if c.endswith("pace")]

    df_clean_final = df_clean.with_columns(
        [parse_hhmmss(c) for c in time_cols] +
        [parse_mmss(c) for c in pace_cols]
    )

    df_clean_final.write_parquet(OUTPUT_PATH)

    print(f"Colonnes supprimées : {cols_existing}")
    print(f"Shape final : {df_clean_final.shape}")


if __name__ == "__main__":
    main()
