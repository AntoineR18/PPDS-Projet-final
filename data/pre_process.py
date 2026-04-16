import polars as pl

INPUT_PATH = "data/marathon_paris_2026.csv"
OUTPUT_PATH = "data/marathon_paris_2026_clean.csv"


def main():
    df = pl.read_csv(INPUT_PATH)

    cols_to_drop = [
        "raceId",
        "registrationCode",
        "photoLink",
        "videoLink",
        "addDistance",
        "team",
    ]

    cols_existing = [c for c in cols_to_drop if c in df.columns]

    df_clean = df.drop(cols_existing)

    df_clean.write_csv(OUTPUT_PATH)

    print(f"Colonnes supprimées : {cols_existing}")
    print(f"Shape final : {df_clean.shape}")


if __name__ == "__main__":
    main()
