import requests
import polars as pl

BASE_URL = "https://sportinnovation.fr/api/races/80626/results"
SPLIT_URL = "https://sportinnovation.fr/api/results/{}?intermediates=1"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

SPLIT_FIELDS = [
    "location", "realTime", "officialTime", "rankGeneral", "rankSex",
    "rankCategory", "pace", "speed", "distance", "position"
]


def fetch_main_results():
    response = requests.get(BASE_URL, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def fetch_splits(result_id):
    response = requests.get(SPLIT_URL.format(result_id), headers=HEADERS)
    response.raise_for_status()
    return response.json().get("intermediates") or []


def flatten_splits(splits: list) -> dict:
    """
    Indexe les splits en excluant START (index 0, toujours nul).
    Produit : split_1_location, split_1_realTime, ..., split_2_location, ...
    """
    flat = {}
    idx = 1
    for split in splits:
        if split.get("location") == "START":
            continue
        for field in SPLIT_FIELDS:
            flat[f"split_{idx}_{field}"] = split.get(field)
        idx += 1
    return flat


def main():
    data = fetch_main_results()
    sample = data[:]

    rows = []
    for i, row in enumerate(sample):
        rid = row["id"]
        try:
            splits = fetch_splits(rid)
        except Exception:
            splits = []

        merged = {k: v for k, v in row.items()}
        merged.update(flatten_splits(splits))
        rows.append(merged)

        print(f"{i+1}/{len(sample)}")

    df = pl.from_dicts(rows, infer_schema_length=len(rows))
    df.write_csv("data/test_splits_clean.csv")
    print(f"Export terminé — {df.shape[0]} lignes, {df.shape[1]} colonnes")


if __name__ == "__main__":
    main()
