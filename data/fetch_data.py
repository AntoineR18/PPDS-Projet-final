import requests
import polars as pl

API_URL = "https://sportinnovation.fr/api/races/80626/results"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

def main():
    response = requests.get(API_URL, headers=HEADERS)
    response.raise_for_status()

    data = response.json()  

    df = pl.from_dicts(data)

    df.write_csv("data/marathon_paris_2026.csv")

    print(f"{df.shape[0]} lignes exportées")


if __name__ == "__main__":
    main()
