import s3fs
import polars as pl
from pre_process import pre_process_splits, INPUT_PATH, OUTPUT_PATH

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"})

MY_BUCKET = "azizseghaier"
FILE_PATH_S3 = f"{MY_BUCKET}/diffusion/test_splits_clean.csv"

if __name__ == "__main__":
    with fs.open(FILE_PATH_S3, 'rb') as file_in:
        df = pl.read_csv(file_in)
        df.write_csv(INPUT_PATH)

    pre_process_splits(df)
