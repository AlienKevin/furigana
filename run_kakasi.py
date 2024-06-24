import polars as pl
from tqdm import tqdm
import pykakasi

RANDOM_SEED = 42
N_TEST_SAMPLES = 1000

pl.set_random_seed(RANDOM_SEED)


def main():
    # Lazily read the Parquet file into a DataFrame
    df = pl.scan_parquet("data/ndlbib.parquet")

    # Randomly select 1000 samples
    df_sample = df.collect().sample(n=N_TEST_SAMPLES, with_replacement=False, seed=RANDOM_SEED)

    # Initialize Kakasi for conversion
    kks = pykakasi.kakasi()

    # Use Kakasi for conversion
    converted_readings = []
    for text in tqdm(df_sample["text"]):
        converted_reading = ''.join(segment['hira'] for segment in kks.convert(text))
        converted_readings.append(converted_reading)

    # Create a new DataFrame with the sampled text and converted readings
    df_sample = df_sample.with_columns(pl.Series("reading_output", converted_readings))

    # Keep only the "text" and "reading_output" columns
    df_sample = df_sample.select(["text", "reading", "reading_output"])

    # Write the new DataFrame to a new Parquet file
    df_sample.write_parquet("results/ndlbib_kakasi.parquet")

if __name__ == "__main__":
    main()
