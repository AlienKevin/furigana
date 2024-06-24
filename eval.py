import polars as pl
import jiwer
from tqdm import tqdm
import regex as re
import sys

# Define a regex pattern to strip away all punctuation characters
punctuation_pattern = re.compile(r'\p{P}')

def normalize(text):
    greek_to_hiragana = {
        "α": "あるふぁ",
        "β": "べーた",
        "γ": "がんま",
        "δ": "でるた",
        "ε": "いぷしろん",
        "ζ": "ぜーた",
        "η": "いーた",
        "θ": "しーた",
        "ι": "いおた",
        "κ": "かっぱ",
        "λ": "らむだ",
        "μ": "みゅー",
        "ν": "にゅー",
        "ξ": "くしー",
        "ο": "おみくろん",
        "π": "ぱい",
        "ρ": "ろー",
        "σ": "しぐま",
        "τ": "たう",
        "υ": "うぷしろん",
        "φ": "ふぁい",
        "χ": "かい",
        "ψ": "ぷさい",
        "ω": "おめが"
    }
    text = text.lower()
    return punctuation_pattern.sub('', text.replace('を', 'お')).translate(str.maketrans(greek_to_hiragana))

def calculate_cer(name, field_name="reading_output"):
    # Lazily read the Parquet file into a DataFrame
    df = pl.scan_parquet(f"results/ndlbib_{name}.parquet")

    # Collect the DataFrame to get the number of rows
    num_rows = df.collect().height

    # Initialize variables for running average calculation
    total_cer = 0.0
    count = 0

    # List to store wrong sentences
    wrong_sentences = []

    # Iterate over the DataFrame in chunks
    pbar = tqdm(total=num_rows//1000)
    for chunk in df.collect().iter_slices(1000):
        readings = [normalize(reading) for reading in chunk["reading"].to_list()]
        output_readings = [normalize(reading) for reading in chunk[field_name].to_list()]

        # Calculate the Character Error Rate (CER) for the chunk
        cer = jiwer.cer(readings, output_readings)

        # Identify wrong sentences
        for i, (reading, output_reading) in enumerate(zip(readings, output_readings)):
            if reading != output_reading:
                wrong_sentences.append({
                    "text": chunk["text"][i],
                    "reading": reading,
                    field_name: output_reading
                })

        # Update running average
        total_cer += cer * len(readings)
        count += len(readings)
        running_average_cer = total_cer / count

        # Print the running average CER
        pbar.set_description(f"Running Average CER: {running_average_cer:.2f}")
        pbar.update(1)

    pbar.close()
    
    # Print the final CER
    print(f"Final Character Error Rate (CER): {running_average_cer}")

    # Create a DataFrame from the wrong sentences and write to a new Parquet file
    wrong_sentences_df = pl.DataFrame(wrong_sentences)
    wrong_sentences_df.write_parquet(f"results/ndlbib_{name}_wrong.parquet")

if __name__ == "__main__":
    name = sys.argv[1]
    calculate_cer(name)
