import openai
import polars as pl
from tqdm import tqdm
import random
import json

random.seed(42)

model_name = "gpt-4o"

with open('models.json') as f:
    config = json.load(f)

client = openai.OpenAI(
    base_url=config[model_name]["base_url"],
    api_key=config[model_name]["api_key"]
)

def run_llm(input):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "次の文章を全てひらがなに変換してください。"},
            {"role": "user", "content": "<入力>彼は5年前にニューヨークへ行きました.</入力>"},
            {"role": "assistant", "content": "<ひらがな>かれは5ねんまえににゅーよーくへいきました.</ひらがな>"},
            {"role": "user", "content": "<入力>αとβはギリシャ文字です。</入力>"},
            {"role": "assistant", "content": "<ひらがな>あるふぁとべーたはぎりしゃもじです。</ひらがな>"},
            {"role": "user", "content": "<入力>私の友達はJohnです.彼はABCの社員です.</入力>"},
            {"role": "assistant", "content": "<ひらがな>わたしのともだちはJohnです.かれはABCのしゃいんです.</ひらがな>"},
            {"role": "user", "content": "<入力>今日は30℃でとても暑いです。</入力>"},
            {"role": "assistant", "content": "<ひらがな>きょうは30どでとてもあついです。</ひらがな>"},
            {"role": "user", "content": f"<入力>{input}</入力>"}
        ]
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

# Lazily read the Parquet file into a DataFrame
df = pl.scan_parquet("data/ndlbib.parquet")

# Collect the DataFrame to get the number of rows
num_rows = df.collect().height

# Randomly select 1000 samples
df_sample = df.collect().sample(n=1000, with_replacement=False)

# Initialize a list to store the results
results = []
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_llm_with_retry(text, retries=3):
    for _ in range(retries):
        try:
            reading_llm = run_llm(text)
            reading_llm = reading_llm.split("<ひらがな>")[1].split("</ひらがな>")[0]
            return reading_llm
        except Exception as e:
            print(f"Error: {e}. Retrying...")
    return None

# Iterate over each sample in the DataFrame
pbar = tqdm(total=1000)
with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_row = {executor.submit(run_llm_with_retry, row["text"]): row for row in df_sample.iter_rows(named=True)}
    for future in as_completed(future_to_row):
        row = future_to_row[future]
        try:
            reading_llm = future.result()
            if reading_llm is not None:
                results.append({
                    "text": row["text"],
                    "reading": row["reading"],
                    "reading_output": reading_llm
                })
        except Exception as e:
            print(f"Error processing row: {e}")
        pbar.update(1)

pbar.close()

# Create a DataFrame from the results and write to a new Parquet file
results_df = pl.DataFrame(results)
results_df.write_parquet(f"data/ndlbib_{model_name}.parquet")
