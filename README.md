# Furigana Annotators

Evaluating various models on the furigana annotations of book titles from the National Diet Library (https://huggingface.co/datasets/AlienKevin/ndlbib-furigana).

Randomly sampled 1,000 samples from the dataset for evaluation.

| Model | Character Error Rate (CER) |
|-------|-----|
| gpt-4o-2024-05-13 | 2.17% |
| pykakasi 2.2.1 | 3.71% |
| deepseek-v2-chat | 10.95% |
| qwen2-7b-instruct | 55.97% |

# Evaluation

1. Create a `models.json` to store your API keys:
```json
{
    "deepseek-chat": {
        "base_url": "https://api.deepseek.com",
        "api_key": "sk-xxxx"
    },
    "gpt-4o": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-xxxx"
    }
}
```

2. Install dependencies:
```
conda install openai polars tqdm pykakasi pyarrow
```

3. Run models to get furigana predictions:
```
python run_kakasi.py
python run_llm.py deepseek-chat
python run_llm.py gpt-4o
```

4. Evaluate predictions. CER will be printed in the console.
```
python eval.py kakasi
python eval.py deepseek-chat
python eval.py gpt-4o
```
