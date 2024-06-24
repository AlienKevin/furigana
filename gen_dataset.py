import glob
import json
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

def parse_tsv_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    segments = []
    titles = []
    lines_after_index = 0
    for line in lines:
        lines_after_index += 1
        if line.startswith("行番号: "):
            if segments != []:
                filtered_segments = []
                for i, segment in enumerate(segments):
                    if segment["category"] == "分かち書き":
                        left_segment = segments[i-1] if i > 0 else None
                        right_segment = segments[i+1] if i < len(segments) - 1 else None
                        if (left_segment and left_segment["category"] in ["英文字", "半角数字"] and right_segment and right_segment["category"] in ["英文字", "半角数字"]):
                            segment["word"] = " "
                            filtered_segments.append(segment)
                    else:
                        filtered_segments.append(segment)
                
                text = "".join([segment["word"] for segment in filtered_segments])
                reading = "".join([segment["pronunciation"] for segment in filtered_segments])
                titles.append({
                    "text": text,
                    "reading": reading,
                    "segments": json.dumps(segments, ensure_ascii=False)
                })
            segments = []
            lines_after_index = 0
        elif lines_after_index > 2:
            parts = line.rstrip().split("\t")
            word, pronunciation, category = parts
            segments.append({"word": word, "pronunciation": pronunciation, "category": category})

    return titles

def build_ndlbib_parquet():
    tsv_files = glob.glob('data/ndlbib/tsv_file*.txt')
    
    # Define schema for the Parquet file
    schema = pa.schema([
        ('text', pa.string()),
        ('reading', pa.string()),
        ('segments', pa.string())
    ])
    
    with pq.ParquetWriter('data/ndlbib.parquet', schema) as writer:
        for tsv_file in tqdm(tsv_files):
            titles = parse_tsv_file(tsv_file)
            arrays = [
                pa.array([title['text'] for title in titles], pa.string()),
                pa.array([title['reading'] for title in titles], pa.string()),
                pa.array([title['segments'] for title in titles], pa.string())
            ]
            table = pa.Table.from_arrays(arrays, schema=schema)
            writer.write_table(table)

if __name__ == "__main__":
    build_ndlbib_parquet()
