import json
import os
import re
from glob import glob
import pandas as pd
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Preprocess Whisper transcript JSONs.")
    parser.add_argument("--json_folder", type=str, required=True, help="Folder containing JSON files.")
    parser.add_argument("--csv_folder", type=str, required=True, help="Folder to save CSV files.")
    args = parser.parse_args()
    json_folder = args.json_folder
    csv_folder = args.csv_folder

    all_json_files = sorted(glob(os.path.join(json_folder, "**", "*.json"), recursive=True))
    for json_file in tqdm(all_json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)

        file_name = os.path.basename(json_file)
        file_name = re.sub(r"\.json$", "", file_name)
        subject_id = file_name.split("_")[0]

        output_path = os.path.join(csv_folder, subject_id, f"{file_name}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if len(data) == 0:
            res_df = pd.DataFrame(columns=['utterance_id', 'utterance', 'token_num', 'token', 'token_start_time', 'token_end_time'])
            res_df.to_csv(output_path, index_label="token_id")
            continue

        token_data = []
        uid = 0

        for u in data:
            # uid = u['id']
            # utt = u['text']

            cur_utt = ""
            all_tokens = []
            all_token_start_times = []
            all_token_end_times = []

            for idx, w in enumerate(u['words']):
                word = w['word']
                all_tokens.append(word)
                all_token_start_times.append(w['start'])
                all_token_end_times.append(w['end'])

                cur_utt = cur_utt + word

                if re.search("[!?\.]$", word) is not None or idx == len(u['words'])-1:
                    for n, (t, s, e) in enumerate(zip(all_tokens, all_token_start_times, all_token_end_times)):
                        token_data.append({
                            'utterance_id': uid,
                            'utterance': cur_utt,
                            'token_num': n,
                            'token': t,
                            'token_start_time': s,
                            'token_end_time': e
                        })
                    
                    uid += 1
                    cur_utt = ""
                    all_tokens = []
                    all_token_start_times = []
                    all_token_end_times = []
        
        res_df = pd.DataFrame(token_data)
        res_df['utterance'] = res_df['utterance'].str.strip()
        res_df['token'] = res_df['token'].str.strip(' .,!?":;-')
        res_df['token'] = res_df['token'].str.lower()

        res_df.to_csv(output_path, index_label="token_id")

if __name__ == "__main__":
    main()
