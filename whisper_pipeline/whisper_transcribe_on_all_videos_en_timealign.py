import torch
from glob import glob
import pandas as pd
import os
import re
import argparse
from tqdm import tqdm
import stable_whisper
import logging
import json
logging.getLogger().setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description="Extract MP3 audio from video files using ffmpeg.")
    parser.add_argument("--mp3_folder", type=str, required=True, help="Folder to save extracted MP3 files.")
    parser.add_argument("--english_subjects_file", type=str, required=False, default="", help="File to save extracted transcripts.")
    parser.add_argument("--transcript_output_folder", type=str, required=True, help="Folder to save extracted transcripts.")
    parser.add_argument("--rank_id", type=int, default=0, help="Rank ID for distributed running.")
    parser.add_argument("--num_parallel", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--is_saycam", type=int, default=0, help="Whether the videos are from SayCam.")
    parser.add_argument("--overwrite", action='store_true', help="Whether to overwrite existing files.")
    args = parser.parse_args()
    mp3_folder = args.mp3_folder
    is_saycam = args.is_saycam
    transcript_output_folder = args.transcript_output_folder
    device = torch.device("cuda")
    rank_id = args.rank_id
    num_parallel = args.num_parallel
    overwrite = args.overwrite
    # model = stable_whisper.load_hf_whisper("turbo", device=device)
    model = stable_whisper.load_model("large-v3", device=device)

    english_subjects = []
    filter_english_subjects = False
    english_subjects_file = args.english_subjects_file
    if os.path.exists(english_subjects_file):
        filter_english_subjects = True
        with open(english_subjects_file, 'r') as f:
            for line in f:
                english_subjects.append(line.strip())
    else:
        filter_english_subjects = False
       
    all_audio_files = sorted(glob(os.path.join(mp3_folder, "**", "*.mp3"), recursive=True))
    en_audio_files = []
    file_name_subject_id = {}
    for audio_file in all_audio_files:
        file_name = os.path.basename(audio_file)
        file_name = re.sub(r"\.mp3$", "", file_name)
        subject_id = file_name.split("_")[0]
        file_name_subject_id[file_name] = subject_id
        if subject_id in english_subjects and filter_english_subjects:
            en_audio_files.append(audio_file)
    
    if filter_english_subjects:
        all_audio_files = en_audio_files

    if not overwrite:
        new_audio_files = []
        for audio_file in all_audio_files:
            file_name = os.path.basename(audio_file)
            file_name = re.sub(r"\.mp3$", "", file_name)
            subject_id = file_name.split("_")[0]
            json_output_path = os.path.join(transcript_output_folder, "json", subject_id, f"{file_name}.json")
            if not os.path.exists(json_output_path):
                new_audio_files.append(audio_file)
        all_audio_files = new_audio_files

    group_size = len(all_audio_files) // num_parallel
    start_idx = rank_id * group_size
    end_idx = start_idx + group_size
    if rank_id == num_parallel - 1:
        end_idx = len(all_audio_files)
    current_group_audio_files = all_audio_files[start_idx:end_idx]

    for idx, audio_file in enumerate(tqdm(current_group_audio_files)):
        file_name = os.path.basename(audio_file)
        file_name = re.sub(r"\.mp3$", "", file_name)
        subject_id = file_name_subject_id[file_name]

        json_output_path = os.path.join(transcript_output_folder, "json", subject_id, f"{file_name}.json")

        with torch.no_grad():
            result = model.transcribe(audio_file, language='en', word_timestamps=True, suppress_silence=True)
        result_dict = result.to_dict()
        
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, "w") as f:
            json.dump(result_dict['ori_dict']['segments'], f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
