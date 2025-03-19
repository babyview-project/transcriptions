#%% convert all of them to mp3
import os
video_fodler = "/ccn2/dataset/babyview/unzip_2025"
mp3_folder = '/ccn2/dataset/babyview/outputs_20250312/mp3'
transcript_output_folder = '/ccn2/dataset/babyview/outputs_20250312/transcripts'
#%%
print("=" * 80)
print("🎬 Step 1: Video to Audio Conversion")
print(f"📥 Input Videos: {video_fodler}")
print(f"📤 Output MP3s: {mp3_folder}")
print("=" * 80)
max_workers = 8
os.makedirs(mp3_folder, exist_ok=True)
os.system(f"conda activate whisper;python3 all_videos_ffmpeg_extract_audios_multithread.py --video_folder {video_fodler} --mp3_folder {mp3_folder} --max_workers {max_workers}")
# %% transcribe all mp3 files
print("=" * 80)
print("🎬 Step 2: Transcribe all MP3 files")
print(f"📥 Input MP3s: {mp3_folder}")
print(f"📤 Output Transcripts: {transcript_output_folder}")
print("=" * 80)
os.makedirs(transcript_output_folder, exist_ok=True)
os.system(f"conda activate whisper;python3 whisper_transcribe_on_all_videos_timealign_parallel.py --mp3_folder {mp3_folder} --transcript_output_folder {transcript_output_folder} --num_parallel 8 --device_ids '[0,1,2,3,4,5,6,7]'")
# %% preprocess all transcripts
print("=" * 80)
print("🎬 Step 3: Preprocess all transcripts")
print(f"📤 Input JSONs: {transcript_output_folder}/json")
print(f"📤 Output CSVs: {transcript_output_folder}/csv")
print("=" * 80)
os.system(f"conda activate whisper;python3 transcript_preprocess.py --json_folder {transcript_output_folder}/json --csv_folder {transcript_output_folder}/csv")