# Environment

- CPU:AMD EPYC 9334 32-Core Processor
- GPU: NVIDIA A40,  Driver Version: 535.104.12 
- Memory: 756GB
- System: Ubuntu 20.04.6
- Pytorch 2.2.2, CUDA 12.1, Python3.10

# Installation

```
conda create --name whisper python=3.10
conda activate whisper
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ffmpeg
pip install numpy pandas tqdm opencv-python pillow
pip install matplotlib
pip install nvidia-pyindex 
pip install --upgrade transformers accelerate
pip install stable-ts
```

## [Optional] Install flash-attn to accelerate the inference of transformers
(Note: Alvin did not manage to make this work)
```
conda install nvidia/label/cuda-12.1.0::cuda
export CUDA_HOME=$(python -c "import os; print(os.path.dirname(os.path.dirname(os.path.dirname(os.__file__))))")
pip install flash-attn --no-build-isolation
```

## Run stable-whisper/large-v3 to get transcripts

### Demo

#### Step1: Extract the audio from the video
replace [video_folder] and [mp3_folder] with the parent directories of the videos and output mp3 files
```
conda activate whisper
python all_videos__ffmpeg_extract_audios_multithread.py --video_folder [video_folder] --mp3_folder [mp3_folder]
```
#### Step2: Run the distil-whisper/distil-large-v3 model to get the transcript
replace [mp3_folder] and [output_csv_root_folder] with the full path of the audio folder and the output csv folder
```
conda activate whisper
python whisper_transcibe_on_all_videos_timealign_parallel.py --mp3_folder [mp3_folder] --transcript_output_folder [output_json_root_folder]
```
#### Step3: Convert output JSON to CSV file
```
python transcript_preprocess.py --json_folder [output_json_root_folder] --csv_folder [output_csv_folder]
```
