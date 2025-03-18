# Environment

- CPU:AMD EPYC 9334 32-Core Processor
- GPU: NVIDIA A40,  Driver Version: 535.104.12 
- Memory: 756GB
- System: Ubuntu 20.04.6
- Pytorch 2.2.2, CUDA 12.1, Python3.10

# Installation

```
conda create --name torch python=3.10
conda activate torch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install ffmpeg
pip install numpy pandas tqdm opencv-python pillow
pip install matplotlib
pip install nvidia-pyindex 
pip install --upgrade transformers accelerate
```

## [Optional] Install flash-attn to accelerate the inference of transformers
```
conda install nvidia/label/cuda-12.1.0::cuda
export CUDA_HOME=$(python -c "import os; print(os.path.dirname(os.path.dirname(os.path.dirname(os.__file__))))")
pip install flash-attn --no-build-isolation
```

## Run distil-whisper/distil-large-v3 or stable-whisper/large-v3 to get transcripts

### Demo: Single video

#### Step1: Extract the audio from the video
replace [video_full_path] and [output_mp3_full_path] with the full path of the video and the output mp3 file
```
ffmpeg -i [video_full_path] -vn -ar 44100 -ac 2 -b:a 192k [output_mp3_full_path]
```
#### Step2: Run the distil-whisper/distil-large-v3 model to get the transcript
replace [audio_full_path] and [output_csv_root_folder] with the full path of the audio file and the output csv folder
```
cd ~/workspace/transcriptions/whisper_pipeline
conda activate torch
python3 whisper_transcribe_single_video.py --input_audio [audio_full_path] --output_transcript_folder [output_csv_root_folder]
```

### Demo: All videos
#### Extract the audio from all of the videos and the stable-whisper/large-v3 model to get the word-level transcripts
```
cd ~/workspace/transcriptions/whisper_pipeline
conda activate torch
python run_whisper.py
```
