#!/usr/bin/env bash
set -e

# Variables
GITHUB_USER="Yona544"
RVC_REPO="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion.git"

# 1. Clone the RVC repository and checkout latest release tag
echo "Cloning RVC repo..."
if [ ! -d Retrieval-based-Voice-Conversion ]; then
    git clone "$RVC_REPO"
fi
cd Retrieval-based-Voice-Conversion

echo "Fetching tags and checking out latest release..."
LATEST_TAG=$(git tag --sort=-v:refname | head -n 1)
[ -n "$LATEST_TAG" ] && git checkout "$LATEST_TAG"

# 2. Set up Python virtual environment and install dependencies
echo "Setting up Python virtual environment..."
python3.10 -m venv rvc_env
source rvc_env/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify that core libraries are available
python3 - <<'PY'
import torch, torchaudio, librosa
print("Torch version:", torch.__version__)
print("Librosa version:", librosa.__version__)
PY

# 4. Prepare dataset directory structure
cd ..
mkdir -p dataset_raw/yona
mkdir -p dataset_raw/tony

# 5. Download audio files
echo "Downloading audio files..."
wget -O dataset_raw/yona/yona_voice.wav \
  "https://raw.githubusercontent.com/${GITHUB_USER}/voicecloning/main/yona_voice.wav"
wget -O dataset_raw/tony/tony.wav \
  "https://raw.githubusercontent.com/${GITHUB_USER}/voicecloning/main/tony.wav"

# 6. Download pretrained RVC v2 checkpoints
mkdir -p models/pretrained/v2
cd models/pretrained/v2
wget -O G40k.pth   https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/pretrained/v2/G40k.pth
wget -O D40k.pth   https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/pretrained/v2/D40k.pth
wget -O f0D40k.pth https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/pretrained/v2/f0D40k.pth
wget -O f0G40k.pth https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/pretrained/v2/f0G40k.pth
cd ../../..

# 7. Generate training configuration file
mkdir -p configs
cat > configs/yona_train.yaml <<'YAML'
dataset:
  data_root: "./dataset_raw"
  train_folder: "yona"
embedding:
  embedder: "contentvec"
  ch: 768
  out_layer: 12
model:
  sample_rate: 44100
  hop_length: 256
  win_length: 1024
  n_fft: 1024
training:
  batch_size: 4
  epochs: 30
  save_every: 10
  fp16: true
checkpoints:
  generator_path: "./models/pretrained/v2/G40k.pth"
  discriminator_path: "./models/pretrained/v2/D40k.pth"
  content_encoder_path: "./models/pretrained/v2/f0D40k.pth"
  phoneme_predictor_path: "./models/pretrained/v2/f0G40k.pth"
output:
  checkpoint_dir: "./models/yona_model"
  log_dir: "./logs/yona"
YAML

# 8. Create directories for training outputs and run training
mkdir -p models/yona_model
mkdir -p logs/yona

echo "Starting training..."
python3 Retrieval-based-Voice-Conversion/train.py --config configs/yona_train.yaml
# 9. Verify final checkpoint
echo "Listing checkpoint directory..."
ls models/yona_model/

# 10. Generate inference configuration file
cat > configs/yona_infer.yaml <<'YAML'
model:
  generator_path: "./models/yona_model/G40k_epoch30.pth"
  embedding_layer: 12
  embedder: "contentvec"
  device: "cuda:0"
inference:
  input_audio: "./dataset_raw/tony/tony.wav"
  output_dir: "./outputs/tony_to_yona"
  loud_norm: true
  f0_method: "crepe"
  trans: false
  index_path: "./models/yona_model/index.faiss"
YAML

# 11. Run inference
mkdir -p outputs/tony_to_yona

echo "Running inference..."
python3 Retrieval-based-Voice-Conversion/infer.py --config configs/yona_infer.yaml

echo "Converted audio saved to outputs/tony_to_yona/tony_converted.wav"
