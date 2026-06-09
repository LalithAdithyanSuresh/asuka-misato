#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set Hugging Face cache directory to a writable local path to avoid permission issues in /home/cks
export HF_HOME="/tmp/cks/.cache/huggingface"

# Automatically run inside a GNU screen session named 'asuka' if screen is available
if [ -z "$STY" ]; then
    if command -v screen >/dev/null 2>&1; then
        echo "[*] Re-running script inside a GNU screen session named 'asuka'..."
        exec screen -S asuka bash "$0" "$@"
    else
        echo "[!] screen command not found. Proceeding in current session..."
    fi
fi

echo "====================================================================="
echo "        ASUKA Environment Setup & Evaluation Script"
echo "====================================================================="

# Ensure we are in the asuka-flux directory
if [ ! -f "test_asuka_flux.py" ] || [ ! -d "src" ]; then
    if [ -d "asuka-flux" ]; then
        echo "[*] Moving into asuka-flux folder..."
        cd asuka-flux
    elif [ -d "asuka-misato/asuka-flux" ]; then
        echo "[*] Moving into asuka-misato/asuka-flux folder..."
        cd asuka-misato/asuka-flux
    else
        echo "Error: Cannot find asuka-flux directory. Please run this script inside the asuka-misato repository folder."
        exit 1
    fi
fi

# Create python virtual environment
ENV_NAME="asuka-env"
if [ ! -d "$ENV_NAME" ] || [ ! -f "$ENV_NAME/bin/pip" ] || [ ! -f "$ENV_NAME/bin/python" ]; then
    echo "[*] Creating/Re-creating virtual environment: $ENV_NAME..."
    rm -rf "$ENV_NAME"
    python3 -m venv "$ENV_NAME"
else
    echo "[*] Virtual environment $ENV_NAME already exists."
fi

# Activate virtual environment
echo "[*] Activating virtual environment..."
source "$ENV_NAME"/bin/activate

# Install dependencies
echo "[*] Upgrading pip..."
"$ENV_NAME"/bin/pip install --upgrade pip

echo "[*] Installing PyTorch with CUDA 12.1 compatibility..."
"$ENV_NAME"/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[*] Installing ASUKA pipeline requirements..."
"$ENV_NAME"/bin/pip install numpy pyyaml tqdm pillow scipy huggingface_hub diffusers==0.30.2 transformers==4.45.2 omegaconf einops accelerate opencv-python invisible-watermark

# Dynamically write check/download weights script
echo "[*] Ensuring checkpoint download helper is written..."
cat << 'EOF' > download_weights.py
import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

os.makedirs("ckpt/flux_fill", exist_ok=True)

# 1. Download ASUKA checkpoints from yikaiwang/ASUKA-FLUX.1-Fill
print("[*] Verifying/Downloading ASUKA checkpoints from Hugging Face...")
try:
    snapshot_download(
        repo_id="yikaiwang/ASUKA-FLUX.1-Fill",
        local_dir="ckpt",
        ignore_patterns=["*.md", "*.git*"]
    )
    print("[*] ASUKA checkpoints verified/downloaded.")
except Exception as e:
    print(f"[!] Error during default download: {e}")
    print("This repository requires a Hugging Face token. Please make sure you have accepted the terms at:")
    print("https://huggingface.co/yikaiwang/ASUKA-FLUX.1-Fill")
    token = input("Enter your Hugging Face Access Token: ").strip()
    if token:
        snapshot_download(
            repo_id="yikaiwang/ASUKA-FLUX.1-Fill",
            local_dir="ckpt",
            ignore_patterns=["*.md", "*.git*"],
            token=token
        )
        print("[*] ASUKA checkpoints verified/downloaded using token.")
    else:
        print("[!] No token provided. Exiting.")
        sys.exit(1)

# 2. Download FLUX.1-Fill-dev base weights
print("[*] Verifying/Downloading FLUX.1-Fill-dev base weights...")
flux_files = ["flux1-fill-dev.safetensors", "ae.safetensors"]
for f in flux_files:
    dest = os.path.join("ckpt/flux_fill", f)
    if not os.path.exists(dest):
        print(f"Downloading {f} from black-forest-labs/FLUX.1-Fill-dev...")
        try:
            hf_hub_download(
                repo_id="black-forest-labs/FLUX.1-Fill-dev",
                filename=f,
                local_dir="ckpt/flux_fill"
            )
        except Exception as e:
            print(f"[!] Error downloading {f}: {e}")
            print("Please make sure you have accepted the terms at:")
            print("https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev")
            token = input("Enter your Hugging Face Access Token: ").strip()
            if token:
                hf_hub_download(
                    repo_id="black-forest-labs/FLUX.1-Fill-dev",
                    filename=f,
                    local_dir="ckpt/flux_fill",
                    token=token
                )
            else:
                print("[!] No token provided. Exiting.")
                sys.exit(1)
    else:
        print(f"[*] Base weight {f} already exists.")

print("[*] Checkpoints verification/download complete.")
EOF

# Run downloader
"$ENV_NAME"/bin/python download_weights.py
rm download_weights.py

# Define default paths (user server paths)
DEFAULT_IMG_DIR="/tmp/cks/SEM-Net/datasets/places365/test_256"
DEFAULT_MASK_DIR="/tmp/cks/SEM-Net/datasets/testing_mask_dataset"
DEFAULT_OUT_DIR="./results"

# Ask user if they want to override paths
echo ""
echo "---------------------------------------------------------------------"
echo "Please confirm or specify dataset paths for evaluation:"
read -p "Enter Places365 images directory [$DEFAULT_IMG_DIR]: " IMG_DIR
IMG_DIR=${IMG_DIR:-$DEFAULT_IMG_DIR}

read -p "Enter Masks directory [$DEFAULT_MASK_DIR]: " MASK_DIR
MASK_DIR=${MASK_DIR:-$DEFAULT_MASK_DIR}

read -p "Enter Output directory [$DEFAULT_OUT_DIR]: " OUT_DIR
OUT_DIR=${OUT_DIR:-$DEFAULT_OUT_DIR}
echo "---------------------------------------------------------------------"

echo "[*] Running ASUKA model validation for the 5 target images..."
"$ENV_NAME"/bin/python validate_asuka.py \
    --image_dir "$IMG_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUT_DIR"

echo "====================================================================="
echo "                    Evaluation Completed!"
echo "====================================================================="
