#!/usr/bin/env bash
# Simple launch script for HF Jobs GRPO training
# Usage: bash scripts/launch_hfjob.sh
#
# This script clones the source repo and runs training on HF Jobs.

set -e

# Training parameters
EPISODES=5
GROUP_SIZE=4
KL_COEF=0.05
LR=5e-6
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_REPO="GeniusPlums/role-drift-runs"
SOURCE_REPO="anishadamane/role-drift-env"
FLAVOR="l40sx1"

echo "=============================================="
echo "HF Jobs GRPO Training Launch"
echo "=============================================="
echo "Source:     $SOURCE_REPO"
echo "Output:     $OUTPUT_REPO"
echo "Model:      $MODEL_NAME"
echo "Episodes:  $EPISODES"
echo "Group size: $GROUP_SIZE"
echo "Flavor:     $FLAVOR"
echo "=============================================="

# Check HF auth
echo "Checking authentication..."
hf auth whoami

# Run the job
echo "Launching job..."
JOB_OUTPUT=$(hf jobs run \
    --flavor "$FLAVOR" \
    --secrets HF_TOKEN \
    -e OUTPUT_REPO="$OUTPUT_REPO" \
    -e EPISODES="$EPISODES" \
    -e GROUP_SIZE="$GROUP_SIZE" \
    -e KL_COEF="$KL_COEF" \
    -e LR="$LR" \
    -e MODEL_NAME="$MODEL_NAME" \
    --namespace GeniusPlums \
    python:3.12 \
    python -c "
import subprocess, sys, os

# Clone the source repo
print('[1/4] Cloning repo...')
subprocess.run(['git', 'clone', 'https://huggingface.co/$SOURCE_REPO', '/app'], check=True)
os.chdir('/app')

# Install dependencies
print('[2/4] Installing deps...')
deps = ['torch', 'transformers>=4.36.0', 'accelerate>=0.20.0', 
       'sentence-transformers>=2.2.0', 'fasttext-langdetect>=0.4.0',
       'langdetect>=1.0.9', 'bitsandbytes', 'huggingface_hub>=1.11.0']
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet'] + deps, check=False)

# Install project
print('[3/4] Installing project...')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)

# Run training
print('[4/4] Running training...')
os.environ['OUTPUT_REPO'] = os.environ.get('OUTPUT_REPO', '$OUTPUT_REPO')
result = subprocess.run([
    sys.executable, 'training/train_grpo_hfjobs.py',
    '--episodes', os.environ.get('EPISODES', '$EPISODES'),
    '--group-size', os.environ.get('GROUP_SIZE', '$GROUP_SIZE'),
    '--kl-coef', os.environ.get('KL_COEF', '$KL_COEF'),
    '--lr', os.environ.get('LR', '$LR'),
    '--model-name', os.environ.get('MODEL_NAME', '$MODEL_NAME'),
    '--output-repo', os.environ.get('OUTPUT_REPO', '$OUTPUT_REPO'),
])
print('Done, exit:', result.returncode)
" \
    --detach)

echo "Output: $JOB_OUTPUT"

# Extract job ID
JOB_ID=$(echo "$JOB_OUTPUT" | grep -oE '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' | head -1 || echo "")

if [ -z "$JOB_ID" ]; then
    echo "Checking recent jobs..."
    JOB_ID=$(hf jobs ps -a 2>/dev/null | head -3 | tail -1 | awk '{print $1}' || echo "unknown")
fi

echo ""
echo "=============================================="
echo "Job launched!"
echo "Job ID: $JOB_ID"
echo ""
echo "To monitor logs:"
echo "  hf jobs logs $JOB_ID -f"
echo ""
echo "To check status:"
echo "  hf jobs inspect $JOB_ID"
echo "=============================================="