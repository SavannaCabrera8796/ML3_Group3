#!/bin/bash

#SBATCH --job-name=vgg16_erasing
#SBATCH --partition=gpu
#SBATCH --account=YOUR_ALLOCATION

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00

#SBATCH --output=erasing_%j.out
#SBATCH --error=erasing_%j.err

set -euo pipefail

# Load Python environment
module purge
module load miniforge

eval "$(conda shell.bash hook)"
conda activate /home/azd9vv/.conda/envs/ml3_vgg16

# Navigate to the project
cd /home/azd9vv/ML3_Group3

# Point to the existing IMDB-WIKI images
export MLIII_DATA_DIR=/scratch/azd9vv

# Avoid excessive CPU threading
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Keep temporary files in scratch
mkdir -p /scratch/azd9vv/tmp
export TMPDIR=/scratch/azd9vv/tmp

echo "=================================="
echo "Random Erasing VGG16 Experiment"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=================================="

# Verify PyTorch GPU access
python -u - <<'PY'
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected")

print("GPU:", torch.cuda.get_device_name(0))
PY

# Execute the random-erasing training notebook
python -m jupyter nbconvert \
    --to notebook \
    --execute random_erasing_train.ipynb \
    --output /scratch/azd9vv/random_erasing_executed.ipynb \
    --ExecutePreprocessor.timeout=-1

echo "Training completed: $(date)"
