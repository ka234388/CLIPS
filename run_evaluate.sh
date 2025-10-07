#!/bin/bash

# This script performs a full setup and execution for the performance evaluation.
# 1. Creates a new Conda environment named 'eval_env'.
# 2. Installs all required Python libraries from a 'requirements.txt' file.
# 3. Runs the evaluate.py script on the specified metrics file.

# --- Configuration ---
CONDA_ENV_NAME="eval_env"
# Change this path to the actual location of your metrics.csv file.
METRICS_FILE="/kaggle/working/a3_sam2_camvid/metrics.csv"
METRICS_FILE1="/kaggle/working/a3_sam2_camvid/aclip-metrics.csv"
METRICS_FILE2="/kaggle/working/a3_sam2_camvid/siglip-metrics.csv"
PYTHON_VERSION="3.9"
REQUIREMENTS_FILE="requirements.txt"

echo "--- Starting Full Evaluation Setup ---"

# --- Step 1: Create Conda Environment ---
# Check if the environment already exists
if conda env list | grep -q -w "$CONDA_ENV_NAME"; then
    echo "Conda environment '$CONDA_ENV_NAME' already exists. Activating it."
else
    echo "Conda environment '$CONDA_ENV_NAME' not found. Creating it now..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create Conda environment. Please check your Conda installation."
        exit 1
    fi
    echo "Environment created successfully."
fi

# --- Step 2: Activate Environment and Install Libraries ---
echo "Activating environment and installing required libraries from $REQUIREMENTS_FILE..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

# Check if requirements.txt file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found. Please create it in the same directory."
    conda deactivate
    exit 1
fi

# Install libraries using pip from the requirements file
pip install -r "$REQUIREMENTS_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Failed to install required Python libraries."
    conda deactivate
    exit 1
fi
echo "All libraries installed successfully."

# --- Step 3: Run the Evaluation Script ---
echo "--- Starting Performance Evaluation ---"

# Check if the metrics file exists before running
if [ ! -f "$METRICS_FILE" ]; then
    echo "Error: Metrics file not found at $METRICS_FILE"
    conda deactivate
    exit 1
fi

# Run the Python evaluation script
python evaluate.py --csv-path "$METRICS_FILE"
python a-clip-evaluate.py  --csv-path "$METRICS_FILE1"
python siglip-evaluate.py  --csv-path "$METRICS_FILE2"

echo "--- Evaluation Script Finished ---"

# --- Step 4: Deactivate the Environment ---
echo "Deactivating Conda environment."
conda deactivate

echo "--- Setup and Execution Complete ---"

