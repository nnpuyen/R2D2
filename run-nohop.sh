#!/usr/bin/env bash
set -euo pipefail

file_path=$1
gpu_id=${2:-auto}  # Second argument: GPU id, -1 for CPU, or 'auto' (default)

# Validate config file exists
if [ ! -f "$file_path" ]; then
    echo "Error: Config file '$file_path' not found"
    exit 1
fi

# Parse configuration file
params=""
while IFS= read -r line || [ -n "$line" ]
do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^#.*$ ]] && continue
    
    name="--$(cut -d '=' -f1 <<<"$line")"
    val="$(cut -d '=' -f2 <<<"$line")"
    val="$(cut -d '"' -f2 <<<"$val")"
    params="$params $name $val"
done < "$file_path"

# Set Python path
export PYTHONPATH="."

# Configure runtime device selection.
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    if [ "$gpu_id" = "auto" ]; then
        if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
            gpu_id=0
            export CUDA_VISIBLE_DEVICES="$gpu_id"
            echo "Using GPU $gpu_id"
        else
            gpu_id=-1
            export CUDA_VISIBLE_DEVICES="-1"
            export TF_DISABLE_CUDA="1"
            echo "No NVIDIA driver detected. Falling back to CPU mode."
        fi
    elif [ "$gpu_id" = "-1" ]; then
        export CUDA_VISIBLE_DEVICES="-1"
        export TF_DISABLE_CUDA="1"
        echo "CPU mode requested."
    else
        export CUDA_VISIBLE_DEVICES="$gpu_id"
    fi
fi

# Reduce TensorFlow backend startup noise (can be overridden by user env).
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

# Create log directory if it doesn't exist
log_dir="logs"
mkdir -p "$log_dir"

# Generate log filename with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
config_name=$(basename "$file_path" .sh)
log_file="${log_dir}/${config_name}_${timestamp}.log"
pid_file="${log_dir}/${config_name}_${timestamp}.pid"

# Construct command
cmd="python3 code/model/trainer.py $params"

echo "========================================"
echo "Starting training with configuration: $file_path"
echo "GPU setting: ${CUDA_VISIBLE_DEVICES:-$gpu_id}"
echo "Log file: $log_file"
echo "PID file: $pid_file"
echo "Command: $cmd"
echo "========================================"

# Run with nohup and redirect output to log file
nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL}" TF_DISABLE_CUDA="${TF_DISABLE_CUDA:-}" $cmd > "$log_file" 2>&1 &

# Save process ID
echo $! > "$pid_file"

echo "Training started with PID: $(cat $pid_file)"
echo "To monitor progress: tail -f $log_file"
echo "To stop training: kill $(cat $pid_file)"
echo "========================================"