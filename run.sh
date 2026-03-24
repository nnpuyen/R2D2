#!/usr/bin/env bash
set -euo pipefail

file_path=$1

if [ ! -f "$file_path" ]; then
    echo "Error: Config file '$file_path' not found"
    exit 1
fi

params=""
while IFS= read -r line || [ -n "$line" ]
do
    [[ -z "$line" || "$line" =~ ^#.*$ ]] && continue
    name="--$(cut -d '=' -f1 <<<"$line")"
    val="$(cut -d '=' -f2 <<<"$line")"
    val="$(cut -d '"' -f2 <<<"$val")"
    params="$params $name $val"
done < "$file_path"

export PYTHONPATH="."

# If user did not provide CUDA_VISIBLE_DEVICES externally, auto-detect GPU support.
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        gpu_id=0
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        echo "Using GPU $gpu_id"
    else
        export CUDA_VISIBLE_DEVICES="-1"
        export TF_DISABLE_CUDA="1"
        echo "No NVIDIA driver detected. Falling back to CPU mode."
    fi
fi

# Reduce TensorFlow backend startup noise (can be overridden by user env).
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

cmd="python3 code/model/trainer.py $params"



echo "Executing $cmd"

$cmd
