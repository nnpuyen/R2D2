#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys


def parse_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        return value[1:-1]
    return value


def load_params(config_path: str) -> list[str]:
    params: list[str] = []
    with open(config_path, "r", encoding="utf-8") as cfg:
        for line in cfg:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            key = key.strip()
            if not key:
                continue
            params.extend([f"--{key}", parse_value(raw_value)])
    return params


def detect_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: run.sh <config_file>")
        return 2

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: Config file '{file_path}' not found")
        return 1

    params = load_params(file_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    # Kaggle images can expose CUDA libraries that trigger noisy XLA plugin
    # registration errors with some TensorFlow builds. Default to CPU there,
    # unless the user explicitly opts in via FORCE_TF_GPU=1.
    is_kaggle = bool(env.get("KAGGLE_URL_BASE") or env.get("KAGGLE_KERNEL_RUN_TYPE"))
    force_gpu = env.get("FORCE_TF_GPU") == "1"

    if is_kaggle and not force_gpu and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env["TF_DISABLE_CUDA"] = "1"
        print("Kaggle environment detected. Using CPU mode by default.")

    # If user did not provide CUDA_VISIBLE_DEVICES externally, auto-detect GPU support.
    if "CUDA_VISIBLE_DEVICES" not in env:
        if detect_gpu():
            env["CUDA_VISIBLE_DEVICES"] = "0"
            print("Using GPU 0")
        else:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            env["TF_DISABLE_CUDA"] = "1"
            print("No NVIDIA driver detected. Falling back to CPU mode.")

    # Reduce TensorFlow backend startup noise (can be overridden by user env).
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    python_exec = sys.executable or "python3"
    cmd = [python_exec, "code/model/trainer.py", *params]
    print(f"Executing {shlex.join(cmd)}")

    completed = subprocess.run(cmd, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
