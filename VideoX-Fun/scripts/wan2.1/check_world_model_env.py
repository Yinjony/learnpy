"""Preflight checks for the vendored Wan2.1 camera world-model trainer."""

import argparse
import importlib.util
import os
from pathlib import Path


VIDEOX_FUN_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Wan2.1-T2V-1.3B"

REQUIRED_MODULES = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "diffusers": "diffusers",
    "einops": "einops",
    "flash-attn": "flash_attn",
    "ftfy": "ftfy",
    "lmdb": "lmdb",
    "omegaconf": "omegaconf",
    "Pillow": "PIL",
    "regex": "regex",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "tensorboard": "tensorboard",
    "torch": "torch",
    "tqdm": "tqdm",
    "transformers": "transformers",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.environ.get("WAN_MODEL_DIR", VIDEOX_FUN_ROOT / "models" / MODEL_NAME)),
        help="Wan2.1 model directory. Defaults to VideoX-Fun/models/Wan2.1-T2V-1.3B.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Camera latent LMDB directory. Defaults to VideoX-Fun/datasets/Wan21/Action2V/data.",
    )
    return parser.parse_args()


def has_lmdb(data_path):
    if not data_path.is_dir():
        return False
    if (data_path / "data.mdb").is_file():
        return True
    return any((child / "data.mdb").is_file() for child in data_path.iterdir())


def report(label, ok, detail):
    status = "OK" if ok else "MISSING"
    print(f"[{status:7}] {label}: {detail}")
    return ok


def main():
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path is not None
        else VIDEOX_FUN_ROOT / "datasets" / "Wan21" / "Action2V" / "data"
    )

    ok = True
    print("Python dependencies")
    for package, module in REQUIRED_MODULES.items():
        found = importlib.util.find_spec(module) is not None
        ok = report(package, found, module) and ok

    print("\nVideoX-Fun world-model source and model files")
    required_paths = [
        VIDEOX_FUN_ROOT / "config" / "wan2.1" / "world_model_default.yaml",
        VIDEOX_FUN_ROOT / "videox_fun" / "world_model" / "model" / "camera_bidirectional_diffusion.py",
        VIDEOX_FUN_ROOT / "videox_fun" / "world_model" / "wan" / "modules" / "model.py",
        VIDEOX_FUN_ROOT / "videox_fun" / "world_model" / "wan" / "modules" / "prope.py",
        VIDEOX_FUN_ROOT / "videox_fun" / "world_model" / "wan_utils",
        VIDEOX_FUN_ROOT / "videox_fun" / "world_model" / "algorithms" / "flow_matching.py",
        model_dir / "config.json",
        model_dir / "diffusion_pytorch_model.safetensors",
        model_dir / "Wan2.1_VAE.pth",
        model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        model_dir / "google" / "umt5-xxl",
    ]
    for path in required_paths:
        ok = report("path", path.exists(), path) and ok
    ok = report("camera LMDB", has_lmdb(data_path), data_path) and ok

    print("\nCUDA")
    if importlib.util.find_spec("torch") is None:
        ok = report("torch CUDA", False, "torch is not installed") and ok
    else:
        import torch

        torch_version = tuple(
            int(part) for part in torch.__version__.split("+")[0].split(".")[:2]
        )
        ok = report(
            "torch version",
            torch_version >= (2, 1),
            f"torch={torch.__version__}, required>=2.1.2",
        ) and ok
        ok = report(
            "torch CUDA",
            torch.cuda.is_available(),
            f"torch={torch.__version__}, available={torch.cuda.is_available()}",
        ) and ok

    if not ok:
        print(
            "\nPreflight failed. Install requirements_world_model.txt, install "
            "flash-attn separately, download the Wan2.1 weights into VideoX-Fun/models, "
            "and build the camera LMDB."
        )
        raise SystemExit(1)

    print("\nWorld-model training environment is ready.")


if __name__ == "__main__":
    main()
