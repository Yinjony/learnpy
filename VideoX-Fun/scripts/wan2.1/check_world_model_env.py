"""Preflight checks for the integrated minWM Wan2.1 world-model trainer."""

import argparse
import importlib.util
import os
from pathlib import Path


VIDEOX_FUN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MINWM_ROOT = VIDEOX_FUN_ROOT.parent / "minWM"
MODEL_NAME = "Wan2.1-T2V-1.3B"

REQUIRED_MODULES = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "diffusers": "diffusers",
    "easydict": "easydict",
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
    "torchvision": "torchvision",
    "tqdm": "tqdm",
    "transformers": "transformers",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minwm-root",
        type=Path,
        default=Path(os.environ.get("MINWM_ROOT", DEFAULT_MINWM_ROOT)),
        help="Path to the minWM repository. Defaults to the sibling minWM folder.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Camera latent LMDB directory. Defaults to minWM/dataset/Wan21/Action2V/data.",
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
    minwm_root = args.minwm_root.expanduser().resolve()
    data_path = (
        args.data_path.expanduser().resolve()
        if args.data_path is not None
        else minwm_root / "dataset" / "Wan21" / "Action2V" / "data"
    )
    model_dir = minwm_root / "Wan21" / "wan_models" / MODEL_NAME

    ok = True
    print("Python dependencies")
    for package, module in REQUIRED_MODULES.items():
        found = importlib.util.find_spec(module) is not None
        ok = report(package, found, module) and ok

    print("\nminWM source and model files")
    required_paths = [
        minwm_root / "Wan21" / "configs" / "default_config.yaml",
        minwm_root / "Wan21" / "wan_utils",
        minwm_root / "shared" / "algorithms" / "flow_matching.py",
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
            torch_version >= (2, 5),
            f"torch={torch.__version__}, required>=2.5",
        ) and ok
        ok = report(
            "torch CUDA",
            torch.cuda.is_available(),
            f"torch={torch.__version__}, available={torch.cuda.is_available()}",
        ) and ok

    if not ok:
        print(
            "\nPreflight failed. Install requirements_world_model.txt, install "
            "flash-attn separately, download the Wan2.1 weights, create the "
            "wan_models link, and build the camera LMDB."
        )
        raise SystemExit(1)

    print("\nWorld-model training environment is ready.")


if __name__ == "__main__":
    main()
