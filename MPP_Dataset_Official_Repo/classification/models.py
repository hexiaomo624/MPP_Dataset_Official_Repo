from typing import Literal

import os
from pathlib import Path
import sys

import torch


ModelName = Literal["vit_b16", "resnet50", "convnext_tiny", "swin_tiny"]


MODEL_ID: dict[ModelName, str] = {
    "vit_b16": "vit_base_patch16_224",
    "resnet50": "resnet50",
    "convnext_tiny": "convnext_tiny",
    "swin_tiny": "swin_tiny_patch4_window7_224",
}

_PROJECT_DIR = Path(__file__).resolve().parent
_DATA_ROOT = _PROJECT_DIR.parent
_OFFLINE_DIR = _DATA_ROOT / "paper_offline_linux_py310_cu124"
if _OFFLINE_DIR.exists():
    os.environ.setdefault("HF_HOME", str((_OFFLINE_DIR / "cache" / "hf").resolve()))
    os.environ.setdefault("TORCH_HOME", str((_OFFLINE_DIR / "cache" / "torch").resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import timm


def _debug(msg: str) -> None:
    if os.environ.get("PNG_PRETRAINED_DEBUG", "0") == "1":
        print(msg, file=sys.stderr, flush=True)


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not prefix:
        return state_dict
    out: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _extract_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]  # type: ignore[return-value]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]  # type: ignore[return-value]
        return obj  # type: ignore[return-value]
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")


def _require_hf_model_safetensors(repo_dirname: str) -> Path:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise FileNotFoundError("HF_HOME is not set")
    hub_dir = Path(hf_home) / "hub"
    repo_dir = hub_dir / repo_dirname
    expected_glob = repo_dir / "snapshots" / "*" / "model.safetensors"
    matches = sorted(repo_dir.glob("snapshots/*/model.safetensors"))
    if not matches:
        raise FileNotFoundError(f"Missing HF weight file: {expected_glob}")
    return matches[0]


def _require_torchhub_checkpoint(pattern: str) -> Path:
    torch_home = os.environ.get("TORCH_HOME")
    if not torch_home:
        raise FileNotFoundError("TORCH_HOME is not set")
    ckpt_dir = Path(torch_home) / "hub" / "checkpoints"
    expected_glob = ckpt_dir / pattern
    matches = sorted(ckpt_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing torch hub checkpoint: {expected_glob}")
    return matches[0]


def _load_torchhub_pretrained(model_name: ModelName, num_classes: int) -> tuple[torch.nn.Module, Path]:
    if model_name == "resnet50":
        weights_path = _require_torchhub_checkpoint("resnet50*.pth")
    elif model_name == "swin_tiny":
        weights_path = _require_torchhub_checkpoint("swin_tiny_patch4_window7_224*.pth")
    else:
        raise ValueError(f"Unsupported torch-hub pretrained model: {model_name}")

    model = timm.create_model(MODEL_ID[model_name], pretrained=False, num_classes=num_classes)
    obj = torch.load(weights_path, map_location="cpu")
    state_dict = _extract_state_dict(obj)
    state_dict = _strip_prefix(state_dict, "module.")
    model_sd = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    return model, weights_path


def create_model(
    model_name: ModelName,
    num_classes: int,
    pretrained: bool = True,
    device: torch.device | None = None,
) -> torch.nn.Module:
    if model_name not in MODEL_ID:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: {sorted(MODEL_ID)}")

    if pretrained:
        if model_name == "vit_b16":
            weights_path = _require_hf_model_safetensors("models--timm--vit_base_patch16_224.augreg2_in21k_ft_in1k")
            _debug(f"pretrained_source model={model_name} path={weights_path}")
            m = timm.create_model(MODEL_ID[model_name], pretrained=True, num_classes=num_classes)
        elif model_name == "convnext_tiny":
            weights_path = _require_hf_model_safetensors("models--timm--convnext_tiny.in12k_ft_in1k")
            _debug(f"pretrained_source model={model_name} path={weights_path}")
            m = timm.create_model(MODEL_ID[model_name], pretrained=True, num_classes=num_classes)
        elif model_name in ("resnet50", "swin_tiny"):
            m, weights_path = _load_torchhub_pretrained(model_name=model_name, num_classes=num_classes)
            _debug(f"pretrained_source model={model_name} path={weights_path}")
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    else:
        m = timm.create_model(MODEL_ID[model_name], pretrained=False, num_classes=num_classes)
    if device is not None:
        m = m.to(device)
    return m


def supported_models() -> list[str]:
    return list(MODEL_ID.keys())
