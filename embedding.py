"""
Stage 3 — Skeleton embedding via pretrained ResNet-18.

get_skeleton_embedding(skeleton_image_path) -> np.ndarray
Embeddings cached to disk by image MD5 hash.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

CACHE_DIR = Path(__file__).resolve().parent / "output" / "embedding_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_transform = None

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_model_and_transform():
    global _model, _transform
    if _model is not None:
        return _model, _transform

    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms

    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the final classification head → 512-dim penultimate features
    layers = list(base.children())[:-1]  # drop the Linear layer
    _model = torch.nn.Sequential(*layers)
    _model.eval()

    _transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return _model, _transform


def _image_hash(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_skeleton_embedding(skeleton_image_path: str) -> np.ndarray:
    """
    Compute the penultimate ResNet-18 feature vector for a skeleton image.
    Results are cached by MD5 hash of the image file.
    """
    img_hash = _image_hash(skeleton_image_path)
    cache_path = CACHE_DIR / f"{img_hash}.npy"

    if cache_path.exists():
        return np.load(str(cache_path))

    import torch

    model, transform = _get_model_and_transform()
    img = Image.open(skeleton_image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        features = model(tensor)  # (1, 512, 1, 1)

    embedding = features.squeeze().numpy()  # (512,)
    np.save(str(cache_path), embedding)
    return embedding


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python embedding.py <skeleton_image_path>")
        sys.exit(1)
    emb = get_skeleton_embedding(path)
    print(f"Embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
