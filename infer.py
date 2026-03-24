"""Inference script for viewpoint scoring and ranking.

Loads a Lightning checkpoint directly — no weight export step required.

Tasks
-----
score   Score one or more images; outputs a value in [0, 1].
rank    Compare two images and report which is preferred and by what margin.

Usage
-----
    # Score
    uv run infer.py score --ckpt path/to/epoch=X.ckpt --image a.png b.png

    # Rank (pairwise)
    uv run infer.py rank --ckpt path/to/epoch=X.ckpt --image0 a.png --image1 b.png
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as TF  # noqa: N812
from PIL import Image

from learning.model import ViewpointRankingModel, ViewpointScoreModel

# EncoderSimple uses AvgPool2d(img_size // 16), trained at 256x256
INPUT_SIZE = 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def preprocess(image_path: Path) -> torch.Tensor:
    """Load an image, resize to INPUT_SIZE, and convert to a (1, 3, H, W) tensor."""
    img = Image.open(image_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    return TF.to_tensor(img).unsqueeze(0)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@torch.no_grad()
def task_score(args: argparse.Namespace) -> None:
    """Predict a viewpoint quality score in [0, 1] for each image."""
    model = ViewpointScoreModel.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    print(f"Loaded  ViewpointScoreModel  hparams: {dict(model.hparams)}\n")

    for img_path in args.image:
        x = preprocess(img_path)
        score = model.model(x).sigmoid().item()
        print(f"{img_path.name:50s}  score = {score:.4f}")


@torch.no_grad()
def task_rank(args: argparse.Namespace) -> None:
    """Compare two images and report which viewpoint is preferred."""
    model = ViewpointRankingModel.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    print(f"Loaded  ViewpointRankingModel  hparams: {dict(model.hparams)}\n")

    x0 = preprocess(args.image0)
    x1 = preprocess(args.image1)

    if model.decoder == "binary":
        logit = model.model(x0, x1)
        prob = logit.sigmoid().item()
        preferred = args.image0.name if prob > 0.5 else args.image1.name  # noqa: PLR2004
        margin = abs(prob - 0.5) * 2  # 0 = tie, 1 = certain
    else:  # goodness
        s0 = model.model(x0).sigmoid().item()
        s1 = model.model(x1).sigmoid().item()
        prob = 1.0 / (1.0 + (s1 - s0))  # sigmoid of score difference
        preferred = args.image0.name if s0 > s1 else args.image1.name
        margin = abs(s0 - s1)

    print(f"image0 : {args.image0.name}")
    print(f"image1 : {args.image1.name}")
    print(f"preferred : {preferred}  (margin = {margin:.4f}, p(image0) = {prob:.4f})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the requested task."""
    parser = argparse.ArgumentParser(description="Viewpoint inference (score / rank)")
    sub = parser.add_subparsers(dest="task", required=True)

    # -- score ----------------------------------------------------------------
    p_score = sub.add_parser("score", help="Predict viewpoint quality score for each image")
    p_score.add_argument("--ckpt", required=True, type=Path, help="Path to Lightning .ckpt file")
    p_score.add_argument("--image", required=True, nargs="+", type=Path, help="Image(s) to score")

    # -- rank -----------------------------------------------------------------
    p_rank = sub.add_parser("rank", help="Compare two images and predict which is preferred")
    p_rank.add_argument("--ckpt", required=True, type=Path, help="Path to Lightning .ckpt file")
    p_rank.add_argument("--image0", required=True, type=Path, help="First image")
    p_rank.add_argument("--image1", required=True, type=Path, help="Second image")

    args = parser.parse_args()

    if args.task == "score":
        task_score(args)
    elif args.task == "rank":
        task_rank(args)


if __name__ == "__main__":
    main()
