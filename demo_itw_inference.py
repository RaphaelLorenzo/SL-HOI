#!/usr/bin/env python3
"""
Run SL-HOI inference on ITW person crops (data/itw/ssup_crops).

Example:
  python demo_itw_inference.py \\
    --input-dir data/itw/ssup_crops \\
    --output-dir outputs/itw_demo \\
    --show
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from datasets.hico import make_hico_transforms
from datasets.hico_text_label import hico_obj_text_label, hico_text_label
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from util.topk import top_k

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "weights/SL-HOI-weights/pretrained/hico/pytorch_model.bin"
DEFAULT_CLASSIFIER_TRAIN = REPO_ROOT / "weights/SL-HOI-weights/params/hico/classifier_default.pt"
DEFAULT_CLASSIFIER_EVAL = REPO_ROOT / "weights/SL-HOI-weights/params/hico/classifier_eval.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SL-HOI demo on ITW SSUP person crops.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data/itw/ssup_crops",
        help="Directory with person crop images (.jpg, .jpeg, .png).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/itw_demo",
        help="Directory for saved visualizations and JSON predictions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="SL-HOI checkpoint trained on HICO-DET.",
    )
    parser.add_argument(
        "--classifier-train",
        type=Path,
        default=DEFAULT_CLASSIFIER_TRAIN,
        help="HOI classifier weights (train split).",
    )
    parser.add_argument(
        "--classifier-eval",
        type=Path,
        default=DEFAULT_CLASSIFIER_EVAL,
        help="HOI classifier weights (eval split).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/hico.yaml",
        help="Model config (HICO).",
    )
    parser.add_argument(
        "--default-config",
        type=Path,
        default=REPO_ROOT / "configs/base.yaml",
        help="Base config merged with --config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for SL-HOI.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Minimum HOI score to draw (after top-k selection).",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=15,
        help="Max HOI predictions per image (same cap as HICO eval).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N crop images (0 = all).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each saved visualization with matplotlib (blocks until closed).",
    )
    return parser.parse_args()


def _pairwise_triplet_nms(
    subs: np.ndarray,
    objs: np.ndarray,
    scores: np.ndarray,
    thres_nms: float,
    nms_alpha: float,
    nms_beta: float,
) -> List[int]:
    """HICO eval triplet NMS (joint suppression on subject + object boxes)."""
    sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
    ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]
    sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
    obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)
    order = scores.argsort()[::-1]
    keep_inds: List[int] = []
    while order.size > 0:
        i = order[0]
        keep_inds.append(int(i))
        sxx1 = np.maximum(sx1[i], sx1[order[1:]])
        syy1 = np.maximum(sy1[i], sy1[order[1:]])
        sxx2 = np.minimum(sx2[i], sx2[order[1:]])
        syy2 = np.minimum(sy2[i], sy2[order[1:]])
        sw = np.maximum(0.0, sxx2 - sxx1 + 1)
        sh = np.maximum(0.0, syy2 - syy1 + 1)
        sub_inter = sw * sh
        sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter
        oxx1 = np.maximum(ox1[i], ox1[order[1:]])
        oyy1 = np.maximum(oy1[i], oy1[order[1:]])
        oxx2 = np.minimum(ox2[i], ox2[order[1:]])
        oyy2 = np.minimum(oy2[i], oy2[order[1:]])
        ow = np.maximum(0.0, oxx2 - oxx1 + 1)
        oh = np.maximum(0.0, oyy2 - oyy1 + 1)
        obj_inter = ow * oh
        obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter
        ovr = np.power(sub_inter / sub_union, nms_alpha) * np.power(obj_inter / obj_union, nms_beta)
        inds = np.where(ovr <= thres_nms)[0]
        order = order[inds + 1]
    return keep_inds


def triplet_nms_filter(entry: Dict[str, Any], cfg) -> Dict[str, Any]:
    pred_bboxes = entry["predictions"]
    pred_hois = entry["hoi_prediction"]
    all_triplets: Dict[int, Dict[str, List]] = defaultdict(lambda: {"subs": [], "objs": [], "scores": [], "indexes": []})
    for index, pred_hoi in enumerate(pred_hois):
        triplet = pred_hoi["category_id"]
        all_triplets[triplet]["subs"].append(pred_bboxes[pred_hoi["subject_id"]]["bbox"])
        all_triplets[triplet]["objs"].append(pred_bboxes[pred_hoi["object_id"]]["bbox"])
        all_triplets[triplet]["scores"].append(pred_hoi["score"])
        all_triplets[triplet]["indexes"].append(index)

    all_keep_inds: List[int] = []
    for values in all_triplets.values():
        keep_inds = _pairwise_triplet_nms(
            np.array(values["subs"]),
            np.array(values["objs"]),
            np.array(values["scores"]),
            cfg.EVAL.THRES_NMS,
            cfg.EVAL.NMS_ALPHA,
            cfg.EVAL.NMS_BETA,
        )
        all_keep_inds.extend(list(np.array(values["indexes"])[keep_inds]))

    entry["hoi_prediction"] = list(np.array(entry["hoi_prediction"])[all_keep_inds])
    return entry


def load_label_font(size: int = 18) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "FreeSansBold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def background_font_for(font: ImageFont.ImageFont, scale: float = 1.35) -> ImageFont.ImageFont:
    base_size = getattr(font, "size", 20)
    return load_label_font(max(base_size + 2, int(round(base_size * scale))))


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.ImageFont) -> None:
    """Yellow caption with larger white text behind for readability on panos/crops."""
    x, y = xy
    bg_font = background_font_for(font)
    fg_bbox = draw.textbbox((0, 0), text, font=font)
    bg_bbox = draw.textbbox((0, 0), text, font=bg_font)
    fg_w = fg_bbox[2] - fg_bbox[0]
    fg_h = fg_bbox[3] - fg_bbox[1]
    bg_w = bg_bbox[2] - bg_bbox[0]
    bg_h = bg_bbox[3] - bg_bbox[1]
    bg_x = x + (fg_w - bg_w) / 2
    bg_y = y + (fg_h - bg_h) / 2
    draw.text((bg_x, bg_y), text, fill="#ffffff", font=bg_font)
    draw.text((x, y), text, fill="#ffff00", font=font)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.classifier_train.is_file() or not args.classifier_eval.is_file():
        raise FileNotFoundError("Classifier weights missing under weights/SL-HOI-weights/params/hico/")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hoi_vis_dir = args.output_dir / "hoi_visualizations"
    hoi_vis_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load(args.default_config), OmegaConf.load(args.config))
    cfg.RUNTIME.DEVICE = args.device
    cfg.RUNTIME.EVAL = True
    cfg.ZERO_SHOT.TYPE = "default"
    cfg.ZERO_SHOT.DEL_UNSEEN = False
    cfg.ZERO_SHOT.CLASSIFIER.TRAIN = str(args.classifier_train)
    cfg.ZERO_SHOT.CLASSIFIER.EVAL = str(args.classifier_eval)

    device = torch.device(args.device)
    model, _, postprocessors = build_model(cfg, is_fresh_train=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and any(k in state for k in ("model", "module")):
        state = state.get("model", state.get("module"))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    postprocessor = postprocessors["hoi"]
    transform = make_hico_transforms("val")

    # Label lookup (same triplet ordering as HICO evaluation).
    hoi_triplet_keys = list(hico_text_label.keys())
    hoi_obj_list = [pair[1] for pair in hoi_triplet_keys]
    obj_name_by_id = {
        idx: text.replace("a photo of a ", "").replace("a photo of an ", "").replace("a photo of ", "")
        for idx, text in hico_obj_text_label
    }

    crop_paths = sorted(args.input_dir.glob("*.jpg"))
    crop_paths += sorted(args.input_dir.glob("*.jpeg"))
    crop_paths += sorted(args.input_dir.glob("*.png"))
    if args.limit > 0:
        crop_paths = crop_paths[: args.limit]
    if not crop_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    font = load_label_font(size=18)

    all_results: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for crop_path in crop_paths:
            pil = Image.open(crop_path).convert("RGB")
            w, h = pil.size

            # --- Forward pass (val preprocessing + SL-HOI) ---
            orig_size = torch.tensor([int(h), int(w)])
            tensor, _ = transform(pil, None)
            samples = nested_tensor_from_tensor_list([tensor.to(device)])
            outputs = model(samples)
            raw = postprocessor(outputs, orig_size.unsqueeze(0).to(device))[0]

            # --- Decode to HOI triplets (HICO eval scoring + top-k + NMS) ---
            numpy_preds = {k: v.cpu().numpy() for k, v in raw.items()}
            bboxes = [{"bbox": list(b)} for b in numpy_preds["boxes"]]
            combined = numpy_preds["hoi_scores"] + (numpy_preds["obj_scores"] ** 2)[:, hoi_obj_list]

            hoi_label_grid = np.tile(np.arange(combined.shape[1]), (combined.shape[0], 1))
            sub_ids = np.tile(numpy_preds["sub_ids"], (combined.shape[1], 1)).T
            obj_ids = np.tile(numpy_preds["obj_ids"], (combined.shape[1], 1)).T
            flat_scores = combined.ravel()
            k = min(args.max_predictions, flat_scores.size)
            topk_scores = top_k(list(flat_scores), k)
            topk_indexes = np.array([np.where(flat_scores == s)[0][0] for s in topk_scores])

            hoi_prediction = []
            for idx, score in zip(topk_indexes, topk_scores):
                if float(score) < args.score_threshold:
                    continue
                hoi_prediction.append(
                    {
                        "subject_id": int(sub_ids.ravel()[idx]),
                        "object_id": int(obj_ids.ravel()[idx]),
                        "category_id": int(hoi_label_grid.ravel()[idx]),
                        "score": float(score),
                    }
                )

            entry = {"filename": crop_path.name, "predictions": bboxes, "hoi_prediction": hoi_prediction}
            if cfg.EVAL.USE_NMS_FILTER and hoi_prediction:
                entry = triplet_nms_filter(entry, cfg)

            hois = []
            for pred in entry["hoi_prediction"]:
                if pred["score"] < args.score_threshold:
                    continue
                cat_id = pred["category_id"]
                verb_obj = hoi_triplet_keys[cat_id]
                label_text = hico_text_label[verb_obj]
                label = label_text.replace("a photo of a person ", "").replace("a photo of ", "")
                hois.append(
                    {
                        "score": float(pred["score"]),
                        "category_id": cat_id,
                        "label": label,
                        "verb_id": int(verb_obj[0]),
                        "object_id": int(verb_obj[1]),
                        "object_name": obj_name_by_id.get(int(verb_obj[1]), f"object_{verb_obj[1]}"),
                        "subject_box_xyxy": [float(x) for x in bboxes[pred["subject_id"]]["bbox"]],
                        "object_box_xyxy": [float(x) for x in bboxes[pred["object_id"]]["bbox"]],
                    }
                )

            # --- Draw: blue = person (subject), red = object ---
            vis = pil.copy()
            draw = ImageDraw.Draw(vis)
            for i, hoi in enumerate(hois):
                sx1, sy1, sx2, sy2 = hoi["subject_box_xyxy"]
                ox1, oy1, ox2, oy2 = hoi["object_box_xyxy"]
                draw.rectangle([sx1, sy1, sx2, sy2], outline="#0066ff", width=2)
                draw.rectangle([ox1, oy1, ox2, oy2], outline="#ff3333", width=2)
                caption = f"{i + 1}. {hoi['label']} ({hoi['score']:.2f})"
                draw_label(draw, (sx1, max(0, sy1 - 22 * (i + 1))), caption, font)

            out_path = hoi_vis_dir / crop_path.name
            vis.save(out_path, quality=95)
            all_results.append(
                {
                    "image": crop_path.name,
                    "visualization": str(out_path.relative_to(args.output_dir)),
                    "num_predictions": len(hois),
                    "predictions": hois,
                }
            )

            logger.info("%s: %d HOI predictions -> %s", crop_path.name, len(hois), out_path)
            for h in hois[:5]:
                logger.info("  %.3f  %s", h["score"], h["label"])

            if args.show:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 8))
                plt.imshow(vis)
                plt.axis("off")
                plt.tight_layout()
                plt.show()

    summary_path = args.output_dir / "predictions.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "input_dir": str(args.input_dir),
                "score_threshold": args.score_threshold,
                "images": all_results,
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
