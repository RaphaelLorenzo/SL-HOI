#!/usr/bin/env python3
"""
Panorama demo: Ultralytics person detection + SL-HOI on data/itw/ssup_panos.

For each YOLO person (height >= min height), SL-HOI runs on a fixed-size square crop
(default 1024×1024) centered on that person—not on the full panorama—then boxes are mapped
back to pano coordinates. Shows the top-1 HOI per person.

Example:
  python demo_itw_inference_panos.py \\
    --input-dir data/itw/ssup_panos \\
    --output-dir outputs/itw_pano_demo
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from ultralytics import YOLO

from datasets.hico import make_hico_transforms
from datasets.hico_text_label import hico_obj_text_label, hico_text_label
from demo_itw_inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CLASSIFIER_EVAL,
    DEFAULT_CLASSIFIER_TRAIN,
    REPO_ROOT,
    draw_label,
    load_label_font,
    triplet_nms_filter,
)
from models import build_model
from util.misc import nested_tensor_from_tensor_list
from util.topk import top_k

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SL-HOI demo on ITW panoramas (YOLO persons + HOI).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "data/itw/ssup_panos",
        help="Directory with panorama images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/itw_pano_demo",
        help="Saved visualizations and predictions.json.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--classifier-train", type=Path, default=DEFAULT_CLASSIFIER_TRAIN)
    parser.add_argument("--classifier-eval", type=Path, default=DEFAULT_CLASSIFIER_EVAL)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/hico.yaml")
    parser.add_argument("--default-config", type=Path, default=REPO_ROOT / "configs/base.yaml")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for SL-HOI.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default="yolo11n.pt",
        help="Ultralytics weights for COCO person detection (class 0).",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=1920, help="YOLO inference size.")
    parser.add_argument(
        "--min-person-height",
        type=float,
        default=256.0,
        help="Drop person boxes whose height (y2 - y1) is below this (pixels).",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1024,
        help="Square crop size (pixels) centered on each YOLO person for SL-HOI.",
    )
    parser.add_argument(
        "--subject-iou-threshold",
        type=float,
        default=0.3,
        help="Min IoU between YOLO person box and SL-HOI subject box to assign an HOI.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Min HOI score after decoding.")
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=100,
        help="Max HOI candidates from SL-HOI before per-person assignment (HICO eval uses 100).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max number of panoramas (0 = all).")
    parser.add_argument("--show", action="store_true", help="Show each result with matplotlib.")
    return parser.parse_args()


def box_iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centered_square_crop(
    image: Image.Image,
    person_xyxy: List[float],
    size: int,
) -> tuple[Image.Image, float, float]:
    """
    Extract a size×size crop centered on the person box center.
    Pads with black when the window extends past image borders.
    Returns (crop, origin_x, origin_y): full-image coords of the crop's top-left corner
    (crop pixel (0,0) ↔ full image (origin_x, origin_y)).
    """
    w, h = image.size
    x1, y1, x2, y2 = person_xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    origin_x = cx - size * 0.5
    origin_y = cy - size * 0.5

    crop = Image.new("RGB", (size, size), (0, 0, 0))
    src_left = max(0, int(origin_x))
    src_top = max(0, int(origin_y))
    src_right = min(w, int(origin_x + size))
    src_bottom = min(h, int(origin_y + size))
    if src_right > src_left and src_bottom > src_top:
        patch = image.crop((src_left, src_top, src_right, src_bottom))
        dst_left = int(round(src_left - origin_x))
        dst_top = int(round(src_top - origin_y))
        crop.paste(patch, (dst_left, dst_top))
    return crop, origin_x, origin_y


def offset_box_xyxy(box: List[float], origin_x: float, origin_y: float) -> List[float]:
    return [box[0] + origin_x, box[1] + origin_y, box[2] + origin_x, box[3] + origin_y]


def decode_slhoi_hois(
    pil: Image.Image,
    model: torch.nn.Module,
    postprocessor: torch.nn.Module,
    transform,
    device: torch.device,
    cfg,
    max_predictions: int,
    score_threshold: float,
    hoi_triplet_keys: List,
    hoi_obj_list: List[int],
    obj_name_by_id: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Run SL-HOI on one image and return decoded HOI dicts in that image's pixel coords."""
    w, h = pil.size
    orig_size = torch.tensor([int(h), int(w)])
    tensor, _ = transform(pil, None)
    samples = nested_tensor_from_tensor_list([tensor.to(device)])
    outputs = model(samples)
    raw = postprocessor(outputs, orig_size.unsqueeze(0).to(device))[0]

    numpy_preds = {k: v.cpu().numpy() for k, v in raw.items()}
    bboxes = [{"bbox": list(b)} for b in numpy_preds["boxes"]]
    combined = numpy_preds["hoi_scores"] + (numpy_preds["obj_scores"] ** 2)[:, hoi_obj_list]

    hoi_label_grid = np.tile(np.arange(combined.shape[1]), (combined.shape[0], 1))
    sub_ids = np.tile(numpy_preds["sub_ids"], (combined.shape[1], 1)).T
    obj_ids = np.tile(numpy_preds["obj_ids"], (combined.shape[1], 1)).T
    flat_scores = combined.ravel()
    k = min(max_predictions, flat_scores.size)
    topk_scores = top_k(list(flat_scores), k)
    topk_indexes = np.array([np.where(flat_scores == s)[0][0] for s in topk_scores])

    hoi_prediction = []
    for idx, score in zip(topk_indexes, topk_scores):
        if float(score) < score_threshold:
            continue
        hoi_prediction.append(
            {
                "subject_id": int(sub_ids.ravel()[idx]),
                "object_id": int(obj_ids.ravel()[idx]),
                "category_id": int(hoi_label_grid.ravel()[idx]),
                "score": float(score),
            }
        )

    entry = {"filename": "", "predictions": bboxes, "hoi_prediction": hoi_prediction}
    if cfg.EVAL.USE_NMS_FILTER and hoi_prediction:
        entry = triplet_nms_filter(entry, cfg)

    hois: List[Dict[str, Any]] = []
    for pred in entry["hoi_prediction"]:
        if pred["score"] < score_threshold:
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
    return hois


def top_hoi_for_person(person_xyxy: List[float], hois: List[Dict[str, Any]], min_iou: float) -> Optional[Dict[str, Any]]:
    """Best-scoring HOI whose subject box overlaps the YOLO person (top-1 per person)."""
    best: Optional[Dict[str, Any]] = None
    best_iou = min_iou
    for hoi in hois:
        iou = box_iou_xyxy(person_xyxy, hoi["subject_box_xyxy"])
        if iou < min_iou:
            continue
        if best is None or hoi["score"] > best["score"] or (hoi["score"] == best["score"] and iou > best_iou):
            best = hoi
            best_iou = iou
    if best is not None:
        return best
    if not hois:
        return None
    return max(hois, key=lambda h: h["score"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.classifier_train.is_file() or not args.classifier_eval.is_file():
        raise FileNotFoundError("Classifier weights missing under SL-HOI-weights/params/hico/")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.merge(OmegaConf.load(args.default_config), OmegaConf.load(args.config))
    cfg.RUNTIME.DEVICE = args.device
    cfg.RUNTIME.EVAL = True
    cfg.ZERO_SHOT.TYPE = "default"
    cfg.ZERO_SHOT.DEL_UNSEEN = False
    cfg.ZERO_SHOT.CLASSIFIER.TRAIN = str(args.classifier_train)
    cfg.ZERO_SHOT.CLASSIFIER.EVAL = str(args.classifier_eval)

    device = torch.device(args.device)
    yolo_device = 0 if device.type == "cuda" else "cpu"

    model, _, postprocessors = build_model(cfg, is_fresh_train=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and any(k in state for k in ("model", "module")):
        state = state.get("model", state.get("module"))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    postprocessor = postprocessors["hoi"]
    transform = make_hico_transforms("val")
    yolo = YOLO(str(args.yolo_weights))

    hoi_triplet_keys = list(hico_text_label.keys())
    hoi_obj_list = [pair[1] for pair in hoi_triplet_keys]
    obj_name_by_id = {
        idx: text.replace("a photo of a ", "").replace("a photo of an ", "").replace("a photo of ", "")
        for idx, text in hico_obj_text_label
    }

    pano_paths = sorted(args.input_dir.glob("*.jpg"))
    pano_paths += sorted(args.input_dir.glob("*.jpeg"))
    pano_paths += sorted(args.input_dir.glob("*.png"))
    if args.limit > 0:
        pano_paths = pano_paths[: args.limit]
    if not pano_paths:
        raise FileNotFoundError(f"No panoramas in {args.input_dir}")

    font = load_label_font(size=20)

    all_results: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for pano_path in pano_paths:
            pil = Image.open(pano_path).convert("RGB")

            # --- YOLO: person boxes (COCO class 0), filter by height ---
            yolo_out = yolo.predict(
                source=str(pano_path),
                classes=[0],
                conf=args.yolo_conf,
                imgsz=args.yolo_imgsz,
                device=yolo_device,
                verbose=False,
            )[0]

            persons: List[Dict[str, Any]] = []
            yolo_person_count = len(yolo_out.boxes) if yolo_out.boxes is not None else 0
            if yolo_out.boxes is not None:
                for box in yolo_out.boxes:
                    xyxy = box.xyxy[0].tolist()
                    height = xyxy[3] - xyxy[1]
                    if height < args.min_person_height:
                        continue
                    persons.append(
                        {
                            "xyxy": [float(v) for v in xyxy],
                            "height": float(height),
                            "confidence": float(box.conf[0]),
                        }
                    )
            persons.sort(key=lambda p: p["confidence"], reverse=True)

            # --- SL-HOI on a person-centered crop per detection ---
            person_results: List[Dict[str, Any]] = []
            total_hoi_candidates = 0
            for person_idx, person in enumerate(persons):
                crop_pil, origin_x, origin_y = centered_square_crop(
                    pil, person["xyxy"], args.crop_size
                )
                crop_hois = decode_slhoi_hois(
                    crop_pil,
                    model,
                    postprocessor,
                    transform,
                    device,
                    cfg,
                    args.max_predictions,
                    args.score_threshold,
                    hoi_triplet_keys,
                    hoi_obj_list,
                    obj_name_by_id,
                )
                pano_hois: List[Dict[str, Any]] = []
                for hoi in crop_hois:
                    pano_hois.append(
                        {
                            **hoi,
                            "subject_box_xyxy": offset_box_xyxy(hoi["subject_box_xyxy"], origin_x, origin_y),
                            "object_box_xyxy": offset_box_xyxy(hoi["object_box_xyxy"], origin_x, origin_y),
                        }
                    )
                total_hoi_candidates += len(pano_hois)
                best_hoi = top_hoi_for_person(person["xyxy"], pano_hois, args.subject_iou_threshold)
                person_results.append(
                    {
                        "person_index": person_idx,
                        "person_box_xyxy": person["xyxy"],
                        "person_confidence": person["confidence"],
                        "person_height": person["height"],
                        "crop_origin_xy": [origin_x, origin_y],
                        "crop_size": args.crop_size,
                        "hoi": best_hoi,
                    }
                )

            # --- Draw: green = YOLO person, red = object, blue = SL-HOI subject (if assigned) ---
            vis = pil.copy()
            draw = ImageDraw.Draw(vis)
            for pr in person_results:
                px1, py1, px2, py2 = pr["person_box_xyxy"]
                # draw.rectangle([px1, py1, px2, py2], outline="#00cc00", width=3)
                tag = f"P{pr['person_index']} det {pr['person_confidence']:.2f}"
                hoi = pr["hoi"]
                label_y = max(0, py1 - 26)
                if hoi is None:
                    draw_label(draw, (px1, label_y), f"{tag} | no HOI", font)
                    continue

                sx1, sy1, sx2, sy2 = hoi["subject_box_xyxy"]
                ox1, oy1, ox2, oy2 = hoi["object_box_xyxy"]
                draw.rectangle([sx1, sy1, sx2, sy2], outline="#0066ff", width=2)
                draw.rectangle([ox1, oy1, ox2, oy2], outline="#ff3333", width=3)
                caption = f"{tag} | {hoi['label']} ({hoi['score']:.2f})"
                draw_label(draw, (px1, label_y), caption, font)

            out_path = vis_dir / pano_path.name
            vis.save(out_path, quality=92)

            record = {
                "image": pano_path.name,
                "visualization": str(out_path.relative_to(args.output_dir)),
                "num_persons_kept": len(persons),
                "num_persons_skipped_small": yolo_person_count - len(persons),
                "num_hoi_candidates": total_hoi_candidates,
                "crop_size": args.crop_size,
                "persons": person_results,
            }
            all_results.append(record)

            logger.info(
                "%s: %d persons (h>=%.0f), %d with top-1 HOI -> %s",
                pano_path.name,
                len(persons),
                args.min_person_height,
                sum(1 for p in person_results if p["hoi"] is not None),
                out_path,
            )

            if args.show:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(16, 8))
                plt.imshow(vis)
                plt.axis("off")
                plt.tight_layout()
                plt.show()

    summary_path = args.output_dir / "predictions.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "yolo_weights": str(args.yolo_weights),
                "input_dir": str(args.input_dir),
                "min_person_height": args.min_person_height,
                "crop_size": args.crop_size,
                "subject_iou_threshold": args.subject_iou_threshold,
                "score_threshold": args.score_threshold,
                "images": all_results,
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
