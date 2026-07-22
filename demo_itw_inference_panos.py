#!/usr/bin/env python3
"""
Panorama demo: Ultralytics person detection + SL-HOI on data/itw/ssup_panos.

For each YOLO person (height >= min height), SL-HOI runs on a fixed-size square crop
(default 1024×1024) centered on that person—not on the full panorama. Horizontal edges
wrap (equirectangular); vertical out-of-bounds regions are padded black. Boxes are mapped
back to pano coordinates. Shows the top-1 HOI per person. Person crops are saved under
``<output-dir>/crops/<pano_stem>/person_XXX.jpg``.

Example:
  python demo_itw_inference_panos.py \\
    --input-dir data/itw/ssup_panos \\
    --output-dir outputs/itw_pano_demo
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
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

    Equirectangular panos: wrap horizontally at image width; pad vertically with black
    when the window extends past the top or bottom.

    Returns (crop, origin_x, origin_y): unwrapped top-left of the crop window
    (crop pixel (dx, dy) samples pano x = origin_x + dx (mod width), y = origin_y + dy).
    """
    w, h = image.size
    x1, y1, x2, y2 = person_xyxy
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    origin_x = cx - size * 0.5
    origin_y = cy - size * 0.5

    crop = Image.new("RGB", (size, size), (0, 0, 0))
    for dy in range(size):
        sy = int(math.floor(origin_y + dy))
        if sy < 0 or sy >= h:
            continue
        dst = 0
        u = origin_x
        while dst < size:
            img_x = int(math.floor(u)) % w
            run = min(size - dst, w - img_x)
            strip = image.crop((img_x, sy, img_x + run, sy + 1))
            crop.paste(strip, (dst, dy))
            dst += run
            u += run
    return crop, origin_x, origin_y


def offset_box_xyxy(
    box: List[float],
    origin_x: float,
    origin_y: float,
    pano_w: int,
    pano_h: int,
) -> List[float]:
    """Map crop-space xyxy to pano pixels (wrap x, clamp y to image)."""
    x1, y1, x2, y2 = box
    ux1 = (float(x1) + origin_x) % pano_w
    ux2 = (float(x2) + origin_x) % pano_w
    uy1 = float(y1) + origin_y
    uy2 = float(y2) + origin_y
    if uy1 > uy2:
        uy1, uy2 = uy2, uy1
    uy1 = max(0.0, min(float(pano_h), uy1))
    uy2 = max(0.0, min(float(pano_h), uy2))
    return [ux1, uy1, ux2, uy2]


def pano_y_range_from_crop_box(
    box: List[float],
    origin_y: float,
    pano_h: int,
) -> Optional[tuple[float, float]]:
    _, y1, _, y2 = (float(v) for v in box)
    if y1 > y2:
        y1, y2 = y2, y1
    y1 = max(0.0, min(float(pano_h), y1 + origin_y))
    y2 = max(0.0, min(float(pano_h), y2 + origin_y))
    if y2 <= y1:
        return None
    return y1, y2


def pano_x_segments_from_crop_box(
    box: List[float],
    origin_x: float,
    pano_w: int,
) -> List[tuple[float, float]]:
    """
    Map crop x extent to one or two pano x intervals when the unwrapped span crosses width.
    """
    x1, _, x2, _ = (float(v) for v in box)
    if x1 > x2:
        x1, x2 = x2, x1
    left = x1 + origin_x
    right = x2 + origin_x
    if right <= left:
        return []

    w = float(pano_w)
    segments: List[tuple[float, float]] = []
    pos = left
    while pos < right:
        x_start = pos % w
        period_end = (math.floor(pos / w) + 1) * w
        end = min(right, period_end)
        span = end - pos
        x_end = x_start + span
        if x_end <= w:
            if x_end > x_start:
                segments.append((x_start, x_end))
        else:
            if w > x_start:
                segments.append((x_start, w))
            tail = x_end - w
            if tail > 0:
                segments.append((0.0, tail))
        pos = end
    return segments


def draw_pano_rectangle_from_crop(
    draw: ImageDraw.ImageDraw,
    box_crop: List[float],
    origin_x: float,
    origin_y: float,
    pano_w: int,
    pano_h: int,
    outline: str,
    width: int,
) -> None:
    """Draw a crop-space box on the equirectangular pano (split at horizontal wrap)."""
    y_range = pano_y_range_from_crop_box(box_crop, origin_y, pano_h)
    if y_range is None:
        return
    y0, y1 = y_range
    for x0, x2 in pano_x_segments_from_crop_box(box_crop, origin_x, pano_w):
        xy = pil_rectangle_xyxy([x0, y0, x2, y1], pano_w, pano_h)
        if xy is not None:
            draw.rectangle(xy, outline=outline, width=width)


def draw_label_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    x, y = xy
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw_label(draw, (x - tw / 2, y - th / 2), text, font)


def verb_label_from_hoi(label: str, object_name: str) -> str:
    """Interaction text minus trailing object phrase (e.g. 'riding a bicycle' -> 'riding')."""
    for suffix in (f"a {object_name}", f"an {object_name}", object_name):
        if label.endswith(suffix):
            verb = label[: -len(suffix)].strip()
            if verb:
                return verb
    return label


def box_center_unwrapped_from_crop(
    box_crop: List[float],
    origin_x: float,
    origin_y: float,
    pano_w: int,
    pano_h: int,
) -> tuple[float, float, float, float]:
    """Unwrapped (cx, cy) and display (cx % w, cy) for the object box center."""
    x1, y1, x2, y2 = (float(v) for v in box_crop)
    cx = (x1 + x2) * 0.5 + origin_x
    cy = max(0.0, min(float(pano_h), (y1 + y2) * 0.5 + origin_y))
    return cx, cy, cx % pano_w, cy


def shortest_unwrapped_x_pair(x0: float, x1: float, pano_w: int) -> tuple[float, float]:
    """Unwrapped x endpoints for the shortest horizontal path on a cylindrical pano."""
    w = float(pano_w)
    best_dist = float("inf")
    best = (x0, x1)
    for shift_a in (-w, 0.0, w):
        for shift_b in (-w, 0.0, w):
            ua = x0 + shift_a
            ub = x1 + shift_b
            dist = abs(ub - ua)
            if dist < best_dist:
                best_dist = dist
                best = (ua, ub)
    return best


def draw_wrapped_pano_line(
    draw: ImageDraw.ImageDraw,
    ux0: float,
    uy0: float,
    ux1: float,
    uy1: float,
    pano_w: int,
    fill: str,
    width: int,
) -> None:
    """Draw a straight line in unwrapped x; split into two segments at x = n * pano_w."""
    w = float(pano_w)
    lo, hi = min(ux0, ux1), max(ux0, ux1)
    seams: List[float] = []
    k = math.ceil(lo / w)
    while k * w < hi:
        sx = k * w
        if lo < sx < hi:
            seams.append(sx)
        k += 1

    points: List[tuple[float, float]] = [(ux0, uy0)]
    dx = ux1 - ux0
    for sx in seams:
        if abs(dx) < 1e-9:
            ty = uy0
        else:
            t = (sx - ux0) / dx
            ty = uy0 + t * (uy1 - uy0)
        points.append((sx, ty))
    points.append((ux1, uy1))

    for i in range(len(points) - 1):
        ax, ay = points[i]
        bx, by = points[i + 1]
        draw.line([(ax % w, ay), (bx % w, by)], fill=fill, width=width)


def person_box_center(person_xyxy: List[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = person_xyxy
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def draw_line_endpoint(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    radius: float,
    fill: str,
) -> None:
    r = radius
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill, outline="#ffffff", width=2)


def draw_hoi_link(
    draw: ImageDraw.ImageDraw,
    person_xyxy: List[float],
    obj_crop: List[float],
    origin_x: float,
    origin_y: float,
    verb_text: str,
    object_text: str,
    pano_w: int,
    pano_h: int,
    font: ImageFont.ImageFont,
    line_color: str = "#ffcc00",
    line_width: int = 3,
) -> None:
    px, py = person_box_center(person_xyxy)
    oux, oy, ox, _oy = box_center_unwrapped_from_crop(
        obj_crop, origin_x, origin_y, pano_w, pano_h
    )
    ux0, ux1 = shortest_unwrapped_x_pair(px, oux, pano_w)
    draw_wrapped_pano_line(draw, ux0, py, ux1, oy, pano_w, line_color, line_width)
    point_r = max(4.0, line_width * 2.5)
    draw_line_endpoint(draw, px, py, point_r, line_color)
    draw_line_endpoint(draw, ox, oy, point_r, line_color)
    mid_ux = (ux0 + ux1) * 0.5
    mid_uy = (py + oy) * 0.5
    draw_label_centered(draw, (mid_ux % pano_w, mid_uy), verb_text, font)
    draw_label(draw, (ox, max(0.0, oy - 22)), object_text, font)


def pil_rectangle_xyxy(
    box: List[float],
    pano_w: int,
    pano_h: int,
) -> Optional[List[float]]:
    """
    xyxy suitable for ImageDraw.rectangle (x0 <= x1, y0 <= y1, inside image).
    Returns None if the box is degenerate after normalization.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(pano_w), x1))
    x2 = max(0.0, min(float(pano_w), x2))
    y1 = max(0.0, min(float(pano_h), y1))
    y2 = max(0.0, min(float(pano_h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


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
        object_name = obj_name_by_id.get(int(verb_obj[1]), f"object_{verb_obj[1]}")
        hois.append(
            {
                "score": float(pred["score"]),
                "category_id": cat_id,
                "label": label,
                "verb_label": verb_label_from_hoi(label, object_name),
                "object_name": object_name,
                "verb_id": int(verb_obj[0]),
                "object_id": int(verb_obj[1]),
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
        raise FileNotFoundError("Classifier weights missing under weights/SL-HOI-weights/params/hico/")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = args.output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    crops_root = args.output_dir / "crops"
    crops_root.mkdir(parents=True, exist_ok=True)

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
            pano_w, pano_h = pil.size

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

            pano_crop_dir = crops_root / pano_path.stem
            pano_crop_dir.mkdir(parents=True, exist_ok=True)

            # --- SL-HOI on a person-centered crop per detection ---
            person_results: List[Dict[str, Any]] = []
            total_hoi_candidates = 0
            for person_idx, person in enumerate(persons):
                crop_pil, origin_x, origin_y = centered_square_crop(
                    pil, person["xyxy"], args.crop_size
                )
                crop_filename = f"person_{person_idx:03d}.jpg"
                crop_path = pano_crop_dir / crop_filename
                crop_pil.save(crop_path, quality=92)

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
                    sub_crop = [float(x) for x in hoi["subject_box_xyxy"]]
                    obj_crop = [float(x) for x in hoi["object_box_xyxy"]]
                    pano_hois.append(
                        {
                            **hoi,
                            "subject_box_crop_xyxy": sub_crop,
                            "object_box_crop_xyxy": obj_crop,
                            "subject_box_xyxy": offset_box_xyxy(
                                sub_crop, origin_x, origin_y, pano_w, pano_h
                            ),
                            "object_box_xyxy": offset_box_xyxy(
                                obj_crop, origin_x, origin_y, pano_w, pano_h
                            ),
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
                        "crop_image": str(crop_path.relative_to(args.output_dir)),
                        "hoi": best_hoi,
                    }
                )

            # --- Draw: person–object link, boxes (blue subject, red object) ---
            vis = pil.copy()
            draw = ImageDraw.Draw(vis)
            for pr in person_results:
                px1, py1, px2, py2 = pr["person_box_xyxy"]
                tag = f"P{pr['person_index']}"
                hoi = pr["hoi"]
                if hoi is None:
                    draw_label(draw, (px1, max(0, py1 - 26)), f"{tag} | no HOI", font)
                    continue

                crop_ox, crop_oy = pr["crop_origin_xy"]
                sub_crop = hoi.get("subject_box_crop_xyxy", hoi["subject_box_xyxy"])
                obj_crop = hoi.get("object_box_crop_xyxy", hoi["object_box_xyxy"])
                draw_pano_rectangle_from_crop(
                    draw, sub_crop, crop_ox, crop_oy, pano_w, pano_h, "#0066ff", 2
                )
                draw_pano_rectangle_from_crop(
                    draw, obj_crop, crop_ox, crop_oy, pano_w, pano_h, "#ff3333", 3
                )
                verb_text = hoi.get("verb_label") or verb_label_from_hoi(hoi["label"], hoi["object_name"])
                object_text = hoi["object_name"]
                draw_hoi_link(
                    draw,
                    pr["person_box_xyxy"],
                    obj_crop,
                    crop_ox,
                    crop_oy,
                    verb_text,
                    object_text,
                    pano_w,
                    pano_h,
                    font,
                )
                draw_label(
                    draw,
                    (px1, max(0, py1 - 26)),
                    f"{tag} | {hoi['score']:.2f}",
                    font,
                )

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
