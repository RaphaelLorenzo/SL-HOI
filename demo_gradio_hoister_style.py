#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
import gradio as gr
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from PIL import Image
from torchvision.ops import box_iou

SLHOI_ROOT = os.path.dirname(os.path.abspath(__file__))
if SLHOI_ROOT not in sys.path:
    sys.path.insert(0, SLHOI_ROOT)
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "slhoi-gradio-matplotlib"))

from cam_reader import CamReader
from datasets.hico import make_hico_transforms
from datasets.hico_text_label import hico_text_label, hico_unseen_index
from datasets.swig import prepare_dataset_text
import datasets.swig_transforms as swig_transforms
from datasets.swig_v1_categories import SWIG_INTERACTIONS
from gradio_utils.app import build_app
from gradio_utils.tracking import (
    InteractionStabilizer,
    ObjectClassStabilizer,
    PersonColorTracker,
    TargetColorStabilizer,
)
from gradio_utils.visualization import (
    build_bars_html,
    display_image_from_state,
    draw_all_pairs,
    keep_live_video_frame,
    resolve_input_image,
    show_live_video_frame,
    show_video_input,
)
from models import build_model
from util.config_manager import load_config
import util.misc as utils


LOGGER = logging.getLogger("slhoi_hoister_style_demo")

DEFAULT_SLHOI_ROOT = SLHOI_ROOT
DEFAULT_DATASET_ROOTS = {
    "hico": "",
    "swig": "",
}
DEFAULT_MODEL_CONFIGS = {
    "hico": {
        "config": "configs/hico.yaml",
        "weights": "weights/SL-HOI-weights/pretrained/hico/pytorch_model.bin",
        "classifier_eval": "weights/SL-HOI-weights/params/hico/classifier_eval.pt",
        "classifier_train": "weights/SL-HOI-weights/params/hico/classifier_default.pt",
    },
    "hico_ov": {
        "config": "configs/hico.yaml",
        "weights": "weights/SL-HOI-weights/pretrained/hico_ov/pytorch_model.bin",
        "classifier_eval": "weights/SL-HOI-weights/params/hico/classifier_eval.pt",
        "classifier_train": "weights/SL-HOI-weights/params/hico/classifier_default.pt",
    },
    "swig": {
        "config": "configs/swig.yaml",
        "weights": "weights/SL-HOI-weights/pretrained/swig/pytorch_model.bin",
        "classifier_eval": "weights/SL-HOI-weights/params/swig/classifier_swig_dict.pt",
        "classifier_train": "weights/SL-HOI-weights/params/swig/classifier_swig_dict.pt",
    },
}
DEFAULT_BASE_CONFIG = "configs/base.yaml"
DEFAULT_CAMERA_URL = "http://root:axis0@172.16.46.6/mjpg/video.mjpg"
DEFAULT_VIDEO_SAMPLE_FPS = 4.0
DEFAULT_VIDEO_INFERENCE_SIZE = 800
DEFAULT_INTERACTION_THRESHOLD = 0.3
PAIR_OBJECT_NMS_IOU = 0.5
NO_INTERACTION_VERB_IDS = {57, 58}
INTERACTION_EMA_ALPHA = 0.25
INTERACTION_HYSTERESIS_EXIT_RATIO = 0.65
PERSON_CLUSTER_IOU = 0.45


@dataclass
class DemoState:
    model: torch.nn.Module
    postprocessors: dict
    device: str
    dataset: str
    cfg: object
    param_dtype: torch.dtype
    compat_by_object: dict
    unseen_pairs: set
    hicodet_pairs: set
    text_embeddings: Optional[torch.Tensor] = None
    dataset_val: Optional[object] = None
    swig_id_to_name: Optional[dict] = None


STATE: Optional[DemoState] = None
MODEL_LOCK = threading.RLock()
CAMERA_TRACKER = None
CAMERA_OBJECT_STABILIZER = None
CAMERA_INTERACTION_STABILIZER = None
CAMERA_TARGET_COLOR_STABILIZER = None
CAMERA_FRAME_IDX = 0
NETWORK_CAMERA_READER = None
NETWORK_CAMERA_SOURCE = None


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    LOGGER.setLevel(logging.INFO)


def _abs_slhoi_path(path: str, slhoi_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(slhoi_root, path)


def _load_model_only_weights(model, pretrained_path: str):
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")
    LOGGER.info("Loading checkpoint from: %s", pretrained_path)
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "module" in ckpt:
        state_dict = ckpt["module"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Missing keys: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys: %s", unexpected)


def _build_swig_text_embeddings(cfg, text_mapper, device, dtype):
    classifier_weights = torch.load(
        cfg.ZERO_SHOT.CLASSIFIER.EVAL,
        map_location="cpu",
        weights_only=False,
    )
    hoi_classifier = classifier_weights["hoi_embeddings"]
    text_ids = list(text_mapper.values())
    text_embeddings = torch.cat(
        [hoi_classifier[text_id].unsqueeze(0) for text_id in text_ids],
        dim=0,
    )
    return text_embeddings.to(device=device, dtype=dtype)


def _build_hico_metadata():
    hicodet_pairs = {tuple(map(int, key)) for key in hico_text_label.keys()}
    triplet_labels = list(hico_text_label.keys())
    unseen_pairs = {
        tuple(map(int, triplet_labels[idx]))
        for idx in hico_unseen_index["rare_first"]
    }
    compat_by_object = {}
    for (verb_id, obj_id), label in hico_text_label.items():
        compat_by_object.setdefault(int(obj_id), []).append((int(verb_id), label))
    return compat_by_object, unseen_pairs, hicodet_pairs


def _build_swig_metadata():
    swig_id_to_name = {int(item["id"]): str(item["name"]) for item in SWIG_INTERACTIONS}
    return {}, set(), set(), swig_id_to_name


def _format_interaction_label(text: str) -> str:
    prefix = "a photo of a person "
    text = str(text)
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _hico_label(verb_id: int, obj_id: int) -> str:
    return _format_interaction_label(
        hico_text_label.get((int(verb_id), int(obj_id)), f"verb_{verb_id} obj_{obj_id}")
    )


def _is_no_interaction_verb(verb_id: int) -> bool:
    if int(verb_id) in NO_INTERACTION_VERB_IDS:
        return True
    label = _hico_label(int(verb_id), 0).lower()
    return "no interaction" in label or "no interactions" in label


def load_demo_state(
    dataset: str,
    variant: str,
    device: str,
    slhoi_root: str,
    config_path: str | None,
    base_config_path: str | None,
    ckpt_path: str | None,
    classifier_eval_path: str | None,
    classifier_train_path: str | None,
    dataset_root: str | None,
) -> DemoState:
    if dataset == "hico" and variant not in {"hico", "hico_ov"}:
        raise ValueError("HICO dataset expects --variant hico or hico_ov.")
    if dataset == "swig":
        variant = "swig"

    defaults = DEFAULT_MODEL_CONFIGS[variant]
    config_path = _abs_slhoi_path(config_path or defaults["config"], slhoi_root)
    base_config_path = _abs_slhoi_path(base_config_path or DEFAULT_BASE_CONFIG, slhoi_root)
    ckpt_path = _abs_slhoi_path(ckpt_path or defaults["weights"], slhoi_root)
    classifier_eval_path = _abs_slhoi_path(
        classifier_eval_path or defaults["classifier_eval"],
        slhoi_root,
    )
    classifier_train_path = _abs_slhoi_path(
        classifier_train_path or defaults["classifier_train"],
        slhoi_root,
    )
    dataset_root = dataset_root or DEFAULT_DATASET_ROOTS[dataset]

    overrides = [
        f"RUNTIME.DEVICE={device}",
        f"RUNTIME.PRETRAINED={ckpt_path}",
        f"RUNTIME.EVAL=true",
        f"ZERO_SHOT.CLASSIFIER.EVAL={classifier_eval_path}",
        f"ZERO_SHOT.CLASSIFIER.TRAIN={classifier_train_path}",
        f"INPUT.DATASET_FILE={dataset}",
        f"INPUT.PATH={dataset_root}",
    ]

    LOGGER.info("Loading SL-HOI %s model once.", variant.upper())
    kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=30))
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[kwargs],
    )
    cfg = load_config(
        accelerator=accelerator,
        config_path=config_path,
        default_config_path=base_config_path,
        cli_config_overrides=overrides,
    )
    model, _criterion, postprocessors = build_model(cfg, is_fresh_train=False)
    model.to(device)
    model.eval()
    _load_model_only_weights(model, ckpt_path)
    param_dtype = next(model.parameters()).dtype

    dataset_val = None
    text_embeddings = None
    swig_id_to_name = None
    if dataset == "swig":
        _texts, text_mapper = prepare_dataset_text("val")
        dataset_val = SimpleNamespace(text_mapper=text_mapper)
        text_embeddings = _build_swig_text_embeddings(
            cfg,
            text_mapper,
            torch.device(device),
            param_dtype,
        )
        compat_by_object, unseen_pairs, hicodet_pairs, swig_id_to_name = _build_swig_metadata()
    else:
        compat_by_object, unseen_pairs, hicodet_pairs = _build_hico_metadata()

    LOGGER.info("Model loaded.")
    return DemoState(
        model=model,
        postprocessors=postprocessors,
        device=device,
        dataset=dataset,
        cfg=cfg,
        param_dtype=param_dtype,
        compat_by_object=compat_by_object,
        unseen_pairs=unseen_pairs,
        hicodet_pairs=hicodet_pairs,
        text_embeddings=text_embeddings,
        dataset_val=dataset_val,
        swig_id_to_name=swig_id_to_name,
    )


def _resize_for_inference(image: Image.Image, max_side: int | float | None):
    image = image.convert("RGB")
    if max_side is None or float(max_side) <= 0:
        return image, image, 1.0, 1.0
    width, height = image.size
    longest = max(width, height)
    if longest <= int(max_side):
        return image, image, 1.0, 1.0
    scale = float(max_side) / float(longest)
    infer_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    infer_image = image.resize(infer_size, resample=resample)
    return image, infer_image, width / infer_size[0], height / infer_size[1]


def _scale_box_xyxy(box, sx: float, sy: float):
    x1, y1, x2, y2 = map(float, box)
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def _scale_result_state_to_display(result_state, display_image: Image.Image, sx: float, sy: float):
    if result_state is None or (sx == 1.0 and sy == 1.0):
        return result_state
    for pair in result_state.get("pairs", []):
        pair["person_box"] = _scale_box_xyxy(pair["person_box"], sx, sy)
        pair["object_box"] = _scale_box_xyxy(pair["object_box"], sx, sy)
    result_state["input_image"] = display_image
    result_state["image"] = display_image
    result_state["display_image"] = draw_all_pairs(display_image, result_state.get("pairs", []))
    return result_state


def _preprocess_hico_pil(image: Image.Image):
    transform = make_hico_transforms("val")
    image_tensor, _target = transform(image.convert("RGB"), None)
    samples = utils.nested_tensor_from_tensor_list([image_tensor])
    return samples


def _preprocess_swig_pil(image: Image.Image, eval_size: int):
    transform = swig_transforms.Compose(
        [
            swig_transforms.RandomResize([eval_size], max_size=eval_size * 1333 // 800),
            swig_transforms.Compose(
                [
                    swig_transforms.ToTensor(),
                    swig_transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225],
                    ),
                ]
            ),
        ]
    )
    target = {
        "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64),
        "size": torch.tensor([image.height, image.width], dtype=torch.int64),
        "image_id": torch.tensor(0),
    }
    image_tensor, target = transform(image.convert("RGB"), target)
    samples = utils.nested_tensor_from_tensor_list([image_tensor])
    return samples, target["orig_size"]


def _box_iou_tuple(a, b) -> float:
    boxes = torch.tensor([a, b], dtype=torch.float32)
    return float(box_iou(boxes[:1], boxes[1:]).item())


def _assign_person_indices(pairs):
    clusters = []
    for pair in sorted(
        pairs,
        key=lambda item: float(item["interactions"][0]["score"] if item.get("interactions") else 0.0),
        reverse=True,
    ):
        assigned_idx = None
        for idx, cluster in enumerate(clusters):
            if _box_iou_tuple(pair["person_box"], cluster["box"]) >= PERSON_CLUSTER_IOU:
                assigned_idx = idx
                break
        if assigned_idx is None:
            assigned_idx = len(clusters)
            clusters.append({"box": pair["person_box"], "score": 0.0})
        clusters[assigned_idx]["score"] = max(
            float(clusters[assigned_idx]["score"]),
            float(pair["interactions"][0]["score"]),
        )
        pair["person_idx"] = int(assigned_idx)
        pair["person_score"] = float(clusters[assigned_idx]["score"])
    return pairs


def _hico_outputs_to_pairs(outputs, target_sizes, score_threshold: float, topk_interactions: int):
    assert STATE is not None
    processed = STATE.postprocessors["hoi"](outputs, target_sizes)[0]
    boxes = processed["boxes"]
    labels = processed["labels"]
    hoi_scores = processed["hoi_scores"].float()
    obj_scores = processed["obj_scores"].float()
    sub_ids = processed["sub_ids"].long()
    obj_ids = processed["obj_ids"].long()
    triplet_labels = list(hico_text_label.keys())
    hoi_obj_ids = torch.tensor([obj_id for _verb_id, obj_id in triplet_labels], dtype=torch.long)
    pairs = []

    for q in range(hoi_scores.shape[0]):
        query_hoi_scores = hoi_scores[q].detach().cpu()
        query_obj_scores = obj_scores[q].detach().cpu()
        valid_obj_ids = hoi_obj_ids.clamp(min=0, max=query_obj_scores.numel() - 1)
        combined = query_hoi_scores * query_obj_scores[valid_obj_ids]
        top_scores, top_indices = torch.topk(
            combined,
            k=min(int(topk_interactions), combined.numel()),
        )
        interactions = []
        for score, class_idx in zip(top_scores, top_indices):
            score_value = float(score.item())
            if score_value < float(score_threshold):
                continue
            verb_id, obj_id = triplet_labels[int(class_idx.item())]
            if _is_no_interaction_verb(int(verb_id)):
                continue
            interactions.append(
                {
                    "score": score_value,
                    "label": _hico_label(int(verb_id), int(obj_id)),
                    "verb_id": int(verb_id),
                    "obj_id": int(obj_id),
                    "q": int(q),
                    "inter": float(query_hoi_scores[int(class_idx)].item()),
                    "obj_prob": float(query_obj_scores[int(obj_id)].item())
                    if int(obj_id) < query_obj_scores.numel()
                    else 1.0,
                    "box_conf": 1.0,
                }
            )
        if not interactions:
            continue

        top_obj_id = int(interactions[0]["obj_id"])
        obj_label = int(labels[int(obj_ids[q])].item()) if int(obj_ids[q]) < len(labels) else top_obj_id
        pairs.append(
            {
                "person_idx": 0,
                "query_idx": int(q),
                "person_box": tuple(map(float, boxes[int(sub_ids[q])].tolist())),
                "object_box": tuple(map(float, boxes[int(obj_ids[q])].tolist())),
                "person_score": None,
                "object_id": top_obj_id,
                "raw_object_label": obj_label,
                "object_prob": float(interactions[0]["obj_prob"]),
                "interaction_score": float(interactions[0]["inter"]),
                "box_conf": 1.0,
                "interactions": interactions,
                "mask_logits": None,
            }
        )
    return _assign_person_indices(pairs)


def _swig_outputs_to_pairs(outputs, orig_size, score_threshold: float, topk_interactions: int):
    assert STATE is not None
    results = STATE.postprocessors["hoi"](
        {
            "pred_logits": outputs["logits_per_hoi"][0],
            "pred_boxes": outputs["pred_boxes"][0],
            "box_scores": outputs["box_scores"][0],
        },
        orig_size,
        STATE.dataset_val.text_mapper,
    )
    pairs = []
    for query_idx, pred in enumerate(sorted(results, key=lambda item: float(item[1]), reverse=True)):
        if len(pred) != 10:
            continue
        score = float(pred[1])
        if score < float(score_threshold):
            continue
        hoi_id = int(pred[0])
        label = STATE.swig_id_to_name.get(hoi_id, f"hoi_{hoi_id}") if STATE.swig_id_to_name else f"hoi_{hoi_id}"
        pairs.append(
            {
                "person_idx": 0,
                "query_idx": int(query_idx),
                "person_box": tuple(map(float, pred[2:6])),
                "object_box": tuple(map(float, pred[6:10])),
                "person_score": None,
                "object_id": int(hoi_id),
                "object_prob": 1.0,
                "interaction_score": score,
                "box_conf": 1.0,
                "interactions": [
                    {
                        "score": score,
                        "label": str(label).replace("_", " "),
                        "verb_id": int(hoi_id),
                        "obj_id": int(hoi_id),
                        "q": int(query_idx),
                        "inter": score,
                        "obj_prob": 1.0,
                        "box_conf": 1.0,
                        "suppress_unseen": True,
                    }
                ],
                "mask_logits": None,
            }
        )
        if len(pairs) >= int(topk_interactions) * 8:
            break
    return _assign_person_indices(pairs)


def _mark_overlapping_pair_visual_representatives(
    pairs,
    object_nms_iou: float = PAIR_OBJECT_NMS_IOU,
):
    if not pairs:
        return []
    groups = {}
    next_visual_group_id = 0
    for pair in pairs:
        pair["show_visual"] = True
        key = (pair["person_idx"], pair["object_id"])
        groups.setdefault(key, []).append(pair)
    for group_pairs in groups.values():
        if len(group_pairs) == 1:
            group_pairs[0]["visual_query_group"] = [group_pairs[0]["query_idx"]]
            group_pairs[0]["visual_group_id"] = next_visual_group_id
            next_visual_group_id += 1
            continue
        boxes = torch.tensor([pair["object_box"] for pair in group_pairs], dtype=torch.float32)
        ious = box_iou(boxes, boxes)
        order = sorted(
            range(len(group_pairs)),
            key=lambda i: group_pairs[i]["interactions"][0]["score"],
            reverse=True,
        )
        assigned = set()
        for seed in order:
            if seed in assigned:
                continue
            cluster = [seed]
            assigned.add(seed)
            changed = True
            while changed:
                changed = False
                for j in range(len(group_pairs)):
                    if j in assigned:
                        continue
                    if any(float(ious[j, k].item()) >= float(object_nms_iou) for k in cluster):
                        cluster.append(j)
                        assigned.add(j)
                        changed = True
            cluster_pairs = [group_pairs[i] for i in cluster]
            rep = max(cluster_pairs, key=lambda pair: pair["interactions"][0]["score"])
            query_group = sorted(pair["query_idx"] for pair in cluster_pairs)
            for pair in cluster_pairs:
                pair["show_visual"] = pair is rep
                pair["visual_query_group"] = query_group
                pair["visual_group_id"] = next_visual_group_id
            next_visual_group_id += 1
    pairs.sort(key=lambda pair: pair["interactions"][0]["score"], reverse=True)
    return pairs


def _dedupe_interactions_by_target_and_label(pairs):
    if not pairs:
        return pairs
    best_by_key = {}
    for pair_idx, pair in enumerate(pairs):
        target_group = int(pair.get("visual_group_id", pair["query_idx"]))
        person_idx = int(pair["person_idx"])
        for item_idx, item in enumerate(pair.get("interactions", [])):
            label_key = str(item.get("label", "")).strip().lower()
            key = (person_idx, target_group, label_key)
            score = float(item.get("score", 0.0))
            prev = best_by_key.get(key)
            if prev is None or score > prev[0]:
                best_by_key[key] = (score, pair_idx, item_idx)
    keep_by_pair = {}
    for _score, pair_idx, item_idx in best_by_key.values():
        keep_by_pair.setdefault(pair_idx, set()).add(item_idx)
    deduped_pairs = []
    for pair_idx, pair in enumerate(pairs):
        keep_items = keep_by_pair.get(pair_idx, set())
        if not keep_items:
            continue
        pair = dict(pair)
        pair["interactions"] = [
            item
            for item_idx, item in enumerate(pair.get("interactions", []))
            if item_idx in keep_items
        ]
        pair["interactions"].sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        deduped_pairs.append(pair)
    deduped_pairs.sort(key=lambda pair: pair["interactions"][0]["score"], reverse=True)
    return deduped_pairs


def _apply_stable_object_labels(result_state):
    if STATE is None or STATE.dataset != "hico" or not result_state:
        return result_state
    for pair in result_state.get("pairs", []):
        stable_obj_id = pair.get("stable_object_id")
        if stable_obj_id is None:
            continue
        stable_obj_id = int(stable_obj_id)
        pair["raw_object_id"] = int(pair.get("object_id", stable_obj_id))
        pair["object_id"] = stable_obj_id
        for item in pair.get("interactions", []):
            item["raw_obj_id"] = int(item.get("obj_id", stable_obj_id))
            item["obj_id"] = stable_obj_id
            item["label"] = _hico_label(int(item["verb_id"]), stable_obj_id)
    return result_state


def infer_image(
    image: Image.Image,
    person_conf: float,
    person_nms_iou: float,
    interaction_score_threshold: float,
    topk_interactions: int,
    image_path: str = "gradio_upload",
    inference_max_side: int | float | None = None,
    object_nms_iou: float = PAIR_OBJECT_NMS_IOU,
    **_unused,
):
    if STATE is None:
        raise RuntimeError("Demo model is not loaded yet.")
    if image is None:
        return None, "Upload an image first.", None

    display_image, model_image, sx, sy = _resize_for_inference(image, inference_max_side)
    target_sizes = torch.tensor([[model_image.height, model_image.width]], device=STATE.device)
    if STATE.dataset == "hico":
        samples = _preprocess_hico_pil(model_image)
        orig_size = None
    else:
        eval_size = int(getattr(STATE.cfg.INPUT, "EVAL_SIZE", 512))
        samples, orig_size = _preprocess_swig_pil(model_image, eval_size)
        orig_size = orig_size.to(STATE.device)
    samples = samples.to(STATE.device)
    samples.tensors = samples.tensors.to(STATE.param_dtype)

    if STATE.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if STATE.device == "cuda"
        else torch.no_grad()
    )
    with torch.no_grad():
        with autocast_ctx:
            with MODEL_LOCK:
                if STATE.dataset == "hico":
                    outputs = STATE.model(samples)
                else:
                    outputs = STATE.model(samples, STATE.text_embeddings)
    if STATE.device == "cuda":
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0

    if STATE.dataset == "hico":
        pairs = _hico_outputs_to_pairs(
            outputs,
            target_sizes,
            score_threshold=float(interaction_score_threshold),
            topk_interactions=int(topk_interactions),
        )
    else:
        pairs = _swig_outputs_to_pairs(
            outputs,
            orig_size,
            score_threshold=float(interaction_score_threshold),
            topk_interactions=int(topk_interactions),
        )
    pairs = _mark_overlapping_pair_visual_representatives(
        pairs,
        object_nms_iou=float(object_nms_iou),
    )
    pairs = _dedupe_interactions_by_target_and_label(pairs)
    num_people = len({int(pair["person_idx"]) for pair in pairs})
    logs = [
        f"SL-HOI direct pair predictions; person threshold/NMS sliders are not used by this backend.",
        f"Inference time: total={total_ms:.1f} ms",
        f"Model output queries: {int(outputs.get('pred_hoi_logits', outputs.get('logits_per_hoi')).shape[1]) if STATE.dataset == 'hico' else int(outputs['pred_boxes'].shape[1])}",
        f"Visible pairs: {len(pairs)}",
    ]
    annotated = draw_all_pairs(model_image, pairs)
    result_state = {
        "input_image": model_image,
        "display_image": annotated,
        "image": model_image,
        "pairs": pairs,
        "summary_masks": [],
        "top_verb_predictions": [],
        "masks_visible": False,
        "clip_masks_to_boxes": False,
        "logs": logs,
        "total_ms": total_ms,
        "num_people": num_people,
    }
    if sx != 1.0 or sy != 1.0:
        logs.append(
            f"Inference resized to {model_image.width}x{model_image.height}; "
            f"display frame is {display_image.width}x{display_image.height}."
        )
        result_state = _scale_result_state_to_display(result_state, display_image, sx, sy)
    bars_html = build_bars_html(
        result_state.get("pairs", []),
        logs=logs,
        total_ms=total_ms,
        num_people=num_people,
        unseen_pairs=STATE.unseen_pairs if STATE is not None else set(),
    )
    return result_state["display_image"], bars_html, result_state


def run_demo_inference(
    image: Image.Image,
    person_conf: float,
    person_nms_iou: float,
    interaction_score_threshold: float,
    topk_interactions: int,
    object_nms_iou: float,
    show_all_combinations: bool,
    unconstrained_verbs: bool,
    result_state,
):
    if image is None:
        return None, "Upload an image first.", None
    image = resolve_input_image(image, result_state)
    return infer_image(
        image=image,
        person_conf=person_conf,
        person_nms_iou=person_nms_iou,
        interaction_score_threshold=interaction_score_threshold,
        topk_interactions=topk_interactions,
        object_nms_iou=float(object_nms_iou),
    )


def add_live_object_class(_object_name: str):
    return "SL-HOI uses fixed classifier embeddings; live object insertion is not available."


def _render_camera_image(
    image: Image.Image,
    person_conf: float,
    person_nms_iou: float,
    interaction_score_threshold: float,
    topk_interactions: int,
    object_nms_iou: float,
    show_masks: bool,
    show_top_verbs: bool,
    unconstrained_verbs: bool,
    inference_max_side: int | float,
    image_path: str,
):
    global CAMERA_TRACKER, CAMERA_OBJECT_STABILIZER, CAMERA_INTERACTION_STABILIZER, CAMERA_TARGET_COLOR_STABILIZER
    candidate_score_threshold = max(
        0.0,
        float(interaction_score_threshold) * float(INTERACTION_HYSTERESIS_EXIT_RATIO),
    )
    annotated, bars_html, result_state = infer_image(
        image=image,
        person_conf=person_conf,
        person_nms_iou=person_nms_iou,
        interaction_score_threshold=candidate_score_threshold,
        topk_interactions=topk_interactions,
        image_path=image_path,
        inference_max_side=inference_max_side,
        object_nms_iou=float(object_nms_iou),
    )
    if CAMERA_TRACKER is not None:
        CAMERA_TRACKER.assign_result_state_color_indices(result_state)
    if CAMERA_OBJECT_STABILIZER is not None:
        CAMERA_OBJECT_STABILIZER.update_result_state(result_state)
        _apply_stable_object_labels(result_state)
    if CAMERA_INTERACTION_STABILIZER is not None:
        CAMERA_INTERACTION_STABILIZER.update_result_state(
            result_state,
            enter_threshold=float(interaction_score_threshold),
        )
    if CAMERA_TARGET_COLOR_STABILIZER is not None:
        CAMERA_TARGET_COLOR_STABILIZER.update_result_state(result_state)
    result_state["display_image"] = draw_all_pairs(result_state["image"], result_state.get("pairs", []))
    bars_html = build_bars_html(
        result_state.get("pairs", []),
        logs=result_state.get("logs", []),
        total_ms=float(result_state.get("total_ms", 0.0)),
        num_people=int(result_state.get("num_people", 0)),
        unseen_pairs=STATE.unseen_pairs if STATE is not None else set(),
        show_logs=False,
        show_pair_headers=False,
        compact=True,
        fixed_person_interaction_slots=int(topk_interactions),
    )
    display_image = display_image_from_state(result_state, False, allow_summary_masks=False)
    return (display_image or annotated).convert("RGB"), bars_html, result_state


def run_camera_frame(
    image: Image.Image | None,
    person_conf: float,
    person_nms_iou: float,
    interaction_score_threshold: float,
    topk_interactions: int,
    object_nms_iou: float,
    show_masks: bool,
    show_top_verbs: bool,
    unconstrained_verbs: bool,
    inference_max_side: int | float,
    result_state,
):
    global CAMERA_TRACKER, CAMERA_OBJECT_STABILIZER, CAMERA_INTERACTION_STABILIZER, CAMERA_TARGET_COLOR_STABILIZER, CAMERA_FRAME_IDX
    if image is None:
        state = {"running": False, "source": "browser"}
        return None, "", "Start the browser USB camera.", state
    if CAMERA_TRACKER is None:
        CAMERA_TRACKER = PersonColorTracker(frame_rate=10)
    if CAMERA_OBJECT_STABILIZER is None:
        CAMERA_OBJECT_STABILIZER = ObjectClassStabilizer()
    if CAMERA_INTERACTION_STABILIZER is None:
        CAMERA_INTERACTION_STABILIZER = InteractionStabilizer(
            alpha=INTERACTION_EMA_ALPHA,
            exit_ratio=INTERACTION_HYSTERESIS_EXIT_RATIO,
        )
    if CAMERA_TARGET_COLOR_STABILIZER is None:
        CAMERA_TARGET_COLOR_STABILIZER = TargetColorStabilizer()
    display_image, bars_html, result_state = _render_camera_image(
        image=image,
        person_conf=person_conf,
        person_nms_iou=person_nms_iou,
        interaction_score_threshold=interaction_score_threshold,
        topk_interactions=topk_interactions,
        object_nms_iou=object_nms_iou,
        image_path=f"camera:frame_{CAMERA_FRAME_IDX}",
        inference_max_side=inference_max_side,
        show_masks=show_masks,
        show_top_verbs=show_top_verbs,
        unconstrained_verbs=unconstrained_verbs,
    )
    CAMERA_FRAME_IDX += 1
    status = (
        f"Browser camera running. Processed {CAMERA_FRAME_IDX} frame(s); "
        f"last inference {float(result_state.get('total_ms', 0.0)):.1f} ms."
    )
    return display_image, bars_html, status, {"running": True, "source": "browser", "frame_idx": CAMERA_FRAME_IDX}


def _network_camera_url(camera_source: str) -> str | None:
    if camera_source == "Axis MJPEG camera 172.16.46.6":
        return DEFAULT_CAMERA_URL
    return None


def select_camera_source(camera_source: str):
    stop_network_camera_stream()
    is_network_camera = _network_camera_url(camera_source) is not None
    if is_network_camera:
        status = "Axis MJPEG camera selected. Press Start selected stream."
        state = {"running": False, "source": "network", "camera_source": camera_source}
    else:
        status = "Browser USB camera selected. Open the camera preview and press Record."
        state = {"running": False, "source": "browser"}
    return (
        gr.update(visible=not is_network_camera),
        None,
        "",
        status,
        state,
        gr.update(active=False),
        gr.update(visible=is_network_camera),
        gr.update(visible=is_network_camera),
    )


def start_network_camera_stream(camera_source: str):
    global CAMERA_TRACKER, CAMERA_OBJECT_STABILIZER, CAMERA_INTERACTION_STABILIZER, CAMERA_TARGET_COLOR_STABILIZER, CAMERA_FRAME_IDX, NETWORK_CAMERA_READER, NETWORK_CAMERA_SOURCE
    stop_network_camera_stream()
    url = _network_camera_url(camera_source)
    if url is None:
        return (
            None,
            "",
            "Select the Axis MJPEG camera to start a Python-side stream.",
            {"running": False, "source": "browser"},
            gr.update(active=False),
        )
    try:
        reader = CamReader(url, color_order="RGB", buffer_size=2)
    except Exception as exc:
        return (
            None,
            "",
            f"Could not open network camera: {exc!r}",
            {"running": False, "source": "network", "url": url},
            gr.update(active=False),
        )
    NETWORK_CAMERA_READER = reader
    NETWORK_CAMERA_SOURCE = camera_source
    CAMERA_TRACKER = PersonColorTracker(frame_rate=10)
    CAMERA_OBJECT_STABILIZER = ObjectClassStabilizer()
    CAMERA_INTERACTION_STABILIZER = InteractionStabilizer(
        alpha=INTERACTION_EMA_ALPHA,
        exit_ratio=INTERACTION_HYSTERESIS_EXIT_RATIO,
    )
    CAMERA_TARGET_COLOR_STABILIZER = TargetColorStabilizer()
    CAMERA_FRAME_IDX = 0
    return (
        None,
        "",
        f"{camera_source} started.",
        {"running": True, "source": "network", "camera_source": camera_source, "frame_idx": 0},
        gr.update(active=True),
    )


def stop_network_camera_stream():
    global NETWORK_CAMERA_READER, NETWORK_CAMERA_SOURCE, CAMERA_TRACKER, CAMERA_OBJECT_STABILIZER, CAMERA_INTERACTION_STABILIZER, CAMERA_TARGET_COLOR_STABILIZER, CAMERA_FRAME_IDX
    if NETWORK_CAMERA_READER is not None:
        try:
            NETWORK_CAMERA_READER.stop()
        except Exception:
            pass
    NETWORK_CAMERA_READER = None
    NETWORK_CAMERA_SOURCE = None
    CAMERA_TRACKER = None
    CAMERA_OBJECT_STABILIZER = None
    CAMERA_INTERACTION_STABILIZER = None
    CAMERA_TARGET_COLOR_STABILIZER = None
    CAMERA_FRAME_IDX = 0
    return None, "", "Network camera stopped.", {"running": False, "source": "network"}, gr.update(active=False)


def run_network_camera_frame(
    person_conf: float,
    person_nms_iou: float,
    interaction_score_threshold: float,
    topk_interactions: int,
    object_nms_iou: float,
    show_masks: bool,
    show_top_verbs: bool,
    unconstrained_verbs: bool,
    inference_max_side: int | float,
    result_state,
):
    global CAMERA_FRAME_IDX, NETWORK_CAMERA_READER, NETWORK_CAMERA_SOURCE
    if NETWORK_CAMERA_READER is None:
        return None, "", "Network camera is not running.", {"running": False, "source": "network"}
    ok, frame_rgb, frame_idx = NETWORK_CAMERA_READER.get_image()
    if not ok or frame_rgb is None:
        state = {
            "running": True,
            "source": "network",
            "camera_source": NETWORK_CAMERA_SOURCE,
            "frame_idx": CAMERA_FRAME_IDX,
        }
        return None, "", "Waiting for a frame from the network camera.", state
    image = Image.fromarray(frame_rgb)
    display_image, bars_html, result_state = _render_camera_image(
        image=image,
        person_conf=person_conf,
        person_nms_iou=person_nms_iou,
        interaction_score_threshold=interaction_score_threshold,
        topk_interactions=topk_interactions,
        object_nms_iou=object_nms_iou,
        image_path=f"{NETWORK_CAMERA_SOURCE or 'network_camera'}:frame_{frame_idx}",
        inference_max_side=inference_max_side,
        show_masks=show_masks,
        show_top_verbs=show_top_verbs,
        unconstrained_verbs=unconstrained_verbs,
    )
    CAMERA_FRAME_IDX += 1
    status = (
        f"{NETWORK_CAMERA_SOURCE or 'Network camera'} running. "
        f"Processed {CAMERA_FRAME_IDX} frame(s); source frame {frame_idx}; "
        f"last inference {float(result_state.get('total_ms', 0.0)):.1f} ms."
    )
    state = {
        "running": True,
        "source": "network",
        "camera_source": NETWORK_CAMERA_SOURCE,
        "frame_idx": CAMERA_FRAME_IDX,
    }
    return display_image, bars_html, status, state


def _video_path(video):
    if video is None:
        return None
    if isinstance(video, (str, os.PathLike)):
        return str(video)
    if isinstance(video, (tuple, list)) and video:
        return str(video[0])
    if isinstance(video, dict):
        for key in ("video", "path", "name"):
            if video.get(key):
                return str(video[key])
    return None


def run_video_inference(
    video,
    person_conf: float,
    person_nms_iou: float,
    interaction_score_threshold: float,
    topk_interactions: int,
    object_nms_iou: float,
    sample_fps: float,
    show_masks: bool,
    show_top_verbs: bool,
    unconstrained_verbs: bool,
    inference_max_side: int | float,
):
    video_path = _video_path(video)
    if not video_path:
        yield show_video_input("Upload or record a video first.", "No video input.")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield show_video_input("Could not open video.", f"Could not open: {video_path}")
        return
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_fps = max(0.1, float(sample_fps))
    stride = max(1, int(round(source_fps / target_fps))) if source_fps > 0 else 1
    output_fps = min(target_fps, source_fps) if source_fps > 0 else target_fps
    person_tracker = PersonColorTracker(frame_rate=max(1, int(round(output_fps))))
    object_stabilizer = ObjectClassStabilizer()
    interaction_stabilizer = InteractionStabilizer(
        alpha=INTERACTION_EMA_ALPHA,
        exit_ratio=INTERACTION_HYSTERESIS_EXIT_RATIO,
    )
    candidate_score_threshold = max(
        0.0,
        float(interaction_score_threshold) * float(INTERACTION_HYSTERESIS_EXIT_RATIO),
    )
    frame_idx = 0
    sampled = 0
    last_bars = ""
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            annotated, bars_html, result_state = infer_image(
                image=frame_pil,
                person_conf=person_conf,
                person_nms_iou=person_nms_iou,
                interaction_score_threshold=candidate_score_threshold,
                topk_interactions=topk_interactions,
                image_path=f"{os.path.basename(video_path)}:frame_{frame_idx}",
                inference_max_side=inference_max_side,
                object_nms_iou=float(object_nms_iou),
            )
            person_tracker.assign_result_state_color_indices(result_state)
            object_stabilizer.update_result_state(result_state)
            _apply_stable_object_labels(result_state)
            interaction_stabilizer.update_result_state(
                result_state,
                enter_threshold=float(interaction_score_threshold),
            )
            result_state["display_image"] = draw_all_pairs(result_state["image"], result_state.get("pairs", []))
            bars_html = build_bars_html(
                result_state.get("pairs", []),
                logs=result_state.get("logs", []),
                total_ms=float(result_state.get("total_ms", 0.0)),
                num_people=int(result_state.get("num_people", 0)),
                unseen_pairs=STATE.unseen_pairs if STATE is not None else set(),
                show_logs=False,
                show_pair_headers=False,
                compact=True,
                fixed_person_interaction_slots=int(topk_interactions),
            )
            display = display_image_from_state(result_state, False, allow_summary_masks=False) or annotated
            display = display.convert("RGB")
            sampled += 1
            last_bars = bars_html
            status = (
                f"Processed {sampled} sampled frame(s)"
                + (f" from {total_frames} source frames" if total_frames else "")
                + f" at ~{output_fps:.2f} FPS output."
            )
            yield show_live_video_frame(display, bars_html, status)
            frame_idx += 1
    finally:
        cap.release()
    if sampled == 0:
        yield show_video_input("No frames were sampled.", "No frames were sampled.")
        return
    yield keep_live_video_frame(last_bars, f"Done. Processed {sampled} sampled frame(s).")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch an SL-HOI Gradio demo with the HOIster UI.")
    parser.add_argument("--dataset", choices=("hico", "swig"), default="hico")
    parser.add_argument("--variant", choices=("hico", "hico_ov", "swig"), default="hico")
    parser.add_argument("--slhoi-root", default=DEFAULT_SLHOI_ROOT)
    parser.add_argument("--config", default=None)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--classifier-eval", default=None)
    parser.add_argument("--classifier-train", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    global STATE
    setup_logger()
    args = parse_args()
    if args.device == "cuda" and torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    STATE = load_demo_state(
        dataset=args.dataset,
        variant=args.variant,
        device=args.device,
        slhoi_root=args.slhoi_root,
        config_path=args.config,
        base_config_path=args.base_config,
        ckpt_path=args.ckpt,
        classifier_eval_path=args.classifier_eval,
        classifier_train_path=args.classifier_train,
        dataset_root=args.dataset_root,
    )
    app = build_app(
        run_demo_inference=run_demo_inference,
        run_video_inference=run_video_inference,
        run_camera_frame=run_camera_frame,
        add_live_object_class=add_live_object_class,
        select_camera_source=select_camera_source,
        start_network_camera_stream=start_network_camera_stream,
        stop_network_camera_stream=stop_network_camera_stream,
        run_network_camera_frame=run_network_camera_frame,
        default_video_sample_fps=DEFAULT_VIDEO_SAMPLE_FPS,
        default_video_inference_size=DEFAULT_VIDEO_INFERENCE_SIZE,
    )
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
