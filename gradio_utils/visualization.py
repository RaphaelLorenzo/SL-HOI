import html
import math
from typing import Sequence, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageDraw, ImageFont


PAIR_COLORS = [
    "#e53935",
    "#1e88e5",
]

TARGET_COLOR_FAMILIES = [
    ("#e53935", "#c62828", "#ff6b5f", "#ff8a80", "#b71c1c", "#ffcdd2"),
    ("#1e88e5", "#0d47a1", "#42a5f5", "#90caf9", "#1565c0", "#bbdefb"),
]


def pair_color(pair):
    color_idx = int(pair.get("color_idx", pair["person_idx"]))
    return PAIR_COLORS[color_idx % len(PAIR_COLORS)]


def person_item_color(item):
    color_idx = int(item.get("color_idx", item["person_idx"]))
    return PAIR_COLORS[color_idx % len(PAIR_COLORS)]


def target_pair_color(pair):
    if pair.get("target_color"):
        return str(pair["target_color"])
    shade_idx = pair.get("target_color_idx")
    if shade_idx is None:
        return pair_color(pair)
    person_family_idx = (
        int(pair.get("color_idx", pair["person_idx"])) % len(TARGET_COLOR_FAMILIES)
    )
    palette = TARGET_COLOR_FAMILIES[person_family_idx]
    return palette[int(shade_idx) % len(palette)]


def _draw_box(draw: ImageDraw.ImageDraw, box, color, width=3):
    if box is None:
        return
    x1, y1, x2, y2 = map(float, box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def _draw_dotted_box(
    draw: ImageDraw.ImageDraw,
    box,
    color,
    width=3,
    dash_len=8,
    gap_len=6,
):
    if box is None:
        return
    x1, y1, x2, y2 = map(float, box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    def draw_dotted_line(start, end):
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        length = math.hypot(dx, dy)
        if length <= 0:
            return
        ux = dx / length
        uy = dy / length
        pos = 0.0
        while pos < length:
            seg_end = min(length, pos + dash_len)
            draw.line(
                [
                    (sx + ux * pos, sy + uy * pos),
                    (sx + ux * seg_end, sy + uy * seg_end),
                ],
                fill=color,
                width=width,
            )
            pos += dash_len + gap_len

    draw_dotted_line((x1, y1), (x2, y1))
    draw_dotted_line((x2, y1), (x2, y2))
    draw_dotted_line((x2, y2), (x1, y2))
    draw_dotted_line((x1, y2), (x1, y1))


def _get_font(size=13):
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text_panel(
    image: Image.Image,
    lines: Sequence[str],
    xy: Tuple[float, float],
    fill=(20, 20, 20, 200),
):
    if not lines:
        return ImageDraw.Draw(image)

    draw = ImageDraw.Draw(image)
    font = _get_font()
    padding = 6
    line_gap = 3
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    widths = [bb[2] - bb[0] for bb in boxes]
    heights = [bb[3] - bb[1] for bb in boxes]
    box_w = max(widths) + padding * 2
    box_h = sum(heights) + line_gap * (len(lines) - 1) + padding * 2

    W, H = image.size
    x0, y0 = xy
    x0 = max(0, min(float(x0), W - box_w))
    y0 = max(0, min(float(y0), H - box_h))

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        [x0, y0, x0 + box_w, y0 + box_h],
        radius=6,
        fill=fill,
    )
    image.alpha_composite(overlay)

    draw = ImageDraw.Draw(image)
    y = y0 + padding
    for line, height in zip(lines, heights):
        draw.text((x0 + padding, y), line, font=font, fill=(255, 255, 255, 255))
        y += height + line_gap
    return draw


def draw_all_pairs(image: Image.Image, pairs):
    viz = image.convert("RGBA")
    draw = ImageDraw.Draw(viz)
    for pair in pairs:
        if not pair.get("show_visual", True):
            continue
        _draw_box(draw, pair["person_box"], color=pair_color(pair), width=4)
        _draw_dotted_box(
            draw,
            pair["object_box"],
            color=target_pair_color(pair),
            width=4,
        )
    return viz


def _draw_predicted_mask_overlay(
    viz: Image.Image,
    hm_logits: torch.Tensor,
    color,
    alpha=120,
    threshold_rel=0.35,
    clip_box=None,
):
    if hm_logits is None:
        return

    W, H = viz.size
    hm = torch.sigmoid(hm_logits.detach().float().cpu())
    hm = F.interpolate(
        hm[None, None],
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    hm_min = hm.min()
    hm_max = hm.max()
    hm_norm = (hm - hm_min) / (hm_max - hm_min).clamp_min(1e-6)
    mask_alpha = torch.where(
        hm_norm >= float(threshold_rel),
        hm_norm * float(alpha),
        torch.zeros_like(hm_norm),
    )
    if clip_box is not None:
        x1, y1, x2, y2 = map(float, clip_box)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1 = max(0, min(W, int(round(x1))))
        x2 = max(0, min(W, int(round(x2))))
        y1 = max(0, min(H, int(round(y1))))
        y2 = max(0, min(H, int(round(y2))))
        box_mask = torch.zeros_like(mask_alpha, dtype=torch.bool)
        if x2 > x1 and y2 > y1:
            box_mask[y1:y2, x1:x2] = True
        mask_alpha = torch.where(box_mask, mask_alpha, torch.zeros_like(mask_alpha))
    alpha_np = mask_alpha.clamp(0, 255).byte().numpy()
    if not np.any(alpha_np):
        return

    r, g, b = ImageColor.getrgb(color)
    overlay_np = np.zeros((H, W, 4), dtype=np.uint8)
    overlay_np[..., 0] = r
    overlay_np[..., 1] = g
    overlay_np[..., 2] = b
    overlay_np[..., 3] = alpha_np
    overlay = Image.fromarray(overlay_np, mode="RGBA")
    viz.alpha_composite(overlay)


def draw_all_pairs_with_masks(image: Image.Image, pairs, summary_masks):
    viz = image.convert("RGBA")
    pair_masks_drawn = False
    for pair in pairs:
        if not pair.get("show_visual", True):
            continue
        mask_logits = pair.get("mask_logits")
        if mask_logits is None:
            continue
        pair_masks_drawn = True
        _draw_predicted_mask_overlay(
            viz,
            mask_logits,
            color=target_pair_color(pair),
            clip_box=pair.get("mask_clip_box"),
        )

    if not pair_masks_drawn:
        for mask_item in summary_masks:
            color = person_item_color(mask_item)
            _draw_predicted_mask_overlay(viz, mask_item["mask_logits"], color=color)

    draw = ImageDraw.Draw(viz)
    for pair in pairs:
        if not pair.get("show_visual", True):
            continue
        _draw_box(draw, pair["person_box"], color=pair_color(pair), width=4)
        _draw_dotted_box(
            draw,
            pair["object_box"],
            color=target_pair_color(pair),
            width=4,
        )
    return viz


def collect_summary_query_masks(outputs):
    pred_heatmaps = outputs["pred_heatmaps"][0]
    if pred_heatmaps is None or pred_heatmaps.numel() == 0:
        return []

    is_interaction_logits = outputs.get("pred_interaction_logits", None)
    if is_interaction_logits is not None:
        is_interaction_logits = is_interaction_logits[0]
        if is_interaction_logits.dim() == 3:
            is_interaction_logits = is_interaction_logits.squeeze(-1)

    summary_masks = []
    P, Q = pred_heatmaps.shape[:2]
    for p in range(P):
        if is_interaction_logits is not None:
            scores = torch.sigmoid(is_interaction_logits[p].detach().float())
        else:
            scores = (
                torch.sigmoid(pred_heatmaps[p].detach().float())
                .flatten(1)
                .max(dim=1)
                .values
            )
        best_q = int(scores.argmax().item())
        summary_masks.append(
            {
                "person_idx": int(p),
                "query_idx": best_q,
                "interaction_score": float(scores[best_q].item()),
                "mask_logits": pred_heatmaps[p, best_q].detach().float().cpu(),
            }
        )
    return summary_masks


def toggle_masks_from_state(result_state):
    if not result_state:
        return None, None
    image = result_state.get("image")
    pairs = result_state.get("pairs", [])
    summary_masks = result_state.get("summary_masks", [])
    if result_state.get("clip_masks_to_boxes"):
        summary_masks = []
    if image is None:
        return None, result_state

    masks_visible = not bool(result_state.get("masks_visible", False))
    result_state["masks_visible"] = masks_visible
    if masks_visible:
        display_image = draw_all_pairs_with_masks(image, pairs, summary_masks)
    else:
        display_image = draw_all_pairs(image, pairs)
    result_state["display_image"] = display_image
    return display_image, result_state


def clear_image_state(_image):
    return "", None


def images_equal(a: Image.Image, b: Image.Image) -> bool:
    if a is None or b is None:
        return False
    a = a.convert("RGB")
    b = b.convert("RGB")
    if a.size != b.size:
        return False
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def resolve_input_image(image: Image.Image, result_state):
    if result_state:
        display_image = result_state.get("display_image")
        input_image = result_state.get("input_image")
        if display_image is not None and input_image is not None:
            if image.size == display_image.size and not images_equal(image, input_image):
                return input_image.convert("RGB")
    return image.convert("RGB")


def display_image_from_state(
    result_state,
    show_masks: bool,
    allow_summary_masks: bool = True,
):
    if not result_state:
        return None

    image = result_state.get("image")
    pairs = result_state.get("pairs", [])
    summary_masks = result_state.get("summary_masks", [])
    if result_state.get("clip_masks_to_boxes"):
        summary_masks = []
    if image is None:
        return None
    if show_masks:
        if not allow_summary_masks and not pairs:
            return draw_all_pairs(image, pairs)
        return draw_all_pairs_with_masks(
            image,
            pairs,
            summary_masks if allow_summary_masks else [],
        )
    return result_state.get("display_image") or draw_all_pairs(image, pairs)


def reset_video_slot():
    return (
        gr.update(value=None, visible=True),
        gr.update(value=None, visible=False),
        "",
        "No video input.",
    )


def show_video_input(message: str, status: str):
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        message,
        status,
    )


def show_live_video_frame(frame, bars_html: str, status: str):
    return (
        gr.update(visible=False),
        gr.update(value=frame, visible=True),
        bars_html,
        status,
    )


def keep_live_video_frame(bars_html: str, status: str):
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        bars_html,
        status,
    )


def build_bars_html(
    pairs,
    logs,
    total_ms: float,
    num_people: int,
    unseen_pairs,
    show_logs: bool = True,
    show_pair_headers: bool = True,
    compact: bool = False,
    top_verb_predictions=None,
    show_top_verbs: bool = False,
    fixed_person_interaction_slots: int | None = None,
):
    panel_style = (
        "font-family: system-ui, sans-serif; padding: 14px; border-radius: 8px; "
        "background: var(--block-background-fill, #1f2937); "
        "border: 1px solid var(--border-color-primary, #374151); "
        "color: var(--body-text-color, #f9fafb);"
    )
    muted_style = "color: var(--body-text-color-subdued, #cbd5e1);"
    log_style = (
        "white-space: pre-wrap; color: var(--body-text-color-subdued, #cbd5e1); "
        "font-size: 12px; line-height: 1.35;"
    )

    pair_gap = 8 if compact else 16
    bar_gap = 4 if compact else 6
    def render_top_verbs():
        return ""

    ordered_pairs = sorted(
        pairs,
        key=lambda pair: (
            int(pair["person_idx"]),
            -float(pair["interactions"][0]["score"] if pair["interactions"] else 0.0),
            int(pair.get("object_id", -1)),
            int(pair.get("query_idx", -1)),
        ),
    )

    def render_interaction_bar(item, color):
        width_pct = max(3.0, min(100.0, 100.0 * float(item["score"])))
        label = html.escape(item["label"])
        is_unseen = (not bool(item.get("suppress_unseen", False))) and (
            bool(item.get("force_unseen", False))
            or (
                int(item["verb_id"]),
                int(item["obj_id"]),
            )
            in unseen_pairs
        )
        unseen_badge = (
            """
            <span title="Unseen or not present in HICO-DET"
                  style="font-size: 10px; font-weight: 750; color: #7c2d12; background: #ffedd5; border: 1px solid #fdba74; border-radius: 3px; padding: 1px 4px; white-space: nowrap;">UNSEEN</span>
            """
            if is_unseen
            else ""
        )
        return f"""
        <div style="margin: {bar_gap}px 0;">
          <div style="display: flex; justify-content: space-between; gap: 10px; font-size: 17px;">
            <span style="display: inline-flex; align-items: center; gap: 6px; min-width: 0; flex-wrap: wrap; color: var(--body-text-color, #f9fafb);">{label}{unseen_badge}</span>
            <span style="font-variant-numeric: tabular-nums; color: var(--body-text-color, #f9fafb);">{item['score']:.3f}</span>
          </div>
          <div style="height: 10px; background: rgba(148,163,184,0.28); border-radius: 999px; overflow: hidden; margin-top: 4px;">
            <div style="height: 100%; width: {width_pct:.1f}%; background: {color}; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.18);"></div>
          </div>
        </div>
        """

    def render_empty_bar_slot():
        return f"""
        <div style="margin: {bar_gap}px 0; opacity: 0.22;">
          <div style="height: 15px;"></div>
          <div style="height: 10px; background: rgba(148,163,184,0.28); border-radius: 999px; overflow: hidden; margin-top: 4px;"></div>
        </div>
        """

    blocks = [
        f"""
        <div style="{panel_style}">
          <div style="font-weight: 700; margin-bottom: 4px;">SL-HOI interactions</div>
          <div style="{muted_style} font-size: 13px; margin-bottom: 12px;">
            {num_people} person(s), {len(pairs)} visible pair(s), inference {total_ms:.1f} ms
          </div>
        """
    ]

    if compact and fixed_person_interaction_slots is not None:
        slots = max(0, int(fixed_person_interaction_slots))
        active_person_slots = [
            int(pair.get("color_idx", pair["person_idx"])) for pair in ordered_pairs
        ]
        slot_count = int(num_people)
        if active_person_slots:
            slot_count = max(slot_count, max(active_person_slots) + 1)
        slot_count = min(len(PAIR_COLORS), max(0, slot_count))

        pairs_by_person = {person_idx: [] for person_idx in range(slot_count)}
        for pair in ordered_pairs:
            person_slot = int(pair.get("color_idx", pair["person_idx"]))
            if 0 <= person_slot < len(PAIR_COLORS):
                pairs_by_person.setdefault(person_slot, []).append(pair)

        for person_idx in range(slot_count):
            person_pairs = pairs_by_person.get(person_idx, [])
            color_item = person_pairs[0] if person_pairs else {"person_idx": person_idx}
            color = pair_color(color_item)
            blocks.append(
                f"""
                <div style="margin-bottom: {pair_gap}px;">
                  <div style="display: flex; align-items: center; gap: 8px; margin: 2px 0 5px 0;">
                    <span style="width: 10px; height: 10px; background: {color}; display: inline-block; border-radius: 2px; flex: 0 0 auto;"></span>
                    <span style="{muted_style}; font-size: 12px;">Person {person_idx}</span>
                  </div>
                """
            )
            entries = []
            for pair in person_pairs:
                for item in pair.get("interactions", []):
                    entries.append((item, target_pair_color(pair)))
            entries.sort(key=lambda entry: float(entry[0].get("score", 0.0)), reverse=True)

            for slot_idx in range(slots):
                if slot_idx < len(entries):
                    item, entry_color = entries[slot_idx]
                    blocks.append(render_interaction_bar(item, entry_color))
                else:
                    blocks.append(render_empty_bar_slot())
            blocks.append("</div>")

        top_verbs_html = render_top_verbs()
        if top_verbs_html:
            blocks.append(top_verbs_html)

        blocks.append("</div>")
        return "\n".join(blocks)

    if not pairs:
        top_verbs_html = render_top_verbs()
        return f"""
        <div style="{panel_style}">
          <div style="font-weight: 700; margin-bottom: 8px;">No visible interactions</div>
          {top_verbs_html}
        </div>
        """

    for pair in ordered_pairs:
        color = target_pair_color(pair)
        person_color = pair_color(pair)
        blocks.append(f"""<div style="margin-bottom: {pair_gap}px;">""")
        blocks.append(
            f"""
              <div style="display: flex; align-items: center; gap: 5px; margin: 2px 0 5px 0;">
                <span style="height: 3px; width: 18px; background: {person_color}; display: inline-block; border-radius: 999px;"></span>
                <span style="height: 3px; width: 28px; background: {color}; display: inline-block; border-radius: 999px;"></span>
              </div>
            """
        )
        for item in pair["interactions"]:
            blocks.append(render_interaction_bar(item, color))
        blocks.append("</div>")

    top_verbs_html = render_top_verbs()
    if top_verbs_html:
        blocks.append(top_verbs_html)

    if show_logs and logs:
        escaped_logs = html.escape("\n".join(logs))
        blocks.append(
            f"""
              <details style="margin-top: 10px;">
                <summary style="cursor: pointer; {muted_style}">Logs</summary>
                <pre style="{log_style}">{escaped_logs}</pre>
              </details>
            </div>
            """
        )
    else:
        blocks.append("</div>")
    return "\n".join(blocks)
