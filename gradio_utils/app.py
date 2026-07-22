import os

import gradio as gr
from PIL import Image

from .visualization import clear_image_state, reset_video_slot, toggle_masks_from_state


SAMPLE_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "samples", "images")
SAMPLE_CLIP_DIR = os.path.join(os.path.dirname(__file__), "samples", "clips")


def _sample_image_paths():
    if not os.path.isdir(SAMPLE_IMAGE_DIR):
        return []
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return [
        os.path.join(SAMPLE_IMAGE_DIR, name)
        for name in sorted(os.listdir(SAMPLE_IMAGE_DIR))
        if name.lower().endswith(exts)
    ]


def _sample_clip_paths():
    if not os.path.isdir(SAMPLE_CLIP_DIR):
        return []
    exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    return [
        os.path.join(SAMPLE_CLIP_DIR, name)
        for name in sorted(os.listdir(SAMPLE_CLIP_DIR))
        if name.lower().endswith(exts)
    ]


def _load_sample_image(path):
    if not path:
        return None, "", None
    return Image.open(path).convert("RGB"), "", None


def _load_gallery_sample(evt: gr.SelectData):
    paths = _sample_image_paths()
    if not paths:
        return None, "", None
    idx = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    if idx is None or idx < 0 or idx >= len(paths):
        return None, "", None
    return _load_sample_image(paths[int(idx)])


def _load_gallery_clip(evt: gr.SelectData):
    paths = _sample_clip_paths()
    if not paths:
        return None, None, "", "No video input."
    idx = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    if idx is None or idx < 0 or idx >= len(paths):
        return None, None, "", "No video input."
    return paths[int(idx)], gr.update(value=None, visible=False), "", "No video input."


def build_app(
    run_demo_inference,
    run_video_inference,
    run_camera_frame,
    add_live_object_class,
    select_camera_source,
    start_network_camera_stream,
    stop_network_camera_stream,
    run_network_camera_frame,
    default_video_sample_fps: float,
    default_video_inference_size: int,
):
    css = """
    #camera-interactions,
    #camera-interactions *,
    #camera-interactions.generating,
    #camera-interactions.pending,
    #camera-interactions.loading {
        opacity: 1 !important;
        filter: none !important;
    }
    #camera-interactions {
        min-height: 180px;
    }
    #camera-interactions .wrap,
    #camera-interactions .html-container,
    #camera-interactions .prose {
        opacity: 1 !important;
        filter: none !important;
    }
    """
    with gr.Blocks(title="SL-HOI Demo", css=css) as demo:
        gr.Markdown("# SL-HOI demo")
        gr.Markdown(
            "Run SL-HOI on an image, a sampled video clip, or a live webcam stream. "
            "The model is loaded once at startup."
        )
        with gr.Row(visible=False):
            live_object_name = gr.Textbox(
                label="Add object class live",
                placeholder="e.g. microwave, skateboard helmet, red mug",
            )
            add_object_button = gr.Button("Add object")
        live_object_status = gr.Markdown()
        add_object_button.click(
            add_live_object_class,
            inputs=[live_object_name],
            outputs=[live_object_status],
        )
        with gr.Tabs():
            with gr.Tab("Image"):
                result_state = gr.State()
                gr.Markdown(
                    """
                    - Load your own image or select one of the samples.
                    - Run the model to visualize predicted person/object pairs and interaction categories.
                    - SL-HOI predicts complete dataset triplets directly; arbitrary verb/object combinations are not available.
                    - Mask visualization is not available.
                    - **UNSEEN** interactions were never observed during training by the open-vocabulary model.
                    - Use the sliders to adjust person confidence, final interaction score, and the number of shown interactions.
                    """
                )
                sample_paths = _sample_image_paths()
                if sample_paths:
                    sample_gallery = gr.Gallery(
                        value=[(path, os.path.basename(path)) for path in sample_paths],
                        label="Sample images",
                        columns=min(4, len(sample_paths)),
                        rows=1,
                        height=170,
                        allow_preview=True,
                        type="filepath",
                    )
                else:
                    sample_gallery = None
                    gr.Markdown("No sample images found.")
                with gr.Row():
                    image = gr.Image(type="pil", label="Image")
                    with gr.Column():
                        bars = gr.HTML(label="Interactions")
                with gr.Row():
                    person_conf = gr.Slider(
                        0.01, 0.9, value=0.8, step=0.01, label="Person threshold"
                    )
                    person_nms_iou = gr.Slider(
                        0.1,
                        1.0,
                        value=0.6,
                        step=0.05,
                        label="Person NMS IoU",
                    )
                    interaction_score_threshold = gr.Slider(
                        0.0,
                        1.0,
                        value=0.6,
                        step=0.01,
                        label="Final interaction score threshold",
                    )
                    topk_interactions = gr.Slider(
                        1, 10, value=5, step=1, label="Top-k interactions"
                    )
                    object_nms_iou = gr.Slider(
                        0.0,
                        1.0,
                        value=0.5,
                        step=0.05,
                        label="Target NMS IoU",
                    )
                with gr.Row():
                    run = gr.Button("Run")
                    show_masks = gr.Button("Show masks", visible=False)
                    image_show_all_combinations = gr.Checkbox(
                        value=False,
                        label="Show all verb/object combinations",
                        visible=False,
                    )
                    image_unconstrained_verbs = gr.Checkbox(
                        value=False,
                        label="Unconstrained verbs",
                        visible=False,
                    )

                run.click(
                    run_demo_inference,
                    inputs=[
                        image,
                        person_conf,
                        person_nms_iou,
                        interaction_score_threshold,
                        topk_interactions,
                        object_nms_iou,
                        image_show_all_combinations,
                        image_unconstrained_verbs,
                        result_state,
                    ],
                    outputs=[image, bars, result_state],
                )
                show_masks.click(
                    toggle_masks_from_state,
                    inputs=[result_state],
                    outputs=[image, result_state],
                )
                image.upload(
                    clear_image_state,
                    inputs=[image],
                    outputs=[bars, result_state],
                )
                if sample_gallery is not None:
                    sample_gallery.select(
                        _load_gallery_sample,
                        inputs=[],
                        outputs=[image, bars, result_state],
                    )

            with gr.Tab("Video Clip"):
                gr.Markdown(
                    """
                    - Upload, or select a video clip, then run the model on sampled frames.
                    - Interaction bars and detected boxes for paired people/objects update for each processed sampled frame.
                    - SL-HOI predicts complete dataset triplets directly; arbitrary verb/object combinations are not available.
                    - Use **Change video** to reset the input.
                    - Use **Sample FPS** to control how densely the clip is sampled; higher values run more model calls.
                    """
                )
                clip_paths = _sample_clip_paths()
                if clip_paths:
                    clip_gallery = gr.Gallery(
                        value=[(path, os.path.basename(path)) for path in clip_paths],
                        label="Sample clips",
                        columns=min(3, len(clip_paths)),
                        rows=1,
                        height=170,
                        allow_preview=True,
                        type="filepath",
                    )
                else:
                    clip_gallery = None
                    gr.Markdown("No sample clips found.")
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(
                            sources=["upload", "webcam"],
                            format="mp4",
                            include_audio=False,
                            label="Video",
                        )
                        video_frame = gr.Image(
                            type="pil",
                            label="Live processed sampled frame",
                            visible=False,
                        )
                    with gr.Column():
                        video_bars = gr.HTML(label="Frame interactions")
                with gr.Row():
                    video_person_conf = gr.Slider(
                        0.01, 0.9, value=0.8, step=0.01, label="Person threshold"
                    )
                    video_person_nms_iou = gr.Slider(
                        0.1,
                        1.0,
                        value=0.6,
                        step=0.05,
                        label="Person NMS IoU",
                    )
                    video_score_threshold = gr.Slider(
                        0.0,
                        1.0,
                        value=0.6,
                        step=0.01,
                        label="Final interaction score threshold",
                    )
                    video_topk = gr.Slider(
                        1, 10, value=5, step=1, label="Top-k interactions"
                    )
                    video_object_nms_iou = gr.Slider(
                        0.0,
                        1.0,
                        value=0.5,
                        step=0.05,
                        label="Target NMS IoU",
                    )
                    video_sample_fps = gr.Slider(
                        0.5,
                        8.0,
                        value=default_video_sample_fps,
                        step=0.5,
                        label="Sample FPS",
                    )
                    video_inference_size = gr.Slider(
                        320,
                        1280,
                        value=default_video_inference_size,
                        step=64,
                        label="Inference max side",
                    )
                with gr.Row():
                    video_show_masks = gr.Checkbox(value=False, label="Show masks", visible=False)
                    video_show_top_verbs = gr.Checkbox(
                        value=False,
                        label="Show all verb/obj combinations",
                        visible=False,
                    )
                    video_unconstrained_verbs = gr.Checkbox(
                        value=False,
                        label="Unconstrained verbs",
                        visible=False,
                    )
                    run_video = gr.Button("Run video")
                    change_video = gr.Button("Change video")
                video_status = gr.Markdown()

                run_video.click(
                    run_video_inference,
                    inputs=[
                        video_input,
                        video_person_conf,
                        video_person_nms_iou,
                        video_score_threshold,
                        video_topk,
                        video_object_nms_iou,
                        video_sample_fps,
                        video_show_masks,
                        video_show_top_verbs,
                        video_unconstrained_verbs,
                        video_inference_size,
                    ],
                    outputs=[
                        video_input,
                        video_frame,
                        video_bars,
                        video_status,
                    ],
                )
                change_video.click(
                    reset_video_slot,
                    inputs=[],
                    outputs=[
                        video_input,
                        video_frame,
                        video_bars,
                        video_status,
                    ],
                )
                if clip_gallery is not None:
                    clip_gallery.select(
                        _load_gallery_clip,
                        inputs=[],
                        outputs=[
                            video_input,
                            video_frame,
                            video_bars,
                            video_status,
                        ],
                    )

            with gr.Tab("Camera"):
                camera_state = gr.State()
                network_camera_timer = gr.Timer(0.05, active=False)
                gr.Markdown(
                    """
                    - Open the browser camera and press **Record** to stream frames to the model.
                    - Or select the Axis MJPEG camera and press **Start selected stream**.
                    - Boxes and interaction bars update as new camera frames are processed.
                    - SL-HOI predicts complete dataset triplets directly; arbitrary verb/object combinations are not available.
                    - Mask visualization is not available.
                    - **UNSEEN** interactions were never observed as training triplets by the open-vocabulary model.
                    - Use the sliders to tune person confidence, final interaction score, and top-k interactions.
                    """
                )
                with gr.Row():
                    browser_camera = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="pil",
                        label="Browser USB camera",
                        webcam_options=gr.WebcamOptions(mirror=False),
                    )
                    camera_frame = gr.Image(
                        type="pil",
                        label="Live processed camera frame",
                    )
                    with gr.Column():
                        camera_bars = gr.HTML(
                            label="Frame interactions",
                            elem_id="camera-interactions",
                        )
                        camera_status = gr.Textbox(
                            label="Camera status",
                            value="Open the browser camera, then press Record.",
                            interactive=False,
                        )
                with gr.Row():
                    camera_source = gr.Dropdown(
                        choices=[
                            "Browser USB camera",
                            "Axis MJPEG camera 172.16.46.6",
                        ],
                        value="Browser USB camera",
                        label="Camera source",
                    )
                    camera_person_conf = gr.Slider(
                        0.01, 0.9, value=0.8, step=0.01, label="Person threshold"
                    )
                    camera_person_nms_iou = gr.Slider(
                        0.1,
                        1.0,
                        value=0.6,
                        step=0.05,
                        label="Person NMS IoU",
                    )
                    camera_score_threshold = gr.Slider(
                        0.0,
                        1.0,
                        value=0.6,
                        step=0.01,
                        label="Final interaction score threshold",
                    )
                    camera_topk = gr.Slider(
                        1, 10, value=5, step=1, label="Top-k interactions"
                    )
                    camera_object_nms_iou = gr.Slider(
                        0.0,
                        1.0,
                        value=0.5,
                        step=0.05,
                        label="Target NMS IoU",
                    )
                    camera_inference_size = gr.Slider(
                        320,
                        1280,
                        value=default_video_inference_size,
                        step=64,
                        label="Inference max side",
                    )
                    camera_show_masks = gr.Checkbox(value=False, label="Show masks", visible=False)
                    camera_show_top_verbs = gr.Checkbox(
                        value=False,
                        label="Show all verb/obj combinations",
                        visible=False,
                    )
                    camera_unconstrained_verbs = gr.Checkbox(
                        value=False,
                        label="Unconstrained verbs",
                        visible=False,
                    )
                with gr.Row():
                    start_selected_camera = gr.Button(
                        "Start selected stream",
                        visible=False,
                    )
                    stop_selected_camera = gr.Button(
                        "Stop selected stream",
                        visible=False,
                    )

                camera_source.change(
                    select_camera_source,
                    inputs=[camera_source],
                    outputs=[
                        browser_camera,
                        camera_frame,
                        camera_bars,
                        camera_status,
                        camera_state,
                        network_camera_timer,
                        start_selected_camera,
                        stop_selected_camera,
                    ],
                    show_progress="hidden",
                )

                browser_camera.stream(
                    run_camera_frame,
                    inputs=[
                        browser_camera,
                        camera_person_conf,
                        camera_person_nms_iou,
                        camera_score_threshold,
                        camera_topk,
                        camera_object_nms_iou,
                        camera_show_masks,
                        camera_show_top_verbs,
                        camera_unconstrained_verbs,
                        camera_inference_size,
                        camera_state,
                    ],
                    outputs=[camera_frame, camera_bars, camera_status, camera_state],
                    stream_every=0.05,
                    trigger_mode="multiple",
                    concurrency_limit=1,
                    show_progress="hidden",
                )
                start_selected_camera.click(
                    start_network_camera_stream,
                    inputs=[camera_source],
                    outputs=[
                        camera_frame,
                        camera_bars,
                        camera_status,
                        camera_state,
                        network_camera_timer,
                    ],
                )
                stop_selected_camera.click(
                    stop_network_camera_stream,
                    inputs=[],
                    outputs=[
                        camera_frame,
                        camera_bars,
                        camera_status,
                        camera_state,
                        network_camera_timer,
                    ],
                )
                network_camera_timer.tick(
                    run_network_camera_frame,
                    inputs=[
                        camera_person_conf,
                        camera_person_nms_iou,
                        camera_score_threshold,
                        camera_topk,
                        camera_object_nms_iou,
                        camera_show_masks,
                        camera_show_top_verbs,
                        camera_unconstrained_verbs,
                        camera_inference_size,
                        camera_state,
                    ],
                    outputs=[camera_frame, camera_bars, camera_status, camera_state],
                    trigger_mode="always_last",
                    concurrency_limit=1,
                    show_progress="hidden",
                )
    return demo
