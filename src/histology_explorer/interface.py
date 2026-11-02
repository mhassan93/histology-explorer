import math
import time
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from .patch_input import (
    MAX_SIDE,
    MAX_SOURCE_PIXELS,
    MIN_SIDE,
    SUPPORTED_EXTENSIONS,
    Crop,
    LoadedImage,
    PatchInputError,
    crop_pixels,
    crop_problem,
    load_image,
)
from .segmentation import SegmentationModel, render_overlay, segment
from .segmentation.outputs import TYPE_COLORS

DISPLAY_TARGET = 512
PREVIEW_TARGET = 560
MAX_UPLOAD_SIZE = "20mb"
CROP_COLOR = (0, 200, 255)
INVALID_CROP_COLOR = (255, 40, 40)
OUTSIDE_DIMMING = 0.45
ANCHOR_ARM = 8

INPUT_DESCRIPTION = (
    "**Accepted input:** one H&E-stained tissue image as **PNG, JPEG (.jpg, .jpeg, .jfif) or single-page TIFF**, "
    f"at least {MIN_SIDE} px per side and at most {MAX_SOURCE_PIXELS / 1e6:.0f} megapixels. "
    "JPEG/TIFF orientation tags are applied and the image is converted to RGB.\n\n"
    f"**Patches of {MIN_SIDE}–{MAX_SIDE} px per side** (rectangular is fine) are segmented directly. "
    f"**For a larger image, select a crop of {MIN_SIDE}–{MAX_SIDE} px per side**: click two opposite corners on "
    "the source image, or type the crop position and size. The crop is processed at its original pixel "
    "resolution; images are never resized.\n\n"
    "**Scale:** the model was trained on 40× tissue (about 0.25 µm per pixel) and works best at that magnification. "
    "Uploaded files carry no reliable physical scale, so the app does not know the µm-per-pixel of your image; "
    "areas are reported in **pixels**."
)

CSS = """
#result-image img { image-rendering: pixelated; }
#replace-upload { display: none !important; }
#image-upload:not(:has(.upload-container)) { cursor: pointer; }
#image-upload.replace-target { outline: 2px dashed var(--color-accent); outline-offset: -2px; }
"""

JS = """
() => {
    const box = target => target.closest?.("#image-upload");
    const occupied = el => el && !el.querySelector(".upload-container");
    const spare = () => document.querySelector("#replace-upload input[type=file]");
    const send = files => {
        const input = spare();
        if (!input || !files || !files.length) return;
        const transfer = new DataTransfer();
        transfer.items.add(files[0]);
        input.files = transfer.files;
        input.dispatchEvent(new Event("change", { bubbles: true }));
    };
    document.addEventListener("dragover", event => {
        const el = box(event.target);
        if (!occupied(el)) return;
        event.preventDefault();
        el.classList.add("replace-target");
    }, true);
    document.addEventListener("dragleave", event => {
        const el = box(event.target);
        if (el && !el.contains(event.relatedTarget)) el.classList.remove("replace-target");
    }, true);
    document.addEventListener("drop", event => {
        const el = box(event.target);
        if (!occupied(el)) return;
        event.preventDefault();
        event.stopPropagation();
        el.classList.remove("replace-target");
        send(event.dataTransfer.files);
    }, true);
    document.addEventListener("click", event => {
        const el = box(event.target);
        if (!occupied(el) || event.target.closest("a, button")) return;
        spare()?.click();
    }, true);
}
"""


def display_scale(height: int, width: int) -> int:
    return max(1, DISPLAY_TARGET // max(height, width))


def enlarge(image: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def describe_image(image: LoadedImage) -> str:
    note = ", EXIF orientation applied" if image.orientation_corrected else ""
    return f"{image.width} × {image.height} px {image.source_format} ({image.source_mode}{note})"


def describe_region(image: LoadedImage, crop: Crop | None) -> str:
    if crop is None:
        return describe_image(image)
    return f"{crop.width} × {crop.height} px crop at (x={crop.left}, y={crop.top}) of a {describe_image(image)}"


def make_preview(image: LoadedImage) -> np.ndarray:
    factor = min(1.0, PREVIEW_TARGET / max(image.width, image.height))
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    if size == (image.width, image.height):
        return image.pixels
    return np.asarray(Image.fromarray(image.pixels).resize(size, Image.Resampling.LANCZOS, reducing_gap=2.0))


def preview_to_source(index: list[int], preview: np.ndarray, image: LoadedImage) -> tuple[int, int]:
    preview_height, preview_width = preview.shape[:2]
    x = int((int(index[0]) + 0.5) * image.width / preview_width)
    y = int((int(index[1]) + 0.5) * image.height / preview_height)
    return min(max(x, 0), image.width - 1), min(max(y, 0), image.height - 1)


def draw_selection(source: dict, selection: dict) -> np.ndarray:
    image, preview = source["image"], source["preview"]
    height, width = preview.shape[:2]
    sx, sy = width / image.width, height / image.height
    canvas = Image.fromarray(preview)
    crop = selection["crop"]
    if crop is not None:
        x0 = min(max(math.floor(crop.left * sx), 0), width - 1)
        y0 = min(max(math.floor(crop.top * sy), 0), height - 1)
        x1 = min(max(math.ceil(crop.right * sx) - 1, x0), width - 1)
        y1 = min(max(math.ceil(crop.bottom * sy) - 1, y0), height - 1)
        dimmed = Image.fromarray((preview * OUTSIDE_DIMMING).astype(np.uint8))
        dimmed.paste(canvas.crop((x0, y0, x1 + 1, y1 + 1)), (x0, y0))
        canvas = dimmed
        valid = crop_problem(crop, image.width, image.height) is None
        ImageDraw.Draw(canvas).rectangle((x0, y0, x1, y1), outline=CROP_COLOR if valid else INVALID_CROP_COLOR, width=2)
    if selection["anchor"] is not None:
        ax, ay = int((selection["anchor"][0] + 0.5) * sx), int((selection["anchor"][1] + 0.5) * sy)
        draw = ImageDraw.Draw(canvas)
        draw.line((ax - ANCHOR_ARM, ay, ax + ANCHOR_ARM, ay), fill=CROP_COLOR, width=2)
        draw.line((ax, ay - ANCHOR_ARM, ax, ay + ANCHOR_ARM), fill=CROP_COLOR, width=2)
    return np.asarray(canvas)


def legend_html(model: SegmentationModel) -> str:
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px">'
        f'<span style="width:14px;height:14px;border-radius:3px;background:rgb{color}"></span>'
        f"{model.nuclei_types[type_id]}</span>"
        for type_id, color in TYPE_COLORS.items()
    )
    return f"<div>{items}</div>"


def summarize(nuclei: list[dict], image: LoadedImage, crop: Crop | None) -> str:
    counts = Counter(n["type_name"] for n in nuclei)
    breakdown = " · ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
    return f"### {len(nuclei)} nuclei detected\n{breakdown}\n\nImage: {describe_region(image, crop)}"


def build_interface(model: SegmentationModel, sample_path: Path) -> gr.Blocks:
    def crop_view(source, crop):
        image = source["image"]
        problem = crop_problem(crop, image.width, image.height)
        if problem:
            return None, f"**Crop not usable.** {problem}", "Fix the crop before running segmentation."
        pixels = crop_pixels(image, crop)
        message = (
            f"**Crop:** x={crop.left}, y={crop.top}, {crop.width} × {crop.height} px of the "
            f"{image.width} × {image.height} px image, processed at full resolution."
        )
        return enlarge(pixels, display_scale(crop.height, crop.width)), message, "Crop ready. Press **Run segmentation**."

    def change_crop(source, selection, crop, echo_numbers):
        new_selection = {"crop": crop, "anchor": None}
        result_image, message, status = crop_view(source, crop)
        preview = draw_selection(source, new_selection)
        numbers = (crop.left, crop.top, crop.width, crop.height) if echo_numbers else (gr.skip(),) * 4
        if crop == selection["crop"]:
            return (new_selection, gr.skip(), preview, *numbers, message, gr.skip(), gr.skip(), gr.skip())
        valid = "valid" if result_image is not None else "not usable"
        print(f"[crop] x={crop.left} y={crop.top} {crop.width}x{crop.height} ({valid}); previous result cleared", flush=True)
        return (new_selection, None, preview, *numbers, message, result_image, status, "")

    def cleared(status):
        return (None, gr.Column(visible=False), None, None, None, None, None, None, None, "", None, status, "")

    def begin_loading(file_path):
        if file_path is None:
            return tuple(gr.skip() for _ in range(13))
        print(f"[load] {Path(file_path).name}: previous image and result cleared", flush=True)
        return cleared(f"Loading {Path(file_path).name}…")

    def load_file(file_path):
        if file_path is None:
            return (gr.skip(), gr.skip(), *(gr.skip() for _ in range(13)))
        name = Path(file_path).name
        try:
            image = load_image(file_path)
        except PatchInputError as error:
            print(f"[load] {name} rejected: {error}", flush=True)
            return (None, None, *cleared(f"**Could not load {name}.** {error}"))
        source = {"image": image, "name": name, "preview": make_preview(image)}
        if not image.needs_crop:
            shown = enlarge(image.pixels, display_scale(image.height, image.width))
            status = f"Loaded {name}: {describe_image(image)}. Press **Run segmentation**."
            return (file_path, None, source, gr.Column(visible=False), None, None, None, None, None, None, None, "", shown, status, "")
        crop = Crop.centred(image.width, image.height)
        selection = {"crop": crop, "anchor": None}
        shown, message, _ = crop_view(source, crop)
        status = (
            f"Loaded {name}: {describe_image(image)}. It is larger than {MAX_SIDE} px, so select a crop of "
            f"{MIN_SIDE}–{MAX_SIDE} px per side: click two opposite corners on the source image or edit the crop "
            "numbers. A centred crop is preselected."
        )
        return (
            file_path, None, source, gr.Column(visible=True), selection, None, draw_selection(source, selection),
            crop.left, crop.top, crop.width, crop.height, message, shown, status, "",
        )

    def select_corner(source, selection, event: gr.SelectData):
        if source is None or selection is None:
            return tuple(gr.skip() for _ in range(11))
        point = preview_to_source(event.index, source["preview"], source["image"])
        if selection["anchor"] is None:
            new_selection = {"crop": selection["crop"], "anchor": point}
            print(f"[crop] first corner (x={point[0]}, y={point[1]})", flush=True)
            message = (
                f"First corner at (x={point[0]}, y={point[1]}). Click the opposite corner; "
                "the current crop stays active until then."
            )
            return (new_selection, gr.skip(), draw_selection(source, new_selection), gr.skip(), gr.skip(), gr.skip(), gr.skip(), message, gr.skip(), gr.skip(), gr.skip())
        return change_crop(source, selection, Crop.from_corners(selection["anchor"], point), echo_numbers=True)

    def apply_numbers(left, top, width, height, source, selection):
        if source is None or selection is None:
            return tuple(gr.skip() for _ in range(11))
        if any(value is None for value in (left, top, width, height)):
            return (gr.skip(),) * 7 + ("Enter whole numbers for x, y, width and height.",) + (gr.skip(),) * 3
        crop = Crop(int(left), int(top), int(width), int(height))
        if crop == selection["crop"] and selection["anchor"] is None:
            return tuple(gr.skip() for _ in range(11))
        return change_crop(source, selection, crop, echo_numbers=False)

    def run_segmentation(source, selection, show_overlay):
        if source is None:
            raise gr.Error("Upload an image or load the sample patch first.")
        image = source["image"]
        crop = None
        if image.needs_crop:
            crop = selection["crop"]
            problem = crop_problem(crop, image.width, image.height)
            if problem:
                raise gr.Error(problem)
            pixels = crop_pixels(image, crop)
        else:
            pixels = image.pixels
        start = time.perf_counter()
        result = segment(model, pixels)
        overlay = render_overlay(pixels, result)
        region = f"{pixels.shape[1]}x{pixels.shape[0]}"
        if crop is not None:
            region += f" crop at (x={crop.left}, y={crop.top}) of {image.width}x{image.height}"
        print(
            f"[inference] {source['name']} {region}: {len(result.nuclei)} nuclei in {time.perf_counter() - start:.2f} s",
            flush=True,
        )
        scale = display_scale(*pixels.shape[:2])
        state = {
            "image": enlarge(pixels, scale),
            "overlay": enlarge(overlay, scale),
            "scale": scale,
            "origin": None if crop is None else (crop.left, crop.top),
            "mask": result.instance_mask,
            "nuclei": {n["id"]: n for n in result.nuclei},
        }
        shown = state["overlay"] if show_overlay else state["image"]
        return state, shown, summarize(result.nuclei, image, crop), "Click a nucleus to see its details."

    def toggle_overlay(show_overlay, state):
        if state is None:
            return gr.skip()
        return state["overlay"] if show_overlay else state["image"]

    def describe_nucleus(state, event: gr.SelectData):
        if state is None:
            return "Run segmentation first, then click a nucleus."
        height, width = state["mask"].shape
        x, y = int(event.index[0]) // state["scale"], int(event.index[1]) // state["scale"]
        if not (0 <= x < width and 0 <= y < height):
            return "Click inside the image."
        nucleus_id = int(state["mask"][y, x])
        print(f"[select] pixel (x={x}, y={y}) -> nucleus {nucleus_id or 'none'} (no inference)", flush=True)
        if state["origin"] is None:
            where = f"pixel (x={x}, y={y})"
        else:
            where = f"crop pixel (x={x}, y={y}), source pixel (x={x + state['origin'][0]}, y={y + state['origin'][1]})"
        if nucleus_id == 0:
            return f"No nucleus at {where}."
        nucleus = state["nuclei"][nucleus_id]
        return (
            f"**Nucleus ID:** {nucleus['id']}  \n"
            f"**Area:** {nucleus['area_px']} px  \n"
            f"**Predicted type:** {nucleus['type_name']}  \n"
            f"**Clicked:** {where}"
        )

    with gr.Blocks(title="Histology Explorer") as demo:
        gr.Markdown("# Histology Explorer — Nuclei segmentation\nCellViT-256 (HIPT encoder) running on CPU. Research demo, not for diagnostic use.")
        source_state = gr.State(None)
        selection_state = gr.State(None)
        result_state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(INPUT_DESCRIPTION)
                file_input = gr.File(
                    label=f"H&E image (PNG, JPEG or TIFF, at least {MIN_SIDE} px per side)",
                    file_types=list(SUPPORTED_EXTENSIONS),
                    type="filepath",
                    elem_id="image-upload",
                )
                replace_input = gr.File(
                    file_types=list(SUPPORTED_EXTENSIONS),
                    type="filepath",
                    elem_id="replace-upload",
                )
                sample_button = gr.Button("Load sample patch")
                with gr.Column(visible=False) as crop_panel:
                    gr.Markdown("**Source image:** click two opposite corners to select a crop.")
                    source_image = gr.Image(
                        type="numpy",
                        format="png",
                        interactive=False,
                        show_label=False,
                        buttons=[],
                        height=PREVIEW_TARGET + 40,
                        elem_id="source-image",
                    )
                    with gr.Row():
                        crop_left = gr.Number(label="Crop x (left)", precision=0, elem_id="crop-left")
                        crop_top = gr.Number(label="Crop y (top)", precision=0, elem_id="crop-top")
                        crop_width = gr.Number(label="Crop width", precision=0, elem_id="crop-width")
                        crop_height = gr.Number(label="Crop height", precision=0, elem_id="crop-height")
                    crop_message = gr.Markdown("", elem_id="crop-message")
                run_button = gr.Button("Run segmentation", variant="primary")
                status = gr.Markdown("", elem_id="status")
            with gr.Column(scale=1):
                gr.Markdown("**Processed patch:** the exact pixels sent to the model. After a run, click a nucleus.")
                result_image = gr.Image(
                    type="numpy",
                    format="png",
                    interactive=False,
                    show_label=False,
                    buttons=[],
                    height=560,
                    elem_id="result-image",
                )
                overlay_toggle = gr.Checkbox(value=True, label="Show nuclei overlay")
                gr.HTML(legend_html(model))
                details = gr.Markdown("", elem_id="details")

        crop_numbers = [crop_left, crop_top, crop_width, crop_height]
        crop_outputs = [selection_state, result_state, source_image, *crop_numbers, crop_message, result_image, status, details]
        queue = {"concurrency_id": "image", "concurrency_limit": 1}

        run_event = run_button.click(
            run_segmentation,
            inputs=[source_state, selection_state, overlay_toggle],
            outputs=[result_state, result_image, status, details],
            **queue,
        )
        load_outputs = [source_state, crop_panel, *crop_outputs]
        for upload in (file_input, replace_input):
            upload.upload(
                begin_loading, inputs=upload, outputs=load_outputs, cancels=[run_event], concurrency_limit=None
            ).then(load_file, inputs=upload, outputs=[file_input, replace_input, *load_outputs], **queue)
        sample_button.click(
            lambda: begin_loading(str(sample_path)), outputs=load_outputs, cancels=[run_event], concurrency_limit=None
        ).then(lambda: load_file(str(sample_path)), outputs=[file_input, replace_input, *load_outputs], **queue)
        file_input.clear(lambda: cleared(""), outputs=load_outputs, cancels=[run_event], concurrency_limit=None)
        source_image.select(
            select_corner, inputs=[source_state, selection_state], outputs=crop_outputs, show_progress="hidden", **queue
        )
        for number in crop_numbers:
            for event in (number.submit, number.blur):
                event(
                    apply_numbers,
                    inputs=[*crop_numbers, source_state, selection_state],
                    outputs=crop_outputs,
                    show_progress="hidden",
                    **queue,
                )
        overlay_toggle.change(toggle_overlay, inputs=[overlay_toggle, result_state], outputs=result_image, concurrency_limit=None)
        result_image.select(describe_nucleus, inputs=result_state, outputs=details, concurrency_limit=None)

    return demo
