#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Web Viewer — multi-model compare (server-side)

- Dataset filter by filename prefix (before first `_`)
- Grid view (16/page) of combined overlays; click a tile for the detailed view
- Download combined overlay (JPG, same basename as source)
- Lazy model loading (no heavy work before UI shows)
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml
import gradio as gr
from ultralytics import YOLO

SHOW_CLASS_NAME = True   # affiche le nom de classe (d'après names.yaml)
SHOW_CLASS_ID = False      # affiche l'ID de classe (0, 1, 2, …)
SHOW_SCORE = False         # affiche le score de prédiction (0.00–1.00)


# ------------------------ Constants & Colors ------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PAGE_SIZE = 16
PALETTE = [
    (0, 255, 0),      # green
    (255, 0, 0),      # red
    (0, 128, 255),    # blue
    (255, 165, 0),    # orange
    (186, 85, 211),   # orchid
    (255, 215, 0),    # gold
    (0, 206, 209),    # turquoise
    (255, 105, 180),  # pink
]

# ------------------------ Utils ------------------------
def iter_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]

def dataset_from_filename(p: Path) -> str:
    base = p.stem
    return base.split("_", 1)[0] if "_" in base else "unknown"

def yolo_txt_for_image(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"

def load_names(names_yaml: Optional[Path]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if names_yaml and Path(names_yaml).exists():
        with open(names_yaml, "r") as f:
            data = yaml.safe_load(f)
        raw = data.get("names") if isinstance(data, dict) else None
        if isinstance(raw, dict):
            names = {int(k): str(v) for k, v in raw.items()}
        elif isinstance(raw, list):
            names = {i: str(v) for i, v in enumerate(raw)}
    return names

def parse_yolo_labels(txt_path: Path):
    if not txt_path.exists():
        return []
    items = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                items.append((cls, cx, cy, w, h))
    return items

def yolo_norm_to_xyxy(box, w: int, h: int) -> Tuple[int, int, int, int]:
    _, cx, cy, bw, bh = box
    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
    x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
    x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)

def _pick_font() -> Optional[ImageFont.FreeTypeFont]:
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, 14)
            except Exception:
                pass
    return None

def draw_boxes(image: Image.Image, boxes, classes, scores, id_to_name, color) -> Image.Image:
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _pick_font()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        cls_i = classes[i] if i < len(classes) else -1

        # Construction du label selon les flags globaux
        label_parts = []

        # Nom de classe (optionnel)
        if SHOW_CLASS_NAME:
            name = id_to_name.get(cls_i, str(cls_i))
            label_parts.append(str(name))

        # ID de classe (optionnel)
        if SHOW_CLASS_ID:
            label_parts.append(str(cls_i))

        # Score (optionnel)
        if SHOW_SCORE and scores is not None and i < len(scores):
            label_parts.append(f"{scores[i]:.2f}")

        label = " ".join(label_parts)

        # Si aucun label à afficher, on ne dessine que le rectangle
        if not label:
            continue

        try:
            # textbbox retourne (x0, y0, x1, y1)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (8 * len(label), 16)

        y0 = max(0, y1 - th - 2)
        draw.rectangle([x1, y0, x1 + tw + 4, y0 + th + 2], fill=color)
        draw.text((x1 + 2, y0 + 1), label, fill=(0, 0, 0), font=font)

    return im


# ------------------------ Config inference ------------------------
def infer_from_training_config(config_path: Optional[Path], split: str, weights_name: str = "best.pt") -> Dict[str, Optional[Path]]:
    res = {"images_dir": None, "labels_dir": None, "model": None, "names_yaml": None}
    if not config_path:
        return res
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    dataset = (cfg or {}).get("dataset", {})
    training = (cfg or {}).get("training", {})
    ds_loc = dataset.get("location")
    if ds_loc:
        ds_root = Path(ds_loc)
        res["images_dir"] = ds_root / split / "images"
        res["labels_dir"] = ds_root / split / "labels"
        for cand in [ds_root / "dataset.yaml", ds_root / "data.yaml", ds_root / "dataset.yml", ds_root / "data.yml"]:
            if cand.exists():
                res["names_yaml"] = cand
                break
    proj = training.get("project")
    name = training.get("name")
    if proj and name:
        weights = Path(proj) / name / "weights" / weights_name
        if weights.exists():
            res["model"] = weights
    return res

# ------------------------ App State ------------------------
class AppState:
    def __init__(self, images_dir: Path, labels_dir: Path, model_paths: List[Path], model_names: Optional[List[str]],
                 names_yaml: Optional[Path], device: str = "cpu"):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.names = load_names(names_yaml)
        self.images = iter_images(images_dir)
        if not self.images:
            raise RuntimeError(f"No images found in {images_dir}")

        # datasets (prefix)
        self.image_datasets = {str(p): dataset_from_filename(p) for p in self.images}
        self.datasets = sorted({v for v in self.image_datasets.values()})
        self.filtered_paths: List[Path] = list(self.images)
        self.page_index: int = 0

        # models (lazy: keep path only)
        self.models = []
        for i, mp in enumerate(model_paths or []):
            self.models.append({
                "name": (model_names[i] if model_names and i < len(model_names) and model_names[i] else Path(mp).stem),
                "path": str(mp),
                "model": None,            # lazy loaded later
                "color": PALETTE[i % len(PALETTE)],
            })
        self.device = device

    def add_model(self, path: str, name: Optional[str] = None, device: str = "cpu") -> str:
        p = Path(path)
        if not p.exists():
            return f"Path not found: {path}"
        idx = len(self.models)
        color = PALETTE[idx % len(PALETTE)]
        nick = name or p.stem
        self.models.append({"name": nick, "path": str(p), "model": None, "color": color})
        return f"Registered model '{nick}' from {p} (will load on first use)"

    def image_list(self) -> List[str]:
        return [str(p) for p in self.images]

    def apply_filter(self, dataset: str):
        if dataset and dataset != "All":
            self.filtered_paths = [p for p in self.images if self.image_datasets[str(p)] == dataset]
        else:
            self.filtered_paths = list(self.images)
        self.page_index = 0

    def page(self) -> List[Path]:
        start = self.page_index * PAGE_SIZE
        end = start + PAGE_SIZE
        return self.filtered_paths[start:end]

    def page_count(self) -> int:
        return max(1, (len(self.filtered_paths) + PAGE_SIZE - 1) // PAGE_SIZE)

# ------------------------ Inference helpers ------------------------
state: Optional[AppState] = None

def _ensure_loaded(mdict, device: str):
    """Load YOLO model once per entry, move to device."""
    if mdict["model"] is None:
        mdl = YOLO(mdict["path"])
        try:
            mdl.to(device)
        except Exception:
            pass
        mdict["model"] = mdl

def run_models_on_image(img: Image.Image, conf: float, iou: float):
    outs = []
    for m in state.models:
        _ensure_loaded(m, state.device)  # lazy load
        res = m["model"](img, conf=conf, iou=iou, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.xyxy is None:
            xyxy, cls, scr = [], [], []
        else:
            xyxy = boxes.xyxy.cpu().numpy().astype(int).tolist()
            cls   = boxes.cls.cpu().numpy().astype(int).tolist()
            scr   = boxes.conf.cpu().numpy().tolist()
        outs.append((m, xyxy, cls, scr))
    return outs

def combine_overlay(img: Image.Image, model_outs) -> Image.Image:
    out = img.copy().convert("RGB")
    for (m, xyxy, cls, scr) in model_outs:
        out = draw_boxes(out, xyxy, cls, scr, state.names, m["color"])
    return out

# ------------------------ Detail handlers ------------------------
def on_select(image_path: str, conf: float, iou: float):
    assert state is not None
    img_p = Path(image_path)
    img = Image.open(img_p).convert("RGB")
    w, h = img.size

    # GT
    gt_items = parse_yolo_labels(yolo_txt_for_image(img_p, state.labels_dir))
    gt_xyxy = [yolo_norm_to_xyxy(b, w, h) for b in gt_items]
    gt_classes = [b[0] for b in gt_items]
    im_gt = draw_boxes(img, gt_xyxy, gt_classes, None, state.names, (255, 215, 0))

    # models
    model_outs = run_models_on_image(img, conf, iou)
    per_model_images = [draw_boxes(img, xyxy, cls, scr, state.names, m["color"]) for (m, xyxy, cls, scr) in model_outs]
    per_model_tables = [[{"model": m["name"], "class_id": c, "class_name": state.names.get(c, str(c)), "score": float(s), "box": b}
                         for c, s, b in zip(cls, scr, xyxy)]
                        for (m, xyxy, cls, scr) in model_outs]
    overlay_img = combine_overlay(img, model_outs)

    legend_items = []
    for (m, _, __, ___) in model_outs:
        r, g, b = m["color"]
        legend_items.append(
            f"<span style='display:inline-block;width:12px;height:12px;background:rgb({r},{g},{b});"
            f"border-radius:3px;margin-right:6px;vertical-align:middle'></span>{m['name']}"
        )
    legend_html = "<b>Legend:</b> " + " &nbsp; ".join(legend_items) if legend_items else "<i>No models loaded</i>"

    return im_gt, per_model_images, overlay_img, gt_items, per_model_tables, legend_html

def on_save_overlay(image_path: str, conf: float, iou: float):
    if not image_path:
        return None
    p = Path(image_path)
    img = Image.open(p).convert("RGB")
    outs = run_models_on_image(img, conf, iou)
    overlay = combine_overlay(img, outs)
    outdir = Path("/tmp/yolo_viewer")
    outdir.mkdir(parents=True, exist_ok=True)
    # Save with same basename, JPG enforced
    outpath = outdir / f"{p.stem}.jpg"
    overlay.save(outpath, format="JPEG")
    return str(outpath)

# ------------------------ Grid handlers ------------------------
def on_filter_or_page(dataset: str, page: int, conf: float, iou: float):
    state.apply_filter(dataset)
    state.page_index = max(0, min(page, state.page_count() - 1))
    page_paths = state.page()

    thumbs = []
    for p in page_paths:
        img = Image.open(p).convert("RGB")
        outs = run_models_on_image(img, conf, iou)
        thumbs.append(combine_overlay(img, outs))

    info = f"Page {state.page_index + 1}/{state.page_count()} — {len(state.filtered_paths)} images in '{dataset if dataset != 'All' else 'All'}'"
    return thumbs, [str(p) for p in page_paths], info

def on_grid_click(evt: gr.SelectData, current_paths: List[str], conf: float, iou: float):
    if not current_paths:
        return None, None, None, None, None, None, None
    idx = evt.index
    if idx is None or idx < 0 or idx >= len(current_paths):
        idx = 0
    path = current_paths[idx]
    im_gt, pred_images, overlay_img, gt_items, pred_tables_list, legend = on_select(path, conf, iou)
    gt_rows = [[int(c), float(cx), float(cy), float(w), float(h)] for (c, cx, cy, w, h) in gt_items]
    return path, im_gt, pred_images, gt_rows, pred_tables_list, overlay_img, legend

# ------------------------ UI ------------------------
def build_ui() -> gr.Blocks:
    assert state is not None

    with gr.Blocks(title="YOLO Web Viewer (multi-model)") as demo:
        gr.Markdown("""
        # YOLO Web Viewer — multi-model compare
        - **Filter by dataset** (detected from filename prefix before `_`)
        - **Grid of 16** combined overlays (click to open the detailed view)
        - **Detailed view**: GT, per-model predictions, combined overlay, legend
        - **Download** the combined overlay of the selected image
        - Use **Refresh grid** to (re)compute thumbnails without blocking initial load
        """)

        # Controls (top)
        with gr.Row():
            dataset_choices = ["All"] + state.datasets
            dataset_dd = gr.Dropdown(choices=dataset_choices, value="All", label="Dataset filter")
            conf_slider = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Confidence")
            iou_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="IoU (NMS)")
            refresh_btn = gr.Button("🔄 Refresh grid")

        with gr.Row():
            prev_page = gr.Button("⬅️ Page")
            next_page = gr.Button("Page ➡️")
            page_jump = gr.Number(value=1, precision=0, label="Go to page", interactive=True)
            go_btn = gr.Button("Go")
            page_info = gr.Markdown()

        # Grid view (16)
        grid_gallery = gr.Gallery(label="Grid (combined overlays)", columns=4, rows=4, preview=True)
        current_page_paths = gr.State([])

        # Detail header: selector + download
        with gr.Row():
            image_selector = gr.Dropdown(choices=state.image_list(), value=None, label="Image (detail)", interactive=True)
            save_btn = gr.Button("💾 Download combined overlay")
            save_file = gr.File(label="Overlay file")

        # Detail: images
        with gr.Row():
            gt_img = gr.Image(label="Ground truth", interactive=False)
            pred_gallery = gr.Gallery(label="Per-model predictions", columns=2, preview=True)
        with gr.Row():
            overlay_all = gr.Image(label="Overlay: all models", interactive=False)
        legend_md = gr.HTML()

        # Detail: tables
        with gr.Row():
            gt_table = gr.Dataframe(headers=["class", "cx", "cy", "w", "h"], label="GT (raw YOLO)")
            pred_tables = gr.JSON(label="Predictions tables (per model)")

        # Add model accordion
        with gr.Accordion("Add a model (server path)", open=False):
            with gr.Row():
                new_model_path = gr.Textbox(label="Model .pt path (server)")
                new_model_name = gr.Textbox(label="Nickname (optional)")
            add_btn = gr.Button("Add model")
            add_status = gr.Markdown()

        # ---- Wiring ----
        def _refresh_grid(ds, conf, iou):
            thumbs, paths, info = on_filter_or_page(ds, state.page_index, conf, iou)
            return thumbs, paths, info

        refresh_btn.click(_refresh_grid, [dataset_dd, conf_slider, iou_slider], [grid_gallery, current_page_paths, page_info])

        dataset_dd.change(_refresh_grid, [dataset_dd, conf_slider, iou_slider], [grid_gallery, current_page_paths, page_info])
        conf_slider.change(_refresh_grid, [dataset_dd, conf_slider, iou_slider], [grid_gallery, current_page_paths, page_info])
        iou_slider.change(_refresh_grid, [dataset_dd, conf_slider, iou_slider], [grid_gallery, current_page_paths, page_info])

        def _prev_page(ds, conf, iou):
            state.page_index = max(0, state.page_index - 1)
            return on_filter_or_page(ds, state.page_index, conf, iou)

        def _next_page(ds, conf, iou):
            state.page_index = min(state.page_count() - 1, state.page_index + 1)
            return on_filter_or_page(ds, state.page_index, conf, iou)

        prev_page.click(_prev_page, [dataset_dd, conf_slider, iou_slider], [grid_gallery, current_page_paths, page_info])
        next_page.click(_next_page, [dataset_dd, conf_slider, iou_slider], [grid_gallery, current_page_paths, page_info])

        # Jump to specific page (1-based)
        def _go_to_page(ds, page_num, conf, iou):
            try:
                pn = int(page_num)
            except Exception:
                pn = 1
            pn = max(1, pn)
            state.page_index = min(state.page_count() - 1, pn - 1)
            return on_filter_or_page(ds, state.page_index, conf, iou)

        go_btn.click(_go_to_page, [dataset_dd, page_jump, conf_slider, iou_slider],
                     [grid_gallery, current_page_paths, page_info])

        # Grid click -> detail
        def _grid_select(evt: gr.SelectData, paths, conf, iou):
            return on_grid_click(evt, paths, conf, iou)

        grid_gallery.select(
            _grid_select,
            [current_page_paths, conf_slider, iou_slider],
            [image_selector, gt_img, pred_gallery, gt_table, pred_tables, overlay_all, legend_md],
        )

        # Detail: manual selection
        def _wrap_on_select(img_path, conf, iou):
            if not img_path:
                return None, None, None, None, None, None
            im_gt, pred_images, overlay_img, gt_items, pred_tables_list, legend = on_select(img_path, conf, iou)
            gt_rows = [[int(c), float(cx), float(cy), float(w), float(h)] for (c, cx, cy, w, h) in gt_items]
            return im_gt, pred_images, gt_rows, pred_tables_list, overlay_img, legend

        image_selector.change(_wrap_on_select, [image_selector, conf_slider, iou_slider],
                              [gt_img, pred_gallery, gt_table, pred_tables, overlay_all, legend_md])

        # Save overlay
        save_btn.click(on_save_overlay, [image_selector, conf_slider, iou_slider], [save_file])

        # Add model at runtime
        def _add_model(path, name, ds, conf, iou):
            msg = state.add_model(path, name, device=state.device)
            thumbs, paths, info = on_filter_or_page(ds, state.page_index, conf, iou)
            return msg, thumbs, paths, info

        add_btn.click(_add_model, [new_model_path, new_model_name, dataset_dd, conf_slider, iou_slider],
                      [add_status, grid_gallery, current_page_paths, page_info])

        # NOTE: no heavy inference at load time. Use "Refresh grid".
    return demo

# ------------------------ Main ------------------------
def main():
    global state

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None, help="Training config YAML to infer paths")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split when using --config")
    p.add_argument("--weights_name", type=str, default="best.pt", help="Which weight file to look for under .../weights/")
    # Explicit overrides
    p.add_argument("--images_dir", type=Path, default=None)
    p.add_argument("--labels_dir", type=Path, default=None)
    p.add_argument("--names_yaml", type=Path, default=None)
    # Multiple models
    p.add_argument("--models", type=Path, nargs='*', default=None, help="One or more model .pt files")
    p.add_argument("--model_names", type=str, nargs='*', default=None, help="Display names for models (optional)")
    # Server & auth
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--username", type=str, default=None)
    p.add_argument("--password", type=str, default=None)
    # Device
    p.add_argument("--device", type=str, default="cpu", help="e.g., cpu, cuda:0")
    a = p.parse_args()

    inferred = infer_from_training_config(a.config, a.split, a.weights_name)
    images_dir = a.images_dir or inferred.get("images_dir")
    labels_dir = a.labels_dir or inferred.get("labels_dir")
    names_yaml = a.names_yaml or inferred.get("names_yaml")

    # Models selection
    model_paths: List[Path] = []
    if a.models:
        model_paths = list(a.models)
    else:
        m = inferred.get("model")
        if m:
            model_paths = [m]

    missing = []
    if not images_dir: missing.append("--images_dir (or config.dataset.location + split)")
    if not labels_dir: missing.append("--labels_dir (or config.dataset.location + split)")
    if not model_paths: missing.append("--models (or training.project/name/weights/<weights_name>)")
    if missing:
        raise SystemExit("Missing required paths:\n - " + "\n - ".join(missing))

    for pth in [images_dir, labels_dir]:
        if isinstance(pth, (str, Path)) and not Path(pth).exists():
            raise SystemExit(f"Path not found: {pth}")
    for mp in model_paths:
        if isinstance(mp, (str, Path)) and not Path(mp).exists():
            raise SystemExit(f"Model path not found: {mp}")

    state = AppState(Path(images_dir), Path(labels_dir), [Path(p) for p in model_paths], a.model_names, names_yaml, device=a.device)

    auth = (a.username, a.password) if a.username and a.password else None
    ui = build_ui()
    # queue() sans arguments pour compatibilité large
    ui.queue().launch(server_name=a.host, server_port=a.port, auth=auth)

if __name__ == "__main__":
    main()
