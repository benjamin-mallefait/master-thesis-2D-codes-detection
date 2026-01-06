# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import csv
import sys
import yaml
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

import pandas as pd
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG = logging.getLogger("test_utils")
if not LOG.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# ------------------------------------------------------------------------------
# Const
# ------------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

CSV_FIELDS = [
    "source", "type", "name", "images",
    "map50_95", "map50", "map75",
    "precision", "recall", "f1",
    "speed_pre_ms", "speed_inf_ms", "speed_post_ms",
    "speed_total_ms", "fps_est",
    "results_dir", "timestamp", "weights"
]

# ------------------------------------------------------------------------------
# Data structures (API attendue par test.py)
# ------------------------------------------------------------------------------
@dataclass
class TestSource:
    label: str            # ex: "internal", "External"
    data_yaml: str        # chemin vers data.yaml
    split: str = "test"   # "test" par défaut


# ------------------------------------------------------------------------------
# Utils fichiers / temps
# ------------------------------------------------------------------------------
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\\-]+", "_", s)[:150]

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ------------------------------------------------------------------------------
# YAML helpers
# ------------------------------------------------------------------------------
def _safe_read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        LOG.warning(f"YAML illisible '{p}': {e}")
        return {}

def _dict_get(d: Dict[str, Any], dotted_path: str, default=None):
    cur = d
    for k in dotted_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ------------------------------------------------------------------------------
# Public helpers demandés par test.py
# ------------------------------------------------------------------------------
def load_training_name(config_path: str) -> str:
    """
    Extrait un nom depuis training.name (ou name), sinon stem du fichier.
    """
    p = Path(config_path)
    cfg = _safe_read_yaml(p)
    for key in ("training.name", "name"):
        val = _dict_get(cfg, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return p.stem


def _get_internal_label_and_yaml(cfg: Dict[str, Any]) -> Tuple[str, str]:
    """
    Retrouve le label et le data.yaml interne selon ta structure:
      dataset.location -> <dir>/data.yaml
      dataset.project_name (si présent) pour le label
    """
    dataset = cfg.get("dataset", {}) or {}
    location = dataset.get("location")
    if not location:
        raise ValueError("`dataset.location` manquant dans la config.")
    data_yaml = os.path.join(location, "data.yaml")
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"data.yaml introuvable: {data_yaml}")
    label = dataset.get("project_name") or os.path.basename(location.rstrip("/"))
    return label, data_yaml


def build_sources_from_config(
    config_path: str,
    use_dataset_test: bool,
    use_external_test: bool,
    external_data_yaml: Optional[str],
    external_label: str,
    skip_global: bool,           # ignoré ici (nous faisons global + per_dataset ensemble)
    group_regex: Optional[str],  # utilisé plus tard au moment d'évaluer
) -> List[TestSource]:
    """
    Construit les sources à partir de ta config (interne + externe).
    """
    cfg = _safe_read_yaml(config_path)
    sources: List[TestSource] = []

    if use_dataset_test:
        internal_label, internal_yaml = _get_internal_label_and_yaml(cfg)
        sources.append(TestSource(label=internal_label, data_yaml=internal_yaml))

    if use_external_test and external_data_yaml:
        sources.append(TestSource(label=str(external_label), data_yaml=str(external_data_yaml)))

    if not sources:
        LOG.warning("Aucune source de test détectée (active --use-dataset-test et/ou --use-external-test).")
    else:
        LOG.info("Sources de test: " + ", ".join(s.label for s in sources))

    return sources


# ------------------------------------------------------------------------------
# Évaluation Ultralytics (mêmes métriques et FPS que ton script)
# ------------------------------------------------------------------------------
def eval_split(
    model: YOLO,
    data_yaml: str,
    iou: float,
    conf,
    max_det: int,
    half: bool,
    dnn: bool,
    plots: bool,
    project: str,
    name: str,
    save_json: bool = True,
    save_txt: bool = True,
    save: bool = True,
):
    return model.val(
        data=data_yaml,
        split="test",
        iou=iou,
        conf=conf,
        max_det=max_det,
        half=half,
        dnn=dnn,
        plots=plots,
        project=project,
        name=name,
        save_json=save_json,
        save_txt=save_txt,
        save=save,
    )


def read_results_csv_metrics(results_dir: str) -> Dict[str, float]:
    """Lit results.csv si présent pour récupérer precision/recall/F1."""
    path = os.path.join(results_dir, "results.csv")
    out = {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return out
        last = rows[-1]
        key_map = {
            "precision": ["metrics/precision", "metrics/precision(B)", "precision"],
            "recall":    ["metrics/recall",    "metrics/recall(B)",    "recall"],
            "f1":        ["metrics/f1",        "metrics/f1(B)",        "f1"],
        }
        def pick(keys):
            for k in keys:
                if k in last and last[k] not in ("", None):
                    try:
                        return float(last[k])
                    except Exception:
                        pass
            return float("nan")
        for k, keys in key_map.items():
            out[k] = pick(keys)
        if (out["f1"] != out["f1"]) and all(x == x for x in (out["precision"], out["recall"])) and (out["precision"]+out["recall"]>0):
            out["f1"] = 2 * (out["precision"] * out["recall"]) / (out["precision"] + out["recall"])
    except Exception:
        pass
    return out


def extract_metrics(results, results_dir: str = None) -> Dict[str, float]:
    """
    Récupère mAP + vitesses + (Precision, Recall, F1) exactement comme ton script,
    y compris le calcul de FPS = 1000 / (pre+inf+post).
    """
    # mAP
    try:
        m = {
            "map50_95": float(results.box.map),
            "map50": float(results.box.map50),
            "map75": float(results.box.map75),
        }
    except Exception:
        m = {"map50_95": float("nan"), "map50": float("nan"), "map75": float("nan")}

    # speed (ms/img)
    pre = inf = post = float("nan")
    try:
        spd = getattr(results, "speed", {}) or {}
        pre = float(spd.get("preprocess", float("nan")))
        inf = float(spd.get("inference", float("nan")))
        post = float(spd.get("postprocess", float("nan")))
    except Exception:
        pass
    total = pre + inf + post if all(x == x for x in (pre, inf, post)) else float("nan")
    fps = 1000.0 / total if (total == total and total > 0.0) else float("nan")

    # Precision / Recall / F1
    prec = rec = f1 = float("nan")

    # 1) via attributs Ultralytics (selon versions)
    try:
        if hasattr(results, "box"):
            if hasattr(results.box, "mp"): prec = float(results.box.mp)
            if hasattr(results.box, "mr"): rec  = float(results.box.mr)
            if hasattr(results.box, "f1"): f1   = float(results.box.f1)
    except Exception:
        pass

    # 2) via results_dict si présent
    try:
        d = getattr(results, "results_dict", None) or {}
        def getd(*keys):
            for k in keys:
                if k in d:
                    try:
                        return float(d[k])
                    except Exception:
                        pass
            return float("nan")
        if prec != prec: prec = getd("metrics/precision", "metrics/precision(B)")
        if rec  != rec:  rec  = getd("metrics/recall", "metrics/recall(B)")
        if f1   != f1:   f1   = getd("metrics/f1", "metrics/f1(B)")
    except Exception:
        pass

    # 3) fallback results.csv
    if results_dir:
        csv_fallback = read_results_csv_metrics(results_dir)
        if prec != prec: prec = csv_fallback["precision"]
        if rec  != rec:  rec  = csv_fallback["recall"]
        if f1   != f1:   f1   = csv_fallback["f1"]

    # 4) calcule F1 si P/R connus
    if (f1 != f1) and all(x == x for x in (prec, rec)) and (prec + rec > 0):
        f1 = 2 * (prec * rec) / (prec + rec)

    m.update({
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "speed_pre_ms": pre,
        "speed_inf_ms": inf,
        "speed_post_ms": post,
        "speed_total_ms": total,
        "fps_est": fps,
    })
    return m


# ------------------------------------------------------------------------------
# Résolution des images / grouping (identique à ton script)
# ------------------------------------------------------------------------------
def ensure_dummy_val_dir(base_dir: str) -> str:
    dummy_val = os.path.join(base_dir, "valid", "images")
    os.makedirs(dummy_val, exist_ok=True)
    return os.path.abspath(dummy_val)

def load_data_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def resolve_test_images(data_cfg: dict, base_dir: str) -> List[str]:
    test_entry = data_cfg.get("test")
    if test_entry is None:
        raise ValueError("La clé `test:` n'est pas définie dans data.yaml.")

    def to_abs(p: str) -> str:
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

    candidates_dirs: List[str] = []
    paths: List[str] = []

    if isinstance(test_entry, list):
        paths = [to_abs(p) for p in test_entry]
    elif isinstance(test_entry, str):
        p0 = to_abs(test_entry)
        p1 = os.path.join(base_dir, "test", "images")
        p2 = os.path.normpath(os.path.join(os.path.dirname(base_dir), "test", "images"))
        p3 = os.path.join(base_dir, "test")

        if os.path.isdir(p0):
            candidates_dirs.append(p0)
        elif os.path.isfile(p0):
            if p0.lower().endswith(".txt"):
                raw = read_lines(p0)
                paths = [to_abs(p) for p in raw]
            elif is_image(p0):
                paths = [p0]
        else:
            candidates_dirs += [p1, p2, p3]

        if not paths and candidates_dirs:
            for d in candidates_dirs:
                if os.path.isdir(d):
                    imgs = [os.path.join(d, f) for f in os.listdir(d)]
                    imgs = [i for i in imgs if is_image(i)]
                    if imgs:
                        LOG.info(f"[INFO] test résolu vers: {os.path.abspath(d)}")
                        paths = imgs
                        break
    else:
        raise TypeError("Format de `test` non supporté (attendu: str | list).")

    paths = [os.path.abspath(p) for p in paths if is_image(p)]
    if not paths:
        LOG.debug("Aucune image trouvée pour test.")
    return paths

def group_by_first_underscore(img_paths: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for p in img_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        key = stem.split('_', 1)[0].strip().lower()
        groups.setdefault(key, []).append(p)
    return groups

def group_by_regex(img_paths: List[str], regex: str) -> Dict[str, List[str]]:
    pattern = re.compile(regex)
    groups: Dict[str, List[str]] = {}
    for p in img_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = pattern.match(stem)
        key = (m.group(1) if m else stem).strip().lower()
        groups.setdefault(key, []).append(p)
    return groups


# ------------------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------------------
def write_metrics_csv_row(csv_path: str, row: Dict[str, Any], header: List[str] = CSV_FIELDS):
    file_exists = os.path.isfile(csv_path)
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

def write_single_metrics_csv(csv_path: str, row: Dict[str, Any], header: List[str] = CSV_FIELDS):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------------------------
# Orchestrateur demandé par test.py
# ------------------------------------------------------------------------------
def evaluate_all_sources(
    config_path: str,
    weights_path: Optional[str],
    out_dir: str,
    run_name: str,
    sources: List[TestSource]
) -> List[Tuple[str, str]]:
    """
    Pour chaque source (interne/externe) :
      - lance une éval GLOBALE -> <out_dir>/<source>/global/
      - lance une éval PAR DATASET (groupes) -> <out_dir>/<source>/per_dataset/<key>/
      - écrit un summary: <out_dir>/test_<source>_summary.csv
        (colonne 'source' = '<label>::<type>::<name>' pour une agrégation propre entre epochs)
    Retourne: [(source_label, summary_csv_path), ...]
    """
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Lecture des hyper-params d'éval depuis la config (comme avant)
    cfg = _safe_read_yaml(config_path)
    training = cfg.get("training", {}) or {}
    iou = training.get("iou", 0.7)
    conf = training.get("conf", None)
    max_det = training.get("max_det", 300)
    half = training.get("half", False)
    dnn = training.get("dnn", False)
    plots = training.get("plots", True)

    # Model
    model = YOLO(weights_path) if weights_path else YOLO()
    LOG.info(f"Évaluation avec weights: {weights_path or '<par défaut>'}")

    out: List[Tuple[str, str]] = []

    for src in sources:
        src_root = out_root / sanitize(src.label)
        global_project = src_root
        per_group_project = src_root / "per_dataset"
        ensure_dir(str(global_project))
        ensure_dir(str(per_group_project))

        # summary pour CETTE source
        summary_csv = out_root / f"test_{sanitize(src.label)}_summary.csv"
        if summary_csv.exists():
            summary_csv.unlink()

        # Charger data.yaml et images test
        data_cfg = load_data_yaml(src.data_yaml)
        data_yaml_dir = os.path.dirname(os.path.abspath(src.data_yaml))
        ensure_dummy_val_dir(data_yaml_dir)

        # ----------------- GLOBAL -----------------
        LOG.info(f"\n=== [{src.label}] Évaluation GLOBALE ===")
        global_name = "global"
        res_global = eval_split(
            model, data_yaml=src.data_yaml, iou=iou, conf=conf, max_det=max_det,
            half=half, dnn=dnn, plots=plots, project=str(global_project), name=global_name,
            save_json=True, save_txt=True, save=True
        )
        metrics = extract_metrics(res_global, results_dir=os.path.join(str(global_project), global_name))
        test_images = resolve_test_images(data_cfg, base_dir=data_yaml_dir)

        # NB: pour l’agrégation inter-epochs on encode le "source" combiné
        combined_source = f"{src.label}::global::global"
        row = {
            "source": combined_source,
            "type": "global",
            "name": "global",
            "images": len(test_images),
            "map50_95": metrics["map50_95"],
            "map50": metrics["map50"],
            "map75": metrics["map75"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "speed_pre_ms": metrics["speed_pre_ms"],
            "speed_inf_ms": metrics["speed_inf_ms"],
            "speed_post_ms": metrics["speed_post_ms"],
            "speed_total_ms": metrics["speed_total_ms"],
            "fps_est": metrics["fps_est"],
            "results_dir": os.path.join(str(global_project), global_name),
            "timestamp": now_iso(),
            "weights": str(weights_path or "")
        }
        write_single_metrics_csv(os.path.join(str(global_project), global_name, "metrics.csv"), row)
        write_metrics_csv_row(str(summary_csv), row)

        # ----------------- GROUPS -----------------
        LOG.info(f"\n=== [{src.label}] Évaluation PAR DATASET (groupes) ===")
        all_test_imgs = resolve_test_images(data_cfg, base_dir=data_yaml_dir)
        # Le regex de grouping est fourni dans test.py (args.group_regex); on ne le reçoit pas ici.
        # On applique le split par 1er underscore par défaut (identique à ton script).
        groups = group_by_first_underscore(all_test_imgs)

        tmp_root = src_root / "_tmp_eval_splits"
        ensure_dir(str(tmp_root))

        for group_key, paths in sorted(groups.items()):
            safe_key = sanitize(group_key)
            group_txt = tmp_root / f"{safe_key}.txt"
            group_yaml = tmp_root / f"{safe_key}.yaml"
            ensure_dir(group_txt.parent.as_posix())
            ensure_dir(group_yaml.parent.as_posix())

            with open(group_txt, "w") as f:
                for p in paths:
                    f.write(p + "\n")

            make_group_data_yaml(data_cfg, str(group_txt), str(group_yaml), base_dir=data_yaml_dir)

            LOG.info(f"\n[{src.label}][GROUP] {group_key}  ({len(paths)} images)")
            res = eval_split(
                model, data_yaml=str(group_yaml), iou=iou, conf=conf, max_det=max_det,
                half=half, dnn=dnn, plots=plots, project=str(per_group_project), name=safe_key,
                save_json=True, save_txt=True, save=True
            )
            m = extract_metrics(res, results_dir=os.path.join(str(per_group_project), safe_key))

            combined_source = f"{src.label}::dataset::{group_key}"
            row = {
                "source": combined_source,
                "type": "dataset",
                "name": group_key,
                "images": len(paths),
                "map50_95": m["map50_95"],
                "map50": m["map50"],
                "map75": m["map75"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "speed_pre_ms": m["speed_pre_ms"],
                "speed_inf_ms": m["speed_inf_ms"],
                "speed_post_ms": m["speed_post_ms"],
                "speed_total_ms": m["speed_total_ms"],
                "fps_est": m["fps_est"],
                "results_dir": os.path.join(str(per_group_project), safe_key),
                "timestamp": now_iso(),
                "weights": str(weights_path or "")
            }
            write_single_metrics_csv(os.path.join(str(per_group_project), safe_key, "metrics.csv"), row)
            write_metrics_csv_row(str(summary_csv), row)

        out.append((src.label, str(summary_csv)))

    return out


# ------------------------------------------------------------------------------
# YAML temporaire pour les splits de groupes
# ------------------------------------------------------------------------------
def make_group_data_yaml(base_data: dict, test_listfile: str, out_yaml: str, base_dir: str):
    def to_abs(p: Any):
        if isinstance(p, str):
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
        elif isinstance(p, list):
            return [to_abs(x) for x in p]
        else:
            return p

    data_copy = dict(base_data)

    for k in ("path", "train", "val", "test"):
        if k in data_copy and data_copy[k]:
            data_copy[k] = to_abs(data_copy[k])

    need_dummy_val = ("val" not in data_copy) or (not data_copy["val"]) or \
                     (isinstance(data_copy["val"], str) and not os.path.isdir(data_copy["val"]))
    if need_dummy_val:
        data_copy["val"] = ensure_dummy_val_dir(base_dir)

    data_copy["test"] = os.path.abspath(test_listfile)

    ensure_dir(os.path.dirname(out_yaml))
    with open(out_yaml, "w") as f:
        yaml.safe_dump(data_copy, f, sort_keys=False)


# ------------------------------------------------------------------------------
# Compat helper (pas indispensable mais conservé)
# ------------------------------------------------------------------------------
def make_outdir(base: str, name: str) -> str:
    p = Path(base) / name
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
