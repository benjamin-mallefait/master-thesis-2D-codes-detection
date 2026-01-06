#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import yaml


from roboflow import Roboflow
from utils import load_config, filter_dataset  # <— on appelle DIRECTEMENT

def _exists_yolov8(root: Path) -> bool:
    return (root / "data.yaml").exists()

def _exists_coco(root: Path) -> bool:
    return (root / "train" / "_annotations.coco.json").exists()

def _same_roboflow_version(yolo_data_yaml: Path, cfg_ds: dict) -> bool:
    try:
        data = load_config(str(yolo_data_yaml))
        rf = data.get("roboflow", {})
        return (
            str(rf.get("version")) == str(cfg_ds.get("version_number"))
            and str(rf.get("project")) == str(cfg_ds.get("project_name"))
        )
    except Exception:
        return False

def download_dmcode_dataset(api_key: str, workspace: str, download_format: str,
                            location: str, project_name: str, version_number: int) -> None:
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace)
    project = ws.project(project_name)
    version = project.version(int(version_number))
    version.download(download_format, location=location)

def _is_empty(val) -> bool:
    return (val is None) or (str(val).strip().lower() in ("", "none", "null"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh-config", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--workspace", type=str, default="industrial-qr")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_config(str(cfg_path)) or {}
    ssh_cfg = load_config(args.ssh_config) or {}

    ds = cfg.get("dataset", {})
    src_root = Path(ds["location"]).resolve()
    src_root.parent.mkdir(parents=True, exist_ok=True)

    print("[DL] START", flush=True)

    # == DOWNLOAD ==
    need_download = True
    if src_root.exists():
        if _exists_yolov8(src_root):
            yml = src_root / "data.yaml"
            if _same_roboflow_version(yml, ds):
                print("[DL] YOLOv8 dataset déjà présent (même version) → skip.", flush=True)
                need_download = False
        elif _exists_coco(src_root):
            print("[DL] COCO dataset déjà présent → skip.", flush=True)
            need_download = False

    if need_download:
        print(f"[DL] Téléchargement Roboflow → {ds['project_name']} v{ds['version_number']} ({ds['download_format']})", flush=True)
        download_dmcode_dataset(
            api_key=ssh_cfg["api_key"],
            workspace=args.workspace,
            download_format=ds["download_format"],
            location=str(src_root),
            project_name=ds["project_name"],
            version_number=ds["version_number"],
        )
        print("[DL] OK", flush=True)
    else:
        print("[DL] OK (déjà présent)", flush=True)

     # == FILTRAGE ==
    print("[FILTER] START", flush=True)
    fmt = str(ds.get("download_format", "yolov8")).lower()
    dst_root = ds.get("dst_root") or None
    splits = tuple(s.strip() for s in str(ds.get("splits", "train,valid,test")).split(",") if s.strip())
    on_conflict = str(ds.get("on_conflict", "overwrite")).lower()
    keep_empty = bool(ds.get("keep_images_without_labels", False))
    remap_to_zero = bool(ds.get("remap_to_zero", True))
    dry_first = bool(ds.get("dry_run_first", False))

    # --- nouveaux champs pour le mode de filtrage ---
    filter_mode = ds.get("filter_mode", None)   # keep_one | merge_all
    if filter_mode is None or str(filter_mode).lower() == "none":
        print("[FILTER] Pas de filtrage demandé → dataset inchangé.", flush=True)
        final_location = src_root
    else:
        filter_mode = str(filter_mode).lower()
    
    merge_all_name = str(ds.get("merge_all_name", "code"))
    keep = ds.get("keep_class")  # utilisé seulement en keep_one

    # Normalisation format
    fmt_norm = "coco" if fmt == "coco" else "yolov8"

    # Décider si on filtre :
    do_filter = (
        (filter_mode == "merge_all") or
        (filter_mode == "keep_one" and not _is_empty(keep))
    )

    final_location = src_root  # par défaut : pas de filtre

    if not do_filter:
        if filter_mode == "keep_one":
            print("[FILTER] Aucune classe demandée → pas de filtrage (keep_one sans keep_class).", flush=True)
        else:
            print("[FILTER] Pas de filtrage dans les labels demandés.", flush=True)
    else:
        # Dry-run (facultatif)
        if dry_first:
            print(f"[FILTER] Dry-run… (mode={filter_mode})", flush=True)
            _ = filter_dataset(
                src_root=str(src_root),
                fmt=fmt_norm,
                keep_class=(keep if isinstance(keep, int) else (None if filter_mode == "merge_all" else str(keep))),
                dst_root=dst_root,
                splits=splits,
                keep_images_without_labels=keep_empty,
                remap_to_zero=remap_to_zero,
                dry_run=True,
                verbose=True,
                on_conflict=on_conflict,
                mode=("merge_all" if filter_mode == "merge_all" else "keep_one"),
                merge_all_name=merge_all_name,
            )
            print("[FILTER] Dry-run OK.", flush=True)

        # Exécution réelle
        rep = filter_dataset(
            src_root=str(src_root),
            fmt=fmt_norm,
            keep_class=(keep if isinstance(keep, int) else (None if filter_mode == "merge_all" else str(keep))),
            dst_root=dst_root,
            splits=splits,
            keep_images_without_labels=keep_empty,
            remap_to_zero=remap_to_zero,
            dry_run=False,
            verbose=True,          # ← laisse parler utils.py (prints)
            on_conflict=on_conflict,
            mode=("merge_all" if filter_mode == "merge_all" else "keep_one"),
            merge_all_name=merge_all_name,
        )
        final_location = Path(rep["dst_root"]).resolve()
        print(f"[FILTER] OK → {final_location}", flush=True)

    print("[DONE] download+filter terminé.", flush=True)

if __name__ == "__main__":
    main()
