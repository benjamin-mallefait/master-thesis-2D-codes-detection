#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime
import yaml
from ultralytics import YOLO

# -------- imports communs depuis utils.py --------
from utils import (
    IMG_EXTS,
    iter_images,
    resolve_path_ignoring_parent,
    extract_fold_token,
    write_list_file,
    build_exp_dir,
    save_config,
    load_config
)

# =========================
# Helpers généraux
# =========================

def set_seed_all(seed: int, deterministic: bool = True):
    """Fixe la seed pour random/NumPy/PyTorch (+ CUDA)."""
    try:
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception as e:
        print(f"[WARN] Seed non appliquée complètement: {e}", file=sys.stderr)


def parse_extra_into_training_kwargs(tokens, base: dict) -> dict:
    """
    Convertit une liste du style:
      ["--epochs", "200", "val=False", "imgsz=1280", "--cache"]
    en kwargs compatibles Ultralytics:
      {"epochs":200, "val":False, "imgsz":1280, "cache":True}
    Priorité: les tokens écrasent les clés déjà dans 'base'.
    """
    out = dict(base)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        key = None
        val = None

        if tok.startswith("--"):
            key = tok[2:]
            # Valeur dans le token suivant ?
            if i + 1 < len(tokens) and not tokens[i+1].startswith("--"):
                val = tokens[i+1]
                i += 2
            else:
                # flag seul => True
                val = True
                i += 1
        elif "=" in tok:
            key, val = tok.split("=", 1)
            i += 1
        else:
            # jeton orphelin -> on ignore
            i += 1
            continue

        # casting simple
        if isinstance(val, str):
            low = val.lower()
            if low in ("true", "false"):
                val = (low == "true")
            else:
                # int / float si possible
                try:
                    if "." in val:
                        val = float(val)
                    else:
                        val = int(val)
                except Exception:
                    pass
        out[key] = val
    return out


# =========================
# YOLO-specific helpers
# =========================

def expand_train_images_only(data_yaml_path: Path, data_yaml: dict) -> list[Path]:
    """
    N’expand QUE le split 'train' du data.yaml.
    - Corrige les chemins relatifs en ignorant les '../' de tête (../train/images -> train/images).
    - Supporte dossiers, fichiers image et .txt listant des images.
    """
    base_dir = data_yaml_path.parent
    train_spec = data_yaml.get("train", None)
    if train_spec is None:
        return []

    def _expand_one(spec):
        out = []
        if isinstance(spec, (list, tuple)):
            for s in spec:
                out += _expand_one(s)
            return out

        p = resolve_path_ignoring_parent(base_dir, str(spec))

        if p.is_dir():
            out += list(iter_images(p))
        elif p.is_file():
            if p.suffix.lower() in IMG_EXTS:
                out.append(p)
            elif p.suffix.lower() == ".txt":
                for line in p.read_text().strip().splitlines():
                    ln = line.strip()
                    if not ln:
                        continue
                    q = resolve_path_ignoring_parent(p.parent, ln)
                    if q.exists() and q.suffix.lower() in IMG_EXTS:
                        out.append(q)
        else:
            # dernier recours: essayer sans resolve() strict
            s = str(spec).replace("\\", "/").lstrip()
            while s.startswith("../"):
                s = s[3:]
            while s.startswith("./"):
                s = s[2:]
            s = s or "."
            cand = (base_dir / s)
            if cand.exists():
                if cand.is_dir():
                    out += list(iter_images(cand))
                elif cand.suffix.lower() in IMG_EXTS:
                    out.append(cand)
        return out

    imgs = _expand_one(train_spec)
    imgs = sorted({p.resolve() for p in imgs if p.exists()})
    return imgs


def read_last_metrics_from_results_csv(run_dir: Path) -> dict:
    """
    Lit le 'results.csv' produit par Ultralytics pour récupérer les dernières métriques.
    Renvoie un dict avec precision, recall, map50, map50_95, f1 (si dispo).
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    last = rows[-1]
    candidates = {
        "precision": ["metrics/precision(B)", "precision", "metrics/precision"],
        "recall": ["metrics/recall(B)", "recall", "metrics/recall"],
        "map50": ["metrics/mAP50(B)", "mAP50(B)", "metrics/mAP50", "mAP50"],
        "map50_95": ["metrics/mAP50-95(B)", "mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"],
    }
    out = {}
    for k, keys in candidates.items():
        val = None
        for kk in keys:
            if kk in last and last[kk] != "":
                try:
                    val = float(last[kk])
                    break
                except ValueError:
                    pass
        out[k] = val

    p, r = out.get("precision"), out.get("recall")
    out["f1"] = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else None
    return out


def write_fold_yaml(out_path: Path, train_imgs, val_imgs, names, nc):
    """
    - Écrit deux fichiers .txt (train/val) listant les images (chemins ABSOLUS).
    - Écrit un YAML qui référence ces deux .txt en chemins ABSOLUS.
    """
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_txt = (out_path.parent / f"{out_path.stem}.train.txt").resolve()
    val_txt   = (out_path.parent / f"{out_path.stem}.val.txt").resolve()

    write_list_file(train_txt, list(train_imgs))  # <- utils.write_list_file
    write_list_file(val_txt,   list(val_imgs))

    spec = {
        "train": str(train_txt),   # ABSOLU
        "val":   str(val_txt),     # ABSOLU
    }
    if names is not None:
        spec["names"] = names
    if nc is not None:
        spec["nc"] = nc

    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)

# =========================
# Training entrypoints
# =========================

def train_single(cfg: dict):
    data_root = Path(cfg["dataset"]["location"]).resolve()
    data_yaml_path = Path(cfg["dataset"]["data_yaml"]).resolve()
    if not data_yaml_path.exists():
        print(f"[INFO] data.yaml introuvable, utilisation de {data_root / 'data.yaml'}", file=sys.stderr)
        data_yaml_path = data_root / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[ERR] data.yaml introuvable: {data_yaml_path}", file=sys.stderr)
        sys.exit(1)

    # Seed (si fournie dans la conf)
    seed = cfg.get("training", {}).get("seed", None)
    if seed is not None:
        set_seed_all(int(seed), deterministic=True)

    # Prépare modèle + kwargs
    train_section = cfg["training"].copy()
    model_spec = train_section.pop("model")
    project = train_section.pop("project", None)
    name = train_section.pop("name", None)

    # Device peut être dans train_section (via overrides)
    model = YOLO(model_spec)

    model.train(
        data=str(data_yaml_path),
        project=project,
        name=name,
        **train_section
    )

    # Sauvegarde une copie de la config dans le dossier d'exp
    exp_dir = build_exp_dir(project, name)      # <- utils.build_exp_dir
    save_config(cfg, exp_dir, filename="config.yaml")  # <- utils.save_config


def train_kfold(cfg: dict):
    data_root = Path(cfg["dataset"]["location"]).resolve()
    data_yaml_path = data_root / "data.yaml"
    if not data_yaml_path.exists():
        print(f"[ERR] data.yaml introuvable: {data_yaml_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_yaml_path, "r") as f:
        data_yaml = yaml.safe_load(f)

    # 1) Images du TRAIN uniquement (normalisation des chemins)
    train_images = expand_train_images_only(data_yaml_path, data_yaml)
    if not train_images:
        print(f"[ERR] Aucune image trouvée dans 'train' (après normalisation). "
              f"Ex. attendu: {data_yaml_path.parent/'train/images'}", file=sys.stderr)
        sys.exit(1)

    # 2) Groupage par fold (préfixe avant '_')
    by_fold = {}
    for img in train_images:
        token = extract_fold_token(img)   # <- utils.extract_fold_token
        by_fold.setdefault(token, []).append(img)

    print("\nRépartition par fold (train uniquement):")
    for k in sorted(by_fold):
        print(f"  - {k}: {len(by_fold[k])} images")

    unique_folds = sorted(by_fold.keys())
    if len(unique_folds) <= 1:
        print(f"[WARN] Un seul fold détecté ({unique_folds}). Le K-Fold a peu d’intérêt.")

    # 3) Préparer modèle et paramètres
    base_kwargs = cfg["training"].copy()
    project_root = base_kwargs.pop("project", "results")
    base_name = base_kwargs.pop("name", "exp")

    # 👉 Regrouper tous les runs sous results/<exp_name>/
    project_dir = Path(project_root) / base_name
    project_dir.mkdir(parents=True, exist_ok=True)

    names = data_yaml.get("names", None)
    nc = data_yaml.get("nc", None)

    # Dossier pour les YAML/txt de folds
    folds_dir = project_dir / "foldspecs"
    folds_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    summary_csv = project_dir / f"{base_name}_kfold_summary.csv"

    base_model_spec = cfg["training"]["model"]   # ex: 'yolov8n.pt' ou 'yolov8n.yaml'
    seed = cfg["training"].get("seed", None)
    if seed is not None:
        set_seed_all(int(seed), deterministic=True)

    for fold_val in unique_folds:
        # Si tu veux skipper des folds de test rapide, dé-commente/ajuste la logique
        # if "1" in fold_val or "2" in fold_val or "3" in fold_val or "4" in fold_val:
        #     print(f"[INFO] Skip fold {fold_val} (pour test rapide)")
        #     continue

        val_imgs = sorted(by_fold[fold_val])
        train_imgs = sorted({p for f, imgs in by_fold.items() if f != fold_val for p in imgs})

        if not train_imgs or not val_imgs:
            print(f"[WARN] Fold {fold_val}: train={len(train_imgs)}, val={len(val_imgs)}. Skip.")
            continue

        fold_yaml_path = folds_dir / f"fold_{fold_val}.yaml"
        write_fold_yaml(fold_yaml_path, train_imgs, val_imgs, names, nc)

        run_name = f"val-{fold_val}"
        print(f"\n=== Run K-Fold: validation sur [{fold_val}] ===")
        print(f"Train images: {len(train_imgs)} | Val images: {len(val_imgs)}")
        print(f"YAML utilisé : {fold_yaml_path.resolve()}")

        # 🔁 Recréer un modèle NEUF pour ce fold
        model = YOLO(base_model_spec)

        # 🔒 Empêcher toute reprise
        train_kwargs = dict(base_kwargs)
        train_kwargs["resume"] = False

        model.train(
            data=str(fold_yaml_path),
            project=str(project_dir),
            name=run_name,
            **train_kwargs
        )

        # 🗃️ Log du run
        run_dir = project_dir / run_name
        run_cfg = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "val_folds": [fold_val],
            "num_train_images": len(train_imgs),
            "num_val_images": len(val_imgs),
            "cfg": cfg,
        }
        with open(run_dir / "kfold_run_config.yaml", "w") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False, allow_unicode=True)

        # 📊 Récup métriques
        metrics = read_last_metrics_from_results_csv(run_dir) or {}
        summary_rows.append({
            "run_name": run_name,
            "val_folds": fold_val,
            "num_train_images": len(train_imgs),
            "num_val_images": len(val_imgs),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "mAP50": metrics.get("map50"),
            "mAP50_95": metrics.get("map50_95"),
        })

        # 🧹 Cleanup GPU entre folds
        try:
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        # Écriture du récapitulatif
        if summary_rows:
            with open(summary_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "run_name", "val_folds", "num_train_images", "num_val_images",
                        "precision", "recall", "f1", "mAP50", "mAP50_95"
                    ]
                )
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"\n[OK] Récap K-Fold écrit : {summary_csv}")
        else:
            print("\n[WARN] Aucun run K-Fold n'a produit de métriques/ligne de résumé.")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--kfold",
        action="store_true",
        help="Active le K-Fold par préfixe avant '_' dans le nom de fichier (train uniquement)."
    )
    # ✅ Nouveaux arguments compatibles avec ton launch_train.sh
    parser.add_argument("--seed", type=int, default=None, help="Seed aléatoire")
    parser.add_argument("--project", type=str, default=None, help="Dossier parent des résultats")
    parser.add_argument("--name", type=str, default=None, help="Nom du sous-dossier pour ce run")
    parser.add_argument("--device", type=str, default=None, help="GPU(s) à utiliser, ex: 0 ou 0,1")

    # Args libres à passer à Ultralytics (epochs, val, imgsz, model, pretrained, etc.)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # Charger config YAML
    cfg = load_config(args.config)

    # Appliquer overrides CLI dans la section training
    cfg.setdefault("training", {})
    t = cfg["training"]

    if args.seed is not None:
        t["seed"] = args.seed
    if args.project is not None:
        t["project"] = args.project
    if args.name is not None:
        t["name"] = args.name
    if args.device is not None:
        t["device"] = args.device

    # Intégrer les tokens libres (ex: --epochs 200 val=False model=yolov10s.pt pretrained=True)
    if args.extra:
        t = parse_extra_into_training_kwargs(args.extra, t)
        cfg["training"] = t  # réaffecte

    # Dossier d'expérience + sauvegarde de la conf (après overrides)
    exp_dir = build_exp_dir(Path(cfg["training"].get("project", "results")), cfg["training"].get("name", "exp"))
    save_config(cfg, exp_dir, filename="config.yaml")

    # Seed (globale) si fournie
    if cfg["training"].get("seed", None) is not None:
        set_seed_all(int(cfg["training"]["seed"]), deterministic=True)

    if args.kfold:
        train_kfold(cfg)
    else:
        train_single(cfg)


if __name__ == "__main__":
    main()
