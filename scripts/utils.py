# scripts/utils.py
from __future__ import annotations
from pathlib import Path
import random, numpy as np, yaml
import torch
import os
import json
import shutil
from typing import Dict, List, Tuple, Optional, Literal, Union, Any
from dataclasses import dataclass, asdict
import time
from tempfile import NamedTemporaryFile
import tqdm

import sys as _sys

def _progress(iterable, desc: str = "", total: int | None = None, enable: bool = True):
    """
    Retourne un itérable éventuellement décoré par tqdm.
    - desc: texte à gauche (ex: "[train] YOLO filter")
    - total: len(iterable) si connu (pour une jauge fiable)
    - enable: permet de couper la barre si besoin
    """
    if  enable:
        return tqdm.tqdm(
            iterable,
            total=total,
            desc=desc,
            unit="img",
            ascii=True,          # compatible terminaux simples / tmux
            mininterval=0.5,     # limite le spam
            leave=False,
            file=_sys.stdout     # s'affiche en "live", n'interfère pas avec stdout
        )
    return iterable

# ---------- Config loading ----------
def load_config(path):
    with open(path , "r") as f:
        return yaml.safe_load(f)
# ---------- Image iteration ----------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

# ---------- Path hygiene ----------
def _strip_leading_parent_hops(spec_str: str) -> str:
    s = spec_str.replace("\\", "/").lstrip()
    while s.startswith("../"): s = s[3:]
    while s.startswith("./"):  s = s[2:]
    return s or "."

def resolve_path_ignoring_parent(base_dir: Path, spec: str) -> Path:
    s = _strip_leading_parent_hops(str(spec))
    return (base_dir / s).resolve()

# ---------- Fold token ----------
def extract_fold_token(pathlike) -> str:
    p = Path(pathlike)
    stem = p.stem
    return stem.split("_", 1)[0] if "_" in stem else "nofold"

# ---------- Generic IO helpers ----------
def write_list_file(path: Path, items: list[Path]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in items:
            f.write(str(Path(p).resolve()) + "\n")  # absolute paths

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def build_exp_dir(project: Path | str, name: str) -> Path:
    exp = Path(project) / name
    ensure_dir(exp)
    return exp

def save_config(cfg: dict, exp_dir: Path, filename: str = "config.yaml"):
    with open(exp_dir / filename, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

# ---------- Repro helper (optionnel) ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _scan_images(p: Path):
    # réutilise l’itérateur déjà défini au-dessus
    yield from iter_images(p)

def make_prefix_split_yaml(
    data_root: Path,
    original_yaml: Path,
    prefixes: str | list[str] = "fold5_",
    out_name_suffix: str | None = None,
) -> Path:
    """
    Construit un data.yaml temporaire pour Ultralytics où:
      - train = toutes les images de train/images SANS les préfixes donnés
      - val   = toutes les images de train/images AVEC l'un des préfixes
    prefixes: str séparé par des virgules ou liste de str (ex: "fold5_,fold4_")
    Renvoie le chemin du YAML temporaire.
    """
    data_root = Path(data_root).resolve()
    original_yaml = Path(original_yaml).resolve()

    # Charger YAML d'origine (pour names/nc)
    with open(original_yaml, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    img_dir = (data_root / "train" / "images").resolve()
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {img_dir}")

    # Normaliser prefixes
    if isinstance(prefixes, str):
        pref_list = [p.strip() for p in prefixes.split(",") if p.strip()]
    else:
        pref_list = [str(p).strip() for p in prefixes if str(p).strip()]
    pref_set = tuple(pref_list)

    # Scanner images
    all_imgs = list(_scan_images(img_dir))
    if not all_imgs:
        raise RuntimeError(f"Aucune image trouvée dans {img_dir}")

    # Split
    val_imgs   = [p for p in all_imgs if any(p.name.startswith(pfx) for pfx in pref_set)]
    train_imgs = [p for p in all_imgs if not any(p.name.startswith(pfx) for pfx in pref_set)]

    if not val_imgs:
        print(f"[WARN] Aucun fichier ne commence par {pref_set} dans {img_dir} → validation vide.", flush=True)
    if not train_imgs:
        raise RuntimeError(f"Train vide après filtrage des prefixes {pref_set}.")

    # Fichiers .txt (absolus)
    suf = out_name_suffix or ("_".join(p.strip("_") for p in pref_list) or "valprefix")
    val_txt   = (data_root / f"val_{suf}.txt").resolve()
    train_txt = (data_root / f"train_wo_{suf}.txt").resolve()
    val_txt.write_text("\n".join(str(p.resolve()) for p in val_imgs) + "\n", encoding="utf-8")
    train_txt.write_text("\n".join(str(p.resolve()) for p in train_imgs) + "\n", encoding="utf-8")

    # YAML temporaire
    tmp = NamedTemporaryFile("w", suffix=".yaml", delete=False)
    spec = {
        "train": str(train_txt),
        "val":   str(val_txt),
    }
    if "names" in base:
        spec["names"] = base["names"]
    if "nc" in base:
        spec["nc"] = base["nc"]

    yaml.safe_dump(spec, tmp, sort_keys=False, allow_unicode=True)
    tmp.flush(); tmp.close()
    return Path(tmp.name)   
    

from pathlib import Path

def _is_empty_keep(v) -> bool:
    """Return True if keep_class is effectively empty (None, '', 'none', 'null')."""
    return v is None or str(v).strip().lower() in ("", "none", "null")

def resolve_dataset_root(cfg: dict, *, verbose: bool = True) -> Path:
    """
    Decide which dataset directory to use at training time, based *only* on
    dataset.location and dataset.filter_mode (we deliberately ignore any dst_root).

    Rules:
      - filter_mode == 'none' (or missing)       -> <location>
      - filter_mode == 'merge_all'               -> <location>__merged-all
      - filter_mode == 'keep_one' AND keep_class -> <location>__only-<keep_class>
      - filter_mode == 'keep_one' but keep_class empty -> fallback to <location>

    Notes:
      - We don't create anything here; we just resolve the *expected* path.
      - If data.yaml is missing at the selected path, we print a warning (so you
        immediately see when download/filter hasn't produced the expected folder).

    Parameters
    ----------
    cfg : dict
        The full experiment config (must contain cfg['dataset']['location']).
    verbose : bool
        If True, prints the decision and a warning when data.yaml is missing.

    Returns
    -------
    Path
        The dataset root directory to pass to training code.
    """
    ds = cfg.get("dataset", {})
    base = Path(ds["location"]).resolve()
    mode = str(ds.get("filter_mode", "none")).lower()  # 'none' | 'keep_one' | 'merge_all'
    keep = ds.get("keep_class", None)

    if mode == "none":
        chosen = base
    elif mode == "merge_all":
        chosen = base.parent / f"{base.name}__merged-all"
    elif mode == "keep_one":
        if _is_empty_keep(keep):
            # keep_one requested but no keep_class -> sensible fallback
            if verbose:
                print("[DATASET] filter_mode=keep_one but keep_class is empty → fallback to base dataset.")
            chosen = base
        else:
            chosen = base.parent / f"{base.name}__only-{keep}"
    else:
        # Unknown value → safety fallback
        if verbose:
            print(f"[DATASET] Unknown filter_mode='{mode}' → fallback to base dataset.")
        chosen = base

    chosen = chosen.resolve()
    if verbose:
        print(f"[DATASET] filter_mode={mode} → selected root: {chosen}")
        # Soft sanity check: tell the user early if the expected folder isn't ready
        dy = chosen / "data.yaml"
        if not dy.exists():
            print(f"[WARN] data.yaml not found at: {dy} "
                  f"(download/filter may not have produced this folder yet)")

    return chosen
    
# ---------- Fonction pour ne garder qu'un type d'annotations ----------

Fmt = Literal["coco", "yolov8"]

@dataclass
class SplitReport:
    images_in: int = 0
    images_kept: int = 0
    images_dropped: int = 0
    ann_in: int = 0
    ann_kept: int = 0
    ann_dropped: int = 0
    missing_label_files: int = 0           # YOLO
    missing_images: int = 0                # COCO
    label_files_processed: int = 0         # YOLO
    images_copied: int = 0
    labels_copied: int = 0                 # YOLO
    json_written: bool = False             # COCO

@dataclass
class GlobalReport:
    fmt: Fmt
    keep_class: Union[str, int]
    keep_class_name: str
    keep_class_id_src: int
    keep_class_id_dst: int
    dst_root: str
    remap_to_zero: bool
    keep_images_without_labels: bool
    dry_run: bool
    splits: Dict[str, SplitReport]

def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in s)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _write_yaml(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _default_dst_root(src_root: Path, keep_class_name: str) -> Path:
    return src_root.parent / f"{src_root.name}__only-{_slug(keep_class_name)}"

def _robust_rmtree(path: Path, retries: int = 5, backoff: float = 0.25):
    """
    Supprime un répertoire de façon robuste (NFS/.nfsXXXX).
    Retente plusieurs fois et nettoie les fichiers .nfs* résiduels.
    """
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path)
            return
        except OSError as e:
            # Nettoyage des fichiers .nfs* qui empêchent la suppression
            try:
                for p in Path(path).rglob(".nfs*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
            except Exception:
                pass
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)
# ---------------------------------------------
# Détection format + introspection des classes
# ---------------------------------------------

def _yolo_list_classes(src_root: Path) -> List[str]:
    data_yaml = src_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml introuvable dans {src_root}")
    data = _read_yaml(data_yaml)
    names = data.get("names")
    if isinstance(names, dict):
        # e.g., {0: 'DM', 1: 'QR'}
        # On retransforme en liste ordonnée par clé
        names = [names[k] for k in sorted(names.keys(), key=int)]
    if not isinstance(names, list):
        raise ValueError("Format inattendu pour 'names' dans data.yaml.")
    return [str(x) for x in names]

def _coco_list_classes(split_json: Path) -> List[Tuple[int, str]]:
    data = _read_json(split_json)
    cats = data.get("categories", [])
    out = []
    for c in cats:
        cid = int(c["id"])
        name = str(c["name"])
        out.append((cid, name))
    # tri par id
    out.sort(key=lambda x: x[0])
    return out

def _detect_fmt_or_validate(fmt: Fmt, src_root: Path) -> Fmt:
    if fmt == "yolov8":
        if not (src_root / "data.yaml").exists():
            raise FileNotFoundError("Format YOLOv8 choisi mais data.yaml absent.")
        return "yolov8"
    elif fmt == "coco":
        # On vérifie la présence d’au moins un _annotations.coco.json
        found = False
        for sp in ("train", "valid", "test"):
            if (src_root / sp / "_annotations.coco.json").exists():
                found = True
                break
        if not found:
            raise FileNotFoundError("Format COCO choisi mais aucun _annotations.coco.json trouvé.")
        return "coco"
    else:
        raise ValueError("fmt doit être 'yolov8' ou 'coco'.")

# ---------------------------------------------
# Mapping classe à conserver (nom -> id, ou id -> nom)
# ---------------------------------------------

def _resolve_keep_class_yolo(src_root: Path, keep_class: Union[str, int]) -> Tuple[int, str]:
    names = _yolo_list_classes(src_root)
    if isinstance(keep_class, int):
        if keep_class < 0 or keep_class >= len(names):
            raise ValueError(f"id de classe hors limites pour YOLO: {keep_class}")
        return keep_class, names[keep_class]
    else:
        if keep_class not in names:
            raise ValueError(f"Classe '{keep_class}' introuvable dans names={names}")
        return names.index(keep_class), keep_class

def _resolve_keep_class_coco(src_root: Path, keep_class: Union[str, int]) -> Tuple[int, str]:
    # On prend le premier split dispo pour lire les categories
    split_json = None
    for sp in ("train", "valid", "test"):
        cand = src_root / sp / "_annotations.coco.json"
        if cand.exists():
            split_json = cand
            break
    if split_json is None:
        raise FileNotFoundError("Aucun JSON COCO trouvé pour lire les catégories.")

    cats = _coco_list_classes(split_json)
    if isinstance(keep_class, int):
        for cid, name in cats:
            if cid == keep_class:
                return cid, name
        raise ValueError(f"id de classe {keep_class} introuvable dans COCO: {cats}")
    else:
        for cid, name in cats:
            if name == keep_class:
                return cid, name
        raise ValueError(f"Classe '{keep_class}' introuvable dans COCO: {cats}")

# ---------------------------------------------
# Traitement YOLOv8
# ---------------------------------------------

def _process_yolo_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    # --- changements ---
    mode: Literal["keep_one","merge_all"] = "keep_one",
    keep_id_src: Optional[int] = None,   # devient optionnel en merge_all
    keep_name: str = "object",
    # -------------------
    remap_to_zero: bool = True,
    keep_images_without_labels: bool = False,
    dry_run: bool = False
) -> SplitReport:
    rep = SplitReport()
    img_dir_src = src_root / split / "images"
    lbl_dir_src = src_root / split / "labels"
    if not img_dir_src.exists():
        return rep  # split absent → vide
    img_paths = sorted([p for p in img_dir_src.glob("*") if p.is_file()])
    rep.images_in = len(img_paths)

    img_dir_dst = dst_root / split / "images"
    lbl_dir_dst = dst_root / split / "labels"
    if not dry_run:
        _ensure_dir(img_dir_dst)
        _ensure_dir(lbl_dir_dst)

    print(f"[FILTER][{split}] YOLO: {len(img_paths)} images à traiter…", flush=True)
    for img_path in _progress(
        img_paths,
        desc=f"[{split}] YOLO filter",
        total=len(img_paths),
        enable=not dry_run
    ):
        stem = img_path.stem
        lbl_path = lbl_dir_src / f"{stem}.txt"

        if not lbl_path.exists():
            rep.missing_label_files += 1
            lines_kept: List[str] = []
            keep = bool(keep_images_without_labels)
        else:
            rep.label_files_processed += 1
            with lbl_path.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            # Compte initial d’annotations
            rep.ann_in += len(lines)

            if mode == "keep_one":
                if keep_id_src is None:
                    raise ValueError("keep_id_src doit être défini en mode 'keep_one'.")
                lines_kept = []
                for ln in lines:
                    parts = ln.split()
                    try:
                        cls_id = int(float(parts[0]))
                    except Exception:
                        continue
                    if cls_id == keep_id_src:
                        if remap_to_zero:
                            parts[0] = "0"
                        lines_kept.append(" ".join(parts))
                rep.ann_kept += len(lines_kept)
                rep.ann_dropped += (len(lines) - len(lines_kept))
            else:
                # merge_all → on garde toutes les lignes, cls := 0
                lines_kept = []
                for ln in lines:
                    parts = ln.split()
                    if not parts:
                        continue
                    parts[0] = "0"
                    lines_kept.append(" ".join(parts))
                rep.ann_kept += len(lines_kept)
                # rien "drop", on remappe tout
                # rep.ann_dropped += 0

            keep = (len(lines_kept) > 0) or keep_images_without_labels

        if keep:
            rep.images_kept += 1
            if not dry_run:
                shutil.copy2(img_path, img_dir_dst / img_path.name)
                rep.images_copied += 1
                with (lbl_dir_dst / f"{stem}.txt").open("w", encoding="utf-8") as f:
                    f.write("\n".join(lines_kept))
                rep.labels_copied += 1
        else:
            rep.images_dropped += 1

    if not dry_run:
        # data.yaml cible (monoclasse)
        src_yaml = _read_yaml(src_root / "data.yaml")

        if mode == "keep_one":
            # récupérer le nom de la classe conservée depuis le YAML source
            if isinstance(src_yaml.get("names"), list):
                cls_name = src_yaml["names"][keep_id_src]  # type: ignore[index]
            elif isinstance(src_yaml.get("names"), dict):
                nmap = src_yaml["names"]
                cls_name = nmap.get(str(keep_id_src), nmap.get(keep_id_src))  # type: ignore[index]
            else:
                raise ValueError("Format inattendu pour names dans data.yaml (YOLO).")
        else:
            cls_name = keep_name

        dst_yaml = dict(src_yaml)
        dst_yaml["nc"] = 1
        dst_yaml["names"] = [cls_name]
        if not (dst_root / "data.yaml").exists():
            _write_yaml(dst_root / "data.yaml", dst_yaml)

    return rep

# ---------------------------------------------
# Traitement COCO
# ---------------------------------------------

def _process_coco_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    # --- changements ---
    mode: Literal["keep_one","merge_all"] = "keep_one",
    keep_id_src: Optional[int] = None,  # optionnel en merge_all
    keep_name: str = "object",
    # -------------------
    remap_to_zero: bool = True,
    keep_images_without_labels: bool = False,
    dry_run: bool = False
) -> SplitReport:
    rep = SplitReport()
    split_dir = src_root / split
    if not split_dir.exists():
        return rep
    anno_src = split_dir / "_annotations.coco.json"
    if not anno_src.exists():
        return rep

    data = _read_json(anno_src)
    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])

    rep.images_in = len(images)
    rep.ann_in = len(anns)

    if mode == "keep_one":
        if keep_id_src is None:
            raise ValueError("keep_id_src doit être défini en mode 'keep_one'.")
        keep_anns = [a for a in anns if int(a.get("category_id", -1)) == int(keep_id_src)]
        dst_cat_id = 0 if remap_to_zero else int(keep_id_src)
        for a in keep_anns:
            a["category_id"] = dst_cat_id
    else:
        # merge_all → on garde toutes les annotations et on remap tout à 0
        keep_anns = list(anns)
        dst_cat_id = 0
        for a in keep_anns:
            a["category_id"] = 0

    # images gardées
    image_ids_kept = {a["image_id"] for a in keep_anns}
    if keep_images_without_labels:
        keep_images_list = images
    else:
        keep_images_list = [im for im in images if im["id"] in image_ids_kept]

    rep.ann_kept = len(keep_anns)
    rep.ann_dropped = len(anns) - len(keep_anns)
    rep.images_kept = len(keep_images_list)
    rep.images_dropped = len(images) - len(keep_images_list)

    # COCO: une seule catégorie
    dst_categories = [{
        "id": dst_cat_id if mode == "keep_one" else 0,
        "name": keep_name,
        "supercategory": (cats[0].get("supercategory","none") if cats else "none")
    }]

    # Copier images + écrire JSON
    img_dir_dst = dst_root / split
    if not dry_run:
        _ensure_dir(img_dir_dst)

    # index pour retrouver les fichiers
    id2img = {im["id"]: im for im in images}

    print(f"[FILTER][{split}] COCO: {len(keep_images_list)} images à traiter…", flush=True)
    for im in _progress(
        keep_images_list,
        desc=f"[{split}] COCO filter",
        total=len(keep_images_list),
        enable=not dry_run
    ):
        file_name = im["file_name"]
        src_img = split_dir / file_name
        if not src_img.exists():
            rep.missing_images += 1
            continue
        if not dry_run:
            dst_img = img_dir_dst / file_name
            _ensure_dir(dst_img.parent)
            shutil.copy2(src_img, dst_img)
            rep.images_copied += 1

    if not dry_run:
        dst_data = {
            "images": keep_images_list,
            "annotations": keep_anns,
            "categories": dst_categories,
        }
        _write_json(img_dir_dst / "_annotations.coco.json", dst_data)
        rep.json_written = True

    return rep

# ---------------------------------------------
# Impression classes « avant / après » en dry-run
# ---------------------------------------------

def _print_classes_before_after(
    fmt: Fmt,
    src_root: Path,
    keep_name: str,
    remap_to_zero: bool,
    dry_run: bool,
    merge_all: bool = False,
):
    if not dry_run:
        return
    print("[DRY RUN] --- Inspection des classes ---")
    if fmt == "yolov8":
        names = _yolo_list_classes(src_root)
        print(f"[DRY RUN] Classes trouvées dans le dataset source (YOLO): {names}")
        if merge_all:
            print(f"[DRY RUN] Classes dans le dataset cible: ['{keep_name}'] (toutes → id=0)")
        else:
            tag = "(remapped -> id=0)" if remap_to_zero else ""
            print(f"[DRY RUN] Classes dans le dataset cible: ['{keep_name}'] {tag}".strip())
    else:
        cats = None
        for sp in ("train", "valid", "test"):
            cand = src_root / sp / "_annotations.coco.json"
            if cand.exists():
                cats = _coco_list_classes(cand)
                break
        if cats is None:
            print("[DRY RUN] Aucune catégorie COCO trouvée.")
        else:
            readable = [f"id={cid}: {name}" for cid, name in cats]
            print(f"[DRY RUN] Classes trouvées dans le dataset source (COCO): {readable}")
            if merge_all:
                print(f"[DRY RUN] Classes dans le dataset cible: ['{keep_name}'] (toutes → id=0)")
            else:
                tag = "(remapped -> id=0)" if remap_to_zero else ""
                print(f"[DRY RUN] Classes dans le dataset cible: ['{keep_name}'] {tag}".strip())

# ---------------------------------------------
# API principale
# ---------------------------------------------

def filter_dataset(
    src_root: str,
    fmt: Fmt,
    keep_class: Union[str, int, None] = None,
    dst_root: Optional[str] = None,
    splits: Tuple[str, ...] = ("train", "valid", "test"),
    keep_images_without_labels: bool = False,
    remap_to_zero: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    on_conflict: Literal["fail","overwrite","skip"] = "fail",
    # 🔥 Nouveaux paramètres
    mode: Literal["keep_one","merge_all"] = "keep_one",
    merge_all_name: str = "code",
) -> Dict[str, Any]:
    """
    Filtre/transforme un dataset Roboflow (YOLOv8 ou COCO) vers un dataset *monoclasse*.

    Modes :
      - mode="keep_one": conserve uniquement `keep_class` (comportement historique).
      - mode="merge_all": conserve TOUTES les instances mais remappe toutes les classes en une seule (id=0).

    Remarques :
      - remap_to_zero continue d’avoir du sens en mode keep_one (0 ou id d’origine).
      - en mode merge_all, toutes les annotations deviennent id=0 (remap_to_zero est de facto True).
      - on_conflict='skip' renvoie immédiatement avec un rapport minimal et sans réécrire la cible.
    """
    src_root_p = Path(src_root).resolve()
    fmt = _detect_fmt_or_validate(fmt, src_root_p)

    # ---------- Résolution des infos "classe" selon le mode ----------
    if mode == "keep_one":
        if keep_class is None:
            raise ValueError("mode='keep_one' exige keep_class (str|int).")
        if fmt == "yolov8":
            keep_id_src, keep_name = _resolve_keep_class_yolo(src_root_p, keep_class)
        else:
            keep_id_src, keep_name = _resolve_keep_class_coco(src_root_p, keep_class)
        # suffixe destination si non fourni
        auto_dst = _default_dst_root(src_root_p, keep_name)
    elif mode == "merge_all":
        # On ne résout pas une classe ; on force un nom unique et nc=1 plus tard
        keep_id_src, keep_name = None, merge_all_name
        auto_dst = src_root_p.parent / f"{src_root_p.name}__merged-all"
    else:
        raise ValueError(f"mode inconnu: {mode}")

    # ---------- Dossier de sortie ----------
    dst_root_p = Path(dst_root).resolve() if dst_root is not None else Path(auto_dst).resolve()

    # ---------- Gestion de conflit ----------
    if dst_root_p.exists() and not dry_run:
        if on_conflict == "fail":
            raise FileExistsError(f"Le dossier cible existe déjà: {dst_root_p}")
        elif on_conflict == "overwrite":
            _robust_rmtree(dst_root_p)
        elif on_conflict == "skip":
            if verbose:
                print(
                    f"[FILTER] Cible déjà présente et on_conflict='skip' → pas de ré-écriture.\n"
                    f"         dst_root = {dst_root_p}",
                    flush=True
                )
            # Rapport minimal (pas de recalcul des stats)
            return {
                "fmt": fmt,
                "mode": mode,
                "keep_class": keep_class,
                "keep_class_name": keep_name,
                "keep_class_id_src": keep_id_src,
                "keep_class_id_dst": (0 if (mode == "merge_all" or remap_to_zero) else keep_id_src),
                "dst_root": str(dst_root_p),
                "remap_to_zero": (True if mode == "merge_all" else remap_to_zero),
                "keep_images_without_labels": keep_images_without_labels,
                "dry_run": False,
                "splits": {},
            }

    # ---------- Affichage classes avant/après (dry-run) ----------
    # En merge_all : "après" = une seule entrée (id=0, name=merge_all_name)
    _print_classes_before_after(
        fmt=fmt,
        src_root=src_root_p,
        keep_name=keep_name if mode == "keep_one" else merge_all_name,
        remap_to_zero=True if mode == "merge_all" else remap_to_zero,
        dry_run=dry_run,
        merge_all=(mode == "merge_all"),
    )

    # ---------- Rapport global ----------
    global_rep = GlobalReport(
        fmt=fmt,
        keep_class=keep_class,
        keep_class_name=keep_name,
        keep_class_id_src=(None if mode == "merge_all" else keep_id_src),
        keep_class_id_dst=0 if (mode == "merge_all" or remap_to_zero) else keep_id_src,
        dst_root=str(dst_root_p),
        remap_to_zero=(True if mode == "merge_all" else remap_to_zero),
        keep_images_without_labels=keep_images_without_labels,
        dry_run=dry_run,
        splits={}
    )

    # ---------- Traitement par split ----------
    for sp in splits:
        if fmt == "yolov8":
            rep = _process_yolo_split(
                src_root=src_root_p,
                dst_root=dst_root_p,
                split=sp,
                # 🔑 Ajouts à faire dans le helper :
                mode=mode,                                   # "keep_one" | "merge_all"
                keep_id_src=None if mode == "merge_all" else keep_id_src,
                keep_name=keep_name,                         # utile pour data.yaml
                remap_to_zero=True if mode == "merge_all" else remap_to_zero,
                keep_images_without_labels=keep_images_without_labels,
                dry_run=dry_run,
            )
        else:
            rep = _process_coco_split(
                src_root=src_root_p,
                dst_root=dst_root_p,
                split=sp,
                # 🔑 Ajouts à faire dans le helper :
                mode=mode,
                keep_id_src=None if mode == "merge_all" else keep_id_src,
                keep_name=keep_name,                         # pour categories[0].name
                remap_to_zero=True if mode == "merge_all" else remap_to_zero,
                keep_images_without_labels=keep_images_without_labels,
                dry_run=dry_run,
            )
        global_rep.splits[sp] = rep

    # ---------- Résumé console ----------
    if verbose:
        print("\n=== SUMMARY ===")
        print(f"Format: {fmt}")
        print(f"Source: {src_root_p}")
        print(f"Cible:  {dst_root_p} {'(dry-run, rien écrit)' if dry_run else ''}")
        if mode == "keep_one":
            print(f"Mode: keep_one | Classe conservée: '{keep_name}' "
                  f"(src id={keep_id_src} -> dst id={0 if remap_to_zero else keep_id_src})")
        else:
            print(f"Mode: merge_all | Toutes les classes → '0:{merge_all_name}'")
        for sp, rep in global_rep.splits.items():
            print(f"\n[{sp}]")
            print(f"  images_in      : {rep.images_in}")
            print(f"  images_kept    : {rep.images_kept}")
            print(f"  images_dropped : {rep.images_dropped}")
            print(f"  ann_in         : {rep.ann_in}")
            print(f"  ann_kept       : {rep.ann_kept}")
            print(f"  ann_dropped    : {rep.ann_dropped}")
            if fmt == "yolov8":
                print(f"  label_files    : {rep.label_files_processed} processed, {rep.missing_label_files} missing")
                print(f"  copied         : {rep.images_copied} images, {rep.labels_copied} labels")
            else:
                print(f"  json_written   : {rep.json_written}")
                print(f"  copied         : {rep.images_copied} images")
                if getattr(rep, "missing_images", None):
                    print(f"  missing_images : {rep.missing_images}")

    # ---------- Renvoi dict (sérialisable) ----------
    out = asdict(global_rep)
    out["splits"] = {k: asdict(v) for k, v in global_rep.splits.items()}
    return out




### Recap des differents datasets avant entrainements:
def _expand_one_spec(base_dir: Path, spec) -> list[Path]:
    """
    Développe un spec Ultralytics (dossier, .txt, image ou liste de specs) en liste d'images.
    - base_dir: dossier du data.yaml (pour résoudre les chemins relatifs)
    """
    out = []
    if isinstance(spec, (list, tuple)):
        for s in spec:
            out.extend(_expand_one_spec(base_dir, s))
        return out

    p = resolve_path_ignoring_parent(base_dir, str(spec))
    if p.is_dir():
        out.extend(iter_images(p))  # récursif
    elif p.is_file():
        if p.suffix.lower() in IMG_EXTS:
            out.append(p)
        elif p.suffix.lower() == ".txt":
            for line in p.read_text().splitlines():
                ln = line.strip()
                if not ln:
                    continue
                q = resolve_path_ignoring_parent(p.parent, ln)
                if q.exists() and q.suffix.lower() in IMG_EXTS:
                    out.append(q)
    else:
        # dernier recours: chemins un peu "bizarres" → essaye best-effort
        s = str(spec).replace("\\", "/").lstrip()
        while s.startswith("../"):
            s = s[3:]
        while s.startswith("./"):
            s = s[2:]
        cand = (base_dir / (s or "."))
        if cand.exists():
            if cand.is_dir():
                out.extend(iter_images(cand))
            elif cand.suffix.lower() in IMG_EXTS:
                out.append(cand)

    # normalisation / dédoublonnage
    uniq = sorted({pp.resolve() for pp in out if pp.exists()})
    return uniq


def _yolo_label_path_from_image(img: Path) -> Path:
    """
    Déduit le chemin du label YOLO à partir de l'image.
    Remplace '/images/' par '/labels/' et extension en '.txt'.
    """
    img = img.resolve()
    parts = list(img.parts)
    # remplace le segment 'images' le plus à droite
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    label_dir = Path(*parts[:-1])
    return (label_dir / (img.stem + ".txt")).resolve()


def summarize_data_yaml(data_yaml_path: Path) -> dict:
    """
    Lit un data.yaml et retourne un dict résumé:
      { split: { images: N, labels_ok: M, labels_missing: K } }
    Imprime aussi un petit tableau lisible.
    """
    data_yaml_path = data_yaml_path.resolve()
    base_dir = data_yaml_path.parent
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    def _summ_for_key(key: str):
        if key not in data:
            return None
        imgs = _expand_one_spec(base_dir, data[key])
        labels_ok = 0
        for im in imgs:
            if _yolo_label_path_from_image(im).exists():
                labels_ok += 1
        return {
            "images": len(imgs),
            "labels_ok": labels_ok,
            "labels_missing": len(imgs) - labels_ok,
        }

    report = {}
    for k in ("train", "val", "test"):
        rep = _summ_for_key(k)
        if rep is not None:
            report[k] = rep

    # log lisible
    print("\n[DATA SUMMARY]")
    print(f"  data.yaml: {data_yaml_path}")
    for k in ("train", "val", "test"):
        if k in report:
            r = report[k]
            print(f"  - {k:<5}: images={r['images']:>6} | labels ok={r['labels_ok']:>6} | missing={r['labels_missing']:>6}")
        else:
            print(f"  - {k:<5}: (absent)")
    print()
    return report