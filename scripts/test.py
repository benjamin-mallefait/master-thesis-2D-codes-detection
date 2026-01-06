#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test multipoids par epochs:
- Permet d'indiquer un dossier de poids (--weights-dir) contenant des fichiers epochXXX.pt
- Permet de sélectionner les epochs à tester (--epochs "160,170,180")
- Pour chaque epoch: lance les évaluations (global + par-dataset si demandé, inchangé)
- Stocke les résultats de chaque epoch dans results/<name>/epochXXX/
- Agrège ensuite toutes les summaries *_summary.csv des epochs sélectionnées et écrit
  results/<name>/epochs_mean_summary.csv avec moyenne et écart-type par "source".

NOTE: Le script préserve les options historiques (--use-dataset-test, --use-external-test,
--skip-global, --group-regex, etc.) et le comportement de test unitaire si --weights-dir
n'est pas fourni.

On suppose l’existence des fonctions utilitaires d’évaluation dans test_utils.py :
- build_sources_from_config(...)
- evaluate_all_sources(...): retourne une liste de tuples (source_label, path_to_summary_csv)
- make_outdir(...)
"""

from __future__ import annotations
import argparse
import os
import sys
import re
from pathlib import Path
import pandas as pd

# === imports locaux (inchangés) ===
import test_utils as T  # doit contenir la logique existante d’eval et des helpers


def parse_epochs(s: str) -> list[int]:
    """
    Parse "160,170,180" -> [160,170,180]
    Accepte aussi espaces. Valide que toutes sont >=0.
    """
    parts = re.split(r"[,\s]+", s.strip())
    eps = []
    for p in parts:
        if not p:
            continue
        try:
            v = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Epoch invalide: {p}")
        if v < 0:
            raise argparse.ArgumentTypeError(f"Epoch négative: {v}")
        eps.append(v)
    if not eps:
        raise argparse.ArgumentTypeError("Aucune epoch fournie.")
    print
    return eps


def add_args(parser: argparse.ArgumentParser) -> None:
    # === args existants (garde ton schéma) ===
    parser.add_argument("--config", required=True, help="Chemin du fichier d'expérience (YAML).")
    parser.add_argument("--project", default="results", help="Dossier racine des résultats.")
    parser.add_argument("--name", default=None, help="Nom de la campagne (dossier sous project).")
    parser.add_argument("--skip-global", action="store_true", help="Ne lance que l'évaluation par dataset.")
    parser.add_argument("--group-regex", default=None, help="Regex de regroupement des images par dataset.")

    # Sélection des sources de test (comme avant)
    parser.add_argument("--use-dataset-test", action="store_true",
                        help="Tester sur le test set défini par data.yaml de la config.")
    parser.add_argument("--use-external-test", action="store_true",
                        help="Tester sur un test set externe.")
    parser.add_argument("--external-data-yaml", default=None, help="Chemin du data.yaml externe.")
    parser.add_argument("--external-name", default="External", help="Label du test set externe.")

    # === NOUVEAUX ARGS ===
    parser.add_argument("--weights-dir", default=None,
                        help="Dossier contenant des poids epochXXX.pt (ex: runs/train/exp/weights_epochs).")
    parser.add_argument("--epochs", type=parse_epochs, default=None,
                        help="Liste des epochs à sélectionner. Exemple: \"160,170,180,190,200\"")

    # Optionnel: override direct d’un unique poids (comportement historique)
    parser.add_argument("--weights", default=None,
                        help="Chemin d'un unique .pt si on ne veut pas le mode multi-epochs.")

    # Nom de sous-dossier parent facultatif pour regrouper les epochs (par défaut: racine de --name)
    parser.add_argument("--epochs-folder", default=None,
                        help="Sous-dossier qui contiendra les epochXXX (défaut: dossier de --name).")


def build_output_layout(project: Path, name: str, epochs_folder: str | None) -> Path:
    root = project / name
    if epochs_folder:
        root = root / epochs_folder
    root.mkdir(parents=True, exist_ok=True)
    return root


def discover_sources(args) -> list[T.TestSource]:
    """
    Construit les sources de test (global + datasets) selon la config et les flags.
    Utilise test_utils.build_sources_from_config (à maintenir).
    """
    return T.build_sources_from_config(
        config_path=args.config,
        use_dataset_test=args.use_dataset_test,
        use_external_test=args.use_external_test,
        external_data_yaml=args.external_data_yaml,
        external_label=args.external_name,
        skip_global=args.skip_global,
        group_regex=args.group_regex
    )


def run_one_epoch_eval(args, out_root: Path, epoch: int, weights_path: Path, sources: list[T.TestSource]) -> list[tuple[str, Path]]:
    """
    Lance l'évaluation pour une epoch donnée et renvoie la liste:
      [(source_label, summary_csv_path), ...]
    Les fichiers vont dans out_root / f"epoch{epoch:03d}".
    """
    outdir = out_root / f"epoch{epoch:03d}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Nom logique de run (affichage/fichiers)
    run_name = f"{T.load_training_name(args.config)}_e{epoch:03d}"

    # evaluate_all_sources sait déjà:
    # - créer les sous-dossiers utiles
    # - retourner pour chaque source un CSV résumé "…_summary.csv"
    eval_info = T.evaluate_all_sources(
        config_path=args.config,
        weights_path=str(weights_path),
        out_dir=str(outdir),
        run_name=run_name,
        sources=sources
    )

    # eval_info: list[(label, summary_csv_path_str)]
    results = []
    for label, summary_csv in eval_info:
        results.append((label, Path(summary_csv)))
    return results


def collect_epoch_summaries(epoch_dirs: list[Path]) -> pd.DataFrame:
    """
    Parcourt chaque dossier epochXXX et récupère tous les CSV finissant par _summary.csv.
    Concatène en ajoutant colonnes: epoch, source (d'après le nom de fichier si absente).
    """
    records = []
    for edir in epoch_dirs:
        epoch_match = re.search(r"epoch(\d{3})$", edir.name)
        epoch_num = int(epoch_match.group(1)) if epoch_match else None
        for csv in edir.glob("**/*_summary.csv"):
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            # source: tente de lire une colonne existante, sinon déduis du nom du fichier
            if "source" in df.columns:
                src = df["source"].iloc[0]
                df["__source"] = src
            else:
                df["__source"] = csv.stem.replace("_summary", "")
            df["__epoch"] = epoch_num
            df["__file"] = str(csv)
            records.append(df)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def mean_and_std_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule mean et std par source pour toutes les colonnes numériques.
    Sortie: colonnes <metric>_mean, <metric>_std + 'source'
    """
    # Normalize source column name
    if "source" in df.columns:
        df["__source"] = df["source"]
    num_cols = [c for c in df.columns if c not in {"__source", "__epoch", "__file", "source"}]
    num_cols = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]

    grouped = df.groupby("__source")[num_cols]
    mean_df = grouped.mean().add_suffix("_mean").reset_index()
    std_df = grouped.std(ddof=1).add_suffix("_std").reset_index()
    out = pd.merge(mean_df, std_df, on="__source", how="left")
    out = out.rename(columns={"__source": "source"})
    return out


def main():
    parser = argparse.ArgumentParser("Evaluate models (single or multiple epochs)")
    add_args(parser)
    args = parser.parse_args()

    project = Path(args.project).resolve()
    training_name = T.load_training_name(args.config)
    name = args.name or f"{training_name}_test"
    out_root = build_output_layout(project, name, args.epochs_folder)

    # Découvre les sources de test (global + par-dataset)
    sources = discover_sources(args)
    if not sources:
        print("Aucune source de test détectée (vérifie tes flags).", file=sys.stderr)
        sys.exit(2)

    ran_epochs = []
    if args.weights_dir and args.epochs:
        weights_dir = Path(args.weights_dir).resolve()
        if not weights_dir.is_dir():
            print(f"--weights-dir introuvable: {weights_dir}", file=sys.stderr)
            sys.exit(1)

        for ep in args.epochs:
            w = weights_dir / f"epoch{ep:03d}.pt"
            if not w.is_file():
                print(f"[WARN] Poids manquant: {w} -> epoch ignorée.")
                continue
            print(f"==> Eval epoch {ep:03d} avec {w}")
            _ = run_one_epoch_eval(args, out_root, ep, w, sources)
            ran_epochs.append(ep)

        if not ran_epochs:
            print("Aucun run lancé: aucun poids correspondant aux epochs fournies.", file=sys.stderr)
            sys.exit(3)

        # Agrégation
        epoch_dirs = [out_root / f"epoch{ep:03d}" for ep in ran_epochs if (out_root / f"epoch{ep:03d}").is_dir()]
        df_all = collect_epoch_summaries(epoch_dirs)
        if df_all.empty:
            print("[WARN] Impossible d’agréger: aucun *_summary.csv trouvé.")
        else:
            agg = mean_and_std_by_source(df_all)
            out_csv = out_root / "epochs_mean_summary.csv"
            agg.to_csv(out_csv, index=False)
            print(f"[OK] Moyennes enregistrées: {out_csv}")

    else:
        # Comportement historique: un seul modèle via --weights (ou celui par défaut du config)
        if args.weights_dir and not args.epochs:
            print("Tu as fourni --weights-dir sans --epochs. Ajoute --epochs \"160,170,...\".", file=sys.stderr)
            sys.exit(4)

        weights_path = args.weights  # peut être None -> logique interne dans evaluate_all_sources
        print(f"==> Eval unique avec weights={weights_path or '<par défaut>'}")
        eval_info = T.evaluate_all_sources(
            config_path=args.config,
            weights_path=weights_path,
            out_dir=str(out_root),
            run_name=name,
            sources=sources
        )
        # Rien à agréger ici; les *_summary.csv sont déposés dans out_root


if __name__ == "__main__":
    main()
