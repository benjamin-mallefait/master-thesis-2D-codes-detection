#!/usr/bin/env bash
set -euo pipefail

SSH_CONFIG="${1:-}"
EXPERIMENT_CONFIG="${2:-}"

if [[ -z "${SSH_CONFIG}" ]]; then
  echo "Usage:"
  echo "  bash scripts/pull_all_results.sh <ssh_config.yaml> [<experiment.yaml>]"
  echo "  bash scripts/pull_all_results.sh <ssh_config.yaml> --name <training.name>"
  exit 1
fi

if [[ ! -f "$SSH_CONFIG" ]]; then
  echo "Fichier SSH config introuvable : $SSH_CONFIG"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

extract_yaml() {
  # Util : extract_yaml <key.path> <yaml_file>
  python3 "$SCRIPT_DIR/extract_yaml_for_bash.py" "$1" "$2"
}

REMOTE_USER=$(extract_yaml "user" "$SSH_CONFIG")
REMOTE_HOST=$(extract_yaml "host" "$SSH_CONFIG")
REMOTE_PORT=$(extract_yaml "port" "$SSH_CONFIG")
REMOTE_DIR=$(extract_yaml "remote_dir" "$SSH_CONFIG")

if [[ -z "$REMOTE_USER" || -z "$REMOTE_HOST" || -z "$REMOTE_PORT" || -z "$REMOTE_DIR" ]]; then
  echo "Informations incomplètes dans le fichier YAML"
  exit 1
fi

# --- Parse option courte --name si on ne passe pas d'experiment.yaml
OVERRIDE_NAME=""
if [[ "${EXPERIMENT_CONFIG:-}" == "--name" ]]; then
  OVERRIDE_NAME="${3:-}"
  if [[ -z "$OVERRIDE_NAME" ]]; then
    echo "[ERR] --name requiert une valeur"
    exit 1
  fi
  EXPERIMENT_CONFIG=""
fi

RSYNC_ARGS=(-avz -e "ssh -p $REMOTE_PORT")

if [[ -n "${EXPERIMENT_CONFIG}" && -f "${EXPERIMENT_CONFIG}" ]]; then
  # Lecture du projet et du nom dans l'EXPERIMENT YAML
  PROJ=$(extract_yaml "training.project" "$EXPERIMENT_CONFIG")
  NAME=$(extract_yaml "training.name" "$EXPERIMENT_CONFIG")
elif [[ -n "$OVERRIDE_NAME" ]]; then
  # Projet par défaut + nom forcé
  PROJ="results/"
  NAME="$OVERRIDE_NAME"
else
  PROJ=""
  NAME=""
fi

# Si on a PROJ/NAME -> copie sélective, sinon copie complète de results/
if [[ -n "$NAME" ]]; then
  PROJ="${PROJ:-results/}"
  # Normalise "results/" -> "results"
  PROJ_TRIMMED="${PROJ%/}"
  REMOTE_PATH="$REMOTE_DIR/$PROJ_TRIMMED/$NAME/"
  LOCAL_PATH="./$PROJ_TRIMMED/$NAME/"

  echo "Téléchargement sélectif: $REMOTE_HOST:$REMOTE_PATH -> $LOCAL_PATH"
  mkdir -p "$LOCAL_PATH"
  rsync "${RSYNC_ARGS[@]}" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH" \
    "$LOCAL_PATH"
else
  echo "Téléchargement du dossier complet 'results/' depuis $REMOTE_HOST..."
  rsync "${RSYNC_ARGS[@]}" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/results/" \
    ./results/
fi

echo "Copie terminée."
