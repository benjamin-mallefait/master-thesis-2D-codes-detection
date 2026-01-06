#!/bin/bash

# --- Vérification des arguments ---
SSH_CONFIG="$1"
TRAIN_CONFIG="$2"

if [[ ! -f "$SSH_CONFIG" || ! -f "$TRAIN_CONFIG" ]]; then
  echo "❌ Usage : bash scripts/pull_test_dataset.sh <ssh_config.yaml> <training_config.yaml>"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


extract_yaml() {
  python3 "$SCRIPT_DIR/extract_yaml_for_bash.py" "$1" "$2"
}


REMOTE_USER=$(extract_yaml "user" "$SSH_CONFIG")
REMOTE_HOST=$(extract_yaml "host" "$SSH_CONFIG")
REMOTE_PORT=$(extract_yaml "port" "$SSH_CONFIG")
REMOTE_DIR=$(extract_yaml "remote_dir" "$SSH_CONFIG")

DATASET_PATH=$(extract_yaml "dataset.location" "$TRAIN_CONFIG" )

if [[ -z "$DATASET_PATH" ]]; then
  echo "Chemin du dataset introuvable dans la config d'entraînement."
  exit 1
fi

REMOTE_DATASET_DIR="$REMOTE_DIR/$DATASET_PATH/test"
LOCAL_DATASET_DIR="$DATASET_PATH"

echo "Récupération du dataset '$REMOTE_DATASET_DIR'..."
rsync -avz -e "ssh -p $REMOTE_PORT" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DATASET_DIR" \
  "$LOCAL_DATASET_DIR"

echo "Dataset téléchargé dans $LOCAL_DATASET_DIR/test"
