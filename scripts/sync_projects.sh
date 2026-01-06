#!/bin/bash

# --- Vérification des arguments ---
USER_CONFIG="$1"
if [[ ! -f "$USER_CONFIG" ]]; then
  echo "Fichier user YAML manquant"
  exit 1
fi

# --- Extraction des infos depuis le YAML ---
extract_yaml() {
  python3 scripts/extract_yaml_for_bash.py "$1" "$2"
}

REMOTE_USER=$(extract_yaml "user" "$USER_CONFIG")
REMOTE_HOST=$(extract_yaml "host" "$USER_CONFIG")
REMOTE_PORT=$(extract_yaml "port" "$USER_CONFIG")
REMOTE_DIR=$(extract_yaml "remote_dir" "$USER_CONFIG")

# --- Rsync des fichiers essentiels ---
echo "Synchronisation vers $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR ..."
rsync -avz -e "ssh -p $REMOTE_PORT" \
  --include 'scripts/***' \
  --include 'configs/***' \
  --include 'Makefile' \
  --include 'requirements.txt' \
  --exclude '*' \
  ./ "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" > /dev/null