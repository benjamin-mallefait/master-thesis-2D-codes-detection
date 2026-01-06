#!/bin/bash
set -euo pipefail

SSH_CONFIG=""
EXPERIMENT_CONFIG=""
PROJECT_OVERRIDE=""
NAME_OVERRIDE=""
SKIP_GLOBAL="false"
GROUP_REGEX=""

# Flags sources
USE_DATASET_TEST="false"
USE_EXTERNAL_TEST="false"
EXTERNAL_DATA_YAML=""
EXTERNAL_NAME="code-detection-test"

# === NOUVEAUX PARAMS ===
WEIGHTS_DIR=""
EPOCHS_LIST=""
EPOCHS_FOLDER=""   # sous-dossier de regroupement optionnel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --ssh-config path/to/ssh.yaml --config path/to/experiment.yaml [options]

Options générales:
  --project <dir>               (legacy) Dossier racine des résultats (passé à test.py).
  --name <name>                 Nom de campagne (=> results/<name>/...).
  --skip-global                 Ne lance que l'évaluation par dataset.
  --group-regex <regex>         Regex de regroupement des images par dataset.

Sélection des sources:
  --use-dataset-test
  --use-external-test
  --external-data-yaml <path>
  --external-name <label>

Sélection des poids/epochs:
  --weights-dir <dir>           Dossier contenant epochXXX.pt
  --epochs "<list>"             Liste des epochs, ex: "160,170,180"
  --epochs-folder <subdir>      Sous-dossier contenant les runs par epoch (facultatif)

Exemples:
  $(basename "$0") --ssh-config configs/config_moedts.yaml --config configs/exp.yaml \\
    --use-dataset-test --use-external-test --name bench-xyz \\
    --weights-dir runs/train/exp/weights_epochs --epochs "160,170,180,190,200" --epochs-folder foldsX

EOF
}

# --- Parse args ---
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ssh-config) SSH_CONFIG="$2"; shift ;;
    --config) EXPERIMENT_CONFIG="$2"; shift ;;
    --project) PROJECT_OVERRIDE="$2"; shift ;;
    --name) NAME_OVERRIDE="$2"; shift ;;
    --skip-global) SKIP_GLOBAL="true" ;;
    --group-regex) GROUP_REGEX="$2"; shift ;;
    --use-dataset-test) USE_DATASET_TEST="true" ;;
    --use-external-test) USE_EXTERNAL_TEST="true" ;;
    --external-data-yaml) EXTERNAL_DATA_YAML="$2"; shift ;;
    --external-name) EXTERNAL_NAME="$2"; shift ;;
    # nouveaux
    --weights-dir) WEIGHTS_DIR="$2"; shift ;;
    --epochs) EPOCHS_LIST="$2"; shift ;;
    --epochs-folder) EPOCHS_FOLDER="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Argument inconnu : $1"; usage; exit 1 ;;
  esac
  shift
done

if [[ -z "$SSH_CONFIG" || -z "$EXPERIMENT_CONFIG" ]]; then
  echo "Erreur: --ssh-config et --config sont requis."
  usage
  exit 1
fi
if [[ ! -f "$SSH_CONFIG" || ! -f "$EXPERIMENT_CONFIG" ]]; then
  echo "Les fichiers spécifiés sont introuvables."
  exit 1
fi

# --- Petite fonction pour extraire des valeurs YAML ---
get_yaml_value() {
  python3 "$SCRIPT_DIR/extract_yaml_for_bash.py" "$1" "$2"
}

# --- Récup infos SSH + nom de session ---
REMOTE_USER=$(get_yaml_value "user" "$SSH_CONFIG")
REMOTE_HOST=$(get_yaml_value "host" "$SSH_CONFIG")
REMOTE_PORT=$(get_yaml_value "port" "$SSH_CONFIG")
REMOTE_DIR=$(get_yaml_value "remote_dir" "$SSH_CONFIG")
PYTHON_ENV=$(get_yaml_value "python_env" "$SSH_CONFIG")

SESSION_BASE=$(get_yaml_value "training.name" "$EXPERIMENT_CONFIG")
if [[ -z "$SESSION_BASE" ]]; then
  SESSION_BASE="yolo_exp"
fi
SESSION_NAME="${SESSION_BASE}_test"

echo "Configuration SSH :"
echo "  SSH Config: $SSH_CONFIG"
echo "  Experiment: $EXPERIMENT_CONFIG"
echo "  Host: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PORT"
echo "  Remote dir: $REMOTE_DIR"
echo "  venv: $PYTHON_ENV"
echo "  tmux session: $SESSION_NAME"

# --- Sync projet vers serveur ---
echo "Synchronisation des fichiers vers le serveur distant..."
bash "$SCRIPT_DIR/sync_projects.sh" "$SSH_CONFIG"
echo "Synchronisation terminée."

# --- Télécharger/MAJ dataset ---
echo "Téléchargement/MAJ du dataset..."
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
"bash -lc \"cd '$REMOTE_DIR' && source '$PYTHON_ENV' && python3 scripts/download_dataset.py --ssh-config '$SSH_CONFIG' --config '$EXPERIMENT_CONFIG'\""
echo "Dataset OK."

# --- Construire les args pour test.py ---
TEST_ARGS=( "--config" "$EXPERIMENT_CONFIG" )
if [[ -n "$PROJECT_OVERRIDE" ]]; then
  TEST_ARGS+=( "--project" "$PROJECT_OVERRIDE" )
fi
if [[ -n "$NAME_OVERRIDE" ]]; then
  TEST_ARGS+=( "--name" "$NAME_OVERRIDE" )
fi
if [[ "$SKIP_GLOBAL" == "true" ]]; then
  TEST_ARGS+=( "--skip-global" )
fi
if [[ -n "$GROUP_REGEX" ]]; then
  TEST_ARGS+=( "--group-regex" "$GROUP_REGEX" )
fi
if [[ "$USE_DATASET_TEST" == "true" ]]; then
  TEST_ARGS+=( "--use-dataset-test" )
fi
if [[ "$USE_EXTERNAL_TEST" == "true" ]]; then
  TEST_ARGS+=( "--use-external-test" )
  if [[ -z "$EXTERNAL_DATA_YAML" ]]; then
    EXTERNAL_DATA_YAML="datasets/testdm-u2ap3/data.yaml"
  fi
  TEST_ARGS+=( "--external-data-yaml" "$EXTERNAL_DATA_YAML" "--external-name" "$EXTERNAL_NAME" )
fi

# Nouveaux args
if [[ -n "$WEIGHTS_DIR" ]]; then
  TEST_ARGS+=( "--weights-dir" "$WEIGHTS_DIR" )
fi
if [[ -n "$EPOCHS_LIST" ]]; then
  TEST_ARGS+=( "--epochs" "$EPOCHS_LIST" )
fi
if [[ -n "$EPOCHS_FOLDER" ]]; then
  TEST_ARGS+=( "--epochs-folder" "$EPOCHS_FOLDER" )
fi

REMOTE_CMD="cd '$REMOTE_DIR' && source '$PYTHON_ENV' && python3 scripts/test.py ${TEST_ARGS[*]} > '${SESSION_NAME}.log' 2>&1"

echo "Lancement des tests dans tmux..."
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
"tmux new-session -d -s \"$SESSION_NAME\" \"bash -lc '$REMOTE_CMD'\""

echo "Session tmux '${SESSION_NAME}' lancée sur le serveur distant."
echo "Logs: ${SESSION_NAME}.log (dans $REMOTE_DIR)"
echo
echo "Commandes utiles :"
echo "  ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST"
echo "  tmux attach -t $SESSION_NAME"
echo "  tmux ls"
echo "  tail -f ${SESSION_NAME}.log"
