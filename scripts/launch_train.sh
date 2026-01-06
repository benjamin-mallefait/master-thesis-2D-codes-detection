#!/bin/bash
set -euo pipefail

SSH_CONFIG=""
EXPERIMENT_CONFIG=""
ENGINE=""                  # yolo | detr | autre...
K_FOLD=false
EXTRA_ARGS=""              # args bruts vers le script Python cible
SEEDS=""                   # ex: "0,1,2"
GPUS=""                    # ex: "0,1,2,3"
MODE="seq"                 # seq | par
PARENT_DIR=""              # dossier parent des runs (contiendra seed_X)
MULTI_PER_SEED="false"     # <-- NEW: "true" => un run utilise TOUS les GPUs listés

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ssh-config) SSH_CONFIG="$2"; shift ;;
    --config) EXPERIMENT_CONFIG="$2"; shift ;;
    --model) ENGINE="$2"; shift ;;
    --kfold) K_FOLD=true ;;
    --extra) EXTRA_ARGS="$2"; shift ;;
    --seeds) SEEDS="$2"; shift ;;
    --gpus) GPUS="$2"; shift ;;
    --mode) MODE="$2"; shift ;;
    --parent-dir) PARENT_DIR="$2"; shift ;;
    --multi-per-seed) MULTI_PER_SEED="$2"; shift ;;  # <-- NEW
    *) echo "Argument inconnu : $1"; exit 1 ;;
  esac
  shift
done

if [[ ! -f "$SSH_CONFIG" || ! -f "$EXPERIMENT_CONFIG" ]]; then
  echo "Les fichiers spécifiés sont introuvables."
  exit 1
fi

get_yaml_value() { python3 "$SCRIPT_DIR/extract_yaml_for_bash.py" "$1" "$2"; }

REMOTE_USER=$(get_yaml_value "user" "$SSH_CONFIG")
REMOTE_HOST=$(get_yaml_value "host" "$SSH_CONFIG")
REMOTE_PORT=$(get_yaml_value "port" "$SSH_CONFIG")
REMOTE_DIR=$(get_yaml_value "remote_dir" "$SSH_CONFIG")
PYTHON_ENV=$(get_yaml_value "python_env" "$SSH_CONFIG")
SESSION_NAME=$(get_yaml_value "training.name" "$EXPERIMENT_CONFIG")

if [[ -z "$ENGINE" ]]; then
  ENGINE=$(get_yaml_value "training.engine" "$EXPERIMENT_CONFIG")
fi
[[ -z "$ENGINE" || "$ENGINE" == "None" ]] && ENGINE="yolo"

# Dossier parent par défaut
if [[ -z "$PARENT_DIR" ]]; then
  PARENT_DIR="results/${SESSION_NAME}"
fi

echo "Configuration :"
echo "  SSH: $REMOTE_USER@$REMOTE_HOST:$REMOTE_PORT  | remote_dir=$REMOTE_DIR"
echo "  EXP: $EXPERIMENT_CONFIG  | engine=$ENGINE  | session=$SESSION_NAME"
echo "  Seeds: ${SEEDS:-<none>}  | GPUs: ${GPUS:-<none>}  | mode: $MODE"
echo "  Parent dir (remote): $PARENT_DIR"
echo "  Multi-GPU par seed: $MULTI_PER_SEED"

if [[ -z "$REMOTE_USER" || -z "$REMOTE_HOST" || -z "$REMOTE_PORT" || -z "$REMOTE_DIR" || -z "$PYTHON_ENV" || -z "$SESSION_NAME" ]]; then
  echo "Une ou plusieurs valeurs de configuration sont manquantes."
  exit 1
fi

# Sync projet
echo "Synchronisation des fichiers vers le serveur distant..."
bash "$SCRIPT_DIR/sync_projects.sh" "$SSH_CONFIG" || exit 1
echo "Synchronisation terminée."

# venv
echo "Gestion de l'environnement virtuel Python..."
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" /bin/bash <<'EOF'
set -euo pipefail
REMOTE_DIR="Thesis"  # ajuste si nécessaire
if [ -d "$REMOTE_DIR/venv" ]; then
  echo 'Environnement virtuel déjà présent.'
else
  echo 'Environnement virtuel manquant. Création en cours... (relancez ensuite)'
  cd "$REMOTE_DIR"
  make venv_tmux
  exit 1
fi
EOF
echo "Environnement virtuel Python configuré."

# Dataset
echo "Téléchargement/synchro du dataset (si prévu)..."
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
"bash -c \"cd '$REMOTE_DIR' && source venv/bin/activate && python3 scripts/download_dataset.py --ssh-config '$SSH_CONFIG' --config '$EXPERIMENT_CONFIG'\"" || exit 1
echo "Dataset OK."

# Construire la commande de base selon engine
BASE_CMD=""
case "$ENGINE" in
  yolo|YOLO|yolov8|yolov10|yolov11)
    if [ "$K_FOLD" = true ]; then
      BASE_CMD="python3 scripts/train.py --config $EXPERIMENT_CONFIG --kfold"
    else
      BASE_CMD="python3 scripts/train.py --config $EXPERIMENT_CONFIG"
    fi
    ;;
  detr|DETR|rt-detr|rtdetr)
    if [[ "$ENGINE" =~ ^(rt-detr|rtdetr)$ ]]; then
      BASE_CMD="python3 scripts/rt_detr_train.py --config $EXPERIMENT_CONFIG"
    else
      BASE_CMD="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/detr_train.py --config $EXPERIMENT_CONFIG"
    fi
    if [ "$K_FOLD" = true ]; then
      echo "[WARN] --kfold ignoré pour engine='$ENGINE'."
    fi
    ;;
  *)
    ENTRYPOINT=$(get_yaml_value "training.entrypoint" "$EXPERIMENT_CONFIG")
    if [[ -z "$ENTRYPOINT" || "$ENTRYPOINT" == "None" ]]; then
      echo "[ERR] Engine inconnu '$ENGINE' et aucune clé training.entrypoint fournie."
      exit 1
    fi
    BASE_CMD="python3 $ENTRYPOINT --config $EXPERIMENT_CONFIG"
    if [ "$K_FOLD" = true ]; then
      echo "[WARN] --kfold ignoré pour engine='$ENGINE'."
    fi
    ;;
esac

# Normalisation listes
IFS=',' read -r -a SEED_ARR <<< "${SEEDS:-}"
IFS=',' read -r -a GPU_ARR  <<< "${GPUS:-}"

# Crée le parent dir distant
ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
"bash -c \"cd '$REMOTE_DIR' && mkdir -p '$PARENT_DIR'\""

launch_one() {
  local seed="$1"
  local gpu_ids_csv="$2"   # peut être vide
  local name="seed_${seed}"
  local log="${SESSION_NAME}-${name}.log"

  local per_run_extra="--seed ${seed} --project ${PARENT_DIR} --name ${name}"

  local device_arg=""
  local env_prefix=""

  if [[ -n "$gpu_ids_csv" ]]; then
    if [[ "$MULTI_PER_SEED" == "true" ]]; then
      # Multi-GPU pour CE run : on passe toute la liste à --device
      device_arg="--device ${gpu_ids_csv}"
      env_prefix=""  # ne pas restreindre CUDA_VISIBLE_DEVICES ici
    else
      # Mono-GPU pour ce run : on prend le premier (ou le mappé)
      # -> le mapping mono-GPU est géré plus bas
      :
    fi
  fi

  local CMD="${env_prefix} ${BASE_CMD} ${EXTRA_ARGS} ${per_run_extra} ${device_arg}"

  echo "  -> Lancement ${name} ${gpu_ids_csv:+(device=${gpu_ids_csv})} : $CMD"
  ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
  "tmux new-session -d -s \"${SESSION_NAME}_${name}\" \"bash -c 'cd $REMOTE_DIR && source venv/bin/activate && $CMD > $log 2>&1'\""
}

if [[ -z "${SEEDS}" ]]; then
  echo "[INFO] Aucune seed fournie. Lancement simple..."
  ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
  "tmux new-session -d -s \"$SESSION_NAME\" \"bash -c 'cd $REMOTE_DIR && source venv/bin/activate && ${BASE_CMD} ${EXTRA_ARGS} > ${SESSION_NAME}-train.log 2>&1'\""
  echo "Session tmux '$SESSION_NAME' lancée."
  exit 0
fi

echo "Lancement des runs par seed..."
if [[ "$MODE" == "par" ]]; then
  # --- MODE PARALLELE ---
  for idx in "${!SEED_ARR[@]}"; do
    seed="${SEED_ARR[$idx]}"
    if [[ "$MULTI_PER_SEED" == "true" ]]; then
      # Un run = tous les GPUs listés
      launch_one "$seed" "$GPUS"
    else
      # Mapping mono-GPU: seed i -> GPU[i % nb_gpus]
      gpu=""
      if [[ ${#GPU_ARR[@]} -gt 0 ]]; then
        gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
      fi
      # mono-GPU: on isole via CUDA_VISIBLE_DEVICES et --device <0>
      name="seed_${seed}"
      log="${SESSION_NAME}-${name}.log"
      per_run_extra="--seed ${seed} --project ${PARENT_DIR} --name ${name}"
      env_prefix=""
      device_arg=""
      if [[ -n "$gpu" ]]; then
        env_prefix="CUDA_VISIBLE_DEVICES=${gpu}"
        device_arg="--device 0"
      fi
      CMD="${env_prefix} ${BASE_CMD} ${EXTRA_ARGS} ${per_run_extra} ${device_arg}"
      echo "  -> Lancement ${name} (GPU ${gpu:-NA}) : $CMD"
      ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
      "tmux new-session -d -s \"${SESSION_NAME}_${name}\" \"bash -c 'cd $REMOTE_DIR && source venv/bin/activate && $CMD > $log 2>&1'\""
    fi
  done
  echo "Sessions tmux par seed lancées (mode parallèle)."
else
  # --- MODE SEQUENTIEL ---
  RUN_SCRIPT="/tmp/run_${SESSION_NAME}_seeds.sh"
  RUN_BODY="set -euo pipefail; cd $REMOTE_DIR; source venv/bin/activate;"
  for idx in "${!SEED_ARR[@]}"; do
    seed="${SEED_ARR[$idx]}"
    if [[ "$MULTI_PER_SEED" == "true" ]]; then
      # Un run = tous les GPUs listés
      RUN_BODY+=" echo '=== Run seed_${seed} (multi-GPU: ${GPUS}) ==='; ${BASE_CMD} ${EXTRA_ARGS} --seed ${seed} --project ${PARENT_DIR} --name seed_${seed} --device ${GPUS} > ${SESSION_NAME}-seed_${seed}.log 2>&1; "
    else
      gpu=""
      if [[ ${#GPU_ARR[@]} -gt 0 ]]; then
        gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
      fi
      env_prefix=""; device_arg=""
      if [[ -n "$gpu" ]]; then
        env_prefix="CUDA_VISIBLE_DEVICES=${gpu}"
        device_arg="--device 0"
      fi
      RUN_BODY+=" echo '=== Run seed_${seed} (GPU ${gpu:-NA}) ==='; ${env_prefix} ${BASE_CMD} ${EXTRA_ARGS} --seed ${seed} --project ${PARENT_DIR} --name seed_${seed} ${device_arg} > ${SESSION_NAME}-seed_${seed}.log 2>&1; "
    fi
  done
  ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "bash -c \"cat > $RUN_SCRIPT <<'EOS'\n${RUN_BODY}\nEOS\nchmod +x $RUN_SCRIPT\""
  ssh -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" \
  "tmux new-session -d -s \"${SESSION_NAME}_seeds\" \"bash -c '$RUN_SCRIPT'\""
  echo "Session tmux '${SESSION_NAME}_seeds' lancée (mode séquentiel)."
fi
