#!/usr/bin/env bash
# Sequential training: coins -> cleanup -> harvest_common.
# Per task: IPPO -> MAPPO -> SVO -> LIO -> LOLA.
# Requires: conda env socialjax, repo root on PYTHONPATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}"

LOGDIR="$REPO_ROOT/training_runs"
TB_ROOT="$LOGDIR/tb"
mkdir -p "$LOGDIR" "$TB_ROOT"
MASTER="$LOGDIR/master.log"

log() { echo "[$(date -Is)] $*" | tee -a "$MASTER"; }

run_step() {
  local name="$1"
  shift
  log "START $name"
  mkdir -p "${TB_ROOT}/${name}"
  if conda run -n socialjax "$@" +TENSORBOARD_DIR="${TB_ROOT}/${name}" >>"$LOGDIR/${name}.log" 2>&1; then
    log "OK   $name"
  else
    log "FAIL $name (see $LOGDIR/${name}.log)"
    exit 1
  fi
}

# Optional smoke test: BENCHMARK_QUICK=1 ./run_marl_benchmark_plan.sh
if [[ "${BENCHMARK_QUICK:-0}" == "1" ]]; then
  EXTRA=(TOTAL_TIMESTEPS=500000)
else
  EXTRA=()
fi

WANDB=(WANDB_MODE=offline)

log "========== COINS =========="
run_step coins_ippo python algorithms/IPPO/ippo_cnn_coins.py "${WANDB[@]}" "${EXTRA[@]}"
run_step coins_mappo python algorithms/MAPPO/mappo_cnn_coins.py "${WANDB[@]}" "${EXTRA[@]}"
run_step coins_svo python algorithms/SVO/svo_cnn_coin.py ENV_NAME=coin_game "${WANDB[@]}" "${EXTRA[@]}"
run_step coins_lio python algorithms/LIO/lio_cnn.py --config-name lio_cnn_coins "${WANDB[@]}" "${EXTRA[@]}"
run_step coins_lola python algorithms/LOLA/lola_cnn.py --config-name lola_cnn_coins "${WANDB[@]}" "${EXTRA[@]}"

log "========== CLEANUP =========="
run_step cleanup_ippo python algorithms/IPPO/ippo_cnn_cleanup.py "${WANDB[@]}" "${EXTRA[@]}"
run_step cleanup_mappo python algorithms/MAPPO/mappo_cnn_cleanup.py "${WANDB[@]}" "${EXTRA[@]}"
run_step cleanup_svo python algorithms/SVO/svo_cnn_cleanup.py "${WANDB[@]}" "${EXTRA[@]}"
run_step cleanup_lio python algorithms/LIO/lio_cnn.py --config-name lio_cnn_cleanup "${WANDB[@]}" "${EXTRA[@]}"
run_step cleanup_lola python algorithms/LOLA/lola_cnn.py --config-name lola_cnn_cleanup "${WANDB[@]}" "${EXTRA[@]}"

log "========== HARVEST COMMON =========="
run_step harvest_ippo python algorithms/IPPO/ippo_cnn_harvest_common.py "${WANDB[@]}" "${EXTRA[@]}"
run_step harvest_mappo python algorithms/MAPPO/mappo_cnn_harvest_common.py "${WANDB[@]}" "${EXTRA[@]}"
run_step harvest_svo python algorithms/SVO/svo_cnn_harvest_open.py --config-name svo_cnn_harvest_common "${WANDB[@]}" "${EXTRA[@]}"
run_step harvest_lio python algorithms/LIO/lio_cnn.py --config-name lio_cnn_harvest_common "${WANDB[@]}" "${EXTRA[@]}"
run_step harvest_lola python algorithms/LOLA/lola_cnn.py --config-name lola_cnn_harvest_common "${WANDB[@]}" "${EXTRA[@]}"

log "ALL DONE"
