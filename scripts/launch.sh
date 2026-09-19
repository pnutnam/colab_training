#!/usr/bin/env bash
# Launch a training run (or smoke test) on a Google Colab VM.
#
# How it works:
#   1. Loads .env via scripts/env.sh
#   2. Generates a composite script = (env header with HF_TOKEN etc.) + train_sft.py
#      (or the smoke test) in /tmp with mode 600
#   3. `colab run` provisions a fresh GPU VM, executes it, and tears the VM
#      down on completion (KEEP=1 preserves it for inspection/reuse)
#
# Env knobs (in .env or inline):
#   COLAB_DEFAULT_GPU  T4|L4|G4|H100|A100   (default T4)
#   COLAB_RUN_TIMEOUT  execution timeout seconds for `colab run` (default 14400 = 4h;
#                      the CLI's own 30s default kills any real training run)
#   KEEP               1 = pass --keep to colab run
#   TRAIN_ARGS         extra argparse flags for train_sft.py (quoted string)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/env.sh"

MODE="${1:-train}"
GPU="${COLAB_DEFAULT_GPU:-T4}"

if [[ "$MODE" == "--smoke" ]]; then
  BODY="$ROOT/scripts/smoke_test.py"
else
  BODY="$ROOT/training/train_sft.py"
  : "${DATASET_REPO:?DATASET_REPO must be set in .env — see .env.example}"
  : "${OUTPUT_REPO:?OUTPUT_REPO must be set in .env — see .env.example}"
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is empty — run 'make auth-hf' or paste a token into .env" >&2
    exit 1
  fi
fi

COMPOSITE="$(mktemp /tmp/colab-train-XXXXXX.py)"
trap 'rm -f "$COMPOSITE"' EXIT

{
  echo "# Generated $(date -u +%FT%TZ) by launch.sh — ephemeral, contains secrets, do not share"
  colab_env_header
  cat "$BODY"
} > "$COMPOSITE"
chmod 600 "$COMPOSITE"

KEEP_ARGS=()
[[ "${KEEP:-0}" == "1" ]] && KEEP_ARGS=(--keep)

# Per-account isolated session state (see scripts/accounts.sh).
CONFIG_ARGS=()
if [[ -n "${COLAB_ACCOUNT:-}" && -f "$HOME/.config/colab-training/registry.json" ]]; then
  ACCT_STATE="$(python3 -c "
import json
reg = json.load(open('$HOME/.config/colab-training/registry.json'))
print(reg.get('$COLAB_ACCOUNT', {}).get('state', ''))" 2>/dev/null || true)"
  [[ -n "$ACCT_STATE" ]] && CONFIG_ARGS=(--config "$ACCT_STATE")
fi

# shellcheck disable=SC2086
set +e
colab --auth "$COLAB_AUTH" "${CONFIG_ARGS[@]}" run --gpu "$GPU" --timeout "${COLAB_RUN_TIMEOUT:-14400}" "${KEEP_ARGS[@]}" "$COMPOSITE" ${TRAIN_ARGS:-}
RC=$?
set -e

# Usage log: one JSON line per launch (account, gpu, outcome).
mkdir -p "$ROOT/runs"
python3 - "$ROOT/runs/runlog.jsonl" "${COLAB_ACCOUNT:-default}" "$GPU" "$MODE" "$RC" <<'PY'
import datetime, json, sys
entry = {
    "ts": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "account": sys.argv[2], "gpu": sys.argv[3], "mode": sys.argv[4],
    "exit": int(sys.argv[5]),
}
with open(sys.argv[1], "a") as f:
    f.write(json.dumps(entry) + "\n")
PY

exit $RC
