#!/usr/bin/env bash
# Shared environment loader for colab_training scripts and the Makefile.
# Sources .env (gitignored) with export semantics, falls back to .env.example
# so scripts still run (with warnings) before the user fills in real values.

COLAB_TRAINING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$COLAB_TRAINING_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$COLAB_TRAINING_ROOT/.env"
  set +a
elif [[ -f "$COLAB_TRAINING_ROOT/.env.example" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$COLAB_TRAINING_ROOT/.env.example"
  set +a
  echo "WARNING: no .env found, using .env.example defaults (HF_TOKEN etc. likely empty)" >&2
fi

export COLAB_AUTH="${COLAB_AUTH:-adc}"
export COLAB_DEFAULT_GPU="${COLAB_DEFAULT_GPU:-T4}"

# Multi-account: when COLAB_ACCOUNT names a registered account (see
# scripts/accounts.sh), point ADC at that account's credentials file. Session
# state stays isolated per account via --config in launch.sh / manual calls.
if [[ -n "${COLAB_ACCOUNT:-}" && -f "$HOME/.config/colab-training/registry.json" ]]; then
  ACCT_CREDS="$(python3 -c "
import json
reg = json.load(open('$HOME/.config/colab-training/registry.json'))
print(reg.get('$COLAB_ACCOUNT', {}).get('creds', ''))" 2>/dev/null || true)"
  if [[ -n "$ACCT_CREDS" && -f "$ACCT_CREDS" ]]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$ACCT_CREDS"
  elif [[ -n "${COLAB_ACCOUNT:-}" ]]; then
    echo "WARNING: COLAB_ACCOUNT='$COLAB_ACCOUNT' not registered or creds missing — using default ADC" >&2
  fi
fi

# Emit a Python prologue that re-exports the whitelisted vars on the Colab VM.
# Used by launch_training.sh — secrets ride only inside the generated
# composite script, which is chmod 600 in /tmp and deleted after the run.
colab_env_header() {
  python3 - <<'PY'
import os

WHITELIST = [
    "HF_TOKEN",
    "HF_USERNAME",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "BASE_MODEL",
    "DATASET_REPO",
    "OUTPUT_REPO",
    "EPOCHS",
    "LEARNING_RATE",
    "MAX_LENGTH",
    "PER_DEVICE_BATCH",
    "GRAD_ACCUM",
    "WANDB_API_KEY",
    "WANDB_PROJECT",
]
pairs = [f'    "{k}": {os.environ[k]!r},' for k in WHITELIST if os.environ.get(k)]
print("# --- injected by launch_training.sh (env from local .env) ---")
print("import os")
print("os.environ.update({")
print("\n".join(pairs))
print("})")
PY
}
