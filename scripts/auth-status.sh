#!/usr/bin/env bash
# Report auth state for both halves of the training stack.
set -uo pipefail
PASS=0; FAIL=0
line() { printf '  %-42s %s\n' "$1" "$2"; }

echo "== Hugging Face =="
if command -v hf >/dev/null 2>&1; then
  WHOAMI="$(hf auth whoami 2>/dev/null | head -1)"
  if [[ -n "$WHOAMI" && "$WHOAMI" != *"not logged"* && "$WHOAMI" != *"Error"* ]]; then
    line "logged in as" "$WHOAMI"; PASS=$((PASS+1))
  else
    TOKEN="${HF_TOKEN:-}"
    if [[ -n "$TOKEN" ]]; then
      line "logged in as" "(via \$HF_TOKEN)"; PASS=$((PASS+1))
    else
      line "auth" "MISSING — run: make auth-hf"; FAIL=$((FAIL+1))
    fi
  fi
else
  line "hf CLI" "NOT INSTALLED"; FAIL=$((FAIL+1))
fi

echo "== Google (Colab CLI) =="
if command -v colab >/dev/null 2>&1; then
  ADC="$HOME/.config/gcloud/application_default_credentials.json"
  OAUTH_TOKEN="$HOME/.config/colab-cli/token.json"
  if [[ -f "$ADC" || -f "$OAUTH_TOKEN" ]]; then
    [[ -f "$ADC" ]] && line "ADC credentials" "present ($ADC)"
    [[ -f "$OAUTH_TOKEN" ]] && line "oauth2 token" "present ($OAUTH_TOKEN)"
    SESS="$(colab --auth "$( [[ -f $ADC ]] && echo adc || echo oauth2 )" sessions 2>/dev/null)"
    line "active sessions" "$(echo "$SESS" | grep -c . ) entries (colab sessions to view)"
    PASS=$((PASS+1))
  else
    line "credentials" "MISSING — run: make auth-colab"; FAIL=$((FAIL+1))
  fi
else
  line "colab CLI" "NOT INSTALLED"; FAIL=$((FAIL+1))
fi

echo
echo "passed: $PASS  failed: $FAIL"
[[ $FAIL -eq 0 ]]
