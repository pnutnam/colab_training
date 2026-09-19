#!/usr/bin/env bash
# Multi-account Colab credentials + registry.
#
# Each Google account gets its own gcloud config dir (via CLOUDSDK_CONFIG)
# holding an isolated ADC file. Runs then select an account with:
#
#   COLAB_ACCOUNT=work make train
#
# which points GOOGLE_APPLICATION_CREDENTIALS at that account's ADC file and
# isolates its session state. The default ADC slot (~/.config/gcloud/...) is
# untouched and still used when COLAB_ACCOUNT is unset.
#
# Usage:
#   accounts.sh add LABEL [EMAIL]   interactive per-account login (browser)
#   accounts.sh adopt LABEL EMAIL   register the current default ADC as LABEL
#   accounts.sh list                registered accounts + creds presence
#   accounts.sh status              live Colab sessions per account
#   accounts.sh env-for LABEL       print export line for manual use
set -euo pipefail

BASE="$HOME/.config/colab-training"
REGISTRY="$BASE/registry.json"
SCOPES="openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"

creds_path() { echo "$BASE/gcloud/$1/application_default_credentials.json"; }
state_path() { echo "$BASE/state/$1.json"; }

reg_read() { python3 -c "import json,sys; print(json.dumps(json.load(open('$REGISTRY'))))" 2>/dev/null || echo '{}'; }
reg_write() { python3 -c "import json,sys; json.dump(json.loads(sys.argv[1]), open('$REGISTRY','w'), indent=2)" "$1"; }

reg_set() {  # reg_set LABEL EMAIL NOTES
  python3 - "$REGISTRY" "$1" "$2" "${3:-}" <<'PY'
import json, os, sys
path, label, email, notes = sys.argv[1:5]
reg = json.load(open(path)) if os.path.exists(path) else {}
reg[label] = {
    "email": email,
    "creds": os.path.expanduser(f"~/.config/colab-training/gcloud/{label}/application_default_credentials.json"),
    "state": os.path.expanduser(f"~/.config/colab-training/state/{label}.json"),
    "notes": notes,
    "added": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
}
json.dump(reg, open(path, "w"), indent=2)
PY
}

fetch_email() {  # fetch_email CREDS_PATH -> prints email or empty
  local vp pb
  vp="$(ls -d "$HOME"/.local/share/uv/tools/google-colab-cli/lib/python*/site-packages 2>/dev/null | head -1)"
  [[ -z "$vp" ]] && return 1
  # site-packages -> python3.x -> lib -> tool root -> bin/python
  pb="$(dirname "$(dirname "$(dirname "$vp")")")/bin/python"
  [[ -x "$pb" ]] || pb=python3
  "$pb" - "$1" <<'PY' 2>/dev/null || true
import sys, warnings
warnings.filterwarnings("ignore")
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession
import google.auth.transport.requests as gtr
creds = Credentials.from_authorized_user_file(sys.argv[1])
creds.refresh(gtr.Request())
sess = AuthorizedSession(creds)
r = sess.get("https://www.googleapis.com/oauth2/v1/userinfo?alt=json", timeout=10)
print(r.json().get("email", ""))
PY
}

cmd_add() {
  local label="${1:?usage: accounts.sh add LABEL [EMAIL]}" email="${2:-}"
  local dir="$BASE/gcloud/$label"
  mkdir -p "$dir" "$BASE/state"
  echo ">>> Browser login for account '$label'. Approve with the Google account you want registered."
  CLOUDSDK_CONFIG="$dir" gcloud auth application-default login --scopes="$SCOPES"
  if [[ -z "$email" ]]; then
    email="$(fetch_email "$(creds_path "$label")" | tail -1)"
    [[ -z "$email" ]] && { read -r -p "Couldn't auto-detect email. Enter it manually: " email; }
  fi
  reg_set "$label" "$email" ""
  echo "registered '$label' -> $email (creds: $(creds_path "$label"))"
}

cmd_adopt() {
  local label="${1:?usage: accounts.sh adopt LABEL EMAIL}" email="${2:?usage: accounts.sh adopt LABEL EMAIL}"
  local src="$HOME/.config/gcloud/application_default_credentials.json"
  [[ -f "$src" ]] || { echo "no default ADC found at $src — run make auth-colab first" >&2; exit 1; }
  mkdir -p "$BASE/gcloud/$label" "$BASE/state"
  cp "$src" "$(creds_path "$label")"
  chmod 600 "$(creds_path "$label")"
  reg_set "$label" "$email" "adopted from default ADC"
  echo "registered '$label' -> $email (copy of default ADC; original left in place)"
}

cmd_list() {
  python3 - "$REGISTRY" <<'PY'
import json, os, sys
reg = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {}
if not reg:
    print("no accounts registered — use: make account-add LABEL EMAIL"); sys.exit()
w = max(len(k) for k in reg) + 2
for label, info in reg.items():
    ok = "creds OK" if os.path.exists(os.path.expanduser(info["creds"])) else "CREDS MISSING"
    print(f"{label:<{w}} {info['email']:<35} {ok}  {info.get('notes','')}")
PY
}

cmd_status() {
  local tmp; tmp="$(mktemp)"
  reg_read > "$tmp"
  python3 -c "import json,sys; [print(k) for k in json.load(open('$tmp'))]" | while read -r label; do
    creds="$(python3 -c "import json;print(json.load(open('$tmp'))['$label']['creds'])")"
    state="$(python3 -c "import json;print(json.load(open('$tmp'))['$label']['state'])")"
    email="$(python3 -c "import json;print(json.load(open('$tmp'))['$label']['email'])")"
    echo "== $label ($email) =="
    if [[ -f "$creds" ]]; then
      GOOGLE_APPLICATION_CREDENTIALS="$creds" colab --auth adc --config "$state" sessions 2>&1 | sed 's/^/  /' || true
    else
      echo "  creds missing at $creds"
    fi
    echo
  done
  rm -f "$tmp"
}

cmd_env() {
  local label="${1:?usage: accounts.sh env-for LABEL}"
  local creds; creds="$(python3 -c "import json;print(json.load(open('$REGISTRY'))['$label']['creds'])" 2>/dev/null)" || { echo "unknown account '$label'" >&2; exit 1; }
  echo "export GOOGLE_APPLICATION_CREDENTIALS=\"$creds\""
  echo "export COLAB_ACCOUNT=\"$label\""
}

case "${1:-help}" in
  add) shift; cmd_add "$@" ;;
  adopt) shift; cmd_adopt "$@" ;;
  list) cmd_list ;;
  status) cmd_status ;;
  env-for) shift; cmd_env "$@" ;;
  *) sed -n '2,20p' "$0" | grep -E "^#( |$)" | sed 's/^# \?//'; exit 1 ;;
esac
