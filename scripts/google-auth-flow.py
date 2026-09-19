#!/usr/bin/env python3
"""Agent-driven Google OAuth login for colab_training accounts.

Two-phase version of google-colab-cli's remote copy-paste flow, so an agent
can generate the authorization URLs and complete the exchange when the user
pastes back codes (one round trip for N accounts).

  google-auth-flow.py url LABEL            # print authorization URL, stash flow state
  google-auth-flow.py complete LABEL CODE  # exchange code, write ADC, register account

MUST run under the google-colab-cli tool venv python (needs google_auth_oauthlib).
Uses Google's public cloud-SDK OAuth client bundled with the CLI and the same
remote landing page redirect; scopes match `make auth-colab`.
"""
from __future__ import annotations

import datetime
import json
import pathlib
import re
import sys

BASE = pathlib.Path.home() / ".config" / "colab-training"
PENDING = BASE / "pending"
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
# Registered to Google's cloud-SDK client (see colab_cli/auth.py); any other
# client id with this redirect fails with redirect_uri_mismatch.
REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"


def tool_venv_site_packages() -> pathlib.Path:
    import glob

    hits = glob.glob(
        str(pathlib.Path.home() / ".local/share/uv/tools/google-colab-cli/lib/python*/site-packages")
    )
    if not hits:
        sys.exit("google-colab-cli tool venv not found — is the CLI installed?")
    return pathlib.Path(hits[0])


def load_flow():
    from google_auth_oauthlib.flow import InstalledAppFlow

    cfg = json.loads(
        (tool_venv_site_packages() / "colab_cli" / "oauth_config.json").read_text()
    )
    flow = InstalledAppFlow.from_client_config(cfg, SCOPES)
    flow.redirect_uri = REMOTE_REDIRECT_URI
    return flow


def cmd_url(label: str) -> None:
    PENDING.mkdir(parents=True, exist_ok=True)
    flow = load_flow()
    auth_url, _state = flow.authorization_url(prompt="consent", token_usage="remote")
    state = {
        "label": label,
        "code_verifier": flow.code_verifier,
        "redirect_uri": flow.redirect_uri,
        "url": auth_url,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (PENDING / f"{label}.json").write_text(json.dumps(state, indent=2))
    print(auth_url)


def registry_label(email: str, fallback: str) -> str:
    local = email.split("@")[0]
    label = re.sub(r"[^a-zA-Z0-9._-]", "", local) or fallback
    reg_path = BASE / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    if label in reg or label == fallback:
        return fallback
    return label


def cmd_complete(label: str, code: str) -> None:
    pending_path = PENDING / f"{label}.json"
    if not pending_path.exists():
        sys.exit(f"no pending flow for label '{label}' — run `url {label}` first")
    state = json.loads(pending_path.read_text())

    flow = load_flow()
    flow.code_verifier = state["code_verifier"]
    flow.redirect_uri = state["redirect_uri"]
    flow.fetch_token(code=code)  # raises on bad/expired code
    creds = flow.credentials

    # Email for the registry (userinfo.email scope was granted).
    email = ""
    try:
        from google.auth.transport.requests import AuthorizedSession

        r = AuthorizedSession(creds).get(
            "https://www.googleapis.com/oauth2/v1/userinfo?alt=json", timeout=10
        )
        email = r.json().get("email", "")
    except Exception:
        pass

    final_label = registry_label(email, fallback=label)
    creds_dir = BASE / "gcloud" / final_label
    creds_dir.mkdir(parents=True, exist_ok=True)
    creds_path = creds_dir / "application_default_credentials.json"
    info = json.loads(creds.to_json())
    info["type"] = "authorized_user"  # google.auth's loader requires it; to_json() omits it
    creds_path.write_text(json.dumps(info, indent=2))
    creds_path.chmod(0o600)
    (BASE / "state").mkdir(parents=True, exist_ok=True)

    reg_path = BASE / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    reg[final_label] = {
        "email": email or "(unknown — fill in manually)",
        "creds": str(creds_path),
        "state": str(BASE / "state" / f"{final_label}.json"),
        "notes": "oauth remote flow",
        "added": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    reg_path.write_text(json.dumps(reg, indent=2))
    pending_path.unlink(missing_ok=True)
    print(f"OK label={final_label} email={email or '?'} creds={creds_path}")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    cmd, label = sys.argv[1], sys.argv[2]
    if cmd == "url":
        cmd_url(label)
    elif cmd == "complete":
        if len(sys.argv) < 4:
            sys.exit("usage: google-auth-flow.py complete LABEL CODE")
        cmd_complete(label, sys.argv[3])
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
