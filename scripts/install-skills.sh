#!/usr/bin/env bash
# Install agent skills into every discovery location agents use on this machine:
#   ~/.zcode/skills          ZCode (user-level, any cwd)
#   <repo>/.agents/skills    Codex / Cursor / OpenCode convention (when in this repo)
#   ~/.agents/skills         global .agents convention (any agent, any cwd)
#   ~/.claude/skills         Claude Code (symlinks)
#
# Skills: colab-training (this repo), hf-cli (official, from hf CLI),
#         colab-cli (official, bundled with google-colab-cli).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZCODE_SKILLS="$HOME/.zcode/skills"
GLOBAL_AGENTS="$HOME/.agents/skills"
REPO_AGENTS="$ROOT/.agents/skills"
CLAUDE_SKILLS="$HOME/.claude/skills"
for D in "$ZCODE_SKILLS" "$GLOBAL_AGENTS" "$REPO_AGENTS" "$CLAUDE_SKILLS"; do mkdir -p "$D"; done

install_all() {  # install_all SRC_DIR NAME — copy skill dir to every location
  local src="$1" name="$2"
  for D in "$ZCODE_SKILLS" "$GLOBAL_AGENTS" "$REPO_AGENTS"; do
    install -D -m 644 "$src/SKILL.md" "$D/$name/SKILL.md"
  done
  rm -f "$CLAUDE_SKILLS/$name"; ln -s "$(dirname "$src")/$name" "$CLAUDE_SKILLS/$name"
  echo "installed: $name -> zcode, ~/.agents, repo .agents, claude(symlink)"
}

# 1. colab-training (repo is the source of truth)
install_all "$ROOT/skills/colab-training" colab-training

# 2. hf-cli (official skill generated from the installed hf version)
if command -v hf >/dev/null 2>&1; then
  hf skills add --dest "$ZCODE_SKILLS" --force >/dev/null
  hf skills add --dest "$GLOBAL_AGENTS" --force >/dev/null
  hf skills add --dest "$REPO_AGENTS" --force >/dev/null
  # hf's own --claude flag handles the Claude symlink for hf-cli
  hf skills add --claude --dest "$ZCODE_SKILLS" --force >/dev/null 2>&1 || true
  echo "installed: hf-cli -> zcode, ~/.agents, repo .agents, claude"
else
  echo "WARN: hf CLI not on PATH — skipping hf-cli skill" >&2
fi

# 3. colab-cli (official skill bundled inside the google-colab-cli package)
SKILL_SRC="$(find "$HOME/.local/share/uv/tools/google-colab-cli" -name COLAB_SKILL.md -print -quit 2>/dev/null || true)"
if [[ -n "$SKILL_SRC" && -f "$SKILL_SRC" ]]; then
  TMP="$(mktemp -d)"
  {
    printf -- '---\nname: colab-cli\nversion: 1.0.0\ndescription: |\n  Official google-colab-cli agent skill (bundled with the package). Operate\n  Colab sessions: provision GPU/TPU VMs, exec code, sync files, monitor logs.\nallowed-tools:\n  - Bash\n  - Read\n---\n\n'
    cat "$SKILL_SRC"
  } > "$TMP/SKILL.md"
  install_all "$TMP" colab-cli
  rm -rf "$TMP"
else
  echo "WARN: bundled COLAB_SKILL.md not found — skipping colab-cli skill" >&2
fi

echo "done. Restart the agent to pick up new skills."
