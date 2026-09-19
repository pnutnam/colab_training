# colab_training

Agentic, off-machine LLM training: **open models from Hugging Face, trained on
Google Colab GPUs** (via your Colab Pro subscription), driven entirely from
this machine — by you or by an AI agent (ZCode skills + MCP servers included).

```
┌──────────────── this machine ────────────────┐
│  ZCode agent (skills: colab-training,       │
│   colab-cli, hf-cli; MCP: huggingface,      │
│   colab-mcp)  +  hf / colab CLIs            │
└──────────┬─────────────────────┬────────────┘
           │ models + datasets   │ `colab run --gpu T4 train.py`
           ▼ in, adapters out    ▼
     Hugging Face Hub        Colab Pro VM (T4/L4/G4/H100/A100)
```

Heavy artifacts never touch this machine's disk: base models and datasets are
pulled from the Hub **on the VM**, and trained adapters are pushed **to the
Hub** before the VM is torn down.

## What's installed

| Piece | Where | Purpose |
| --- | --- | --- |
| `google-colab-cli` (`colab`) | `uv tool` (global) | Provision/exec/teardown Colab VMs headlessly |
| `hf` CLI | `uv tool` (global) | Hub auth, model/dataset search, upload/download |
| `.mcp.json` | this repo | `huggingface` (remote MCP, Hub search/jobs/docs) + `colab-mcp` (browser-session bridge) |
| skills | `~/.zcode/skills/{colab-training,colab-cli,hf-cli}` + `.agents/skills/` | Agent know-how; repo copy in `skills/` is source of truth |
| `.env` → `~/.bashrc` | this repo | `HF_TOKEN`, GPU tier, training defaults (gitignored) |
| `training/train_sft.py` | this repo | Self-contained QLoRA/SFT script that runs on the VM |
| `scripts/launch.sh` | this repo | Env-inject + `colab run` ephemeral job launcher |

Update both CLIs anytime with `make update`; refresh skills after updates with
`make skills-install`.

## One-time auth (the only manual steps)

### 1. Hugging Face

```bash
make auth-hf
```

Prints a URL + code (hf.co/oauth/device). Approve in the browser; the CLI
stores the token. Alternatively paste a token with **write** scope from
<https://huggingface.co/settings/tokens> into `.env` as `HF_TOKEN=`.

### 2. Google Colab (same account as your Colab Pro subscription)

```bash
make auth-colab
```

Runs `gcloud auth application-default login` with the four scopes the Colab
backend requires (`colaboratory`, `userinfo.email`, `cloud-platform`,
`openid`). One-time; ADC refreshes thereafter. The alternative `oauth2` mode
(`colab --auth oauth2 ...`) does a browser consent + code paste and caches to
`~/.config/colab-cli/token.json` — fine for humans, ADC is better for agents.

Check everything: `make auth-status`.

> **Note:** the `hf` MCP server in `.mcp.json` reads `${HF_TOKEN}` — start
> ZCode from a shell **after** auth (the `~/.bashrc` block exports it), or the
> Hub MCP tools will 401. The settings page at
> <https://huggingface.co/settings/mcp> can generate a client-specific snippet
> if you prefer a dedicated MCP token.

## Usage

### Smoke test (≈1 min, 1 compute unit or less)

```bash
make smoke        # fresh T4: CUDA check, TFLOPS matmul, Hub connectivity, teardown
```

### Train (QLoRA SFT)

1. Edit `.env`: `DATASET_REPO` (a HF dataset with `text` or `messages`
   column), `OUTPUT_REPO` (e.g. `yourname/qwen3-1.7b-qlora-mytask`),
   optionally `BASE_MODEL` / `EPOCHS` / `PER_DEVICE_BATCH` / `COLAB_DEFAULT_GPU`.
2. ```bash
   make train                  # ephemeral: fresh VM → train → push → teardown
   make train-keep             # keep the VM for inspection (stop it after!)
   TRAIN_ARGS="--epochs=2 --batch=4" make train   # one-off overrides
   ```
3. Training logs stream live. The final `[train] RESULT {...}` line carries
   the pushed adapter URL.

To get a quick SFT dataset up: put a `train.jsonl` (one record per line, with
`{"text": "..."}` or `{"messages": [{"role": ..., "content": ...}, ...]}`)
into a folder and run `hf upload yourname/my-sft-data ./folder --repo-type dataset`.

### Persistent sessions / manual driving

```bash
make sessions               # list VMs
colab --auth adc new -s dev --gpu L4
colab --auth adc exec -s dev -f training/train_sft.py   # kernel state persists across execs
colab --auth adc status -s dev
colab --auth adc stop -s dev    # idle VMs burn compute units — always stop!
```

### Pulling artifacts locally

```bash
hf download yourname/qwen3-1.7b-qlora-mytask --local-dir runs/mytask
```

## Agent integration notes

- The **colab-training** skill (see `skills/colab-training/SKILL.md`) encodes
  the full runbook: pre-flight auth check → smoke → launch → monitor →
  artifact push/pull → teardown. Agents should follow it for any training ask.
- `hf` CLI auto-detects agent callers (`--format agent`), and `--json` is
  available on nearly every subcommand for parseable output.
- **Cost rule:** every provisioned VM must end with `colab stop` unless
  launched via plain `make train` (self-teardown). Agents are told to check
  `colab sessions` before finishing any task.
- GPU entitlements are tier-gated. T4 is the safe default; if `colab new`
  returns 400 for an accelerator, fall back (the skill's failure playbook
  covers the rest).
- Two upstream gotchas already worked around here: colab-cli 0.6.0 breaks with
  jupyter-kernel-client 1.x (pinned to 0.15.0 via `make update`), and
  `colab run`'s execution timeout defaults to 30s (`launch.sh` passes
  `--timeout` from `COLAB_RUN_TIMEOUT`, default 4h).

## Multi-account rotation

`make auth-colab` writes a **single default ADC slot** — re-running it
*overwrites* (never stacks). For multiple Google accounts, each account gets
an isolated credentials file + session state via `scripts/accounts.sh`:

```bash
make account-adopt LABEL=pro EMAIL=you@gmail.com   # register the current login (done for 'pro')
make account-add LABEL=alt1 EMAIL=other@gmail.com  # browser login for another account
make account-list                                  # registered accounts
make account-status                                # live Colab sessions per account
COLAB_ACCOUNT=alt1 make train                       # run as a specific account
COLAB_ACCOUNT=alt1 make sessions                   # manual commands too (via scripts/env.sh)
```

Mechanics: per-account ADC files live under
`~/.config/colab-training/gcloud/<label>/`, selected via the
`GOOGLE_APPLICATION_CREDENTIALS` env var (honored by `google.auth.default()`,
the keep-alive daemon inherits it); session state is isolated with
`--config ~/.config/colab-training/state/<label>.json`. Every `make train`
launch appends a line to `runs/runlog.jsonl` (`{ts, account, gpu, mode,
exit}`) so you can see which account has been spending.

Practical notes: keep the **Pro account as the default** for real runs (compute
units + A100/L4 entitlements); free accounts are T4-at-best and often
CPU-only, with shorter sessions and dynamic availability. Also be aware
Google's Colab terms explicitly disallow using multiple accounts to evade
usage limits — rotating between accounts you genuinely use for different
projects is normal; systematic quota-evasion rotation risks account
suspension. Decide accordingly.

## Repo layout

```
.mcp.json                  MCP servers (huggingface, colab-mcp)
.env / .env.example        tokens + training defaults (never commit .env)
Makefile                   setup | auth-* | smoke | train | sessions | stop
scripts/env.sh             .env loader + VM env-header generator
scripts/launch.sh          composite-script builder + colab run launcher
scripts/smoke_test.py      GPU/Hub connectivity check (also shebang-runnable)
scripts/auth-status.sh     auth state report
scripts/install-skills.sh  (re)install agent skills
training/train_sft.py      the QLoRA/SFT trainer that runs on the VM
training/requirements-colab.txt   VM-side pins (embedded in train_sft.py too)
skills/colab-training/     canonical agent skill (installed to ~/.zcode/skills)
runs/                      local artifacts (gitignored)
```
