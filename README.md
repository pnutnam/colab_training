# colab_training

Agentic, off-machine LLM training: **open models from Hugging Face, trained on
Google Colab GPUs**, driven entirely from your terminal — by you or by an AI
agent (ZCode/Claude Code/Codex skills + MCP servers included).

```
┌──────────────── this machine ────────────────┐
│  AI agent (skills: colab-training,          │
│   colab-cli, hf-cli; MCP: huggingface,      │
│   colab-mcp)  +  hf / colab CLIs            │
└──────────┬─────────────────────┬────────────┘
           │ models + datasets   │ `colab run --gpu T4 train.py`
           ▼ in, adapters out    ▼
     Hugging Face Hub        Colab runtime (T4/L4/G4/A100)
```

Heavy artifacts never touch your machine's disk: base models and datasets are
pulled from the Hub **on the VM**, and trained adapters are pushed **to the
Hub** before the VM is torn down. Your laptop stays a control plane.

## What this actually leverages

| Piece | What it is | What it costs you |
| --- | --- | --- |
| [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) | **Google's official open-source CLI** for driving consumer Colab runtimes headlessly (provision, exec, files, teardown) | Free (Apache-2.0) |
| **Google Colab subscription** | The actual GPUs. Free tier, **Pro ($9.99/mo)**, or **Pro+ ($49.99/mo)** — billed in *compute units* (CUs) | Your CUs — see caps below |
| [`hf` CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) + **HF Hub** | Model/dataset source of truth: base models in, trained adapters/datasets out | Free; a free HF account with a **write** token |
| `colab-mcp` (Google) + HF MCP server | Optional MCP bridges for interactive agents | Free |

That's the whole stack — no cloud accounts beyond Google + Hugging Face, no
service accounts, no GCP project, no billing APIs. If you have a Colab
subscription and an HF account, you have everything.

## Usage caps — read this before launching

**Colab compute units (the real currency):**

| Plan | Price | CUs included | Notes |
| --- | --- | --- | --- |
| Free | $0 | none (dynamic quotas) | T4 "when available", short sessions, no guarantee |
| **Pro** | $9.99/mo | **~100 CU/mo** | T4/L4 access, up to 24 h runtimes |
| **Pro+** | $49.99/mo | **~500 CU/mo** | A100 priority, background execution |

**Approximate burn rates** (as of late 2026 — check Colab's signup page; high-RAM variants burn more):

| GPU | VRAM | ~CU/hour | ~hours on a Pro 100 CU |
| --- | --- | --- | --- |
| T4 | 16 GB | ~1.2 | ~84 h |
| L4 | 24 GB | ~1.7 | ~58 h |
| A100 | 40 GB | ~5.4 | ~18 h |
| A100 | 80 GB | ~8.5+ | ~12 h |

Practical rules baked into this repo:

- **Iterate on T4, finish on bigger.** A full QLoRA run of a 0.5B model on 500
  examples took 2.2 min on a T4 (~0.04 CU). A 7B/8B QLoRA (e.g. Llama-3.1-8B,
  Qwen3-8B) fits a T4 at ~2–8 h ≈ 3–10 CU. Reserve A100 for models >14B.
- **Ephemeral by default:** `make train` provisions a fresh VM and tears it
  down automatically, even on failure. An *idle* VM burns CUs forever (up to a
  24 h hard cap), so `colab stop` / `make stop` is the habit that saves you.
- **GPU availability is dynamic** — paying doesn't guarantee a GPU. If
  `colab new` 400s, fall back to T4/CPU or retry later (the skill's failure
  playbook covers this).
- **Hugging Face side:** free accounts can push adapters fine; large private
  model repos are limited by [HF storage quotas](https://huggingface.co/docs/hub/storage-limits).
  The HF token needs **write** scope (read-only tokens can't push).

## Quickstart (two auths, then train)

```bash
make setup          # installs local deps + skills; creates .env from example
make auth-hf        # Hugging Face: opens hf.co/oauth/device, approve in browser
make auth-colab     # Google: approve with your Colab account (same one as your subscription)
make auth-status    # both should be green
make smoke          # ~1 min on a T4: CUDA check, Hub connectivity, auto-teardown
```

Then set `DATASET_REPO` + `OUTPUT_REPO` in `.env` and:

```bash
make train          # QLoRA SFT: fresh GPU VM → train → push adapter to HF → teardown
```

Any HF dataset with a `text` column or a `messages` (chat) column works. To
make one: `hf upload yourname/my-data ./folder --repo-type dataset` (JSONL of
`{"text": "..."}` or `{"messages": [{"role": ..., "content": ...}, ...]}`).
Config knobs (model, epochs, LR, batch, GPU tier, timeout) all live in
`.env` with comments — see [`.env.example`](.env.example).

`BASE_MODEL` can be **any causal-LM on the Hub** (Qwen, Llama, Gemma, Mistral…).
Default script does 4-bit QLoRA; `TRAIN_ARGS="--full" make train` does full
fine-tuning if the GPU is big enough.

Long runs survive preemption: checkpoints push to `OUTPUT_REPO` every
`SAVE_STEPS` (default 250). If a runtime dies mid-training, relaunch with
`TRAIN_ARGS="--resume-from checkpoint-500" make train` (checkpoint name from
your output repo's `checkpoints/` folder).

## Multiple Google accounts (parallel runs)

Each Colab account gets its own credentials + isolated session state, so you
can run **one project per account, concurrently** — useful because GPU
availability and CU budgets are per-account:

```bash
make account-adopt LABEL=pro EMAIL=you@gmail.com   # register your current login
make account-add LABEL=alt1 EMAIL=other@gmail.com  # add another account (browser consent)
make account-list        # all registered accounts
make account-status      # live VMs per account
COLAB_ACCOUNT=alt1 make train   # run under a specific account
```

Every launch is logged to `runs/runlog.jsonl` (timestamp, account, GPU,
exit) so you always know which account spent what.

**Honest ToS note:** Google's Colab terms disallow using multiple accounts to
*evade usage limits*. Running genuinely separate projects on separate accounts
you own is normal usage; systematic quota-evasion risks suspension. Your call.

## Daily driving

```bash
make sessions / make status / make logs   # what's running
make stop                                 # release the active VM (do this!)
hf download yourname/your-adapter --local-dir runs/   # pull artifacts locally
```

## Agent integration

Skills are installed machine-wide by `make setup` (`make skills-install` to
re-sync after clone or skill edits): **`colab-training`** (the full runbook:
pre-flight → launch → monitor → push → teardown, failure playbook,
multi-account rotation), **`colab-cli`** (Google's official skill),
**`hf-cli`** (HF's official skill) — into `~/.zcode/skills`,
`~/.agents/skills`, and `~/.claude/skills`. MCP
servers (`huggingface`, `colab-mcp`) are configured in
[`.mcp.json`](.mcp.json). `hf` auto-detects agent callers; nearly every
subcommand has `--json`.

## Troubleshooting

Two upstream gotchas are already handled here (documented so you can fix
elsewhere): colab-cli 0.6.0 needs `jupyter-kernel-client==0.15.0` pinned
(`make update` re-applies it), and `colab run`'s execution timeout defaults to
30 s (`launch.sh` passes `--timeout` from `COLAB_RUN_TIMEOUT`, default 4 h).
403s against `colab.pa.googleapis.com` mean your Google token lacks the
`colaboratory` scope → re-run `make auth-colab`. OOM → lower
`PER_DEVICE_BATCH`, raise `GRAD_ACCUM`, shorter `MAX_LENGTH`. HF 401 on push →
token needs write scope.

## Repo layout

```
.mcp.json                  MCP servers (huggingface, colab-mcp)
.env / .env.example        tokens + training defaults (never commit .env)
Makefile                   setup | auth-* | smoke | train | sessions | stop | account-*
scripts/                   launcher, multi-account registry, OAuth onboarding, installers
training/train_sft.py      self-contained QLoRA/SFT trainer that runs ON the VM
skills/colab-training/     canonical agent skill (installed to all agent dirs)
runs/                      local artifacts + runlog (gitignored)
```

MIT licensed — see [LICENSE](LICENSE).

## Disclaimer

Personal project. Not affiliated with or endorsed by Google or Hugging Face.
`google-colab-cli` and `colab-mcp` are Google open-source projects, but Colab
is a consumer product: no SLA, runtimes can be preempted, and its terms of
service govern your usage — including the multi-account note above. Training
quality is on you; this repo automates plumbing, not judgment.
