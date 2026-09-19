---
name: colab-training
version: 0.1.0
description: |
  Agentic off-machine LLM training using Google Colab GPUs (via the user's
  Colab Pro subscription, driven by google-colab-cli) and the Hugging Face Hub
  (models in, adapters/datasets out, via the hf CLI). Use when: "train a
  model", "fine-tune", "QLoRA", "run on Colab", "push to hub", "download a
  dataset", "spin up a GPU". Covers: provisioning GPU sessions, launching
  training runs from this repo, monitoring, artifact push/pull, and cost
  hygiene.
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
triggers:
  - train on colab
  - fine-tune model
  - launch training
  - run on gpu
  - push adapter to hub
---

# Agentic Colab + HuggingFace Training

Repo: `/home/nate/projects/colab_training` — read `.env` for configured
defaults (BASE_MODEL, DATASET_REPO, OUTPUT_REPO, GPU tier) and
`README.md` for full docs. Companion skills: `colab-cli` (session operation),
`hf-cli` (Hub CLI). MCP servers `huggingface` and `colab-mcp` are configured
in `.mcp.json`.

## Machine map (all machine-global — any agent, any cwd)

- Repo / Makefile entrypoint: `/home/nate/projects/colab_training`
- CLIs on PATH: `colab` (Google Colab VMs), `hf` (Hugging Face Hub)
- Account registry: `~/.config/colab-training/registry.json` — one entry per
  authorized Google account: `{label: {email, creds, state}}`. Per-account
  ADC creds + isolated session state live under `~/.config/colab-training/`.
- Run history (which account ran what): `runs/runlog.jsonl` in the repo.
- Multi-account usage: `COLAB_ACCOUNT=<label> make train` (or export
  GOOGLE_APPLICATION_CREDENTIALS=<registry creds> for raw `colab` calls, plus
  `--config <registry state>` to keep session state separated per account).
- Accounts default `COLAB_ACCOUNT` is unset = default ADC slot = `pro`
  (the Colab Pro account; the only one with L4/A100 entitlements as of
  2026-09-05 — free accounts: T4 confirmed, availability is dynamic).

## Pre-flight (always run first)

```bash
make -C /home/nate/projects/colab_training auth-status
```

Checks: `hf` token (`hf auth whoami`), Google ADC for the Colab CLI, and
active sessions. If HF auth is missing → tell the user to run `make auth-hf`.
If ADC is missing → `make auth-colab` (interactive, one-time).

## Onboarding a new Google account (agent-driven)

```bash
PB=$(dirname "$(dirname "$(dirname "$(ls -d ~/.local/share/uv/tools/google-colab-cli/lib/python*/site-packages | head -1)")")")/bin/python
"$PB" /home/nate/projects/colab_training/scripts/google-auth-flow.py url LABEL1   # prints a URL
# user opens URL with that Google account, approves, pastes back a 4/0... code
"$PB" /home/nate/projects/colab_training/scripts/google-auth-flow.py complete LABEL1 '4/0...'
```

Auto-detects the email, registers a readable label, writes per-account creds.
Validate after: `GOOGLE_APPLICATION_CREDENTIALS=<creds> colab --auth adc
--config <state> sessions`. (First validation call can fail transiently —
retry once.)

## GPU capability probe (verify, don't assume)

Availability is dynamic per account. To check what an account can get right
now (allocates briefly, stops immediately):

```bash
GOOGLE_APPLICATION_CREDENTIALS=<creds> colab --auth adc --config <state> new -s probe --gpu T4
GOOGLE_APPLICATION_CREDENTIALS=<creds> colab --auth adc --config <state> stop -s probe
```

400 = no entitlement for that accelerator. Known map (2026-09-05): all six
registered accounts had T4; `pro` additionally had L4 and A100.

## The canonical training loop

1. **Pick/verify data + base model** — `hf models ls --search <name> --json`,
   `hf datasets ls --search <topic> --json`. The dataset needs a `text` or
   `messages` column.
2. **Set config** — edit `.env` (BASE_MODEL, DATASET_REPO, OUTPUT_REPO,
   COLAB_DEFAULT_GPU) or pass `TRAIN_ARGS="--epochs=2 --batch=4"` to override.
3. **Launch** — `make -C /home/nate/projects/colab_training train`
   (ephemeral: fresh VM → train → auto-teardown). `make train-keep` keeps the
   VM for inspection. Output streams live; the final line is
   `[train] RESULT {...json...}` with the pushed adapter URL.
4. **Optional smoke first** — `make smoke` (T4, ~1 min, verifies CUDA + Hub
   connectivity before burning compute units on a real run).

GPU tiers: T4 (16 GB, cheapest), L4 (24 GB), A100 (40 GB, Pro), G4/H100
(higher tiers). QLoRA in 4-bit fits roughly: T4 ≤ ~8B params, L4 ≤ ~14B,
A100 ≤ ~32B. If `colab new` 400s on an accelerator → no entitlement, fall
back to T4.

## Persistent sessions (multi-step / exploratory work)

Use `colab` directly (kernel state persists across `exec` calls — build state
incrementally, don't re-import):

```bash
colab --auth adc new -s train1 --gpu L4
colab --auth adc install -s train1 -r training/requirements-colab.txt
echo "import torch; print(torch.cuda.get_device_name(0))" | colab --auth adc exec -s train1
colab --auth adc exec -s train1 -f some_script.py
colab --auth adc status -s train1      # monitor
colab --auth adc log -s train1 -n 20   # structured events on failure
colab --auth adc stop -s train1        # ALWAYS do this when done
```

MCP `colab-mcp` instead bridges to an interactive browser Colab session —
prefer the CLI for headless training work.

## Artifacts

- **Push (VM → Hub)**: train_sft.py pushes the adapter automatically when
  OUTPUT_REPO + HF_TOKEN are set. For anything else:
  `colab exec` a snippet using `huggingface_hub.HfApi().upload_folder(...)`.
- **Pull (Hub → local)**: `hf download <repo_id> --local-dir runs/<name>`.
- **VM → local file**: `colab download -s <session> /content/file ./local`
  (avoid for multi-GB weights — Hub is the transfer channel).

## Cost hygiene (Colab compute units are finite)

- `colab run` without `--keep` self-cleans; with `--keep`, you MUST
  `colab stop` when finished.
- Check idle sessions before ending any task: `colab sessions` → stop
  everything not actively needed.
- Prefer T4 for iteration, bigger GPUs only for the final run.
- Multi-account: `make account-list` shows registered Google accounts;
  `COLAB_ACCOUNT=<label> make train` runs as one (isolated creds + state);
  `make account-status` shows live VMs per account; `runs/runlog.jsonl`
  logs every launch (account, gpu, exit). Default to `pro`; free accounts
  are T4-at-best/CPU-only.

## Failure playbook

| Symptom | Fix |
| --- | --- |
| 403 on `colab.pa.googleapis.com` | Missing `colaboratory` scope → `make auth-colab` |
| `colab new` 400 with GPU | No entitlement for that accelerator → `--gpu T4` or CPU |
| Session 404/401 on exec | VM was pruned → `colab sessions`, re-provision |
| OOM during training | Lower `PER_DEVICE_BATCH`, raise `GRAD_ACCUM`, shorter `MAX_LENGTH`, or smaller base model |
| HF 401 on push | Token lacks write scope → new token at hf.co/settings/tokens |
| `jupyter_kernel_client has no attribute 'KernelClient'` | colab-cli 0.6.0 vs jupyter-kernel-client 1.x rename → `uv tool install --force google-colab-cli --with 'jupyter-kernel-client==0.15.0'` |
| `TimeoutError: Timeout waiting for output` | `colab run`/`exec` default execution timeout is 30s → pass `--timeout` (launch.sh uses `COLAB_RUN_TIMEOUT`, default 4h) |
