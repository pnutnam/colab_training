.PHONY: setup auth-status auth-hf auth-colab smoke train train-keep sessions status stop logs skills-install update account-add account-adopt account-list account-status

# Most targets route through scripts/env.sh, which loads .env.

setup: ## Install local deps + skills; copy .env from example if missing
	@test -f .env || { cp .env.example .env && chmod 600 .env && echo "created .env — edit it to add HF_TOKEN"; }
	uv sync
	@$(MAKE) --no-print-directory skills-install

auth-status: ## Check HuggingFace + Google auth state
	@bash scripts/auth-status.sh

auth-hf: ## Interactive HuggingFace login (prints URL + code, waits for approval)
	hf auth login --format agent

auth-colab: ## One-time Google ADC login (DEFAULT slot only — for multi-account use account-add)
	@echo "Running gcloud ADC login with colaboratory scopes..."
	@echo "(A browser URL will be shown; approve as the same account as your Colab Pro subscription.)"
	gcloud auth application-default login \
		--scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

# Multi-account rotation (scripts/accounts.sh). Pick a run's account with:
#   COLAB_ACCOUNT=<label> make train
account-add: ## Register another Google account: make account-add LABEL=mylabel EMAIL=me@gmail.com
	@test -n "$(LABEL)" || { echo "usage: make account-add LABEL=mylabel EMAIL=me@gmail.com"; exit 1; }
	bash scripts/accounts.sh add "$(LABEL)" "$(EMAIL)"

account-adopt: ## Register the current default ADC as an account: LABEL= EMAIL=
	@test -n "$(LABEL)" && test -n "$(EMAIL)" || { echo "usage: make account-adopt LABEL=pro EMAIL=me@gmail.com"; exit 1; }
	bash scripts/accounts.sh adopt "$(LABEL)" "$(EMAIL)"

account-list: ## List registered accounts
	@bash scripts/accounts.sh list

account-status: ## Live Colab sessions per registered account
	@bash scripts/accounts.sh status

smoke: ## Provision a T4, verify CUDA + HF connectivity, auto-teardown
	@bash scripts/launch.sh --smoke

train: ## Launch QLoRA SFT run on Colab (config via .env; KEEP=1 to retain VM)
	@bash scripts/launch.sh

KEEP ?= 0
train-keep: ## Same as `make train` but keep the VM around afterwards
	KEEP=1 bash scripts/launch.sh

sessions: ## List active Colab sessions
	colab --auth $(COLAB_AUTH) sessions

status: ## Show status of the active session
	colab --auth $(COLAB_AUTH) status

stop: ## Stop the active session (DO THIS when done — idle VMs burn compute units)
	colab --auth $(COLAB_AUTH) stop

logs: ## Tail recent session events
	colab --auth $(COLAB_AUTH) log -n 50

skills-install: ## Install skills into ~/.zcode/skills (and .agents/skills for other agents)
	@bash scripts/install-skills.sh

update: ## Update both CLIs (re-applies the jupyter-kernel-client pin — see README troubleshooting)
	uv tool update hf
	uv tool install --force google-colab-cli --with 'jupyter-kernel-client==0.15.0'
