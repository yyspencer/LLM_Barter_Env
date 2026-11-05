# LLM Barter Experiment

A multi-agent barter market for measuring **preference drift** in LLM agents. Six agents (Builders, Weavers, Merchants) hold goods A, B and C, have hidden Cobb-Douglas utility functions described to them in plain language, and negotiate pairwise trades over many rounds. Agents are periodically asked to state their preferences, and the experiment checks whether those stated preferences — and their actual trading behaviour — shift under manipulations such as a public market bulletin board.

Supported providers: OpenAI (GPT), Google Gemini, and Anthropic Claude.

## Repository layout

| Path | Contents |
|------|----------|
| `src/` | Experiment code. Entry point is `src/main.py`. See [src/README.md](src/README.md). |
| `configs/` | YAML configs for the experiment, models, players and prompts. See [configs/README.md](configs/README.md). |
| `analysis/` | Analysis notebooks/scripts and generated figures. See [analysis/README.md](analysis/README.md). |
| `data/` | Saved run outputs used in the analysis, grouped by provider and condition. |
| `test/` | Example single-run outputs per provider/condition. |

## Environment setup

This project supports both **Conda** and **pip** workflows.

### Option 1: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate barter
```

### Option 2: pip
```bash
pip install -r requirements.txt
```

The analysis code uses Jupyter notebooks. Register the environment as a kernel:
```bash
python -m ipykernel install --user --name barter --display-name "barter"
```

## API keys

Never put API keys in the repository or in config files. Instead:

1. Copy `.env.example` to `.env` in the project root (`.env` is git-ignored).
2. Fill in the keys for the providers you use. The variable names must match `api_env_var` in `configs/models.yaml` (`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`).

Config loading checks for a key for **every model that is `enabled: true`** in `models.yaml`, so disable providers you aren't using.

## Running the experiment

All runs are launched from the project root via `src/main.py`. Exactly one **run mode** flag is required.

### Run modes

| Flag | Description | API key required |
|------|-------------|------------------|
| `--dry-run` | Validate configs and print the pairing schedule. Nothing is written to disk. | No |
| `--mock-run` | Full experiment with deterministic rule-based agents (no LLM calls). Good for testing the pipeline. | No |
| `--run` | Full experiment with OpenAI agents (`gpt` entry in `models.yaml`). | `OPENAI_API_KEY` |
| `--run-gemini` | Full experiment with Gemini agents (`gemini` entry). | `GEMINI_API_KEY` |
| `--run-claude` | Full experiment with Claude agents (`claude` entry). | `ANTHROPIC_API_KEY` |
| `--random` | Random welfare baseline: in each pair the first speaker proposes a uniformly random feasible 1-for-1 trade and the partner always accepts. No LLM calls, no preference probes. | No |
| `--probe-only` | Repeated preference probes with no trading, to measure whether stated preferences drift under repeated questioning alone. Always uses the OpenAI `gpt` model. | `OPENAI_API_KEY` |

The chosen provider's entry must be `enabled: true` in `models.yaml`, and every player in `players.yaml` must use a `model_id` that is enabled (the `count` for each model must match how many players use it).

### Quick-start examples

```bash
# Sanity-check configs and preview the pairing schedule
python src/main.py --dry-run

# Fast offline test of the full pipeline
python src/main.py --mock-run

# Real LLM experiments
python src/main.py --run
python src/main.py --run-gemini
python src/main.py --run-claude

# 10 back-to-back runs (different seeds and display orders)
python src/main.py --run-claude --num-runs 10

# Resume a 40-run batch that crashed during run 36
python src/main.py --run-claude --start-run-index 36 --num-runs 5

# Force the broadcast condition on without editing experiment.yaml
python src/main.py --run-gemini --broadcast

# Random welfare baseline (10 runs for averaging)
python src/main.py --random --num-runs 10

# Preference drift with no trading — 15 probes per agent, no prior-answer context
python src/main.py --probe-only --probe-only-count 15

# Same, but each probe sees all of the agent's prior answers
python src/main.py --probe-only --probe-only-count 15 --probe-only-context
```

### All flags

#### Run control

| Flag | Default | Description |
|------|---------|-------------|
| `--num-runs N` | `1` | Execute the chosen mode N times back-to-back. Each run gets its own output directory (suffixed `_run_<n>`), the seed is incremented by 1 per run, and the goods display order rotates (see below). Starting inventories are reset from `players.yaml` before every run. Ignored by `--dry-run`. |
| `--start-run-index N` | `1` | 1-indexed run number to start from. Use it to resume an interrupted batch so seeds, folder numbering and display-order counterbalancing continue where the original batch left off. |
| `--broadcast` | off | Turn on the broadcast condition (`mechanism.broadcast_completed_trades`) regardless of `experiment.yaml`. |
| `--probe-only-count N` | `10` | Number of probe iterations per player in `--probe-only` mode. |
| `--probe-only-context` | off | In `--probe-only` mode, feed each player's previous probe answers into later probes. The conversation grows as `[system, Q1, A1, Q2, A2, …, QN]`. Without this flag every probe is an independent call. Output directories are suffixed `_probe_only` vs `_probe_only_context`. |

#### Config paths (rarely need changing)

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `configs/experiment.yaml` | Path to experiment config. |
| `--models` | `configs/models.yaml` | Path to model config. |
| `--players` | `configs/players.yaml` | Path to player config. |
| `--env` | `.env` | Path to the file containing API keys. |
| `--skip-api-key-check` | off | Load configs without requiring API keys. Keys are only checked for `--run`, `--run-gemini`, `--run-claude` and `--probe-only` anyway. |

`prompts.yaml` is always loaded from `configs/prompts.yaml`, and the config snapshot saved with each run is always copied from the four files in `configs/`, even if other paths are passed.

## How a run works

1. **Pre-run probe.** Every agent answers the preference probe once before any trading (round 0).
2. **Rounds.** Each round pairs every agent with one partner using a round-robin schedule (no repeated pairs within a cycle; player order reshuffled each cycle). The number of rounds is `round_multiplier × (num_players − 1)` for an even player count, or `rounds.max_rounds_override` if set. Pairs are negotiated one after another.
3. **Negotiation.** The two agents alternate turns (up to `negotiation_turns_per_agent` each). On a turn an agent can send a message, make an offer, or end the negotiation (`no_trade` / `reject`).
4. **Commitment.** Every valid offer goes straight to the partner, who must accept or reject it right then. An accepted trade executes immediately and ends the pair's negotiation. Invalid offers (wrong shape, insufficient inventory) are never shown for a decision; both agents get a note explaining why.
5. **Mid-run probes** fire after rounds chosen by `preference_drift.probe_schedule` (`interval_rounds` or `midpoint`), plus any washout probe points.
6. **Washout (optional).** `washout.num_rounds` extra rounds are appended to the schedule. With `washout.disable_broadcast`, the bulletin board is hidden during these rounds, to test whether a manipulation's effects persist once it is removed.
7. **Post-run probe** (if `include_post_probe` is true), then the final summary.

### Experimental manipulations and measurements

- **Broadcast / market bulletin board.** When `mechanism.broadcast_completed_trades` is true, accepted trades produce a bulletin from a `prompts.yaml` template, and the board is shown in negotiation, commitment and probe prompts. Bulletins from round N become visible from round N+1. Trades can be filtered by participant (`broadcast_filter_players`), and `bulletin_rules` can choose a different template depending on who traded and what they gained (if rules are set, trades matching no rule are not broadcast). `prompts.static_broadcast_message` adds a fixed message to the board in every prompt. The board is never shown in the pre-run probe, so the baseline is uncontaminated.
- **Preference probe.** Three questions: rate A/B/C 1–10 given the current inventory, rate them 1–10 setting inventory aside, and choose an ideal 6-unit bundle. Each probe record also stores the marginal utility of each good predicted by the agent's true utility function, for comparison.
- **Shadow trades.** If `preference_drift.shadow_trades.enabled`, selected players also receive a battery of hypothetical accept/reject offers alongside each probe (for example "give 1×B for 2×A?"). These look like real commitment decisions but are never executed and never appear in later prompts. They give extra signal on goods that rarely come up in real trades.
- **Display-order counterbalancing.** The order in which goods are listed in all prompts (and probe JSON schemas) rotates through the 6 permutations of A/B/C by run number (run 1 → A,B,C; run 2 → B,C,A; …), to avoid order bias.

## Output

Each run writes to a timestamped subdirectory of `logging.output_dir` in `experiment.yaml` (default `runs/`; currently set to `C:/ExperimentRuns/LLM_Barter_Env/runs`):

```
<output_dir>/
  20261014_120034_preference_drift_v1_parallel_pairs_run_1/
    transcripts.txt           # human-readable play-by-play
    trades.json               # one record per pair per round (accepted or not)
    events.jsonl              # structured event stream (one JSON object per line)
    summary.json              # final inventories, utilities, welfare change
    preference_probes.json    # every probe response (with predicted marginal utilities)
    shadow_trades.json        # every shadow-trade offer and response
    raw_prompts.jsonl         # every prompt sent to a model (LLM runs only)
    raw_model_outputs.jsonl   # every raw and parsed model reply (LLM runs only)
    config_snapshot/          # copies of the config files + redacted loaded config
```

Directory names are `<timestamp>_<experiment name>`, plus `_run_<n>` when running more than one run, `_random` for the random baseline, and `_probe_only` / `_probe_only_context` for probe-only runs.

## Config fields that are not currently used

These fields are parsed and validated but do not change behaviour:

- `mechanism`: `execution_mode` / `synchronization` (pairs always run sequentially within a round), `anonymous`, `mediator_enabled`, `allow_no_trade`, `require_mutual_acceptance`, `enforce_inventory_constraints`, `integer_goods_only`
- `trade_rules`: everything except `min_units_per_side`, `max_units_per_side` and `enforce_inventory_constraints`
- `utility.terminal_utility.normalize_weights`, `preference_drift.save_probe_responses`, the whole `initialization` block, and `stopping` (except `stop_if_no_trades_for_n_rounds`)
- `logging.save_*` flags (all outputs are always written)
- `players.yaml`'s `utility_formula`, and `prompts.yaml`'s `logging` section

 `utility.terminal_utility.shift` is only used for probe/shadow-trade utility calculations; trade and summary utilities always use a shift of 1.
