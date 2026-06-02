# LLM Barter Experiment

## Environment Setup

This project supports both **Conda** and **pip** workflows.  
Most users should use the **Conda** setup.

### Option 1: Conda (recommended)
Create and activate the environment
```bash
conda env create -f environment.yml
conda activate barter
```

### Option 2: Pip for non conda users, install the dependencies with: 
```bash
pip install -r requirements.txt
```
Analysis code requires jupyter notebook, register the environment as a notebook kernel:
```bash
python -m ipykernel install --user --name barter --display-name "barter"
```

## API Keys
API keys should not be stored directly in the repository or inside config files.
Instead:
Create a local .env file in the project root, and put your real API keys there.
Make sure .env is ignored by Git through .gitignore.

An example template is provided in .env.example.
Copy it and create your own private .env file with the same variable names. Just change the API keys there. 

## Run code (src)

All runs are launched from the project root via `src/main.py`. A **run mode** flag is required; all other flags are optional.

### Run modes

| Flag | Description | API key required |
|------|-------------|-----------------|
| `--dry-run` | Validate configs and preview the pairing schedule. Nothing is written to disk. | No |
| `--mock-run` | Full experiment with deterministic rule-based agents (no LLM calls). Good for testing the pipeline. | No |
| `--run` | Full experiment with real LLM agents. | Yes |
| `--random` | Random-baseline run: every agent proposes a uniformly random feasible 1-for-1 trade each round and always accepts. No intelligence, no LLM calls. Use this to establish a welfare baseline that LLM agents should outperform. | No |
| `--probe-only` | Repeated preference probes with no trading at all. Measures whether LLM agents' stated preferences drift over repeated questioning in isolation. | Yes |

### Quick-start examples

```bash
# Sanity-check configs
python src/main.py --dry-run

# Fast offline test of the full pipeline
python src/main.py --mock-run

# Real LLM experiment
python src/main.py --run

# Random welfare baseline (10 independent runs for averaging)
python src/main.py --random --num-runs 10

# Preference drift — 15 probes per agent, no prior-answer context
python src/main.py --probe-only --probe-only-count 15

# Preference drift — 15 probes per agent, each probe sees all prior answers
python src/main.py --probe-only --probe-only-count 15 --probe-only-context
```

### All flags

#### Mode-specific

| Flag | Default | Description |
|------|---------|-------------|
| `--probe-only-count N` | `10` | Number of probe iterations per player in `--probe-only` mode. |
| `--probe-only-context` | off | In `--probe-only` mode, feed each player's previous probe responses into the context window of subsequent probes. Without this flag every probe is a fresh independent call (the model has no memory of prior answers). With this flag the conversation grows as `[system, Q1, A1, Q2, A2, …, QN]`, letting you compare drift with and without self-memory. Output directories are suffixed `_probe_only` vs `_probe_only_context` so the two conditions are always kept separate. |
| `--num-runs N` | `1` | Execute the chosen mode N times back-to-back. Each run produces its own output directory; the random seed is incremented by 1 per run so pairing schedules and random draws genuinely differ across runs. Useful for averaging out variance — especially with `--random` and `--run`. Ignored by `--dry-run`. |

#### Config paths (rarely need changing)

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `configs/experiment.yaml` | Path to experiment config. |
| `--models` | `configs/models.yaml` | Path to model config. |
| `--players` | `configs/players.yaml` | Path to player config. |
| `--env` | `.env` | Path to the file containing API keys. |
| `--skip-api-key-check` | off | Load configs without requiring API keys to be present. Useful when running `--mock-run` on a machine that has no `.env` file. |

### Output

Every run writes to a timestamped subdirectory under `runs/`:

```
runs/
  20240601_143022_my_experiment/
    transcripts.txt          # human-readable play-by-play
    trades.json              # one record per negotiated pair
    events.jsonl             # structured event stream (one JSON object per line)
    summary.json             # final welfare summary
    preference_probes.json   # all preference probe responses
    config_snapshot/         # copy of all config files used for this run
```

With `--num-runs 3` you get three sibling directories suffixed `_run_1`, `_run_2`, `_run_3`.

