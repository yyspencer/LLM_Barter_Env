# src

Core Python source files for the LLM Barter Experiment.

## Entry point

**`main.py`** — CLI entry point. See the [root README](../README.md) for the full list of run modes and flags.

## Source files

| File | Purpose |
|------|---------|
| `main.py` | Argument parsing and run-mode dispatch. |
| `runner.py` | Experiment loops for all modes (`run_gpt_experiment`, `run_mock_experiment`, `run_random_experiment`, `run_probe_only_experiment`) plus the shared `_run_experiment_loop` used by the first two. |
| `config.py` | Pydantic models for loading and validating all YAML configs. |
| `prompt_render.py` | Assembles the message lists sent to the LLM for each phase (world state, negotiation, commitment, preference probe). |
| `openai_agent.py` | OpenAI-specific agent callables (negotiation, commitment, probe, contextual probe). |
| `mock_agent.py` | Rule-based deterministic agents used by `--mock-run`. |
| `pairing.py` | Round-robin pairing schedule generation. |
| `logger.py` | Structured run logging (`RunLogger`). Writes `events.jsonl`, `trades.json`, `transcripts.txt`, `summary.json`, `preference_probes.json`. |
| `utility.py` | Shifted Cobb-Douglas utility function and welfare helpers. |

## Run modes in runner.py

### `run_gpt_experiment` (`--run`)
Full experiment with real LLM agents. Agents negotiate in natural language, make commitment decisions, and are periodically probed for their preferences. Requires `OPENAI_API_KEY`.

### `run_mock_experiment` (`--mock-run`)
Same pipeline as `run_gpt_experiment` but all agent decisions are made by deterministic rule-based callables in `mock_agent.py`. No API calls. Useful for validating the pipeline and testing config changes quickly.

### `run_random_experiment` (`--random`)
No-intelligence welfare baseline. Each pair attempts one uniformly random feasible 1-for-1 trade per round; the responder always accepts. The proposer samples from every `(give_good, receive_good)` pair that both sides can currently fulfil, ignoring preferences entirely. Preference probes are disabled. Use `--num-runs N` to collect multiple seeds for averaging.

### `run_probe_only_experiment` (`--probe-only`)
Probes every agent `--probe-only-count` times (default 10) with no trading. Used to measure whether LLM preference representations drift under repeated self-questioning in isolation, as a baseline for comparing against drift observed during actual trading runs.

Two context conditions are available:

- **No context** (default): each probe is an independent fresh API call — `[system, user]`. The model has no memory of prior answers.
- **With context** (`--probe-only-context`): each probe extends a growing per-player conversation — `[system, Q1, A1, Q2, A2, …, QN]`. The model sees all its previous answers and may anchor to or diverge from them.

Output directories are suffixed `_probe_only` or `_probe_only_context` so the two conditions are always stored separately.