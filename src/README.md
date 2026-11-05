# src

Core Python source files for the LLM Barter Experiment.

## Entry point

**`main.py`** — CLI entry point. See the [root README](../README.md) for the full list of run modes and flags.

## Source files

| File | Purpose |
|------|---------|
| `main.py` | Argument parsing, the `--num-runs` loop (seed / name / display-order per run), and run-mode dispatch. |
| `runner.py` | Experiment loops for all modes (`run_llm_experiment`, `run_mock_experiment`, `run_random_experiment`, `run_probe_only_experiment`), the shared `_run_experiment_loop` used by the first three, trade validation/execution, broadcast/washout logic, and shadow-trade execution. |
| `config.py` | Pydantic models for loading and validating all YAML configs and API keys. |
| `prompt_render.py` | Builds the message lists sent to the LLM for each phase (world state, negotiation, commitment, preference probe), including the `*_cacheable` variants used by `claude_agent.py`. |
| `openai_agent.py` | OpenAI agent functions, plus the provider-agnostic JSON parser, response validators and parse-retry helper reused by the other two agents. |
| `gemini_agent.py` | Gemini agent functions (same interface as `openai_agent.py`). |
| `claude_agent.py` | Claude agent functions (same interface as `openai_agent.py`), with prompt caching. |
| `mock_agent.py` | Rule-based deterministic agents used by `--mock-run`. |
| `shadow_trades.py` | Generates the battery of hypothetical shadow-trade offers from config and filters it to what a player can afford. |
| `display_order.py` | Run-wide counterbalancing of the order goods are shown in (6-run cycle over the permutations of A/B/C). |
| `pairing.py` | Round-robin disjoint-pair schedule generation. |
| `logger.py` | Structured run logging (`RunLogger`). Writes `events.jsonl`, `trades.json`, `transcripts.txt`, `summary.json`, `preference_probes.json`, `shadow_trades.json`, `raw_prompts.jsonl`, `raw_model_outputs.jsonl`. |
| `utility.py` | Shifted Cobb-Douglas utility, predicted marginal utilities, and trade/inventory helpers. |

The three provider modules export the same function names (`gpt_negotiation_action`, `gpt_commitment_decision`, `gpt_preference_probe`, `gpt_preference_probe_contextual`; the `gpt_` prefix is historical), so `runner.run_llm_experiment` loads whichever one is requested via `_PROVIDER_REGISTRY`.

## Run modes in runner.py

### `run_llm_experiment(cfg, provider)` (`--run`, `--run-gemini`, `--run-claude`)
Full experiment with real LLM agents. `provider` is `"gpt"`, `"gemini"` or `"claude"` and must match an enabled `model_id` in `models.yaml`. Agents negotiate in natural language, make commitment decisions, and are probed for their preferences on the configured schedule (with shadow trades if enabled).

### `run_mock_experiment` (`--mock-run`)
Same pipeline as `run_llm_experiment`, but all agent decisions are made by deterministic rule-based callables in `mock_agent.py`. No API calls. Useful for validating the pipeline and testing config changes quickly.

### `run_random_experiment` (`--random`)
No-intelligence welfare baseline. In each pair the first speaker proposes one uniformly random feasible 1-for-1 trade (sampled from every `(give_good, receive_good)` pair both sides can currently fulfil, ignoring preferences) and the partner always accepts. Preference probes are disabled. Use `--num-runs N` to collect multiple seeds for averaging.

### `run_probe_only_experiment` (`--probe-only`)
Probes every agent `--probe-only-count` times (default 10) with no trading. Used to measure whether LLM stated preferences drift under repeated self-questioning in isolation, as a baseline for drift observed during trading runs. Always uses the OpenAI `gpt` model.

Two context conditions are available:

- **No context** (default): each probe is an independent fresh API call — `[system, user]`. The model has no memory of prior answers.
- **With context** (`--probe-only-context`): each probe extends a growing per-player conversation — `[system, Q1, A1, Q2, A2, …, QN]`. The model sees all its previous answers and may anchor to or diverge from them.

Output directories are suffixed `_probe_only` or `_probe_only_context` so the two conditions are always stored separately.
