## Configs
This project is configuration-driven. The main config files control the experiment, model assignments, prompts, and player definitions.

### models.yaml
Defines:
- which providers/models are available
- which environment variable holds each provider's API key
- generation settings (temperature, token limits, timeout)
- how many players use each model

### experiment.yaml
Defines the market setup:
- number of players and goods
- number of rounds (`round_multiplier` × full round-robin cycle length, or `max_rounds_override` to cap it at a fixed number)
- matching/pairing rule
- mechanism settings (sequential vs. simultaneous execution, broadcast, anonymity)
- random seed and experiment name

Key fields relevant to the run modes:

**`rounds.max_rounds_override`** — set to an integer to cap the number of rounds regardless of `round_multiplier`. Useful for quick tests (e.g. `max_rounds_override: 1` for a single-round run) without editing the CLI.

**`experiment.seed`** — integer seed for the pairing schedule and (in `--random` mode) the random trade draws. When `--num-runs N` is used from the CLI, the seed is incremented by 1 per run (seed, seed+1, …, seed+N-1) so each run produces a genuinely different sequence.

**`preference_drift.probe_schedule`** — controls when mid-run preference probes fire during a `--run` or `--mock-run` experiment. Two modes:
- `mode: midpoint` — fires once after the middle round, computed as `(total_rounds + 1) // 2`. Auto-adjusts if you change `round_multiplier` or `num_players`.
- `mode: interval_rounds` — fires every `interval_rounds` completed rounds (recurring). Use `interval_rounds: 1` to probe after every round.

These mid-run probes are separate from `--probe-only` mode, which runs probes with no trading at all.

### players.yaml
Defines each agent:
- display name, role, and starting inventory
- `utility_weights` — the exponents used in the shifted Cobb-Douglas utility function `(A+1)^a * (B+1)^b * (C+1)^c`. **This is what the code actually uses for all utility calculations.**
- `utility_formula` — human-readable string for documentation only; never parsed or executed. Must be kept consistent with `utility_weights` manually.
- `preference_description` — natural language description of the agent's preferences, injected into every LLM prompt. Should also be consistent with `utility_weights`.

### prompts.yaml
Defines all prompt templates:
- system prompt and persona template
- negotiation prompts (opening message and response turns)
- commitment prompt (accept/reject a specific offer)
- preference elicitation prompt (the five-question probe sent in `--probe-only` and mid-run probes)
- JSON response format schemas for each phase

### eval.yaml

