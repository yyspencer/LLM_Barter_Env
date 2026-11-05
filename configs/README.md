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
- number of rounds (`round_multiplier` × full round-robin cycle length, or `max_rounds_override` to set a fixed number)
- optional washout rounds appended after the main schedule
- matching/pairing rule
- mechanism settings (action space, negotiation turns, broadcast / bulletin board)
- preference probe schedule and shadow trades
- random seed, experiment name, and output directory

Key fields relevant to the run modes:

**`rounds.max_rounds_override`** — set to an integer to cap the number of rounds regardless of `round_multiplier`. Useful for quick tests (e.g. `max_rounds_override: 1` for a single-round run) without editing the CLI.

**`experiment.seed`** — integer seed for the pairing schedule and (in `--random` mode) the random trade draws. When `--num-runs N` is used from the CLI, the seed is incremented by 1 per run (seed, seed+1, …, seed+N-1) so each run produces a genuinely different sequence.

**`preference_drift.probe_schedule`** — controls when mid-run preference probes fire during a `--run` or `--mock-run` experiment. Two modes:
- `mode: midpoint` — fires once after the middle round, computed as `(total_rounds + 1) // 2`. Auto-adjusts if you change `round_multiplier` or `num_players`.
- `mode: interval_rounds` — fires every `interval_rounds` completed rounds (recurring). Use `interval_rounds: 1` to probe after every round.

`include_pre_probe` / `include_post_probe` add a probe before the first round and after the last. If the last round is also an interval probe point, turning on `include_post_probe` probes twice at the same state.

These mid-run probes are separate from `--probe-only` mode, which runs probes with no trading at all.

**`washout`** — when `enabled`, appends `num_rounds` extra rounds after the main schedule. `disable_broadcast: true` hides the bulletin board during those rounds. `probe_after_washout_rounds` lists 1-indexed positions within the washout block after which to probe (e.g. `[2, 4]`).

**`mechanism.action_space`** — `one_for_one` (exactly 1 unit for 1 unit), `small_bundle` (one good type per side, any quantity), or `any_bundle` (any mix of goods per side). Quantities are further limited by `trade_rules.min_units_per_side` / `max_units_per_side` and inventories.

**Broadcast** — `mechanism.broadcast_completed_trades` turns on the market bulletin board. `broadcast_filter_players` restricts which trades are posted (by participant). `bulletin_rules` picks a template from `prompts.yaml` by who traded and what they gained. If rules are set, trades matching none of them are not posted; `market_bulletin_template` is only used when no rules are set.

**`preference_drift.shadow_trades`** — hypothetical accept/reject offers sent alongside every probe to the players in `player_ids`. They are never executed or shown again. `focal_goods`, `counter_goods`, `counter_quantities` and `directions` define the battery of offers.

**`logging.output_dir`** — where run folders are written.

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
- preference elicitation prompt (the three-question probe sent in all preference probes)
- market bulletin templates (`market_bulletin_template`, optional `bulletin_rules` templates, `static_broadcast_message`)
- JSON response format schemas for each phase and per-action-space offer examples
- action-space descriptions

Templates can use `{good_1}`, `{good_2}`, `{good_3}`, which are filled in with the run's display order so no fixed A-B-C ordering leaks into prompts.

