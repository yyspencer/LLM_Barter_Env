"""
config.py

Loads and validates:
- experiment.yaml
- models.yaml
- players.yaml
- .env API keys

Prompts are intentionally optional/deferred for now.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


# ------------------------------------------------------------
# Basic YAML loading
# ------------------------------------------------------------

def load_yaml(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML object/dict: {path}")

    return data


# ------------------------------------------------------------
# Model config schemas
# ------------------------------------------------------------

class GenerationConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 400
    timeout_seconds: int = 60


class ModelSpec(BaseModel):
    id: str
    enabled: bool = True
    provider: str
    model: str
    api_env_var: str
    count: int = Field(ge=0)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    # Filled after loading .env / environment
    api_key: Optional[str] = None


class ModelAssignmentConfig(BaseModel):
    mode: str = "expand_by_count"


class ModelsConfig(BaseModel):
    version: int
    model_pool: List[ModelSpec]
    assignment: ModelAssignmentConfig = Field(default_factory=ModelAssignmentConfig)

    @model_validator(mode="after")
    def validate_model_ids_unique(self) -> "ModelsConfig":
        ids = [m.id for m in self.model_pool]
        duplicates = {x for x in ids if ids.count(x) > 1}
        if duplicates:
            raise ValueError(f"Duplicate model ids found: {duplicates}")
        return self

    def enabled_models(self) -> List[ModelSpec]:
        return [m for m in self.model_pool if m.enabled]

    def model_ids(self) -> set[str]:
        return {m.id for m in self.enabled_models()}

    def get_model(self, model_id: str) -> ModelSpec:
        for model in self.enabled_models():
            if model.id == model_id:
                return model
        raise KeyError(f"Model id not found or not enabled: {model_id}")


# ------------------------------------------------------------
# Experiment config schemas
# ------------------------------------------------------------

class MarketConfig(BaseModel):
    goods: List[str]
    num_players: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_goods_unique(self) -> "MarketConfig":
        if len(self.goods) != len(set(self.goods)):
            raise ValueError(f"Duplicate goods found: {self.goods}")
        return self


class RoundsConfig(BaseModel):
    mode: str = "round_robin"
    round_multiplier: int = Field(default=1, ge=1)
    max_rounds_override: Optional[int] = None


class PairingConfig(BaseModel):
    mode: str = "round_robin_disjoint_pairs"
    allow_repeat_pairings_within_cycle: bool = False
    reshuffle_between_runs: bool = True
    handle_odd_player_count: str = "bye"


class MechanismConfig(BaseModel):
    execution_mode: str = "parallel_pairs"
    synchronization: str = "end_of_round"
    negotiation_turns_per_agent: int = Field(default=5, ge=1)
    action_space: str = "one_for_one"
    allow_no_trade: bool = True
    require_mutual_acceptance: bool = True
    enforce_inventory_constraints: bool = True
    integer_goods_only: bool = True
    anonymous: bool = False
    broadcast_completed_trades: bool = False
    mediator_enabled: bool = False


class TradeRulesConfig(BaseModel):
    require_mutual_acceptance: bool = True
    allow_counteroffers: bool = True
    enforce_inventory_constraints: bool = True
    integer_goods_only: bool = True
    min_units_per_side: int = Field(default=1, ge=0)
    max_units_per_side: int = Field(default=1, ge=1)
    conflict_resolution: Dict[str, Any] = Field(default_factory=dict)


class TerminalUtilityConfig(BaseModel):
    type: str = "shifted_cobb_douglas"
    shift: float = 1.0
    normalize_weights: bool = True


class UtilityConfig(BaseModel):
    terminal_utility: TerminalUtilityConfig


class ProbeScheduleConfig(BaseModel):
    mode: str = "interval_rounds"
    interval_rounds: int = Field(default=5, ge=1)
    include_pre_probe: bool = True
    include_post_probe: bool = True


class PreferenceDriftConfig(BaseModel):
    enabled: bool = True
    probe_schedule: ProbeScheduleConfig
    save_probe_responses: bool = True


class InitializationModeConfig(BaseModel):
    mode: str
    require_weights_sum_to_one: Optional[bool] = None
    require_nonnegative_integer_entries: Optional[bool] = None
    avoid_initial_optimal_bundle: Optional[bool] = None
    require_gains_from_trade_exist: Optional[bool] = None
    include_exact_formula_in_prompt: Optional[bool] = None


class InitializationConfig(BaseModel):
    utilities: InitializationModeConfig
    endowments: InitializationModeConfig
    personas: InitializationModeConfig


class StoppingConfig(BaseModel):
    stop_after_all_rounds: bool = True
    stop_if_no_trades_for_n_rounds: Optional[int] = None
    stop_if_market_converged: bool = False


class LoggingConfig(BaseModel):
    output_dir: str = "runs"
    save_config_snapshot: bool = True
    save_event_jsonl: bool = True
    save_trade_json: bool = True
    save_summary_json: bool = True
    save_raw_prompts: bool = True
    save_raw_model_outputs: bool = True
    save_preference_probes: bool = True
    pretty_print_transcripts: bool = True
    filenames: Dict[str, str] = Field(default_factory=dict)


class ExperimentMetaConfig(BaseModel):
    name: str
    description: str = ""
    seed: int = 42
    num_runs: int = Field(default=1, ge=1)


class ExperimentConfig(BaseModel):
    version: int
    experiment: ExperimentMetaConfig
    market: MarketConfig
    rounds: RoundsConfig
    pairing: PairingConfig
    mechanism: MechanismConfig
    trade_rules: TradeRulesConfig
    utility: UtilityConfig
    preference_drift: PreferenceDriftConfig
    initialization: InitializationConfig
    stopping: StoppingConfig
    logging: LoggingConfig


# ------------------------------------------------------------
# Player config schemas
# ------------------------------------------------------------

class PlayerSpec(BaseModel):
    id: str
    display_name: str
    model_id: str
    inventory: Dict[str, int]
    utility_weights: Dict[str, float]
    preference_description: str

    # Filled after linking to models.yaml
    model: Optional[ModelSpec] = None


class PlayersConfig(BaseModel):
    version: int
    players: List[PlayerSpec]

    @model_validator(mode="after")
    def validate_player_ids_unique(self) -> "PlayersConfig":
        ids = [p.id for p in self.players]
        duplicates = {x for x in ids if ids.count(x) > 1}
        if duplicates:
            raise ValueError(f"Duplicate player ids found: {duplicates}")
        return self


# ------------------------------------------------------------
# Combined loaded config
# ------------------------------------------------------------

class LoadedConfig(BaseModel):
    experiment: ExperimentConfig
    models: ModelsConfig
    players: PlayersConfig


# ------------------------------------------------------------
# Cross-file validation
# ------------------------------------------------------------

def attach_api_keys(models: ModelsConfig) -> None:
    """
    Load real API keys from environment variables.

    The YAML only stores env var names, never real secrets.
    """
    for model in models.enabled_models():
        api_key = os.getenv(model.api_env_var)

        if not api_key:
            raise ValueError(
                f"Missing API key for model '{model.id}'. "
                f"Expected environment variable: {model.api_env_var}. "
                f"Check your .env file."
            )

        model.api_key = api_key


def validate_players_against_experiment(
    experiment: ExperimentConfig,
    players: PlayersConfig,
) -> None:
    goods = set(experiment.market.goods)

    if len(players.players) != experiment.market.num_players:
        raise ValueError(
            f"market.num_players={experiment.market.num_players}, "
            f"but players.yaml defines {len(players.players)} players."
        )

    for player in players.players:
        inv_goods = set(player.inventory.keys())
        weight_goods = set(player.utility_weights.keys())

        missing_inv = goods - inv_goods
        extra_inv = inv_goods - goods
        if missing_inv or extra_inv:
            raise ValueError(
                f"Player {player.id} inventory goods mismatch. "
                f"Missing: {missing_inv}, Extra: {extra_inv}"
            )

        missing_weights = goods - weight_goods
        extra_weights = weight_goods - goods
        if missing_weights or extra_weights:
            raise ValueError(
                f"Player {player.id} utility weight goods mismatch. "
                f"Missing: {missing_weights}, Extra: {extra_weights}"
            )

        for good, amount in player.inventory.items():
            if not isinstance(amount, int) or amount < 0:
                raise ValueError(
                    f"Player {player.id} has invalid inventory amount: "
                    f"{good}={amount}. Must be nonnegative integer."
                )

        weight_sum = sum(player.utility_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Player {player.id} utility weights must sum to 1. "
                f"Current sum={weight_sum}"
            )

        for good, weight in player.utility_weights.items():
            if weight < 0:
                raise ValueError(
                    f"Player {player.id} has negative utility weight: "
                    f"{good}={weight}"
                )


def validate_players_against_models(
    models: ModelsConfig,
    players: PlayersConfig,
) -> None:
    model_ids = models.model_ids()

    for player in players.players:
        if player.model_id not in model_ids:
            raise ValueError(
                f"Player {player.id} references model_id='{player.model_id}', "
                f"but this model id is not enabled/found in models.yaml. "
                f"Available enabled model ids: {sorted(model_ids)}"
            )

        player.model = models.get_model(player.model_id)


def validate_model_counts_against_players(
    models: ModelsConfig,
    players: PlayersConfig,
) -> None:
    """
    Optional sanity check:
    If models.yaml says count=1 for gpt, players.yaml should reference gpt once.
    """
    actual_counts: Dict[str, int] = {}
    for player in players.players:
        actual_counts[player.model_id] = actual_counts.get(player.model_id, 0) + 1

    for model in models.enabled_models():
        actual = actual_counts.get(model.id, 0)
        expected = model.count

        if actual != expected:
            raise ValueError(
                f"Model count mismatch for model_id='{model.id}'. "
                f"models.yaml count={expected}, but players.yaml uses it {actual} times."
            )


def validate_parallel_pairing_settings(experiment: ExperimentConfig) -> None:
    """
    Basic consistency checks for parallel pair mode.
    """
    if experiment.mechanism.execution_mode == "parallel_pairs":
        if experiment.pairing.mode != "round_robin_disjoint_pairs":
            raise ValueError(
                "mechanism.execution_mode='parallel_pairs' requires "
                "pairing.mode='round_robin_disjoint_pairs' for v1."
            )

        if experiment.mechanism.synchronization != "end_of_round":
            raise ValueError(
                "parallel_pairs execution should use synchronization='end_of_round'."
            )


def validate_all(config: LoadedConfig) -> None:
    validate_players_against_experiment(config.experiment, config.players)
    validate_players_against_models(config.models, config.players)
    validate_model_counts_against_players(config.models, config.players)
    validate_parallel_pairing_settings(config.experiment)


# ------------------------------------------------------------
# Public loading function
# ------------------------------------------------------------

def load_config(
    experiment_path: str | Path = "configs/experiment.yaml",
    models_path: str | Path = "configs/models.yaml",
    players_path: str | Path = "configs/players.yaml",
    env_path: str | Path = ".env",
    require_api_keys: bool = True,
) -> LoadedConfig:
    """
    Load all required config files for v1.

    prompts.yaml is intentionally skipped for now.
    """

    env_path = Path(env_path)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Not always fatal because keys may already be in the OS environment.
        load_dotenv()

    experiment_raw = load_yaml(experiment_path)
    models_raw = load_yaml(models_path)
    players_raw = load_yaml(players_path)

    experiment = ExperimentConfig.model_validate(experiment_raw)
    models = ModelsConfig.model_validate(models_raw)
    players = PlayersConfig.model_validate(players_raw)

    if require_api_keys:
        attach_api_keys(models)

    config = LoadedConfig(
        experiment=experiment,
        models=models,
        players=players,
    )

    validate_all(config)

    return config


# ------------------------------------------------------------
# Dry-run helper
# ------------------------------------------------------------

def print_config_summary(config: LoadedConfig) -> None:
    exp = config.experiment

    print("=" * 60)
    print("CONFIG VALIDATION PASSED")
    print("=" * 60)

    print(f"Experiment: {exp.experiment.name}")
    print(f"Description: {exp.experiment.description}")
    print(f"Seed: {exp.experiment.seed}")
    print(f"Runs: {exp.experiment.num_runs}")

    print("\nMarket")
    print(f"  Players: {exp.market.num_players}")
    print(f"  Goods: {', '.join(exp.market.goods)}")

    print("\nMechanism")
    print(f"  Pairing: {exp.pairing.mode}")
    print(f"  Execution: {exp.mechanism.execution_mode}")
    print(f"  Synchronization: {exp.mechanism.synchronization}")
    print(f"  Negotiation turns per agent: {exp.mechanism.negotiation_turns_per_agent}")

    print("\nModels")
    for model in config.models.enabled_models():
        print(
            f"  {model.id}: provider={model.provider}, "
            f"model={model.model}, count={model.count}, "
            f"api_env_var={model.api_env_var}, "
            f"api_key_loaded={'yes' if model.api_key else 'no'}"
        )

    print("\nPlayers")
    for player in config.players.players:
        print(f"  {player.id} ({player.display_name})")
        print(f"    model_id: {player.model_id}")
        print(f"    inventory: {player.inventory}")
        print(f"    utility_weights: {player.utility_weights}")

    print("=" * 60)


if __name__ == "__main__":
    cfg = load_config(
        experiment_path="configs/experiment.yaml",
        models_path="configs/models.yaml",
        players_path="configs/players.yaml",
        env_path=".env",
        require_api_keys=True,
    )
    print_config_summary(cfg)