"""
logger.py

Logging utilities for the LLM barter experiment.

Purpose:
- Create a timestamped run directory.
- Save config snapshots for reproducibility.
- Write structured JSONL events.
- Save trade logs, preference probe logs, summaries, and human-readable transcripts.

This module does not call any LLM APIs and does not run the experiment.
It provides reusable logging functions for runner.py.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


JsonDict = Dict[str, Any]


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def filesystem_timestamp() -> str:
    """Return a filesystem-safe timestamp for run directory names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """Write an object as pretty JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as one JSONL line."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_text(path: str | Path, text: str) -> None:
    """Append text to a UTF-8 text file."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def safe_model_dump(obj: Any) -> Any:
    """
    Convert Pydantic models / dataclasses / plain objects into JSON-safe objects.

    Important:
    - API keys are redacted.
    """
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
    elif hasattr(obj, "__dict__"):
        data = dict(obj.__dict__)
    else:
        return obj

    return redact_api_keys(data)


def redact_api_keys(data: Any) -> Any:
    """
    Recursively remove API key values from data before saving config snapshots.

    This protects against accidentally writing real secrets into run outputs.
    """
    if isinstance(data, dict):
        redacted = {}
        for key, value in data.items():
            lower_key = str(key).lower()
            if lower_key in {"api_key", "apikey", "secret", "token"} or "api_key" in lower_key:
                # Keep env var names but redact actual loaded API key values.
                if lower_key == "api_env_var":
                    redacted[key] = value
                else:
                    redacted[key] = "***REDACTED***"
            else:
                redacted[key] = redact_api_keys(value)
        return redacted

    if isinstance(data, list):
        return [redact_api_keys(x) for x in data]

    return data


@dataclass
class RunLogger:
    """
    Logger for a single experiment run.

    Expected output structure:

    runs/
      <run_id>/
        config_snapshot/
          experiment.yaml
          models.yaml
          players.yaml
          prompts.yaml
          loaded_config.redacted.json
        events.jsonl
        trades.json
        preference_probes.json
        summary.json
        transcripts.txt
        raw_prompts.jsonl
        raw_model_outputs.jsonl
    """

    output_dir: Path
    run_id: str
    filenames: Mapping[str, str] = field(default_factory=dict)

    events: list[JsonDict] = field(default_factory=list)
    trades: list[JsonDict] = field(default_factory=list)
    preference_probes: list[JsonDict] = field(default_factory=list)
    summary: JsonDict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    @property
    def config_snapshot_dir(self) -> Path:
        return self.run_dir / "config_snapshot"

    def path_for(self, key: str, default_filename: str) -> Path:
        filename = self.filenames.get(key, default_filename)
        return self.run_dir / filename

    @classmethod
    def create(
        cls,
        output_dir: str | Path = "runs",
        experiment_name: Optional[str] = None,
        filenames: Optional[Mapping[str, str]] = None,
    ) -> "RunLogger":
        """
        Create a new timestamped logger.

        run_id format:
            <timestamp>_<experiment_name>
        """
        timestamp = filesystem_timestamp()
        clean_name = sanitize_name(experiment_name or "run")
        run_id = f"{timestamp}_{clean_name}"

        logger = cls(
            output_dir=Path(output_dir),
            run_id=run_id,
            filenames=filenames or {},
        )
        ensure_dir(logger.run_dir)
        ensure_dir(logger.config_snapshot_dir)
        return logger

    def save_config_files(
        self,
        config_paths: Iterable[str | Path],
        loaded_config: Optional[Any] = None,
    ) -> None:
        """
        Copy raw YAML config files into config_snapshot/ and optionally save
        a redacted JSON dump of the loaded config object.
        """
        ensure_dir(self.config_snapshot_dir)

        for path in config_paths:
            path = Path(path)
            if path.exists():
                shutil.copy2(path, self.config_snapshot_dir / path.name)

        if loaded_config is not None:
            write_json(
                self.config_snapshot_dir / "loaded_config.redacted.json",
                safe_model_dump(loaded_config),
            )

    def log_event(
        self,
        event_type: str,
        payload: Optional[Mapping[str, Any]] = None,
        round_index: Optional[int] = None,
        player_id: Optional[str] = None,
        pair_id: Optional[str] = None,
    ) -> JsonDict:
        """
        Log one structured event.

        Events are appended immediately to events.jsonl and also kept in memory.
        """
        event = {
            "timestamp": utc_timestamp(),
            "event_type": event_type,
            "round_index": round_index,
            "player_id": player_id,
            "pair_id": pair_id,
            "payload": dict(payload or {}),
        }

        self.events.append(event)
        append_jsonl(self.path_for("events", "events.jsonl"), event)
        return event

    def log_prompt(
        self,
        player_id: str,
        prompt_type: str,
        messages: list[Mapping[str, str]],
        round_index: Optional[int] = None,
        pair_id: Optional[str] = None,
    ) -> None:
        """
        Log rendered prompts sent to a model.

        This is separate from events.jsonl because prompts can be large.
        """
        record = {
            "timestamp": utc_timestamp(),
            "round_index": round_index,
            "pair_id": pair_id,
            "player_id": player_id,
            "prompt_type": prompt_type,
            "messages": messages,
        }
        append_jsonl(self.path_for("raw_prompts", "raw_prompts.jsonl"), record)

    def log_model_output(
        self,
        player_id: str,
        output_type: str,
        raw_output: str,
        parsed_output: Optional[Mapping[str, Any]] = None,
        round_index: Optional[int] = None,
        pair_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Log raw and optionally parsed model output.
        """
        record = {
            "timestamp": utc_timestamp(),
            "round_index": round_index,
            "pair_id": pair_id,
            "player_id": player_id,
            "provider": provider,
            "model": model,
            "output_type": output_type,
            "raw_output": raw_output,
            "parsed_output": dict(parsed_output or {}) if parsed_output else None,
        }
        append_jsonl(self.path_for("raw_model_outputs", "raw_model_outputs.jsonl"), record)

    def log_trade(self, trade_record: Mapping[str, Any]) -> None:
        """
        Add one trade record to the in-memory trade list and event log.

        Call finalize() to write trades.json.
        """
        record = {
            "timestamp": utc_timestamp(),
            **dict(trade_record),
        }
        self.trades.append(record)
        self.log_event(
            event_type="trade",
            payload=record,
            round_index=record.get("round_index"),
            pair_id=record.get("pair_id"),
        )

    def log_preference_probe(self, probe_record: Mapping[str, Any]) -> None:
        """
        Add one preference probe record to memory and event log.

        Call finalize() to write preference_probes.json.
        """
        record = {
            "timestamp": utc_timestamp(),
            **dict(probe_record),
        }
        self.preference_probes.append(record)
        self.log_event(
            event_type="preference_probe",
            payload=record,
            round_index=record.get("round_index"),
            player_id=record.get("player_id"),
        )

    def append_transcript(self, text: str) -> None:
        """
        Append human-readable transcript text.
        """
        if text and not text.endswith("\n"):
            text += "\n"
        append_text(self.path_for("transcripts", "transcripts.txt"), text)

    def set_summary(self, summary: Mapping[str, Any]) -> None:
        """Set the final summary object."""
        self.summary = dict(summary)

    def finalize(self) -> None:
        """
        Write final aggregate JSON files.

        This should be called at the end of a run.
        """
        write_json(self.path_for("trades", "trades.json"), self.trades)
        write_json(self.path_for("preference_probes", "preference_probes.json"), self.preference_probes)
        write_json(self.path_for("summary", "summary.json"), self.summary)

        self.log_event(
            event_type="run_finalized",
            payload={
                "num_events": len(self.events),
                "num_trades": len(self.trades),
                "num_preference_probes": len(self.preference_probes),
                "summary_path": str(self.path_for("summary", "summary.json")),
            },
        )


def sanitize_name(name: str) -> str:
    """
    Convert a run name into a filesystem-safe string.
    """
    allowed = []
    for ch in name:
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        elif ch.isspace():
            allowed.append("_")
    cleaned = "".join(allowed).strip("_")
    return cleaned or "run"


def build_pair_id(round_index: int, player_a: str, player_b: str) -> str:
    """Build a stable pair id for logs."""
    return f"round_{round_index}_{player_a}_vs_{player_b}"


if __name__ == "__main__":
    # Optional smoke test. This block only runs if executing:
    #   python src/logger.py
    logger = RunLogger.create(output_dir="runs", experiment_name="logger_smoke_test")
    logger.log_event("smoke_test_started", {"message": "Logger is working."})
    logger.log_trade(
        {
            "round_index": 1,
            "pair_id": "round_1_player_1_vs_player_2",
            "accepted": True,
            "players": ["player_1", "player_2"],
            "trade": {
                "player_1_gives": {"C": 1},
                "player_2_gives": {"A": 1},
            },
        }
    )
    logger.log_preference_probe(
        {
            "round_index": 0,
            "player_id": "player_1",
            "ratings": {"A": 10, "B": 5, "C": 1},
            "desired_bundle_6_units": {"A": 5, "B": 1, "C": 0},
        }
    )
    logger.set_summary({"status": "ok"})
    logger.finalize()
    print(f"Smoke test run written to: {logger.run_dir}")
