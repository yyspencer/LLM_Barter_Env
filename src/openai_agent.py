"""
openai_agent.py
 
OpenAI provider wrapper for the LLM barter experiment.
 
Handles:
  - Calling the OpenAI chat completions API (GPT-5.4)
  - JSON parsing with best-effort repair for common model output issues
  - Retry with exponential backoff on transient API errors
  - Logging raw outputs via RunLogger
 
Public interface (mirrors mock_agent.py signatures):
  gpt_negotiation_action(...)   -> dict
  gpt_commitment_decision(...)  -> dict
  gpt_preference_probe(...)     -> dict
 
These are called by runner.py exactly like the mock equivalents, so
swapping providers later only requires a new agent module.
"""
 
from __future__ import annotations
 
import json
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

 
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError
 
from logger import RunLogger
from prompt_render import (
    build_commitment_messages,
    build_negotiation_first_messages,
    build_negotiation_response_messages,
    build_preference_probe_messages,
)
 
 
# ---------------------------------------------------------------------------
# JSON repair helpers
# ---------------------------------------------------------------------------
 
def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences that models sometimes add."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()
 
 
def _extract_first_json_object(text: str) -> str:
    """
    Pull the first {...} block out of a string that has surrounding prose.
    Returns the substring if found, otherwise returns text unchanged.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text
 
 
def _coerce_quantities_to_int(obj: Any) -> Any:
    """
    Recursively convert float quantities that are whole numbers (e.g. 1.0 → 1)
    to int, since the trade validator requires int quantities.
    """
    if isinstance(obj, dict):
        return {k: _coerce_quantities_to_int(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_quantities_to_int(v) for v in obj]
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj
 
 
def parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Parse a model's raw text output into a dict.
 
    Repair sequence:
      1. Strip markdown fences
      2. Try direct parse
      3. Extract first {...} block and try again
      4. Raise ValueError if still unparseable
    """
    cleaned = _strip_markdown_fences(raw)
 
    try:
        result = json.loads(cleaned)
        return _coerce_quantities_to_int(result)
    except json.JSONDecodeError:
        pass
 
    extracted = _extract_first_json_object(cleaned)
    try:
        result = json.loads(extracted)
        return _coerce_quantities_to_int(result)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse JSON from model output.\n"
            f"Raw output (first 500 chars): {raw[:500]}\n"
            f"Error: {exc}"
        ) from exc
 
 
# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------
 
_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)
_MAX_RETRIES = 3
_BASE_BACKOFF = 2.0   # seconds; doubles each retry
 
 
def _call_openai(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_completion_tokens: int,
    timeout: float,
    response_format={"type": "json_object"},
) -> str:
    """
    Call the OpenAI chat completions endpoint with retry on transient errors.
 
    Returns the raw text content of the first choice.
    Raises RuntimeError if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
 
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,          # type: ignore[arg-type]
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
                response_format=response_format,
            )
            return response.choices[0].message.content or ""
 
        except _RETRYABLE as exc:
            last_exc = exc
            wait = _BASE_BACKOFF * (2 ** attempt)
            print(
                f"  [openai_agent] Transient error ({type(exc).__name__}), "
                f"retry {attempt + 1}/{_MAX_RETRIES} in {wait:.0f}s..."
            )
            time.sleep(wait)
 
        except Exception as exc:
            # Non-retryable (auth errors, bad requests, etc.)
            raise RuntimeError(f"OpenAI API call failed: {exc}") from exc
 
    raise RuntimeError(
        f"OpenAI API call failed after {_MAX_RETRIES} retries. "
        f"Last error: {last_exc}"
    )
 
 
# ---------------------------------------------------------------------------
# Response validators
# ---------------------------------------------------------------------------
 
def _validate_negotiation_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure required fields are present and action_type is a known value.
    Fill safe defaults for missing optional fields.
    """
    required = {"action_type", "message_to_partner", "reasoning_summary"}
    missing = required - set(parsed.keys())
    if missing:
        raise ValueError(f"Negotiation response missing fields: {missing}")
 
    # "accept" is no longer a valid standalone action (acceptance happens
    # only at the moment an offer is made). We still recognise it here so a
    # stray "accept" passes through to the runner, which turns it into a
    # harmless no-op with an explanatory note, rather than being silently
    # coerced to "message" and losing that feedback. The runner never
    # executes a stale prior offer.
    allowed_types = {"message", "offer", "counteroffer", "accept", "reject", "no_trade"}
    if parsed.get("action_type") not in allowed_types:
        # Coerce unknown types to "message" rather than crashing
        parsed["action_type"] = "message"
 
    parsed.setdefault("proposed_trade", None)
    parsed.setdefault("accept_trade", None)
    return parsed
 
 
def _validate_commitment_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure decision is exactly 'accept' or 'reject'."""
    if "decision" not in parsed:
        raise ValueError("Commitment response missing 'decision' field.")

    if parsed["decision"] not in {"accept", "reject"}:
        # Coerce loose answers like "yes"/"no"/"accepted".
        raw = str(parsed["decision"]).lower()
        parsed["decision"] = "accept" if "accept" in raw else "reject"

    parsed.setdefault("reasoning_summary", "")
    return parsed
 
 
def _validate_probe_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the three current probe fields are present and well-formed.

    Current probe (see prompts.yaml preference_elicitation_prompt):
      Q1: ratings_inventory      — value of each good given current inventory
      Q2: ratings_general     — value of each good setting inventory aside
      Q3: desired_bundle — preferred bundle summing to 4
    """
    required = {"ratings_inventory", "ratings_general", "desired_bundle"}
    missing = required - set(parsed.keys())
    if missing:
        raise ValueError(f"Preference probe response missing fields: {missing}")

    # Both ratings blocks: coerce to int, clamp to [1, 10]
    for field in ("ratings_inventory", "ratings_general"):
        for good, val in parsed[field].items():
            try:
                parsed[field][good] = max(1, min(10, int(val)))
            except (TypeError, ValueError):
                parsed[field][good] = 5
                print(f"  [openai_agent] Warning: {field}[{good}] was not an int, defaulting to 5.")

    # Desired bundle must sum to 4; scale + fix rounding drift if not
    bundle = parsed["desired_bundle"]
    total = sum(bundle.values())
    if total != 4 and total > 0:
        print(f"  [openai_agent] Warning: desired_bundle sums to {total}, scaling to sum=4.")
        goods = sorted(bundle.keys())
        scaled = {g: round(bundle[g] / total * 4) for g in goods}
        diff = 4 - sum(scaled.values())
        if diff != 0:
            top = max(goods, key=lambda g: bundle[g])
            scaled[top] = max(0, scaled[top] + diff)
        parsed["desired_bundle"] = scaled

    return parsed
 
 
# ---------------------------------------------------------------------------
# Public agent functions
# ---------------------------------------------------------------------------
 
def gpt_negotiation_action(
    player: Any,
    prompts: Any,
    goods: List[str],
    round_index: int,
    partner: Any,
    negotiation_history: Optional[List[Mapping[str, Any]]],
    turn_index: int,
    model_spec: Any,
    client: OpenAI,
    logger: RunLogger,
    pair_id: str,
    action_space: str = "one_for_one",
    trade_history: Optional[List[Mapping[str, Any]]] = None,
    board_history: Optional[List[Mapping[str, Any]]] = None,
    broadcast: bool = False,
) -> Dict[str, Any]:
    """
    Call GPT for one negotiation turn and return a parsed action dict.
 
    On turn 0, uses the first-message prompt.
    On subsequent turns, uses the response prompt with the partner's last message.

    action_space must be passed in by the caller (sourced from
    cfg.experiment.mechanism.action_space) so the prompt text always matches
    the mechanism actually enforced in validate_trade.
    """
 
    # Build messages
    if turn_index == 0 or not negotiation_history:
        messages = build_negotiation_first_messages(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            partner_name=partner.display_name,
            action_space=action_space,
            trade_history=trade_history,
            negotiation_history=negotiation_history,
            board_history=board_history,
            broadcast=broadcast,
        )
    else:
        # Last message from partner
        last_partner_msg = ""
        for entry in reversed(negotiation_history):
            if entry.get("speaker_id") != player.id:
                last_partner_msg = entry.get("message", "")
                break
 
        messages = build_negotiation_response_messages(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            partner_name=partner.display_name,
            partner_message=last_partner_msg,
            action_space=action_space,
            trade_history=trade_history,
            negotiation_history=negotiation_history,
            board_history=board_history,
            broadcast=broadcast,
        )
 
    # Log prompt
    logger.log_prompt(
        player_id=player.id,
        prompt_type="negotiation",
        messages=messages,
        round_index=round_index,
        pair_id=pair_id,
    )
 
    gen = model_spec.generation
    raw = _call_openai(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_completion_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )
 
    try:
        parsed = parse_json_response(raw)
        parsed = _validate_negotiation_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [openai_agent] Parse error for {player.id} negotiation: {exc}")
        parsed = {
            "action_type": "no_trade",
            "message_to_partner": "I'm unable to respond right now.",
            "proposed_trade": None,
            "accept_trade": False,
            "reasoning_summary": f"Parse error: {exc}",
        }
 
    logger.log_model_output(
        player_id=player.id,
        output_type="negotiation",
        raw_output=raw,
        parsed_output=parsed,
        round_index=round_index,
        pair_id=pair_id,
        provider="openai",
        model=model_spec.model,
    )
 
    return parsed
 
 
def gpt_commitment_decision(
    player: Any,
    prompts: Any,
    goods: List[str],
    proposed_trade: Mapping[str, Any],
    model_spec: Any,
    client: OpenAI,
    logger: RunLogger,
    round_index: int,
    pair_id: str,
    partner_name: str = "your partner",
    negotiation_history: Optional[List[Dict[str, Any]]] = None,
    board_history: Optional[List[str]] = None,
    broadcast: bool = False,
) -> Dict[str, Any]:
    """Call GPT for a commitment decision (accept/reject a finalised trade).

    Now receives the full negotiation history with the partner (so the
    decision is made in context of the full exchange) and, under broadcast,
    the public market bulletin board.
    """
    messages = build_commitment_messages(
        player=player,
        prompts=prompts,
        goods=goods,
        proposed_trade=proposed_trade,
        round_index=round_index,
        partner_name=partner_name,
        negotiation_history=negotiation_history,
        board_history=board_history,
        broadcast=broadcast,
    )
 
    logger.log_prompt(
        player_id=player.id,
        prompt_type="commitment",
        messages=messages,
        round_index=round_index,
        pair_id=pair_id,
    )
 
    gen = model_spec.generation
    raw = _call_openai(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_completion_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )
 
    try:
        parsed = parse_json_response(raw)
        parsed = _validate_commitment_decision(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [openai_agent] Parse error for {player.id} commitment: {exc}")
        parsed = {
            "decision": "reject",
            "reasoning_summary": f"Parse error: {exc}",
        }
 
    logger.log_model_output(
        player_id=player.id,
        output_type="commitment",
        raw_output=raw,
        parsed_output=parsed,
        round_index=round_index,
        pair_id=pair_id,
        provider="openai",
        model=model_spec.model,
    )
 
    return parsed
 
 
def _validate_commitment_decision(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Alias kept for clarity inside this module."""
    return _validate_commitment_response(parsed)
 
 
def gpt_preference_probe(
    player: Any,
    prompts: Any,
    model_spec: Any,
    client: OpenAI,
    logger: RunLogger,
    round_index: int,
    goods: Iterable[str] = ("A", "B", "C"),
    trade_history: Optional[List[Dict[str, Any]]] = None,
    board_history: Optional[List[str]] = None,
    broadcast: bool = False,
) -> Dict[str, Any]:
    """Call GPT for a preference elicitation probe.

    The probe now runs in the situated context the agent has just lived
    through: round, current inventory, own trade history, and (under
    broadcast) the public market bulletin board. Without this, the probe was
    run in a vacuum and the broadcast condition's social framing was
    invisible to it, making drift unmeasurable.
    """
    messages = build_preference_probe_messages(
        player=player,
        prompts=prompts,
        goods=goods,
        round_index=round_index,
        trade_history=trade_history,
        board_history=board_history,
        broadcast=broadcast,
    )
 
    logger.log_prompt(
        player_id=player.id,
        prompt_type="preference_probe",
        messages=messages,
        round_index=round_index,
        pair_id=None,
    )
 
    gen = model_spec.generation
    raw = _call_openai(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_completion_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )
 
    try:
        parsed = parse_json_response(raw)
        parsed = _validate_probe_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [openai_agent] Parse error for {player.id} probe — skipping this probe. {exc}")
        logger.log_model_output(
            player_id=player.id,
            output_type="preference_probe",
            raw_output=raw,
            parsed_output=None,
            round_index=round_index,
            pair_id=None,
            provider="openai",
            model=model_spec.model,
        )
        return None
 
    logger.log_model_output(
        player_id=player.id,
        output_type="preference_probe",
        raw_output=raw,
        parsed_output=parsed,
        round_index=round_index,
        pair_id=None,
        provider="openai",
        model=model_spec.model,
    )
 
    return parsed

def gpt_preference_probe_contextual(
    player: Any,
    prompts: Any,
    model_spec: Any,
    client: "OpenAI",
    logger: Any,
    round_index: int,
    prior_history: List[Dict[str, str]],
    goods: Iterable[str] = ("A", "B", "C"),
    trade_history: Optional[List[Dict[str, Any]]] = None,
    board_history: Optional[List[str]] = None,
    broadcast: bool = False,
) -> Tuple[Dict[str, Any], str]:
    """
    Like gpt_preference_probe but threads the player's prior probe responses
    into the conversation, so the model sees how it answered in all previous
    iterations. Also accepts the situated context (inventory, trade history,
    bulletin board) like gpt_preference_probe.

    The message list becomes:
        [system]
        [user_1]  [assistant_1]   <- iteration 1 Q&A
        [user_2]  [assistant_2]   <- iteration 2 Q&A
        ...
        [user_N]                  <- current iteration question (no reply yet)

    The current user turn carries the most up-to-date context. Each prior
    user turn was sent with the context as it stood at that time and stays
    fixed in history. Returns (parsed_dict, raw_response_text) so the caller
    can append the raw text as the next assistant message in prior_history.
    """
    from prompt_render import build_preference_probe_messages

    base_messages = build_preference_probe_messages(
        player=player,
        prompts=prompts,
        goods=goods,
        round_index=round_index,
        trade_history=trade_history,
        board_history=board_history,
        broadcast=broadcast,
    )
    system_msg = base_messages[0]   # {"role": "system", "content": ...}
    user_msg   = base_messages[1]   # {"role": "user",   "content": probe text}

    # Insert prior history between system and the current user turn.
    messages = [system_msg] + prior_history + [user_msg]

    logger.log_prompt(
        player_id=player.id,
        prompt_type="preference_probe_contextual",
        messages=messages,
        round_index=round_index,
        pair_id=None,
    )

    gen = model_spec.generation
    raw = _call_openai(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_completion_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_probe_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [openai_agent] Parse error for {player.id} contextual probe: {exc}")
        parsed = {
            "ratings": {g: 5 for g in ["A", "B", "C"]},
            "ratings_reasoning": {g: "" for g in ["A", "B", "C"]},
            "desired_bundle_4_units": {"A": 2, "B": 1, "C": 1},
            "role_important_goods": f"Parse error: {exc}",
        }

    logger.log_model_output(
        player_id=player.id,
        output_type="preference_probe_contextual",
        raw_output=raw,
        parsed_output=parsed,
        round_index=round_index,
        pair_id=None,
        provider="openai",
        model=model_spec.model,
    )

    return parsed, raw