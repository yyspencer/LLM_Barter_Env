"""
claude_agent.py

Anthropic Claude provider wrapper for the LLM barter experiment.

Same responsibilities as openai_agent.py / gemini_agent.py, but talks to the
Anthropic Messages API (Claude Sonnet 5) instead:
  - Calling the Messages API, with structured output for the preference
    probe implemented via forced tool use (Anthropic has no OpenAI/Gemini-
    style response_format/response_schema knob — a forced tool call is the
    standard way to get schema-conformant JSON out of Claude)
  - JSON parsing with best-effort repair for common model output issues
    (reuses openai_agent's parser/validators — that logic is provider-agnostic,
    so fixes made there automatically apply here too)
  - Retry with exponential backoff on transient API errors
  - Logging raw outputs via RunLogger

Public interface (mirrors mock_agent.py / openai_agent.py / gemini_agent.py
signatures):
  gpt_negotiation_action(...)            -> dict
  gpt_commitment_decision(...)           -> dict
  gpt_preference_probe(...)              -> dict
  gpt_preference_probe_contextual(...)   -> (dict, str)

These are called by runner.py exactly like the other provider modules, so
swapping providers only requires importing this module's functions and
passing an anthropic.Anthropic client instead.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from anthropic import Anthropic, APIConnectionError, APITimeoutError, RateLimitError

from logger import RunLogger
from openai_agent import (
    _validate_commitment_response,
    _validate_negotiation_response,
    _validate_probe_response,
    parse_json_response,
)
from prompt_render import (
    build_commitment_messages,
    build_negotiation_first_messages,
    build_negotiation_response_messages,
    build_preference_probe_messages,
)

DEFAULT_MODEL = "claude-sonnet-5"

_TOOL_NAME = "submit_response"


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

_RETRYABLE = (APITimeoutError, APIConnectionError, RateLimitError)
_MAX_RETRIES = 3
_BASE_BACKOFF = 2.0   # seconds; doubles each retry


def _to_claude_messages(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], List[Dict[str, str]]]:
    """
    Translate this project's OpenAI-shaped message list
    ([{"role": "system"/"user"/"assistant", "content": str}, ...]) into
    Anthropic's shape: a top-level system string plus a `messages` list.
    Claude's role names ("user" / "assistant") already match, unlike Gemini's.
    """
    system_prompt: Optional[str] = None
    claude_messages: List[Dict[str, str]] = []

    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_prompt = (
                text if system_prompt is None else f"{system_prompt}\n\n{text}"
            )
            continue
        claude_messages.append({"role": role, "content": text})

    return system_prompt, claude_messages


def _call_claude(
    client: Anthropic,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    response_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call the Anthropic Messages API with retry on transient errors.

    When response_schema is given, forces a tool call shaped by that schema
    and returns the tool input re-serialized as a JSON string, so callers can
    feed it through the same parse_json_response() pipeline used everywhere
    else regardless of provider.

    Returns the raw text content of the response.
    Raises RuntimeError if all retries are exhausted.
    """
    system_prompt, claude_messages = _to_claude_messages(messages)

    # claude-sonnet-5 rejects `temperature` outright ("deprecated for this
    # model" — 400 error): it manages its own sampling. `temperature` stays
    # in this function's signature for interface parity with the other two
    # provider modules, but is intentionally not forwarded to the API.
    #
    # It also reasons by default (adaptive thinking) even when we never asked
    # for it — thinking tokens draw from the same max_tokens ceiling as the
    # visible answer, same failure mode as Gemini 3.1 Pro's thinking budget,
    # except here there is no numeric budget to reserve against (only a
    # qualitative "effort" level with no hard cap), so a bounded reserve
    # can't guarantee no truncation on longer, later-round prompts. We
    # disable thinking outright instead: max_tokens then means exactly what
    # it says, matching the other two provider modules.
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": claude_messages,
        "timeout": timeout,
        "thinking": {"type": "disabled"},
    }
    if system_prompt is not None:
        kwargs["system"] = system_prompt
    if response_schema is not None:
        kwargs["tools"] = [{
            "name": _TOOL_NAME,
            "description": "Submit the structured response for this turn.",
            "input_schema": response_schema,
        }]
        kwargs["tool_choice"] = {"type": "tool", "name": _TOOL_NAME}

    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(**kwargs)

            if response_schema is not None:
                for block in response.content:
                    if block.type == "tool_use" and block.name == _TOOL_NAME:
                        return json.dumps(block.input)
                raise RuntimeError(
                    "Claude did not return the expected tool call for a "
                    "structured response."
                )

            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        except _RETRYABLE as exc:
            last_exc = exc
            wait = _BASE_BACKOFF * (2 ** attempt)
            print(
                f"  [claude_agent] Transient error ({type(exc).__name__}), "
                f"retry {attempt + 1}/{_MAX_RETRIES} in {wait:.0f}s..."
            )
            time.sleep(wait)

        except Exception as exc:
            # Non-retryable (auth errors, bad requests, etc.)
            raise RuntimeError(f"Claude API call failed: {exc}") from exc

    raise RuntimeError(
        f"Claude API call failed after {_MAX_RETRIES} retries. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

def _build_probe_schema(display_order: List[str]) -> Dict[str, Any]:
    """
    Build a JSON Schema for the probe with goods in the given display order,
    used as the input_schema of the forced tool call.
    """
    def _int_object(order: List[str]) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {g: {"type": "integer"} for g in order},
            "required": list(order),
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "ratings_inventory": _int_object(display_order),
            "ratings_general":   _int_object(display_order),
            "desired_bundle":    _int_object(display_order),
        },
        "required": [
            "ratings_inventory",
            "ratings_general",
            "desired_bundle",
        ],
        "additionalProperties": False,
    }


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
    client: Anthropic,
    logger: RunLogger,
    pair_id: str,
    display_order: List[str],
    action_space: str = "one_for_one",
    trade_history: Optional[List[Mapping[str, Any]]] = None,
    board_history: Optional[List[Mapping[str, Any]]] = None,
    broadcast: bool = False,
) -> Dict[str, Any]:
    """
    Call Claude for one negotiation turn and return a parsed action dict.

    On turn 0, uses the first-message prompt.
    On subsequent turns, uses the response prompt with the partner's last message.

    action_space must be passed in by the caller (sourced from
    cfg.experiment.mechanism.action_space) so the prompt text always matches
    the mechanism actually enforced in validate_trade.
    """

    if turn_index == 0 or not negotiation_history:
        messages = build_negotiation_first_messages(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            partner_name=partner.display_name,
            action_space=action_space,
            display_order=display_order,
            trade_history=trade_history,
            negotiation_history=negotiation_history,
            board_history=board_history,
            broadcast=broadcast,
        )
    else:
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
            display_order=display_order,
            trade_history=trade_history,
            negotiation_history=negotiation_history,
            board_history=board_history,
            broadcast=broadcast,
        )

    logger.log_prompt(
        player_id=player.id,
        prompt_type="negotiation",
        messages=messages,
        round_index=round_index,
        pair_id=pair_id,
    )

    gen = model_spec.generation
    raw = _call_claude(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_negotiation_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [claude_agent] Parse error for {player.id} negotiation: {exc}")
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
        provider="anthropic",
        model=model_spec.model,
    )

    return parsed


def gpt_commitment_decision(
    player: Any,
    prompts: Any,
    goods: List[str],
    proposed_trade: Mapping[str, Any],
    model_spec: Any,
    client: Anthropic,
    logger: RunLogger,
    round_index: int,
    pair_id: str,
    display_order: List[str],
    partner_name: str = "your partner",
    negotiation_history: Optional[List[Dict[str, Any]]] = None,
    board_history: Optional[List[str]] = None,
    broadcast: bool = False,
    prompt_type: str = "commitment",
    output_type: str = "commitment",
) -> Dict[str, Any]:
    """Call Claude for a commitment decision (accept/reject a finalised trade).

    Receives the full negotiation history with the partner (so the decision
    is made in context of the full exchange) and, under broadcast, the
    public market bulletin board.

    prompt_type/output_type are overridable so callers that reuse this exact
    prompt shape for a different purpose (e.g. shadow_trades.py's hypothetical
    offers) can tag their logs distinctly from real commitment decisions,
    without duplicating this function.
    """
    messages = build_commitment_messages(
        player=player,
        prompts=prompts,
        goods=goods,
        proposed_trade=proposed_trade,
        display_order=display_order,
        round_index=round_index,
        partner_name=partner_name,
        negotiation_history=negotiation_history,
        board_history=board_history,
        broadcast=broadcast,
    )

    logger.log_prompt(
        player_id=player.id,
        prompt_type=prompt_type,
        messages=messages,
        round_index=round_index,
        pair_id=pair_id,
    )

    gen = model_spec.generation
    raw = _call_claude(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_commitment_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [claude_agent] Parse error for {player.id} commitment: {exc}")
        parsed = {
            "decision": "reject",
            "reasoning_summary": f"Parse error: {exc}",
        }

    logger.log_model_output(
        player_id=player.id,
        output_type=output_type,
        raw_output=raw,
        parsed_output=parsed,
        round_index=round_index,
        pair_id=pair_id,
        provider="anthropic",
        model=model_spec.model,
    )

    return parsed


def gpt_preference_probe(
    player: Any,
    prompts: Any,
    model_spec: Any,
    client: Anthropic,
    logger: RunLogger,
    round_index: int,
    display_order: List[str],
    goods: Iterable[str] = ("A", "B", "C"),
    trade_history: Optional[List[Dict[str, Any]]] = None,
    board_history: Optional[List[str]] = None,
    broadcast: bool = False,
) -> Optional[Dict[str, Any]]:
    """Call Claude for a preference elicitation probe.

    display_order: run-wide goods order used for the question text, the
    inventory display, the schema, and the example.
    """
    messages = build_preference_probe_messages(
        player=player,
        prompts=prompts,
        display_order=display_order,
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
    raw = _call_claude(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
        response_schema=_build_probe_schema(display_order),
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_probe_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [claude_agent] Parse error for {player.id} probe — skipping. {exc}")
        logger.log_model_output(
            player_id=player.id,
            output_type="preference_probe",
            raw_output=raw,
            parsed_output=None,
            round_index=round_index,
            pair_id=None,
            provider="anthropic",
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
        provider="anthropic",
        model=model_spec.model,
    )
    return parsed


def gpt_preference_probe_contextual(
    player: Any,
    prompts: Any,
    model_spec: Any,
    client: "Anthropic",
    logger: Any,
    round_index: int,
    prior_history: List[Dict[str, str]],
    display_order: List[str],
    goods: Iterable[str] = ("A", "B", "C"),
    trade_history: Optional[List[Dict[str, Any]]] = None,
    board_history: Optional[List[str]] = None,
    broadcast: bool = False,
) -> Tuple[Optional[Dict[str, Any]], str]:
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
    base_messages = build_preference_probe_messages(
        player=player,
        prompts=prompts,
        display_order=display_order,
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
    raw = _call_claude(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_probe_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [claude_agent] Parse error for {player.id} contextual probe: {exc}")
        parsed = None

    logger.log_model_output(
        player_id=player.id,
        output_type="preference_probe_contextual",
        raw_output=raw,
        parsed_output=parsed,
        round_index=round_index,
        pair_id=None,
        provider="anthropic",
        model=model_spec.model,
    )

    return parsed, raw
