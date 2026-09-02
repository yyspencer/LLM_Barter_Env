"""
gemini_agent.py

Google Gemini provider wrapper for the LLM barter experiment.

Same responsibilities as openai_agent.py, but talks to the Gemini API
(google-genai SDK, Gemini 3.1 Pro) instead of OpenAI's:
  - Calling the Gemini generateContent API
  - JSON parsing with best-effort repair for common model output issues
    (reuses openai_agent's parser/validators — that logic is provider-agnostic,
    so fixes made there, e.g. how malformed probe fields are handled,
    automatically apply here too)
  - Retry with exponential backoff on transient API errors
  - Logging raw outputs via RunLogger

Public interface (mirrors mock_agent.py / openai_agent.py signatures):
  gpt_negotiation_action(...)            -> dict
  gpt_commitment_decision(...)           -> dict
  gpt_preference_probe(...)              -> dict
  gpt_preference_probe_contextual(...)   -> (dict, str)

These are called by runner.py exactly like the OpenAI/mock equivalents, so
swapping providers only requires importing this module's functions and
passing a genai.Client instead of openai_agent's and an OpenAI client.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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

DEFAULT_MODEL = "gemini-3.1-pro-preview"


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_BACKOFF = 2.0   # seconds; doubles each retry

# Gemini 3.1 Pro is a reasoning model: its internal "thinking" tokens are
# deducted from the same max_output_tokens budget as the visible answer. A
# tight budget (e.g. max_tokens: 400 in models.yaml, sized for a plain
# non-reasoning completion) gets entirely consumed by thinking, leaving the
# model to return truncated/empty text (finish_reason=MAX_TOKENS). We cap
# thinking at a fixed budget and add it on top, so a caller's max_tokens
# continues to mean "budget for the visible answer" as it does for OpenAI.
_THINKING_BUDGET = 1000


def _to_gemini_contents(
    messages: List[Dict[str, str]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Translate this project's OpenAI-shaped message list
    ([{"role": "system"/"user"/"assistant", "content": str}, ...]) into
    Gemini's shape: a system_instruction string plus a `contents` list
    using Gemini's role names ("user" / "model").
    """
    system_instruction: Optional[str] = None
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_instruction = (
                text if system_instruction is None else f"{system_instruction}\n\n{text}"
            )
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})

    return system_instruction, contents


def _call_gemini(
    client: genai.Client,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout: float,
    response_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Call the Gemini generateContent endpoint with retry on transient errors.

    Returns the raw text content of the response.
    Raises RuntimeError if all retries are exhausted.
    """
    system_instruction, contents = _to_gemini_contents(messages)

    config_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens + _THINKING_BUDGET,
        "thinking_config": genai_types.ThinkingConfig(
            thinking_budget=_THINKING_BUDGET,
            include_thoughts=False,
        ),
        "http_options": genai_types.HttpOptions(timeout=int(timeout * 1000)),
    }
    if system_instruction is not None:
        config_kwargs["system_instruction"] = system_instruction
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = genai_types.GenerateContentConfig(**config_kwargs)

    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text or ""

        except genai_errors.APIError as exc:
            if getattr(exc, "code", None) in _RETRYABLE_CODES:
                last_exc = exc
                wait = _BASE_BACKOFF * (2 ** attempt)
                print(
                    f"  [gemini_agent] Transient error ({exc.code}), "
                    f"retry {attempt + 1}/{_MAX_RETRIES} in {wait:.0f}s..."
                )
                time.sleep(wait)
                continue
            # Non-retryable API error (auth, bad request, etc.)
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    raise RuntimeError(
        f"Gemini API call failed after {_MAX_RETRIES} retries. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

def _build_probe_schema(display_order: List[str]) -> Dict[str, Any]:
    """
    Build a Gemini structured-output schema (OpenAPI-subset, per Gemini's
    response_schema) for the probe with goods in the given display order.
    """
    def _int_object(order: List[str]) -> Dict[str, Any]:
        return {
            "type": "OBJECT",
            "properties": {g: {"type": "INTEGER"} for g in order},
            "required": list(order),
        }

    return {
        "type": "OBJECT",
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
    client: genai.Client,
    logger: RunLogger,
    pair_id: str,
    display_order: List[str],
    action_space: str = "one_for_one",
    trade_history: Optional[List[Mapping[str, Any]]] = None,
    board_history: Optional[List[Mapping[str, Any]]] = None,
    broadcast: bool = False,
) -> Dict[str, Any]:
    """
    Call Gemini for one negotiation turn and return a parsed action dict.

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
    raw = _call_gemini(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_output_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_negotiation_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [gemini_agent] Parse error for {player.id} negotiation: {exc}")
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
        provider="google",
        model=model_spec.model,
    )

    return parsed


def gpt_commitment_decision(
    player: Any,
    prompts: Any,
    goods: List[str],
    proposed_trade: Mapping[str, Any],
    model_spec: Any,
    client: genai.Client,
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
    """Call Gemini for a commitment decision (accept/reject a finalised trade).

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
    raw = _call_gemini(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_output_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_commitment_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [gemini_agent] Parse error for {player.id} commitment: {exc}")
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
        provider="google",
        model=model_spec.model,
    )

    return parsed


def gpt_preference_probe(
    player: Any,
    prompts: Any,
    model_spec: Any,
    client: genai.Client,
    logger: RunLogger,
    round_index: int,
    display_order: List[str],
    goods: Iterable[str] = ("A", "B", "C"),
    trade_history: Optional[List[Dict[str, Any]]] = None,
    board_history: Optional[List[str]] = None,
    broadcast: bool = False,
) -> Optional[Dict[str, Any]]:
    """Call Gemini for a preference elicitation probe.

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
    raw = _call_gemini(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_output_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
        response_schema=_build_probe_schema(display_order),
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_probe_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [gemini_agent] Parse error for {player.id} probe — skipping. {exc}")
        logger.log_model_output(
            player_id=player.id,
            output_type="preference_probe",
            raw_output=raw,
            parsed_output=None,
            round_index=round_index,
            pair_id=None,
            provider="google",
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
        provider="google",
        model=model_spec.model,
    )
    return parsed


def gpt_preference_probe_contextual(
    player: Any,
    prompts: Any,
    model_spec: Any,
    client: "genai.Client",
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
    raw = _call_gemini(
        client=client,
        messages=messages,
        model=model_spec.model,
        temperature=gen.temperature,
        max_output_tokens=gen.max_tokens,
        timeout=gen.timeout_seconds,
    )

    try:
        parsed = parse_json_response(raw)
        parsed = _validate_probe_response(parsed)
    except (ValueError, KeyError) as exc:
        print(f"  [gemini_agent] Parse error for {player.id} contextual probe: {exc}")
        parsed = None

    logger.log_model_output(
        player_id=player.id,
        output_type="preference_probe_contextual",
        raw_output=raw,
        parsed_output=parsed,
        round_index=round_index,
        pair_id=None,
        provider="google",
        model=model_spec.model,
    )

    return parsed, raw
