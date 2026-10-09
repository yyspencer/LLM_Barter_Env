"""
claude_agent.py

Anthropic Claude provider wrapper for the LLM barter experiment.

Same responsibilities as openai_agent.py / gemini_agent.py, but talks to the
Anthropic Messages API (Claude Sonnet 5) instead:
  - Calling the Messages API, with schema-constrained JSON output
    (output_config.format) for negotiation, commitment and probe calls
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

import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from anthropic import (
    Anthropic,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OverloadedError,
    RateLimitError,
)

from logger import RunLogger
from openai_agent import (
    _validate_commitment_response,
    _validate_negotiation_response,
    _validate_probe_response,
    call_until_parsed,
    parse_json_response,
)
from prompt_render import (
    build_commitment_messages_cacheable,
    build_negotiation_first_messages_cacheable,
    build_negotiation_response_messages_cacheable,
    build_preference_probe_messages,
    build_preference_probe_messages_cacheable,
)

DEFAULT_MODEL = "claude-sonnet-5"

# Reasoning effort for adaptive thinking: low | medium | high | xhigh | max.
_EFFORT = "medium"

# Extra max_tokens reserved for thinking, on top of the caller's max_tokens.
# Matches gemini_agent.py's _THINKING_BUDGET so both providers get the same
# total (thinking + visible answer) for a given models.yaml max_tokens.
_THINKING_HEADROOM = 1000


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

_RETRYABLE = (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    OverloadedError,      # 529 — Anthropic's servers are temporarily over capacity
    InternalServerError,  # 500 — transient server-side error
)
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
    cache_history_text: Optional[str] = None,
) -> str:
    """
    Call the Anthropic Messages API with retry on transient errors.

    When response_schema is given, the reply is constrained to JSON matching
    that schema (output_config.format) and returned as text, so callers can
    feed it through the same parse_json_response() pipeline used everywhere
    else regardless of provider.

    cache_history_text, when given, is the growing-but-append-only prefix
    of the final user message (world-state / negotiation-so-far / trade
    history — see the *_cacheable prompt_render builders). It's split out
    into its own cache_control-marked content block so repeated calls in
    the same round reuse it from Anthropic's cache instead of paying full
    input price for history that hasn't changed. If the last user message
    doesn't actually start with cache_history_text (should not happen, but
    prompt_render.py is the source of truth, not this string check), the
    message is sent unsplit rather than risk corrupting it.

    Returns the raw text content of the response.
    Raises RuntimeError if all retries are exhausted.
    """
    system_prompt, claude_messages = _to_claude_messages(messages)

    if cache_history_text is not None and claude_messages:
        last = claude_messages[-1]
        if (
            last["role"] == "user"
            and isinstance(last["content"], str)
            and last["content"].startswith(cache_history_text)
        ):
            volatile_text = last["content"][len(cache_history_text):]
            last["content"] = [
                {
                    "type": "text",
                    "text": cache_history_text,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": volatile_text},
            ]

    # claude-sonnet-5 rejects `temperature` outright ("deprecated for this
    # model" — 400 error): it manages its own sampling. `temperature` stays
    # in this function's signature for interface parity with the other two
    # provider modules, but is intentionally not forwarded to the API.
    #
    # Thinking is on (adaptive, at _EFFORT) so Claude can reason privately
    # before answering, like the other two providers. With it disabled the
    # model did its reasoning out loud inside message_to_partner, which
    # produced trade-less offers and leaked tool-call markup into the message.
    # Thinking tokens draw from the same max_tokens ceiling as the visible
    # answer, and this model accepts no numeric thinking budget (only the
    # qualitative effort level), so _THINKING_HEADROOM is added on top. The
    # total then matches gemini_agent.py, but unlike Gemini's thinking_budget
    # the split is not enforced: a long think can use the answer's share.
    output_config: Dict[str, Any] = {"effort": _EFFORT}
    if response_schema is not None:
        # Schema-constrained JSON text rather than a forced tool call: the
        # schema guarantee is the same (a required field such as
        # proposed_trade cannot be dropped), but a forced tool call
        # suppresses thinking entirely.
        output_config["format"] = {"type": "json_schema", "schema": response_schema}
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens + _THINKING_HEADROOM,
        "messages": claude_messages,
        "timeout": timeout,
        "thinking": {"type": "adaptive"},
        "output_config": output_config,
    }
    if system_prompt is not None:
        # system_prompt (game rules + persona) depends only on player and
        # display_order, both fixed for the whole run — byte-identical
        # across every call this player makes. Cache it: everything up to
        # and including this block becomes a cache hit on this player's
        # next call of the same type, instead of full-price input tokens.
        kwargs["system"] = [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]

    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(**kwargs)

            if response.stop_reason == "max_tokens":
                # Thinking used up the ceiling before the answer finished.
                # The partial/empty text fails parsing downstream, which
                # triggers call_until_parsed's re-ask.
                print(
                    f"  [claude_agent] Response cut off at max_tokens "
                    f"({response.usage.output_tokens} output tokens)."
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
    used as the output_config.format schema.
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


_NEGOTIATION_ACTION_TYPES = [
    "message", "offer", "counteroffer", "accept", "reject", "no_trade",
]


def _build_negotiation_schema(display_order: List[str]) -> Dict[str, Any]:
    """
    Build a JSON Schema for a negotiation turn, used as the
    output_config.format schema. Mirrors the fields _validate_negotiation_response
    (openai_agent.py) requires/defaults, so a schema-conformant reply
    always passes validation without falling through to the parse-error path.

    proposed_trade's give/receive sides allow *any subset* of display_order
    with integer quantities (not "all goods required") — validate_trade
    (runner.py) is what enforces nonnegative quantities and the
    action-space-specific shape rules (e.g. one_for_one's
    exactly-one-good-each-side, any_bundle's no restriction), so the schema
    only needs to be permissive enough to cover every configured
    action_space, not encode one specific one. No "minimum" here:
    structured-output schemas do not support numeric constraints.
    """
    def _trade_side() -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {g: {"type": "integer"} for g in display_order},
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "action_type": {"type": "string", "enum": _NEGOTIATION_ACTION_TYPES},
            "message_to_partner": {"type": "string"},
            "proposed_trade": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "give": _trade_side(),
                            "receive": _trade_side(),
                        },
                        "required": ["give", "receive"],
                        "additionalProperties": False,
                    },
                ]
            },
            "accept_trade": {"anyOf": [{"type": "null"}, {"type": "boolean"}]},
            "reasoning_summary": {"type": "string"},
        },
        "required": [
            "action_type",
            "message_to_partner",
            "proposed_trade",
            "accept_trade",
            "reasoning_summary",
        ],
        "additionalProperties": False,
    }


def _build_commitment_schema() -> Dict[str, Any]:
    """
    Build a JSON Schema for a commitment (accept/reject) decision, used as
    the output_config.format schema. Mirrors the fields
    _validate_commitment_response (openai_agent.py) requires.
    """
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["accept", "reject"]},
            "reasoning_summary": {"type": "string"},
        },
        "required": ["decision", "reasoning_summary"],
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
        system, history_text, volatile_text = build_negotiation_first_messages_cacheable(
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

        system, history_text, volatile_text = build_negotiation_response_messages_cacheable(
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

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": history_text + "\n\n" + volatile_text},
    ]

    logger.log_prompt(
        player_id=player.id,
        prompt_type="negotiation",
        messages=messages,
        round_index=round_index,
        pair_id=pair_id,
    )

    gen = model_spec.generation
    raw, parsed, exc = call_until_parsed(
        lambda: _call_claude(
            client=client,
            messages=messages,
            model=model_spec.model,
            temperature=gen.temperature,
            max_tokens=gen.max_tokens,
            timeout=gen.timeout_seconds,
            response_schema=_build_negotiation_schema(display_order),
            cache_history_text=history_text,
        ),
        validate=_validate_negotiation_response,
    )

    if exc is not None:
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
    trade_history: Optional[List[Mapping[str, Any]]] = None,
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
    system, history_text, volatile_text = build_commitment_messages_cacheable(
        player=player,
        prompts=prompts,
        goods=goods,
        proposed_trade=proposed_trade,
        display_order=display_order,
        round_index=round_index,
        partner_name=partner_name,
        negotiation_history=negotiation_history,
        trade_history=trade_history,
        board_history=board_history,
        broadcast=broadcast,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": history_text + "\n\n" + volatile_text},
    ]

    logger.log_prompt(
        player_id=player.id,
        prompt_type=prompt_type,
        messages=messages,
        round_index=round_index,
        pair_id=pair_id,
    )

    gen = model_spec.generation
    raw, parsed, exc = call_until_parsed(
        lambda: _call_claude(
            client=client,
            messages=messages,
            model=model_spec.model,
            temperature=gen.temperature,
            max_tokens=gen.max_tokens,
            timeout=gen.timeout_seconds,
            response_schema=_build_commitment_schema(),
            cache_history_text=history_text,
        ),
        validate=_validate_commitment_response,
    )

    if exc is not None:
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
    system, history_text, volatile_text = build_preference_probe_messages_cacheable(
        player=player,
        prompts=prompts,
        display_order=display_order,
        goods=goods,
        round_index=round_index,
        trade_history=trade_history,
        board_history=board_history,
        broadcast=broadcast,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": history_text + "\n\n" + volatile_text},
    ]

    logger.log_prompt(
        player_id=player.id,
        prompt_type="preference_probe",
        messages=messages,
        round_index=round_index,
        pair_id=None,
    )

    gen = model_spec.generation
    raw, parsed, exc = call_until_parsed(
        lambda: _call_claude(
            client=client,
            messages=messages,
            model=model_spec.model,
            temperature=gen.temperature,
            max_tokens=gen.max_tokens,
            timeout=gen.timeout_seconds,
            response_schema=_build_probe_schema(display_order),
            cache_history_text=history_text,
        ),
        validate=_validate_probe_response,
    )

    if exc is not None:
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
    raw, parsed, exc = call_until_parsed(
        lambda: _call_claude(
            client=client,
            messages=messages,
            model=model_spec.model,
            temperature=gen.temperature,
            max_tokens=gen.max_tokens,
            timeout=gen.timeout_seconds,
        ),
        validate=_validate_probe_response,
    )

    if exc is not None:
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
