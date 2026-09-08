"""
prompt_renderer.py

Prompt rendering utilities for the LLM barter experiment.

Purpose:
- Turn templates from prompts.yaml into concrete strings for each agent call.
- Render persona, world state, negotiation prompts, commitment prompts, and
  preference elicitation prompts.
- Keep prompt construction separate from model API calls.

This module does not call any LLM APIs and does not mutate game state.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Optional


Inventory = Mapping[str, int]
TradeHistory = Iterable[Mapping[str, Any]]
NegotiationHistory = Iterable[Mapping[str, Any]]


# -------------------------------------------------------------------
# Generic formatting helpers
# -------------------------------------------------------------------

def safe_format(template: str, **kwargs: Any) -> str:
    """
    Format a template with clear errors when a required placeholder is missing.
    """
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(
            f"Missing template variable '{missing}' while rendering prompt. "
            f"Available variables: {sorted(kwargs.keys())}"
        ) from exc


def format_inventory_for_prompt(
    inventory: Inventory,
    goods: Iterable[str],
    style: str = "compact",
) -> str:
    """
    Format inventory for prompt injection, iterating goods in the order given.
    The caller is responsible for passing goods in the run's display order.
    """
    goods_list = list(goods)

    if style == "compact":
        return ", ".join(f"{g}×{inventory.get(g, 0)}" for g in goods_list)

    if style == "lines":
        return "\n".join(f"- {g}: {inventory.get(g, 0)}" for g in goods_list)

    raise ValueError(f"Unknown inventory format style: {style}")

def format_trade_history(
    trade_history: Optional[TradeHistory],
    empty_text: str = "None yet.",
) -> str:
    """
    Format completed trade history.

    Accepts dicts with flexible fields. Common expected fields:
    - round_index
    - partner_id / partner_name
    - decision
    - give
    - receive
    """
    if not trade_history:
        return empty_text

    lines: list[str] = []
    for idx, trade in enumerate(trade_history, start=1):
        round_index = trade.get("round_index", "?")
        partner = trade.get("partner_name") or trade.get("partner_id") or "unknown partner"
        decision = trade.get("decision", "completed")
        give = trade.get("give", {})
        receive = trade.get("receive", {})

        lines.append(
            f"Trade {idx} [round {round_index}] with {partner}: "
            f"{decision}; gave {format_goods_dict(give)}, received {format_goods_dict(receive)}."
        )

    return "\n".join(lines)


def format_negotiation_history(
    negotiation_history: Optional[NegotiationHistory],
    empty_text: str = "No messages yet.",
) -> str:
    """
    Format current pair's negotiation history.

    Expected flexible fields:
    - speaker / speaker_id / speaker_name
    - message / content

    Entries with source == "commitment_outcome" are rendered as a distinct
    system-style note (no "speaker:" prefix) so that an agent can clearly
    see which offers it has already declined, and why.
    """
    if not negotiation_history:
        return empty_text

    lines: list[str] = []
    for msg in negotiation_history:
        if msg.get("source") == "commitment_outcome":
            note = msg.get("message") or msg.get("content") or ""
            lines.append(f"[OFFER DECLINED] {note}")
            continue

        if msg.get("source") == "accept_failed":
            note = msg.get("message") or msg.get("content") or ""
            lines.append(f"[ACCEPT DID NOT REGISTER] {note}")
            continue

        speaker = (
            msg.get("speaker")
            or msg.get("speaker_name")
            or msg.get("speaker_id")
            or "Unknown"
        )
        content = msg.get("message") or msg.get("content") or ""
        lines.append(f"{speaker}: {content}")

    return "\n".join(lines)


def format_board_history(
    board_history: Optional[Any],
    empty_text: str = "No public trades have been posted yet.",
) -> str:
    """
    Format the public market bulletin board for the broadcast condition.

    board_history is a list of pre-formatted bulletin strings (one per
    completed trade), e.g.:
        "Market Update: Agent 1 exchanged 2×C for 1×A with Agent 4.
         Trade described by participants as 'fair and necessary.'"
    They are shown newest-last, one per line. For backward compatibility a
    list of structured trade dicts is also accepted and delegated to
    format_trade_history.
    """
    if not board_history:
        return empty_text

    # List of plain strings (the bulletin format) -> join directly.
    if all(isinstance(entry, str) for entry in board_history):
        return "\n".join(board_history)

    # Fallback: structured trade dicts.
    return format_trade_history(board_history, empty_text=empty_text)


def format_goods_dict(goods_dict: Optional[Mapping[str, int]]) -> str:
    """
    Format a goods dict like {"A": 1, "C": 2} as '1×A, 2×C'.
    """
    if not goods_dict:
        return "nothing"

    parts = []
    for good, amount in goods_dict.items():
        parts.append(f"{amount}×{good}")
    return ", ".join(parts)


def format_json_block(obj: Any) -> str:
    """
    Format an object as pretty JSON for prompt injection.
    """
    return json.dumps(obj, indent=2, ensure_ascii=False)


# -------------------------------------------------------------------
# Prompt renderers
# -------------------------------------------------------------------

def render_system_prompt(prompts, display_order: List[str]) -> str:
    """
    Render the shared system prompt.

    The goods list ("There are 3 types of goods: ...") is rendered in
    display_order so the very first mention of A/B/C in the whole prompt
    doesn't anchor on a fixed order.
    """
    g1, g2, g3 = display_order
    return safe_format(prompts.system_prompt, good_1=g1, good_2=g2, good_3=g3).strip()


def render_persona_prompt(player, prompts, display_order: List[str]) -> str:
    """
    Render persona prompt from persona_template and player fields.

    preference_description (from players.yaml) is itself formatted with
    good_1/good_2/good_3 first. Builder/Weaver descriptions don't use these
    placeholders (their goods mentions are tied to that character's actual
    weight ranking, not display order) so formatting is a no-op for them.
    The Merchant description has no weight-based ranking to justify a fixed
    order, so it uses {good_1}/{good_2}/{good_3} to avoid anchoring.
    """
    g1, g2, g3 = display_order
    description = safe_format(
        player.preference_description, good_1=g1, good_2=g2, good_3=g3
    )
    return safe_format(
        prompts.persona_template,
        display_name=player.display_name,
        preference_description=description,
        role=getattr(player, "role", None) or "",
        player_id=player.id,
    ).strip()


def render_world_state_prompt(
    player,
    prompts,
    goods: Iterable[str],
    round_index: int,
    partner_name: str,
    display_order: List[str],
    trade_history: Optional[TradeHistory] = None,
    negotiation_history: Optional[NegotiationHistory] = None,
    board_history: Optional[TradeHistory] = None,
    broadcast: bool = False,
) -> str:
    """
    Render the world-state prompt for an agent.

    Uses:
    - world_state_control_template when broadcast=False
    - world_state_broadcast_template when broadcast=True

    display_order: run-wide goods order (e.g. ['C', 'A', 'B']). Used for the
    inventory display so we don't leak an A-first anchor.
    """
    inventory = player.inventory

    template = (
        prompts.world_state_broadcast_template
        if broadcast
        else prompts.world_state_control_template
    )

    # Build the ordered inventory line once, then feed it into the template.
    inventory_line = format_inventory_for_prompt(
        inventory, display_order, style="compact"
    )

    values = {
        "round_index": round_index,
        "display_name": player.display_name,
        "player_id": player.id,
        "partner_name": partner_name,
        "inventory_line": inventory_line,
        "trade_history": format_trade_history(trade_history),
        "negotiation_history": format_negotiation_history(negotiation_history),
        "board_history": format_board_history(board_history),
    }

    return safe_format(template, **values).strip()


def render_action_space_description(action_space: str, prompts) -> str:
    """
    Render the action-space description selected by experiment.yaml.
    """
    descriptions = prompts.action_space_descriptions
    if action_space not in descriptions:
        raise KeyError(
            f"Action space '{action_space}' not found in prompts.action_space_descriptions. "
            f"Available: {sorted(descriptions.keys())}"
        )
    return descriptions[action_space].strip()


def render_negotiation_first_message_prompt(
    prompts,
    partner_name: str,
    action_space: str,
) -> str:
    """
    Render the prompt for the first message in a negotiation.
    """
    return safe_format(
        prompts.negotiation_first_message_prompt,
        partner_name=partner_name,
        action_space_description=render_action_space_description(action_space, prompts),
    ).strip()


def render_negotiation_response_prompt(
    prompts,
    partner_name: str,
    partner_message: str,
    action_space: str,
) -> str:
    """
    Render the prompt for responding to a partner's message.
    """
    return safe_format(
        prompts.negotiation_response_prompt,
        partner_name=partner_name,
        partner_message=partner_message,
        action_space_description=render_action_space_description(action_space, prompts),
    ).strip()


def render_commitment_prompt(
    player,
    prompts,
    goods: Iterable[str],
    proposed_trade: Mapping[str, Any],
    display_order: List[str],
) -> str:
    """
    Render the commitment prompt for accepting/rejecting a proposed trade.

    proposed_trade is from the *proposer's* perspective (see original doc).
    display_order controls only the current-inventory line; trade descriptions
    stay in the order the trade was actually proposed in (retrospective).
    """
    inventory_text = format_inventory_for_prompt(
        player.inventory, display_order, style="lines"
    )

    partner_gives = proposed_trade.get("give", {}) or {}
    partner_receives = proposed_trade.get("receive", {}) or {}

    they_give_text = format_goods_dict(partner_gives)
    you_give_text  = format_goods_dict(partner_receives)

    return safe_format(
        prompts.commitment_prompt,
        they_give=they_give_text,
        you_give=you_give_text,
        inventory=inventory_text,
        preference_description=player.preference_description,
        display_name=player.display_name,
        player_id=player.id,
        role=getattr(player, "role", None) or "",
    ).strip()


def render_commitment_context(
    prompts,
    round_index: int,
    partner_name: str,
    negotiation_history: Optional[NegotiationHistory],
) -> str:
    """
    Render the context block prepended to the commitment prompt: round number
    and the negotiation history with the partner up to this offer. Without
    this, the commitment decision is made on the bare offer text only — with
    it, the responder evaluates the offer in light of the whole exchange.
    """
    return safe_format(
        prompts.commitment_context_template,
        round_index=round_index,
        partner_name=partner_name,
        negotiation_history=format_negotiation_history(negotiation_history),
    ).strip()


def render_probe_context(
    player,
    prompts,
    goods: Iterable[str],
    round_index: int,
    display_order: List[str],
    trade_history: Optional[TradeHistory],
) -> str:
    """
    Render the context block prepended to the preference probe.
    Inventory line uses display_order to avoid A-first anchoring.
    """
    inventory_line = format_inventory_for_prompt(
        player.inventory, display_order, style="compact"
    )
    return safe_format(
        prompts.probe_context_template,
        round_index=round_index,
        inventory_line=inventory_line,
        trade_history=format_trade_history(trade_history),
    ).strip()


def render_bulletin_board_section(
    prompts,
    board_history: Optional[Any],
) -> str:
    """
    Render the public market bulletin board section. Appended to commitment or
    probe contexts under the broadcast condition only — this is the channel
    through which the broadcast experimental condition introduces the
    "fair and necessary" social framing.
    """
    return safe_format(
        prompts.bulletin_board_section_template,
        board_history=format_board_history(board_history),
    ).strip()


def render_preference_elicitation_prompt(prompts, display_order: List[str]) -> str:
    """
    Render the preference elicitation probe.

    The probe question is templated so that the three goods appear in the
    run's display order. This keeps the *question text* itself consistent
    with the JSON schema ordering, so nothing leaks an A-first bias.
    """
    g1, g2, g3 = display_order
    return safe_format(
        prompts.preference_elicitation_prompt,
        good_1=g1, good_2=g2, good_3=g3,
    ).strip()


def render_response_format_instruction(
    prompts,
    format_name: str,
    action_space: Optional[str] = None,
    display_order: Optional[List[str]] = None,
) -> str:
    """
    Render the schema description for one response type.

    For 'preference_probe', if display_order is given, the schema description
    and example are re-rendered in that order to prevent A-first bias.
    """
    if format_name not in prompts.response_formats:
        raise KeyError(
            f"Response format '{format_name}' not found. "
            f"Available: {sorted(prompts.response_formats.keys())}"
        )

    spec = prompts.response_formats[format_name]
    text = spec.schema_description.strip()

    if format_name == "negotiation" and action_space is not None:
        examples = getattr(prompts, "negotiation_offer_examples", None)
        if examples and action_space in examples:
            text = text + "\n\n" + examples[action_space].strip()

    if format_name == "preference_probe" and display_order is not None:
        text = _render_probe_schema_text(display_order)

    return text


def _render_probe_schema_text(display_order: List[str]) -> str:
    """
    Build the probe schema description + example with goods in display_order.
    Kept here (not in prompts.yaml) so the ordering logic and the text stay
    together — if you edit the schema description, edit it here.
    """
    g1, g2, g3 = display_order
    # Example values keyed by good, then re-emitted in display order.
    current_vals  = {"A": 4, "B": 9, "C": 3}
    abstract_vals = {"A": 7, "B": 8, "C": 2}
    bundle_vals   = {"A": 3, "B": 2, "C": 1}

    def _fmt_dict(vals):
        return "{" + ", ".join(f'"{g}": {vals[g]}' for g in display_order) + "}"

    return (
        f"Return one JSON object with exactly these three top-level keys:\n"
        f"ratings_inventory, ratings_general, desired_bundle.\n\n"
        f"ratings_inventory contains {g1}, {g2}, and {g3} with integer scores "
        f"from 1 to 10, reflecting how valuable each good is to you given "
        f"your CURRENT inventory.\n\n"
        f"ratings_general contains {g1}, {g2}, and {g3} with integer scores "
        f"from 1 to 10, reflecting how valuable each good is to you setting "
        f"aside your current inventory.\n\n"
        f"desired_bundle contains {g1}, {g2}, and {g3} with "
        f"nonnegative integers summing to exactly 6 — the bundle of 6 total "
        f"units you would choose if you could pick any combination.\n\n"
        f"Example:\n"
        f"{{\n"
        f'  "ratings_inventory": {_fmt_dict(current_vals)},\n'
        f'  "ratings_general": {_fmt_dict(abstract_vals)},\n'
        f'  "desired_bundle": {_fmt_dict(bundle_vals)}\n'
        f"}}"
    )


# -------------------------------------------------------------------
# Message builders
# -------------------------------------------------------------------

def build_negotiation_first_messages(
    player,
    prompts,
    goods: Iterable[str],
    round_index: int,
    partner_name: str,
    action_space: str,
    display_order: List[str],
    trade_history: Optional[TradeHistory] = None,
    negotiation_history: Optional[NegotiationHistory] = None,
    board_history: Optional[TradeHistory] = None,
    broadcast: bool = False,
) -> list[dict[str, str]]:
    """
    Build chat messages for the first negotiation turn.

    Generic message format:
    [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ]

    Provider wrappers can adapt this to OpenAI / Anthropic / Gemini / xAI.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts, display_order),
            render_persona_prompt(player, prompts, display_order),
        ]
    )

    user_parts = [
        render_world_state_prompt(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            partner_name=partner_name,
            display_order=display_order,
            trade_history=trade_history,
            negotiation_history=negotiation_history,
            board_history=board_history,
            broadcast=broadcast,
        ),
        render_negotiation_first_message_prompt(
            prompts=prompts,
            partner_name=partner_name,
            action_space=action_space,
        ),
        render_response_format_instruction(prompts, "negotiation", action_space=action_space),
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_negotiation_first_messages_cacheable(
    player,
    prompts,
    goods: Iterable[str],
    round_index: int,
    partner_name: str,
    action_space: str,
    display_order: List[str],
    trade_history: Optional[TradeHistory] = None,
    negotiation_history: Optional[NegotiationHistory] = None,
    board_history: Optional[TradeHistory] = None,
    broadcast: bool = False,
) -> tuple[str, str, str]:
    """
    Like build_negotiation_first_messages, but split into
    (system, history_text, volatile_text) instead of one joined user
    string, for callers that support prompt caching (claude_agent.py).

    history_text is the world-state block (round/inventory/trade
    history/negotiation history[/board]). Within a single round's
    negotiation, round_index/inventory/trade_history are constant and
    negotiation_history only ever grows by appending lines, so history_text
    is an exact-prefix match of the previous turn's history_text plus new
    content at the end — the shape prompt caching is built to reuse.
    volatile_text (the turn-specific prompt + response format instructions)
    changes every call and should never be cached.

    history_text + "\n\n" + volatile_text reproduces the exact same string
    build_negotiation_first_messages would have put in the user message, so
    callers can still log/inspect the flat form losslessly.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts, display_order),
            render_persona_prompt(player, prompts, display_order),
        ]
    )

    history_text = render_world_state_prompt(
        player=player,
        prompts=prompts,
        goods=goods,
        round_index=round_index,
        partner_name=partner_name,
        display_order=display_order,
        trade_history=trade_history,
        negotiation_history=negotiation_history,
        board_history=board_history,
        broadcast=broadcast,
    )

    volatile_text = "\n\n".join(
        [
            render_negotiation_first_message_prompt(
                prompts=prompts,
                partner_name=partner_name,
                action_space=action_space,
            ),
            render_response_format_instruction(prompts, "negotiation", action_space=action_space),
        ]
    )

    return system, history_text, volatile_text


def build_negotiation_response_messages(
    player,
    prompts,
    goods: Iterable[str],
    round_index: int,
    partner_name: str,
    partner_message: str,
    action_space: str,
    display_order: List[str],
    trade_history: Optional[TradeHistory] = None,
    negotiation_history: Optional[NegotiationHistory] = None,
    board_history: Optional[TradeHistory] = None,
    broadcast: bool = False,
) -> list[dict[str, str]]:
    """
    Build chat messages for responding during negotiation.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts, display_order),
            render_persona_prompt(player, prompts, display_order),
        ]
    )

    user_parts = [
        render_world_state_prompt(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            partner_name=partner_name,
            display_order=display_order,
            trade_history=trade_history,
            negotiation_history=negotiation_history,
            board_history=board_history,
            broadcast=broadcast,
        ),
        render_negotiation_response_prompt(
            prompts=prompts,
            partner_name=partner_name,
            partner_message=partner_message,
            action_space=action_space,
        ),
        render_response_format_instruction(prompts, "negotiation", action_space=action_space),
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_negotiation_response_messages_cacheable(
    player,
    prompts,
    goods: Iterable[str],
    round_index: int,
    partner_name: str,
    partner_message: str,
    action_space: str,
    display_order: List[str],
    trade_history: Optional[TradeHistory] = None,
    negotiation_history: Optional[NegotiationHistory] = None,
    board_history: Optional[TradeHistory] = None,
    broadcast: bool = False,
) -> tuple[str, str, str]:
    """
    Cache-aware counterpart to build_negotiation_response_messages — see
    build_negotiation_first_messages_cacheable for the rationale. Note
    partner_message lives in volatile_text (it's the newest, just-arrived
    turn, not part of the growing history block).
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts, display_order),
            render_persona_prompt(player, prompts, display_order),
        ]
    )

    history_text = render_world_state_prompt(
        player=player,
        prompts=prompts,
        goods=goods,
        round_index=round_index,
        partner_name=partner_name,
        display_order=display_order,
        trade_history=trade_history,
        negotiation_history=negotiation_history,
        board_history=board_history,
        broadcast=broadcast,
    )

    volatile_text = "\n\n".join(
        [
            render_negotiation_response_prompt(
                prompts=prompts,
                partner_name=partner_name,
                partner_message=partner_message,
                action_space=action_space,
            ),
            render_response_format_instruction(prompts, "negotiation", action_space=action_space),
        ]
    )

    return system, history_text, volatile_text


def build_commitment_messages(
    player,
    prompts,
    goods: Iterable[str],
    proposed_trade: Mapping[str, Any],
    display_order: List[str],
    round_index: int = 0,
    partner_name: str = "your partner",
    negotiation_history: Optional[NegotiationHistory] = None,
    board_history: Optional[Any] = None,
    broadcast: bool = False,
) -> list[dict[str, str]]:
    """
    Build chat messages for the commitment phase.

    The commitment now sees the same negotiation history the negotiation
    phase produced, so the accept/reject decision is made in light of the
    full back-and-forth rather than the bare offer text. Under the broadcast
    condition the public market bulletin board is also shown.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts, display_order),
            render_persona_prompt(player, prompts, display_order),
        ]
    )

    user_parts = [
        render_commitment_context(
            prompts=prompts,
            round_index=round_index,
            partner_name=partner_name,
            negotiation_history=negotiation_history,
        ),
    ]
    if broadcast:
        user_parts.append(render_bulletin_board_section(prompts, board_history))
    user_parts.append(
        render_commitment_prompt(
            player=player,
            prompts=prompts,
            goods=goods,
            proposed_trade=proposed_trade,
            display_order=display_order,
        )
    )
    user_parts.append(render_response_format_instruction(prompts, "commitment"))

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_commitment_messages_cacheable(
    player,
    prompts,
    goods: Iterable[str],
    proposed_trade: Mapping[str, Any],
    display_order: List[str],
    round_index: int = 0,
    partner_name: str = "your partner",
    negotiation_history: Optional[NegotiationHistory] = None,
    board_history: Optional[Any] = None,
    broadcast: bool = False,
) -> tuple[str, str, str]:
    """
    Cache-aware counterpart to build_commitment_messages. history_text is
    the round + negotiation-so-far context (grows the same append-only way
    as the negotiation calls' world-state block, so it's incrementally
    cacheable within a round). volatile_text is the specific offer being
    decided on (they_give/you_give) plus the response format instructions —
    the actual decision point, never cached.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts, display_order),
            render_persona_prompt(player, prompts, display_order),
        ]
    )

    history_parts = [
        render_commitment_context(
            prompts=prompts,
            round_index=round_index,
            partner_name=partner_name,
            negotiation_history=negotiation_history,
        ),
    ]
    if broadcast:
        history_parts.append(render_bulletin_board_section(prompts, board_history))
    history_text = "\n\n".join(history_parts)

    volatile_text = "\n\n".join(
        [
            render_commitment_prompt(
                player=player,
                prompts=prompts,
                goods=goods,
                proposed_trade=proposed_trade,
                display_order=display_order,
            ),
            render_response_format_instruction(prompts, "commitment"),
        ]
    )

    return system, history_text, volatile_text


def build_preference_probe_messages(
    player,
    prompts,
    display_order: List[str],
    goods: Iterable[str] = ("A", "B", "C"),
    round_index: int = 0,
    trade_history: Optional[TradeHistory] = None,
    board_history: Optional[Any] = None,
    broadcast: bool = False,
) -> list[dict[str, str]]:
    """
    Build chat messages for preference elicitation.

    display_order controls the ordering of every A/B/C mention in the probe:
    the inventory line in the context, the question wording, the schema
    description, and the example JSON. Together these prevent the model from
    anchoring on an A-first order.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts, display_order),
            render_persona_prompt(player, prompts, display_order),
        ]
    )

    user_parts = [
        render_probe_context(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            display_order=display_order,
            trade_history=trade_history,
        ),
    ]
    if broadcast:
        user_parts.append(render_bulletin_board_section(prompts, board_history))
    user_parts.append(render_preference_elicitation_prompt(prompts, display_order))
    user_parts.append(
        render_response_format_instruction(
            prompts, "preference_probe", display_order=display_order
        )
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_preference_probe_messages_cacheable(
    player,
    prompts,
    display_order: List[str],
    goods: Iterable[str] = ("A", "B", "C"),
    round_index: int = 0,
    trade_history: Optional[TradeHistory] = None,
    board_history: Optional[Any] = None,
    broadcast: bool = False,
) -> tuple[str, str, str]:
    """
    Cache-aware counterpart to build_preference_probe_messages. history_text
    is the round + inventory + trade-history-so-far context (grows
    append-only across rounds). volatile_text is the fixed probe question
    plus its response format instructions — static per run, but kept
    separate from history_text since it doesn't grow and isn't worth
    chasing here.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts, display_order),
            render_persona_prompt(player, prompts, display_order),
        ]
    )

    history_parts = [
        render_probe_context(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            display_order=display_order,
            trade_history=trade_history,
        ),
    ]
    if broadcast:
        history_parts.append(render_bulletin_board_section(prompts, board_history))
    history_text = "\n\n".join(history_parts)

    volatile_text = "\n\n".join(
        [
            render_preference_elicitation_prompt(prompts, display_order),
            render_response_format_instruction(
                prompts, "preference_probe", display_order=display_order
            ),
        ]
    )

    return system, history_text, volatile_text