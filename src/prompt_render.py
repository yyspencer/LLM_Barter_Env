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
from typing import Any, Iterable, Mapping, Optional


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
    Format inventory for prompt injection.

    compact:
        A×1, B×0, C×3

    lines:
        A: 1
        B: 0
        C: 3
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
    """
    if not negotiation_history:
        return empty_text

    lines: list[str] = []
    for msg in negotiation_history:
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
    board_history: Optional[TradeHistory],
    empty_text: str = "No public trades have been posted.",
) -> str:
    """
    Format public trade-board history for broadcast condition.
    """
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

def render_system_prompt(prompts) -> str:
    """Render the shared system prompt."""
    return prompts.system_prompt.strip()


def render_persona_prompt(player, prompts) -> str:
    """
    Render persona prompt from persona_template and player fields.
    """
    return safe_format(
        prompts.persona_template,
        display_name=player.display_name,
        preference_description=player.preference_description,
        role=getattr(player, "role", None) or "",
        player_id=player.id,
    ).strip()


def render_world_state_prompt(
    player,
    prompts,
    goods: Iterable[str],
    round_index: int,
    partner_name: str,
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
    """
    goods_list = list(goods)
    inventory = player.inventory

    template = (
        prompts.world_state_broadcast_template
        if broadcast
        else prompts.world_state_control_template
    )

    values = {
        "round_index": round_index,
        "display_name": player.display_name,
        "player_id": player.id,
        "partner_name": partner_name,
        "inventory": format_inventory_for_prompt(inventory, goods_list, style="lines"),
        "inventory_compact": format_inventory_for_prompt(inventory, goods_list, style="compact"),
        "trade_history": format_trade_history(trade_history),
        "negotiation_history": format_negotiation_history(negotiation_history),
        "board_history": format_board_history(board_history),
    }

    # Also expose num_A, num_B, num_C, etc. for templates like:
    # A×{num_A}, B×{num_B}, C×{num_C}
    for good in goods_list:
        values[f"num_{good}"] = inventory.get(good, 0)

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
) -> str:
    """
    Render the commitment prompt for accepting/rejecting a proposed trade.
    """
    inventory_text = format_inventory_for_prompt(player.inventory, goods, style="lines")
    proposed_trade_text = format_json_block(proposed_trade)

    return safe_format(
        prompts.commitment_prompt,
        proposed_trade=proposed_trade_text,
        inventory=inventory_text,
        preference_description=player.preference_description,
        display_name=player.display_name,
        player_id=player.id,
        role=getattr(player, "role", None) or "",
    ).strip()


def render_preference_elicitation_prompt(prompts) -> str:
    """
    Render the preference elicitation probe.

    This is intentionally separated from current holdings / negotiation context
    because the probe asks the agent to set aside current holdings.
    """
    return prompts.preference_elicitation_prompt.strip()


def render_response_format_instruction(prompts, format_name: str) -> str:
    """
    Render the schema description for one response type.

    format_name examples:
    - negotiation
    - commitment
    - preference_probe
    """
    if format_name not in prompts.response_formats:
        raise KeyError(
            f"Response format '{format_name}' not found. "
            f"Available: {sorted(prompts.response_formats.keys())}"
        )

    spec = prompts.response_formats[format_name]
    return spec.schema_description.strip()


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
            render_system_prompt(prompts),
            render_persona_prompt(player, prompts),
        ]
    )

    user_parts = [
        render_world_state_prompt(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            partner_name=partner_name,
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
        render_response_format_instruction(prompts, "negotiation"),
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_negotiation_response_messages(
    player,
    prompts,
    goods: Iterable[str],
    round_index: int,
    partner_name: str,
    partner_message: str,
    action_space: str,
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
            render_system_prompt(prompts),
            render_persona_prompt(player, prompts),
        ]
    )

    user_parts = [
        render_world_state_prompt(
            player=player,
            prompts=prompts,
            goods=goods,
            round_index=round_index,
            partner_name=partner_name,
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
        render_response_format_instruction(prompts, "negotiation"),
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_commitment_messages(
    player,
    prompts,
    goods: Iterable[str],
    proposed_trade: Mapping[str, Any],
) -> list[dict[str, str]]:
    """
    Build chat messages for the commitment phase.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts),
            render_persona_prompt(player, prompts),
        ]
    )

    user_parts = [
        render_commitment_prompt(
            player=player,
            prompts=prompts,
            goods=goods,
            proposed_trade=proposed_trade,
        ),
        render_response_format_instruction(prompts, "commitment"),
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_preference_probe_messages(player, prompts) -> list[dict[str, str]]:
    """
    Build chat messages for preference elicitation.

    The persona is included so the model remembers its assigned role, but the
    actual prompt asks it to set aside current holdings.
    """
    system = "\n\n".join(
        [
            render_system_prompt(prompts),
            render_persona_prompt(player, prompts),
        ]
    )

    user_parts = [
        render_preference_elicitation_prompt(prompts),
        render_response_format_instruction(prompts, "preference_probe"),
    ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
