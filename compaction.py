"""Context window management: two-layer compression for long conversations."""
from __future__ import annotations

import providers


# ── Token estimation ──────────────────────────────────────────────────────

def estimate_tokens(messages: list) -> int:
    """Estimate token count by summing content lengths / 3.5.

    Args:
        messages: list of message dicts with "content" field (str or list of dicts)
    Returns:
        approximate token count, int
    """
    total_chars = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    # Sum all string values in the block
                    for v in block.values():
                        if isinstance(v, str):
                            total_chars += len(v)
        # Also count tool_calls if present
        for tc in m.get("tool_calls", []):
            if isinstance(tc, dict):
                for v in tc.values():
                    if isinstance(v, str):
                        total_chars += len(v)
    return int(total_chars / 3.5)


def get_context_limit(model: str) -> int:
    """Look up context window size for a model.

    Args:
        model: model string (e.g. "claude-opus-4-6", "ollama/llama3.3")
    Returns:
        context limit in tokens
    """
    provider_name = providers.detect_provider(model)
    prov = providers.PROVIDERS.get(provider_name, {})
    return prov.get("context_limit", 128000)


# ── Layer 1: Snip old tool results ────────────────────────────────────────

def snip_old_tool_results(
    messages: list,
    max_chars: int = 2000,
    preserve_last_n_turns: int = 6,
) -> list:
    """Truncate tool-role messages older than preserve_last_n_turns from end.

    For old tool messages whose content exceeds max_chars, keep the first half
    and last quarter, inserting '[... N chars snipped ...]' in between.
    Mutates in place and returns the same list.

    Args:
        messages: list of message dicts (mutated in place)
        max_chars: maximum character length before truncation
        preserve_last_n_turns: number of messages from end to preserve
    Returns:
        the same messages list (mutated)
    """
    cutoff = max(0, len(messages) - preserve_last_n_turns)
    for i in range(cutoff):
        m = messages[i]
        if m.get("role") != "tool":
            continue
        content = m.get("content", "")
        if not isinstance(content, str) or len(content) <= max_chars:
            continue
        first_half = content[: max_chars // 2]
        last_quarter = content[-(max_chars // 4):]
        snipped = len(content) - len(first_half) - len(last_quarter)
        m["content"] = f"{first_half}\n[... {snipped} chars snipped ...]\n{last_quarter}"
    return messages


# ── Layer 2: Auto-compact ─────────────────────────────────────────────────

def find_split_point(messages: list, keep_ratio: float = 0.3) -> int:
