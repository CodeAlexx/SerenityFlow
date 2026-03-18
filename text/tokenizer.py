"""Prompt parsing, weighting, and tokenization utilities."""

from __future__ import annotations

import logging

__all__ = [
    "build_token_weight_map",
    "create_token_chunks",
    "has_non_default_weights",
    "parse_prompt_weights",
    "split_segments_at_break",
    "truncate_or_pad",
]

logger = logging.getLogger(__name__)

# Default weight for un-annotated text
_DEFAULT_WEIGHT = 1.0
# Weight boost for bare parentheses (word) without explicit weight
_BARE_PAREN_WEIGHT = 1.1


# ---------------------------------------------------------------------------
# Prompt weight parsing
# ---------------------------------------------------------------------------


def parse_prompt_weights(prompt: str) -> list[tuple[str, float]]:
    """Parse prompt weighting syntax into (text, weight) segments.

    Syntax rules:
      - Plain text gets weight 1.0
      - ``(word:1.5)`` sets weight to 1.5
      - ``(word)`` without explicit weight applies 1.1 multiplier
      - Nested parentheses multiply weights
      - ``BREAK`` keyword splits segments
      - Escaped parentheses ``\\(`` ``\\)`` are treated as literal text

    Returns:
        List of (text_segment, weight) tuples.
    """
    if not prompt:
        return [("", _DEFAULT_WEIGHT)]

    result: list[tuple[str, float]] = []
    weight_stack: list[float] = [_DEFAULT_WEIGHT]
    current_text: list[str] = []
    i = 0
    length = len(prompt)

    while i < length:
        char = prompt[i]

        # Handle escaped parentheses
        if char == "\\" and i + 1 < length and prompt[i + 1] in ("(", ")"):
            current_text.append(prompt[i + 1])
            i += 2
            continue

        # Handle BREAK keyword
        if prompt[i:i + 5] == "BREAK":
            before_ok = i == 0 or not prompt[i - 1].isalnum()
            after_ok = i + 5 >= length or not prompt[i + 5].isalnum()
            if before_ok and after_ok:
                text = "".join(current_text)
                if text:
                    result.append((text, weight_stack[-1]))
                    current_text = []
                result.append(("BREAK", _DEFAULT_WEIGHT))
                i += 5
                continue

        # Opening parenthesis — push weight
        if char == "(":
            text = "".join(current_text)
            if text:
                result.append((text, weight_stack[-1]))
                current_text = []

            i += 1

            # Find matching close paren considering nesting
            depth = 1
            j = i
            while j < length and depth > 0:
                if prompt[j] == "\\" and j + 1 < length and prompt[j + 1] in ("(", ")"):
                    j += 2
                    continue
                if prompt[j] == "(":
                    depth += 1
                elif prompt[j] == ")":
                    depth -= 1
                j += 1

            if depth == 0:
                inner = prompt[i:j - 1]

                colon_pos = _find_weight_colon(inner)
                if colon_pos is not None:
                    weight_text = inner[colon_pos + 1:]
                    inner_prompt = inner[:colon_pos]
                    try:
                        w = float(weight_text)
                        weight_stack.append(w)
                        inner_segments = parse_prompt_weights(inner_prompt)
                        for seg_text, seg_weight in inner_segments:
                            if seg_text:
                                result.append((seg_text, w * (seg_weight / _DEFAULT_WEIGHT) if seg_weight == _DEFAULT_WEIGHT else seg_weight))
                        weight_stack.pop()
                        i = j
                        continue
                    except ValueError:
                        pass

                # Bare parentheses — apply 1.1 multiplier
                new_weight = weight_stack[-1] * _BARE_PAREN_WEIGHT
                weight_stack.append(new_weight)
                inner_segments = parse_prompt_weights(inner)
                for seg_text, seg_weight in inner_segments:
                    if seg_text:
                        if seg_weight == _DEFAULT_WEIGHT:
                            result.append((seg_text, new_weight))
                        else:
                            result.append((seg_text, seg_weight))
                weight_stack.pop()
                i = j
                continue

            # Unmatched paren — treat as literal
            current_text.append("(")
            continue

        # Closing parenthesis without matching open — treat as literal
        if char == ")":
            current_text.append(")")
            i += 1
            continue

        current_text.append(char)
        i += 1

    # Flush remaining text
    text = "".join(current_text)
    if text:
        result.append((text, weight_stack[-1]))

    if not result:
        return [("", _DEFAULT_WEIGHT)]

    return result


def _find_weight_colon(text: str) -> int | None:
    """Find the position of a weight-specifying colon in parenthesised text."""
    depth = 0
    last_colon = None
    for i, ch in enumerate(text):
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in ("(", ")"):
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            last_colon = i

    if last_colon is not None:
        remainder = text[last_colon + 1:]
        try:
            float(remainder)
            return last_colon
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Weight utilities
# ---------------------------------------------------------------------------


def has_non_default_weights(segments: list[tuple[str, float]]) -> bool:
    """Return ``True`` if any segment has a weight different from 1.0."""
    return any(w != _DEFAULT_WEIGHT for _, w in segments)


def split_segments_at_break(
    segments: list[tuple[str, float]],
) -> list[list[tuple[str, float]]]:
    """Split parsed segments into groups separated by BREAK."""
    groups: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = []
    for text, weight in segments:
        if text == "BREAK":
            if current:
                groups.append(current)
            current = []
        else:
            current.append((text, weight))
    if current:
        groups.append(current)
    if not groups:
        groups.append([("", _DEFAULT_WEIGHT)])
    return groups


def build_token_weight_map(
    segments: list[tuple[str, float]],
    tokenize_fn: object,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    max_length: int = 77,
) -> tuple[list[int], list[float]]:
    """Build per-token weight array from weighted text segments.

    Tokenizes each segment individually to determine token boundaries,
    then assembles a combined token list and matching weight list.

    Special tokens (BOS, EOS, PAD) always get weight 1.0.
    """
    all_tokens: list[int] = []
    all_weights: list[float] = []

    content_capacity = max_length
    if bos_token_id is not None:
        content_capacity -= 1
    if eos_token_id is not None:
        content_capacity -= 1

    for text, weight in segments:
        if not text:
            continue
        seg_tokens = tokenize_fn(text)  # type: ignore[operator]
        for tok in seg_tokens:
            if len(all_tokens) >= content_capacity:
                break
            all_tokens.append(tok)
            all_weights.append(weight)

    final_tokens: list[int] = []
    final_weights: list[float] = []

    if bos_token_id is not None:
        final_tokens.append(bos_token_id)
        final_weights.append(_DEFAULT_WEIGHT)

    final_tokens.extend(all_tokens)
    final_weights.extend(all_weights)

    if eos_token_id is not None:
        if len(final_tokens) < max_length:
            final_tokens.append(eos_token_id)
            final_weights.append(_DEFAULT_WEIGHT)

    if pad_token_id is not None:
        while len(final_tokens) < max_length:
            final_tokens.append(pad_token_id)
            final_weights.append(_DEFAULT_WEIGHT)

    final_tokens = final_tokens[:max_length]
    final_weights = final_weights[:max_length]

    return final_tokens, final_weights


# ---------------------------------------------------------------------------
# Token manipulation
# ---------------------------------------------------------------------------


def truncate_or_pad(
    tokens: list[int],
    max_length: int,
    pad_token: int,
) -> list[int]:
    """Truncate or pad a token list to exactly *max_length*."""
    if len(tokens) >= max_length:
        return tokens[:max_length]
    return tokens + [pad_token] * (max_length - len(tokens))


def create_token_chunks(
    tokens: list[int],
    weights: list[float],
    max_chunk_length: int,
) -> list[tuple[list[int], list[float]]]:
    """Split tokens and weights into fixed-size chunks for multi-pass encoding."""
    if not tokens:
        return [([], [])]

    chunks: list[tuple[list[int], list[float]]] = []
    for start in range(0, len(tokens), max_chunk_length):
        end = start + max_chunk_length
        chunks.append((tokens[start:end], weights[start:end]))

    return chunks
