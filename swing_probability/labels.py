"""Explicit Statcast description-to-swing taxonomy."""

from __future__ import annotations

from collections.abc import Iterable


# Covers every description printed by the committed 2022-2024 notebook, plus
# the pitchout swing variants used by some Statcast seasons.
SWING_DESCRIPTIONS = frozenset(
    {
        "bunt_foul_tip",
        "foul",
        "foul_bunt",
        "foul_pitchout",
        "foul_tip",
        "hit_into_play",
        "missed_bunt",
        "swinging_pitchout",
        "swinging_strike",
        "swinging_strike_blocked",
    }
)

TAKE_DESCRIPTIONS = frozenset(
    {
        "ball",
        "blocked_ball",
        "called_strike",
        "hit_by_pitch",
        "pitchout",
    }
)


def classify_description(description: str) -> int:
    """Return 1 for a swing and 0 for a take; reject unknown outcomes."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Pitch descriptions must be nonempty strings")
    normalized = description.strip().casefold()
    if normalized in SWING_DESCRIPTIONS:
        return 1
    if normalized in TAKE_DESCRIPTIONS:
        return 0
    raise ValueError(
        f"Unknown pitch description {description!r}; update the taxonomy explicitly"
    )


def label_descriptions(descriptions: Iterable[str]) -> tuple[int, ...]:
    """Classify an iterable while reporting the position of invalid data."""
    labels: list[int] = []
    for position, description in enumerate(descriptions):
        try:
            labels.append(classify_description(description))
        except ValueError as exc:
            raise ValueError(f"Invalid description at position {position}: {exc}") from exc
    return tuple(labels)
