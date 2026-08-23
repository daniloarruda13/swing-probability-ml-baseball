"""Validated feature selection for pitch-level swing models."""

from __future__ import annotations


PLAYER_ID_FEATURES = ("batter", "pitcher")
CATEGORICAL_FEATURES = ("pitch_type", "stand", "p_throws", "inning_topbot")
NUMERIC_FEATURES = (
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "release_extension",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "sz_top",
    "sz_bot",
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "bat_score",
    "fld_score",
)


def feature_columns(include_player_ids: bool = False) -> tuple[str, ...]:
    """Return the stable model schema in deterministic column order."""
    ids = PLAYER_ID_FEATURES if include_player_ids else ()
    return ids + CATEGORICAL_FEATURES + NUMERIC_FEATURES


def prepare_pitch_data(frame, *, require_target: bool = True, include_player_ids: bool = False):
    """Select model columns and optionally derive the strict binary target.

    Missing feature values are retained for train-fitted pipeline imputation.
    Player IDs are excluded by default to avoid treating identifiers as ordered
    numeric measurements and to support predictions for unseen players.
    """
    import pandas as pd

    from .labels import label_descriptions

    columns = feature_columns(include_player_ids)
    required = set(columns)
    if require_target:
        required.add("description")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    features = frame.loc[:, columns].copy()
    if include_player_ids:
        for column in PLAYER_ID_FEATURES:
            features[column] = features[column].astype("string")

    if not require_target:
        return features
    labels = label_descriptions(frame["description"].tolist())
    target = pd.Series(labels, index=frame.index, name="swing", dtype="int8")
    return features, target


def middle_middle_pitches(
    frame,
    *,
    horizontal_half_width: float = 0.5,
    vertical_half_width: float = 0.5,
):
    """Return pitches within a configurable box around each strike-zone center."""
    if horizontal_half_width <= 0 or vertical_half_width <= 0:
        raise ValueError("Middle-middle half-widths must be positive")
    required = {"plate_x", "plate_z", "sz_top", "sz_bot"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    usable = frame.dropna(subset=sorted(required)).copy()
    center = (usable["sz_top"] + usable["sz_bot"]) / 2
    mask = (
        usable["plate_x"].abs().lt(horizontal_half_width)
        & usable["plate_z"].sub(center).abs().lt(vertical_half_width)
    )
    result = usable.loc[mask].copy()
    result["center_strike_zone"] = center.loc[mask]
    return result
