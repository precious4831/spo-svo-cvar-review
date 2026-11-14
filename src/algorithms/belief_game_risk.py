"""Pure NumPy social-belief / Level-1 game-risk urgency.

This module has no dependency on CARLA, PyTorch, or the training loop.  It
consumes the already-produced SVO posterior, belief-conditioned response
trajectories, and the flat observation used by the encoder.

The project trajectory encoder stores yaw in radians (it rotates trajectories
with ``sin(yaw)`` and ``cos(yaw)``).  An optional ``encoder_cfg.yaw_unit``
override is accepted for isolated degree-valued fixtures.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


RISK_CONDITIONING_MODES = ("fixed", "svo", "belief_game")


def _section(config: Any, name: str) -> Any:
    """Return a nested config section, or ``config`` for section-only tests."""

    section = getattr(config, name, None)
    return config if section is None else section


def _as_finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _encoder_layout(encoder_cfg: Any) -> Tuple[int, int, int, int, int, int]:
    """Read and validate the flat-observation layout.

    Returns ``(N, T, M, A+B, feature_dim)`` where N is the neighbour count,
    T is the history length, M is the trajectory segment length, and A+B is
    the auxiliary plus BV-action segment length.
    """

    try:
        n = int(getattr(encoder_cfg, "num_neighbours"))
        history_steps = int(getattr(encoder_cfg, "history_steps"))
        feature_dim = int(getattr(encoder_cfg, "bv_action_feat_dim", 3))
        trajs_dim = int(getattr(
            encoder_cfg,
            "trajs_flat_dim",
            (1 + n) * history_steps * 5,
        ))
        map_dim = int(getattr(encoder_cfg, "map_flat_dim"))
        aux_dim = int(getattr(encoder_cfg, "aux_state_dim"))
        use_bv = bool(getattr(encoder_cfg, "use_bv_actions_in_obs", True))
        bv_dim = int(getattr(
            encoder_cfg,
            "bv_actions_flat_dim",
            n * feature_dim if use_bv else 0,
        ))
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid encoder observation layout") from exc

    if n < 0 or history_steps < 1:
        raise ValueError(
            "num_neighbours must be non-negative and history_steps must be positive"
        )
    if min(trajs_dim, map_dim, aux_dim, bv_dim) < 0:
        raise ValueError("observation segment lengths must be non-negative")
    return n, history_steps, trajs_dim, map_dim, aux_dim + bv_dim, feature_dim


def _expected_obs_dim(encoder_cfg: Any) -> int:
    """Return the configured flat observation size without importing models."""

    _, _, trajs_dim, map_dim, _, _ = _encoder_layout(encoder_cfg)
    aux_dim = int(getattr(encoder_cfg, "aux_state_dim"))
    bv_dim = int(getattr(encoder_cfg, "bv_actions_flat_dim", 0))
    return int(trajs_dim + map_dim + aux_dim + bv_dim)


def _flat_float_array(flat_obs: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(flat_obs, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if arr.ndim != 1 or not np.isfinite(arr).all():
        return None
    return arr


def _extract_bv_actions_details(
    flat_obs: Any,
    encoder_cfg: Any,
) -> Tuple[Optional[np.ndarray], str]:
    arr = _flat_float_array(flat_obs)
    if arr is None:
        return None, "flat_obs_not_finite_1d"

    try:
        n, _, trajs_dim, map_dim, _, feature_dim = _encoder_layout(encoder_cfg)
        use_bv = bool(getattr(encoder_cfg, "use_bv_actions_in_obs", True))
        aux_dim = int(getattr(encoder_cfg, "aux_state_dim"))
        bv_dim = int(getattr(encoder_cfg, "bv_actions_flat_dim", 0))
    except ValueError as exc:
        return None, str(exc)

    if not use_bv:
        return None, "bv_actions_disabled"
    if feature_dim != 3:
        return None, "bv_action_feature_dim_is_not_3"
    if bv_dim != n * 3:
        return None, "bv_actions_flat_dim_mismatch"

    offset = trajs_dim + map_dim + aux_dim
    end = offset + bv_dim
    if arr.size < end:
        return None, "flat_obs_too_short_for_bv_actions"

    values = arr[offset:end]
    if values.size != n * 3 or not np.isfinite(values).all():
        return None, "bv_actions_not_finite_or_wrong_size"
    try:
        actions = values.reshape(n, 3).copy()
    except ValueError:
        return None, "bv_actions_reshape_failed"
    if not np.isfinite(actions[:, 2]).all():
        return None, "bv_action_valid_mask_not_finite"
    return actions, ""


def extract_bv_actions(
    flat_obs: Any,
    encoder_cfg: Any,
) -> Tuple[Optional[np.ndarray], bool]:
    """Extract aligned BV actions as ``(N, 3)`` and a validity flag.

    Invalid input returns ``(None, False)`` rather than exposing a low-level
    reshape exception.  The main entry point uses the reason-bearing internal
    variant so fallback diagnostics remain actionable.
    """

    actions, reason = _extract_bv_actions_details(flat_obs, encoder_cfg)
    return actions, bool(actions is not None and not reason)


def extract_latest_ego_state(
    flat_obs: Any,
    encoder_cfg: Any,
) -> Optional[np.ndarray]:
    """Extract latest ego state with velocity expressed in ego coordinates.

    The returned shape is ``(5,)`` and keeps the state order
    ``[x, y, yaw, v_ego_x, v_ego_y]``.  The source trajectory order is
    ``[x, y, yaw, vx, vy]`` in global coordinates.
    """

    arr = _flat_float_array(flat_obs)
    if arr is None:
        return None
    try:
        n, history_steps, trajs_dim, _, _, _ = _encoder_layout(encoder_cfg)
    except ValueError:
        return None
    expected_trajs_dim = (1 + n) * history_steps * 5
    if trajs_dim != expected_trajs_dim or arr.size < trajs_dim:
        return None

    try:
        trajs = arr[:trajs_dim].reshape(1 + n, history_steps, 5)
    except ValueError:
        return None
    latest = trajs[0, -1].copy()
    if not np.isfinite(latest).all():
        return None

    yaw = float(latest[2])
    yaw_unit = str(getattr(encoder_cfg, "yaw_unit", "radians")).lower()
    if yaw_unit in ("degree", "degrees", "deg"):
        yaw_rad = np.deg2rad(yaw)
    elif yaw_unit in ("radian", "radians", "rad"):
        yaw_rad = yaw
    else:
        return None

    vx, vy = float(latest[3]), float(latest[4])
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    latest[3] = vx * c + vy * s
    latest[4] = -vx * s + vy * c
    return latest if np.isfinite(latest).all() else None


def _stable_sigmoid(value: float) -> float:
    """Numerically stable sigmoid that always returns a finite scalar."""

    if np.isnan(value):
        return 0.0
    if value >= 0.0:
        if np.isposinf(value):
            return 1.0
        exp_value = np.exp(-value)
        result = 1.0 / (1.0 + exp_value)
    else:
        if np.isneginf(value):
            return 0.0
        exp_value = np.exp(value)
        result = exp_value / (1.0 + exp_value)
    return float(np.clip(result, 0.0, 1.0)) if np.isfinite(result) else 0.0


def validate_game_risk_config(config: Any) -> str:
    """Validate risk mode and scalar parameters, returning the mode."""

    ppo = _section(config, "ppo")
    mode = str(getattr(ppo, "risk_conditioning_mode", "svo"))
    if mode not in RISK_CONDITIONING_MODES:
        raise ValueError(
            "risk_conditioning_mode must be one of 'fixed', 'svo', or "
            f"'belief_game'; got {mode!r}"
        )

    nonnegative = (
        "game_risk_base_weight",
        "game_risk_proximity_weight",
        "game_risk_response_brake_weight",
        "game_risk_sigma_cap",
        "game_risk_urgency_max",
    )
    for name in nonnegative:
        defaults = {
            "game_risk_base_weight": 0.0,
            "game_risk_proximity_weight": 0.7,
            "game_risk_response_brake_weight": 0.3,
            "game_risk_sigma_cap": 1.0,
            "game_risk_urgency_max": 1.0,
        }
        value = _as_finite_float(getattr(ppo, name, defaults[name]), np.nan)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")

    floor = _as_finite_float(getattr(ppo, "game_risk_floor", 0.1), np.nan)
    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("game_risk_floor must be finite and in [0, 1]")

    safe_distance = _as_finite_float(
        getattr(ppo, "game_risk_safe_distance_m", 12.0), np.nan
    )
    if not np.isfinite(safe_distance) or safe_distance < 0.0:
        raise ValueError("game_risk_safe_distance_m must be finite and non-negative")
    for name in ("game_risk_distance_temperature_m", "game_risk_brake_scale"):
        defaults = {
            "game_risk_distance_temperature_m": 2.0,
            "game_risk_brake_scale": 1.0,
        }
        value = _as_finite_float(getattr(ppo, name, defaults[name]), np.nan)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    return mode


def _coerce_vector(value: Any, n: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if arr.ndim != 1 or arr.size != n or not np.isfinite(arr).all():
        return None
    return arr


def _coerce_mask(value: Any, n: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if raw.ndim != 1 or raw.size != n:
        return None
    if raw.dtype.kind in "fc" and not np.isfinite(raw).all():
        return None
    try:
        return raw.astype(bool)
    except (TypeError, ValueError):
        return None


def _coerce_pred_trajs(value: Any, n: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if arr.ndim != 3 or arr.shape[0] != n or arr.shape[1] < 1 or arr.shape[2] != 5:
        return None
    return arr


def _prior_sigma(config: Any) -> float:
    svo = _section(config, "svo")
    return _as_finite_float(getattr(svo, "prior_sigma", 1.0), 1.0)


def _empty_diagnostics(n: int, mode: str) -> Dict[str, Any]:
    zeros = np.zeros(n, dtype=np.float64)
    return {
        "mode": mode,
        "fallback": False,
        "fallback_reason": "",
        "critical_slot": -1,
        "valid_interaction_count": 0,
        "per_vehicle_social": zeros.copy(),
        # Zero is a finite invalid-distance sentinel; callers can use the
        # proximity and valid-trajectory arrays to distinguish it from 0 m.
        "per_vehicle_min_distance": zeros.copy(),
        "per_vehicle_proximity": zeros.copy(),
        "per_vehicle_response_brake": zeros.copy(),
        "per_vehicle_game_conflict": zeros.copy(),
        "per_vehicle_urgency": zeros.copy(),
        "per_vehicle_valid_interaction": np.zeros(n, dtype=bool),
        "per_vehicle_valid_trajectory": np.zeros(n, dtype=bool),
        "per_vehicle_valid_action": np.zeros(n, dtype=bool),
    }


def _set_fallback(diagnostics: Dict[str, Any], reason: str) -> None:
    diagnostics["fallback"] = True
    diagnostics["fallback_reason"] = str(reason)[:256]


def _svo_only_values(
    svo_mu: Any,
    svo_sigma: Any,
    interact_mask: Any,
    ppo: Any,
    prior_sigma: float,
    n: int,
) -> Tuple[float, np.ndarray, np.ndarray, int, int]:
    """Compute SVO-only urgency on a normalized selfishness scale."""

    social = np.zeros(n, dtype=np.float64)
    mu = _coerce_vector(svo_mu, n)
    sigma = _coerce_vector(svo_sigma, n)
    mask = _coerce_mask(interact_mask, n)
    if mu is None or sigma is None or mask is None:
        return 0.0, social, np.zeros(n, dtype=bool), 0, -1
    active = mask & np.isfinite(mu) & np.isfinite(sigma)
    if not active.any():
        return 0.0, social, active, 0, -1

    mu_deg = np.clip(mu[active].astype(np.float32), 0.0, 90.0)
    selfishness = (90.0 - mu_deg) / 90.0
    sg = np.clip(
        sigma[active].astype(np.float32) / max(float(prior_sigma), 1e-6),
        0.0,
        float(getattr(ppo, "game_risk_sigma_cap", 1.0)),
    )
    weights = (
        float(getattr(ppo, "svo_mu_budget_weight", 0.0))
        * selfishness
        + float(getattr(ppo, "svo_sigma_budget_weight", 0.0)) * sg
    )
    finite = np.isfinite(weights)
    active_indices = np.flatnonzero(active)
    social[active_indices[finite]] = weights[finite].astype(np.float64)
    valid = np.zeros(n, dtype=bool)
    valid[active_indices[finite]] = True
    if not finite.any():
        return 0.0, social, valid, 0, -1
    max_value = float(np.max(weights[finite]))
    urgency = max(0.0, max_value) if np.isfinite(max_value) else 0.0
    finite_indices = active_indices[finite]
    critical = int(finite_indices[np.argmax(weights[finite])])
    return urgency, social, valid, int(valid.sum()), critical


def _get_num_neighbours(config: Any) -> int:
    encoder = _section(config, "encoder")
    try:
        return max(0, int(getattr(encoder, "num_neighbours")))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0


def _fixed_result(n: int, mode: str) -> Tuple[float, Dict[str, Any]]:
    return 0.0, _empty_diagnostics(n, mode)


def compute_belief_game_urgency(
    flat_obs: Any,
    svo_mu: Any,
    svo_sigma: Any,
    pred_trajs: Any,
    interact_mask: Any,
    config: Any,
) -> Tuple[float, Dict[str, Any]]:
    """Compute scene-level social-belief / Level-1 game urgency.

    Expected input shapes are ``flat_obs=(D,)``, ``svo_mu=(N,)``,
    ``svo_sigma=(N,)``, ``pred_trajs=(N,H,5)``, and ``interact_mask=(N,)``.
    ``bv_actions=(N,3)`` is extracted from ``flat_obs`` using the encoder
    layout.  In ``belief_game`` mode malformed derived data falls back to the
    legacy SVO-only maximum; no interacting slot returns exactly zero without
    being marked as a fallback.
    """

    ppo = _section(config, "ppo")
    encoder = _section(config, "encoder")
    n = _get_num_neighbours(config)
    mode = validate_game_risk_config(config)
    diagnostics = _empty_diagnostics(n, mode)

    if mode == "fixed":
        return _fixed_result(n, mode)

    prior_sigma = _prior_sigma(config)
    svo_urgency, svo_social, svo_valid, svo_count, svo_critical = _svo_only_values(
        svo_mu, svo_sigma, interact_mask, ppo, prior_sigma, n
    )
    diagnostics["per_vehicle_valid_interaction"] = svo_valid.copy()

    if mode == "svo":
        diagnostics["per_vehicle_social"] = svo_social
        diagnostics["per_vehicle_urgency"] = np.maximum(svo_social, 0.0)
        diagnostics["valid_interaction_count"] = svo_count
        diagnostics["critical_slot"] = svo_critical if svo_count else -1
        return float(svo_urgency) if np.isfinite(svo_urgency) else 0.0, diagnostics

    # No active interaction means no risk, not a non-zero SVO fallback.
    mask = _coerce_mask(interact_mask, n)
    if mask is None:
        _set_fallback(diagnostics, "interact_mask_invalid")
        diagnostics["per_vehicle_social"] = svo_social
        return float(svo_urgency), diagnostics
    if not mask.any():
        return 0.0, diagnostics

    if not svo_valid.any():
        _set_fallback(diagnostics, "svo_posterior_invalid")
        return float(svo_urgency), diagnostics

    try:
        expected_dim = _expected_obs_dim(encoder)
    except ValueError as exc:
        _set_fallback(diagnostics, f"observation_layout_invalid:{exc}")
        diagnostics["per_vehicle_social"] = svo_social
        return float(svo_urgency), diagnostics
    flat_arr = _flat_float_array(flat_obs)
    if flat_arr is None:
        _set_fallback(diagnostics, "observation_not_finite_1d")
        diagnostics["per_vehicle_social"] = svo_social
        return float(svo_urgency), diagnostics
    if flat_arr.size != expected_dim:
        _set_fallback(
            diagnostics,
            f"observation_length_mismatch:expected={expected_dim},got={flat_arr.size}",
        )
        diagnostics["per_vehicle_social"] = svo_social
        return float(svo_urgency), diagnostics

    actions, action_reason = _extract_bv_actions_details(flat_arr, encoder)
    if actions is None:
        _set_fallback(diagnostics, action_reason or "bv_actions_unavailable")
        diagnostics["per_vehicle_social"] = svo_social
        return float(svo_urgency), diagnostics

    ego_state = extract_latest_ego_state(flat_arr, encoder)
    if ego_state is None:
        _set_fallback(diagnostics, "ego_trajectory_invalid")
        diagnostics["per_vehicle_social"] = svo_social
        return float(svo_urgency), diagnostics

    pred = _coerce_pred_trajs(pred_trajs, n)
    if pred is None:
        _set_fallback(diagnostics, "pred_trajs_missing_or_wrong_shape")
        diagnostics["per_vehicle_social"] = svo_social
        return float(svo_urgency), diagnostics

    valid_svo = mask & svo_valid
    valid_traj = valid_svo & np.isfinite(pred[:, :, 0:2]).all(axis=(1, 2))
    valid_action = valid_svo & np.isfinite(actions).all(axis=1) & (actions[:, 2] > 0.5)
    if not (valid_traj | valid_action).any():
        _set_fallback(diagnostics, "no_valid_interaction_data")
        diagnostics["per_vehicle_social"] = svo_social
        return float(svo_urgency), diagnostics

    diagnostics["per_vehicle_valid_trajectory"] = valid_traj
    diagnostics["per_vehicle_valid_action"] = valid_action

    dt = _as_finite_float(
        getattr(_section(config, "carla"), "fixed_delta_seconds", 0.05),
        0.05,
    )
    ppo_float = {
        "game_risk_safe_distance_m": float(
            getattr(ppo, "game_risk_safe_distance_m", 12.0)
        ),
        "game_risk_distance_temperature_m": float(
            getattr(ppo, "game_risk_distance_temperature_m", 2.0)
        ),
        "game_risk_brake_scale": float(
            getattr(ppo, "game_risk_brake_scale", 1.0)
        ),
        "game_risk_sigma_cap": float(getattr(ppo, "game_risk_sigma_cap", 1.0)),
        "game_risk_base_weight": float(
            getattr(ppo, "game_risk_base_weight", 0.0)
        ),
        "game_risk_proximity_weight": float(
            getattr(ppo, "game_risk_proximity_weight", 0.7)
        ),
        "game_risk_response_brake_weight": float(
            getattr(ppo, "game_risk_response_brake_weight", 0.3)
        ),
        "game_risk_floor": float(getattr(ppo, "game_risk_floor", 0.1)),
        "game_risk_urgency_max": float(
            getattr(ppo, "game_risk_urgency_max", 1.0)
        ),
    }
    safe_distance = ppo_float["game_risk_safe_distance_m"]
    distance_temperature = ppo_float["game_risk_distance_temperature_m"]
    brake_scale = ppo_float["game_risk_brake_scale"]

    # Social-risk tendency is SVO posterior plus uncertainty, aligned by slot.
    social = np.zeros(n, dtype=np.float64)
    mu = _coerce_vector(svo_mu, n)
    sigma = _coerce_vector(svo_sigma, n)
    assert mu is not None and sigma is not None  # established by svo_valid
    mu_clip = np.clip(mu, 0.0, 90.0)
    sigma_ratio = np.clip(
        sigma / max(prior_sigma, 1e-6),
        0.0,
        ppo_float["game_risk_sigma_cap"],
    )
    social[:] = (
        ppo_float["game_risk_base_weight"]
        + float(getattr(ppo, "svo_mu_budget_weight", 0.0))
        * ((90.0 - mu_clip) / 90.0)
        + float(getattr(ppo, "svo_sigma_budget_weight", 0.0)) * sigma_ratio
    )
    social = np.where(mask & svo_valid & np.isfinite(social), np.maximum(social, 0.0), 0.0)

    proximity = np.zeros(n, dtype=np.float64)
    min_distance = np.zeros(n, dtype=np.float64)
    ego_vx, ego_vy = float(ego_state[3]), float(ego_state[4])
    steps = np.arange(1, pred.shape[1] + 1, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ego_future_x = steps * dt * ego_vx
        ego_future_y = steps * dt * ego_vy
    if not np.isfinite(ego_future_x).all() or not np.isfinite(ego_future_y).all():
        valid_traj[:] = False
    else:
        for i in np.flatnonzero(valid_traj):
            with np.errstate(over="ignore", invalid="ignore"):
                rel_x = pred[i, :, 0] - ego_future_x
                rel_y = pred[i, :, 1] - ego_future_y
                distances = np.hypot(rel_x, rel_y)
            if not np.isfinite(distances).all():
                valid_traj[i] = False
                continue
            d_min = float(np.min(distances))
            if not np.isfinite(d_min):
                valid_traj[i] = False
                continue
            min_distance[i] = d_min
            z = (safe_distance - d_min) / distance_temperature
            proximity[i] = _stable_sigmoid(float(z))

    response_brake = np.zeros(n, dtype=np.float64)
    for i in np.flatnonzero(valid_action):
        throttle_brake = float(actions[i, 1])
        response_brake[i] = float(np.clip(
            max(0.0, -throttle_brake) / brake_scale,
            0.0,
            1.0,
        ))

    valid_data = valid_svo & (valid_traj | valid_action)
    conflict = np.zeros(n, dtype=np.float64)
    conflict[valid_data] = np.clip(
        ppo_float["game_risk_proximity_weight"] * proximity[valid_data]
        + ppo_float["game_risk_response_brake_weight"] * response_brake[valid_data],
        0.0,
        1.0,
    )
    gate = np.zeros(n, dtype=np.float64)
    gate[valid_data] = ppo_float["game_risk_floor"] + (
        1.0 - ppo_float["game_risk_floor"]
    ) * conflict[valid_data]
    urgency = np.zeros(n, dtype=np.float64)
    urgency[valid_data] = np.clip(
        gate[valid_data] * social[valid_data],
        0.0,
        ppo_float["game_risk_urgency_max"],
    )

    diagnostics["per_vehicle_valid_trajectory"] = valid_traj
    diagnostics["per_vehicle_valid_action"] = valid_action
    diagnostics["per_vehicle_valid_interaction"] = valid_data.copy()
    diagnostics["per_vehicle_social"] = np.where(valid_svo, social, 0.0)
    diagnostics["per_vehicle_min_distance"] = min_distance
    diagnostics["per_vehicle_proximity"] = proximity
    diagnostics["per_vehicle_response_brake"] = response_brake
    diagnostics["per_vehicle_game_conflict"] = conflict
    diagnostics["per_vehicle_urgency"] = urgency
    diagnostics["valid_interaction_count"] = int(valid_data.sum())

    if valid_data.any():
        valid_indices = np.flatnonzero(valid_data)
        critical_index = int(valid_indices[np.argmax(urgency[valid_indices])])
        diagnostics["critical_slot"] = critical_index
        value = float(np.clip(
            np.max(urgency[valid_indices]),
            0.0,
            ppo_float["game_risk_urgency_max"],
        ))
    else:
        _set_fallback(diagnostics, "trajectory_coordinates_invalid")
        diagnostics["per_vehicle_social"] = svo_social
        diagnostics["per_vehicle_urgency"][:] = 0.0
        return float(svo_urgency), diagnostics

    if not np.isfinite(value):
        _set_fallback(diagnostics, "non_finite_risk_intermediate")
        return float(svo_urgency), diagnostics
    return value, diagnostics


compute_game_risk_urgency = compute_belief_game_urgency


__all__ = [
    "RISK_CONDITIONING_MODES",
    "compute_belief_game_urgency",
    "compute_game_risk_urgency",
    "extract_bv_actions",
    "extract_latest_ego_state",
    "validate_game_risk_config",
]
