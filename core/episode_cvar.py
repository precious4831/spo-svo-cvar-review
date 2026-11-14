"""Episode budget, tail weighting and dual update from the training implementation."""

from typing import Dict

import numpy as np


def _empirical_upper_tail(values: np.ndarray, alpha: float):
    order = np.argsort(-values, kind='stable')
    remaining = alpha * len(values)
    coeff = np.zeros(len(values), dtype=np.float64)
    for idx in order:
        if remaining <= 1e-12:
            break
        take = min(1.0, remaining)
        coeff[idx] = take / alpha
        remaining -= take
    estimate = float(np.dot(coeff, values) / len(values))
    boundary = min(max(int(np.ceil(alpha * len(values))) - 1, 0), len(values) - 1)
    return estimate, coeff, float(values[order[boundary]])


def _episode_risk(pending: dict, ppo_cfg) -> dict:
    count = max(1, pending['count'])
    cost = pending['cost'] - pending['dense'] * (1.0 - 1.0 / count)
    urgency = float(np.quantile(pending['urgency'], 0.9))
    d0 = float(ppo_cfg.cvar_budget_base)
    budget = max(
        d0 * np.exp(-float(ppo_cfg.svo_budget_beta) * urgency),
        d0 * float(ppo_cfg.cvar_budget_min_ratio),
    )
    return {
        'cost': float(cost),
        'urgency': urgency,
        'budget': float(budget),
        'normalised_cost': float(cost / budget),
    }

def update_episode_cvar(agent) -> Dict[str, float]:
    """Update CVaR from complete episodes, including rollout boundaries.

    约束写法 (per-episode normalized):
        CVaR_alpha( {C_e / d_e} ) <= 1
        lambda <- max(0, lambda + lr * (CVaR_hat_norm - 1.0))

    ``cvar_alpha`` is upper-tail mass; alpha=0.1 corresponds to the
    common paper notation CVaR_0.9.  The episode budget is jointly
    conditioned by SVO belief and Level-1 response evidence.  Tail
    weights act on the safety-cost advantage, while formal SPO keeps a
    fixed epsilon and SVO does not modify that epsilon.

    A rollout often begins or ends in the middle of an episode.  Those
    fragments are accumulated in ``_cvar_pending_episodes`` and enter the
    empirical tail distribution only after a terminal transition arrives.
    """
    ppo_cfg = agent.config.ppo
    n = agent.buffer.ptr
    agent.buffer.cvar_sample_weights[:n] = 0.0
    alpha = float(ppo_cfg.cvar_alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f'cvar_alpha must be in (0, 1], got {alpha}')
    reference_history = list(agent._cvar_episode_history)
    rollout_risk = {}
    episode_discount = float(
        getattr(ppo_cfg, "cvar_episode_discount", 1.0)
    )
    if not 0.0 < episode_discount <= 1.0:
        raise ValueError(
            "cvar_episode_discount must be in (0, 1], got "
            f"{episode_discount}"
        )
    completed_now = 0
    for i in range(n):
        eid = int(agent.buffer.episode_ids[i])
        pending = agent._cvar_pending_episodes.setdefault(eid, {
            'cost': 0.0,
            'dense': 0.0,
            'urgency': [],
            'count': 0,
            'step': 0,
        })
        step_coefficient = episode_discount ** pending['step']
        step_cost = float(agent.buffer.costs[i])
        pending['cost'] += step_cost * step_coefficient
        pending['dense'] += float(agent.buffer.cost_dense[i]) * step_coefficient
        pending['urgency'].append(float(agent.buffer.svo_budget_terms[i]))
        pending['count'] += 1
        pending['step'] += 1
        if bool(agent.buffer.dones[i] > 0.5):
            risk = _episode_risk(pending, ppo_cfg)
            item = {
                'length': int(pending['count']),
                **risk,
            }
            agent._cvar_episode_history.append(item)
            rollout_risk[eid] = risk
            agent._cvar_pending_episodes.pop(eid, None)
            completed_now += 1

    if len(agent._cvar_episode_history) == 0:
        return {'cvar_hat_norm': 0.0, 'cvar_lambda': agent.cvar_lambda,
                'avg_budget': float(ppo_cfg.cvar_budget_base),
                'avg_episode_cost': 0.0,
                'avg_risk_urgency': 0.0,
                'n_episodes_in_rollout': 0,
                'n_episodes_in_cvar_window': 0,
                'cvar_actor_threshold_norm': 0.0,
                'cvar_weighted_step_fraction': 0.0}

    history = list(agent._cvar_episode_history)
    C_arr = np.asarray([item['cost'] for item in history], dtype=np.float64)
    U_arr = np.asarray([item['urgency'] for item in history], dtype=np.float64)
    d_arr = np.asarray([item['budget'] for item in history], dtype=np.float64)
    Cn_arr = np.asarray(
        [item['normalised_cost'] for item in history], dtype=np.float64
    )

    M = len(Cn_arr)
    cvar_hat_norm, _, _ = _empirical_upper_tail(Cn_arr, alpha)

    threshold_history = reference_history or history
    threshold_values = np.asarray(
        [item['normalised_cost'] for item in threshold_history],
        dtype=np.float64,
    )
    _, _, actor_threshold = _empirical_upper_tail(threshold_values, alpha)
    for eid in np.unique(agent.buffer.episode_ids[:n]):
        eid = int(eid)
        if eid not in rollout_risk and eid in agent._cvar_pending_episodes:
            rollout_risk[eid] = _episode_risk(
                agent._cvar_pending_episodes[eid], ppo_cfg
            )

    for i in range(n):
        risk = rollout_risk.get(int(agent.buffer.episode_ids[i]))
        if risk is not None and risk['normalised_cost'] >= actor_threshold:
            agent.buffer.cvar_sample_weights[i] = 1.0 / (
                alpha * risk['budget']
            )

    # Do not repeatedly update lambda when a rollout contains no newly
    # completed episode; the window can still be logged for diagnostics.
    if completed_now > 0:
        violation = cvar_hat_norm - 1.0
        new_lambda = max(
            0.0,
            agent.cvar_lambda
            + float(ppo_cfg.cvar_lambda_lr) * violation,
        )
        lambda_max = getattr(ppo_cfg, 'cvar_lambda_max', None)
        if lambda_max is not None:
            new_lambda = min(new_lambda, float(lambda_max))
        agent.cvar_lambda = new_lambda

    return {
        'cvar_hat_norm':       float(cvar_hat_norm),
        'cvar_lambda':         float(agent.cvar_lambda),
        'avg_budget':          float(d_arr.mean()),
        'avg_episode_cost':    float(C_arr.mean()),
        'avg_risk_urgency': float(U_arr.mean()),
        'n_episodes_in_rollout': int(completed_now),
        'n_episodes_in_cvar_window': int(M),
        'cvar_actor_threshold_norm': float(actor_threshold),
        'cvar_weighted_step_fraction': float(
            np.mean(agent.buffer.cvar_sample_weights[:n] > 0.0)
        ) if n > 0 else 0.0,
    }
