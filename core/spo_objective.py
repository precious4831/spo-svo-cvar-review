def spo_cvar_actor_loss(
    ratio, reward_advantage, cost_advantage, tail_weight, dual_weight,
    cost_coefficient, epsilon,
):
    """SPO actor loss with the tail-weighted Lagrangian advantage."""
    effective_advantage = reward_advantage - (
        dual_weight * cost_coefficient * (cost_advantage * tail_weight)
    )
    objective = (
        ratio * effective_advantage
        - effective_advantage.abs() * (ratio - 1.0).pow(2) / (2.0 * epsilon)
    )
    return -objective.mean()
