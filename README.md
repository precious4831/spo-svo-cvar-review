# SPO–SVO–CVaR

Selected implementation modules for inspecting the method in the paper. The modules use PyTorch and NumPy.

| Module | Contents |
| --- | --- |
| [`recurrent_beta_belief.py`](core/recurrent_beta_belief.py) | Recurrent state, predictive Beta prior, observation-conditioned posterior and balanced KL losses. |
| [`belief_game_attention.py`](core/belief_game_attention.py) | Belief-game cross-attention, including confidence gating, response trajectories and interaction masking. |
| [`belief_game_risk.py`](core/belief_game_risk.py) | Risk urgency from SVO beliefs, interaction geometry and predicted responses. |
| [`episode_cvar.py`](core/episode_cvar.py) | Episode accumulation across rollout boundaries, adaptive budget, normalized tail cost, actor weights and dual update. |
| [`spo_objective.py`](core/spo_objective.py) | SPO actor objective with the tail-weighted safety advantage. |

The original short examples remain in [`excerpts/`](excerpts/). See the [method overview](docs/method_overview.md), [source map](docs/source_map.md) and [release scope](docs/disclosure_scope.md) for the inputs and module relationships.
