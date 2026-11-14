# Source map

| Released module | Research source and adaptation |
| --- | --- |
| [`recurrent_beta_belief.py`](../core/recurrent_beta_belief.py) | `src/models/svo_belief.py` — continuous Beta path and shared state helpers. |
| [`belief_game_attention.py`](../core/belief_game_attention.py) | `src/models/encoder.py` — `GameAwareCrossAttention` and its attention helper. |
| [`belief_game_risk.py`](../core/belief_game_risk.py) | `src/algorithms/belief_game_risk.py` — source module. |
| [`episode_cvar.py`](../core/episode_cvar.py) | `src/algorithms/ppo_model.py` — episode-risk and dual-update routines extracted from the agent; takes the caller's agent state and rollout buffer. |
| [`spo_objective.py`](../core/spo_objective.py) | `src/algorithms/spo_model.py` — constrained actor objective extracted as a tensor function; pass the fixed SPO epsilon used in the reported method. |

The short examples in `excerpts/` isolate the Beta moments, posterior feature calculations and empirical tail statistic from these modules.
