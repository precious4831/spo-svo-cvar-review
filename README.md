# SPO–SVO–CVaR

SPO–SVO–CVaR studies how inferred social preferences inform driving decisions in CARLA.

## Demonstration Videos

### Lane Change

**Low SVO.** The ego vehicle yields to a neighboring vehicle. [Watch MP4](video/lane-change/low-svo.mp4).

<img src="video/lane-change/low-svo.gif" alt="Lane Change, low SVO: ego vehicle yields" width="720">

**High SVO.** The ego vehicle changes lanes ahead of the neighboring vehicle. [Watch MP4](video/lane-change/high-svo.mp4).

<img src="video/lane-change/high-svo.gif" alt="Lane Change, high SVO: ego vehicle goes first" width="720">

### Ramp Merge

**Low SVO.** The ego vehicle yields to a neighboring vehicle. [Watch MP4](video/ramp/low-svo.mp4).

<img src="video/ramp/low-svo.gif" alt="Ramp Merge, low SVO: ego vehicle yields" width="720">

**High SVO.** The ego vehicle merges ahead of the neighboring vehicle. [Watch MP4](video/ramp/high-svo.mp4).

<img src="video/ramp/high-svo.gif" alt="Ramp Merge, high SVO: ego vehicle goes first" width="720">

## Source Files

| Source file | Contents |
| --- | --- |
| [`src/models/svo_belief.py`](src/models/svo_belief.py) | Recurrent Beta belief network and prior/posterior updates |
| [`src/models/svo_model.py`](src/models/svo_model.py) | Interaction encoder, response decoder and SVO inference |
| [`src/models/encoder.py`](src/models/encoder.py) | Hierarchical observation encoder and belief-game cross-attention |
| [`src/algorithms/belief_game_risk.py`](src/algorithms/belief_game_risk.py) | Belief- and response-conditioned interaction urgency |
| [`src/algorithms/ppo_model.py`](src/algorithms/ppo_model.py) | Rollout buffer, episode risk budgets and CVaR dual update |
| [`src/algorithms/spo_model.py`](src/algorithms/spo_model.py) | SPO objective and integration of the CVaR cost advantage |
