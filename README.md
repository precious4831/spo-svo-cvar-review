# Risk-Constrained Autonomous Driving Decisions Under Social Interaction Uncertainty

**SPO–SVO–CVaR** estimates how neighboring drivers are likely to respond and uses that information to adapt safety budgets for mandatory lane changing and ramp merging in CARLA.

[Method](#method) · [CARLA results](#carla-results) · [Demonstration videos](#demonstration-videos) · [Source files](#source-files)

<p align="center">
  <img src="assets/framework.png" alt="SPO–SVO–CVaR framework: recurrent social belief, belief-game cross-attention, and interaction-conditioned CVaR–SPO" width="900">
</p>

## Method

- **Recurrent Beta Belief Network (RBBN)** updates a distribution over each neighboring driver's social value orientation (SVO) from observed interactions. The model also predicts the vehicle's response trajectory.
- **Belief-Game Cross-Attention (BGCA)** combines each driver's posterior, recurrent state, and predicted response with the ego vehicle's maneuver context. Posterior uncertainty scales the contribution of each vehicle.
- **Interaction-conditioned CVaR–SPO** tightens the episode safety budget as interaction urgency rises. A conditional value-at-risk (CVaR) constraint weights high-cost episodes in the Simple Policy Optimization update.

## CARLA Results

Success and collision rates are reported as mean ± standard deviation (%).

| Scenario | Method | Success ↑ | Collision ↓ |
| --- | --- | ---: | ---: |
| Mandatory lane change | SPO | 61.33 ± 4.70 | 32.67 ± 4.90 |
| Mandatory lane change | **SPO–SVO–CVaR** | **94.00 ± 1.75** | **6.00 ± 1.34** |
| Ramp merge | SPO | 92.67 ± 3.16 | 5.33 ± 2.36 |
| Ramp merge | **SPO–SVO–CVaR** | **96.67 ± 2.19** | **1.67 ± 0.53** |

In the lane-change ablation, the full method also improved on SPO–CVaR without SVO (86.00% success, 14.00% collisions) and SPO–SVO without CVaR (64.67% success, 35.33% collisions).

## Demonstration Videos

### Mandatory Lane Change

| Low-SVO neighbor | High-SVO neighbor |
| :---: | :---: |
| <img src="video/lane-change/low-svo.gif?v=67dd439a941c" alt="Ego vehicle yields to a low-SVO neighbor" width="390"> | <img src="video/lane-change/high-svo.gif?v=548aeca7fa9c" alt="Ego vehicle proceeds ahead of a high-SVO neighbor" width="390"> |
| Ego yields to the neighboring vehicle. [MP4](video/lane-change/low-svo.mp4) | Ego changes lanes ahead of the neighboring vehicle. [MP4](video/lane-change/high-svo.mp4) |

### Ramp Merge

| Low-SVO neighbor | High-SVO neighbor |
| :---: | :---: |
| <img src="video/ramp/low-svo.gif?v=b40355ec9a75" alt="Ego vehicle yields to a low-SVO neighbor" width="390"> | <img src="video/ramp/high-svo.gif?v=0293238fdcde" alt="Ego vehicle proceeds ahead of a high-SVO neighbor" width="390"> |
| Ego yields to the neighboring vehicle. [MP4](video/ramp/low-svo.mp4) | Ego merges ahead of the neighboring vehicle. [MP4](video/ramp/high-svo.mp4) |

## Source Files

| Component | Source |
| --- | --- |
| Recurrent Beta belief and posterior update | [`svo_belief.py`](src/models/svo_belief.py) |
| Interaction encoder and response prediction | [`svo_model.py`](src/models/svo_model.py) |
| Observation encoder and BGCA | [`encoder.py`](src/models/encoder.py) |
| Belief- and response-conditioned urgency | [`belief_game_risk.py`](src/algorithms/belief_game_risk.py) |
| Episode budgets and CVaR dual update | [`ppo_model.py`](src/algorithms/ppo_model.py) |
| SPO objective with tail-weighted cost advantage | [`spo_model.py`](src/algorithms/spo_model.py) |
