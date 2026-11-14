# Method overview

1. The recurrent Beta belief network consumes interaction embeddings and maintains separate predictive and observation-conditioned distributions over continuous SVO.
2. Belief-game cross-attention combines the posterior, recurrent belief state, predicted response trajectories and vehicle relations to update the ego representation.
3. The risk module estimates interaction urgency from beliefs and response evidence. The episode CVaR module aggregates urgency and safety costs, assigns a risk budget and estimates upper-tail risk over budget-normalized episode costs. Its alpha is the upper-tail mass.
4. The SPO objective combines reward advantage with the tail-weighted cost advantage using the CVaR dual multiplier. The reported method uses a fixed SPO epsilon.
