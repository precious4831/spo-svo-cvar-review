# Source map

The excerpts were selected from the local research code on 2026-11-14.
Paths below identify source modules; the source modules themselves are not
included. The release has no access to the original project at runtime.

| Released file | Original source | Selection / adaptation |
| --- | --- | --- |
| `excerpts/beta_moments.py` | `src/models/svo_belief.py`, `RecurrentBetaBeliefNetwork._params` | Statistical formula extracted into a function accepting supplied shape parameters; neural parameterization and configured concentration bounds omitted. Mean is calculated from supplied alpha/beta. |
| `excerpts/belief_features.py` | `src/models/encoder.py`, `GameAwareCrossAttention.forward` | Feature construction and confidence expressions extracted into functions; `self.prior_sigma` becomes an argument. Learned projections, recurrence, attention, and evidence generation omitted. |
| `excerpts/tail_statistics.py` | `src/utils/risk.py`, `empirical_upper_tail` | Function body copied verbatim; standalone imports and input semantics added. Episode cost finalization and budget logic omitted. |

The extracted statistical functions are not substitute implementations of
the withheld models or training system.
