# SPO–SVO–CVaR: selected implementation excerpts for review

This repository contains selected mathematical implementation excerpts and a
module-level reading guide for the CARLA driving research project. It is a
partial code disclosure for methodological inspection. It does not provide
end-to-end training, policy inference, or experimental reproduction.

## Reading guide

| File | What can be inspected |
| --- | --- |
| `excerpts/beta_moments.py` | Bounded-belief distribution moments |
| `excerpts/belief_features.py` | Posterior feature construction and confidence weighting |
| `excerpts/tail_statistics.py` | Empirical upper-tail statistic with fractional boundary weighting |
| `docs/method_overview.md` | Module relationships and tensor meanings |
| `docs/source_map.md` | Relationship between released excerpts and original implementation |
| `docs/disclosure_scope.md` | Exact scope of the release |

The excerpts operate on supplied tensors or cost arrays. Their only external
dependencies are NumPy and PyTorch, listed in `requirements.txt`.

The upper-tail function uses **tail mass** alpha: alpha = 0.1 selects the
worst 10% of samples. Its statistic describes the costs supplied by the caller;
the release does not provide the experiment-specific episode-cost collector.

No experiment results or performance claims are introduced by this repository.
SPO is an existing optimization method, rather than an original contribution
of these statistical excerpts.

## 审稿展示范围

提供信念分布统计、后验特征、置信度与尾部风险统计的局部实现，以及模块关系说明。
完整网络、策略更新、训练入口、环境、配置、数据及权重未包含在本仓库中。
代码来源及改写范围见 `docs/source_map.md`。
