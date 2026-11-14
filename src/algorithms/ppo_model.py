"""
ppo_model.py -- PPO Agent with Belief-Game Integration

========================================================================
在完整训练流程中的位置:
  本文件定义PPOAgent, 被 train.py 和 test.py 调用.
  训练: python train.py --stage l1 / svo_only / l2
  验证: python test.py  --l1_model ... --svo_model ... --l2_model ...
========================================================================

架构:
    shared_encoder → GameAwareCrossAttention → actor_head / critic_head
    observed Beta-RSSM (SVO推断) + L1/L2 relational Transformer

核心设计:
  1. Actor/Critic 共享 Encoder (减半参数, 一致表征)
  2. 单PPO优化器 + 合并损失 (解决共享Encoder梯度冲突)
  3. SVO posterior、RSSM deterministic belief、response trajectory 与
     Level-1 BV action data 缓存在 Buffer 中 (update 时不重复计算)
  4. config.svo.enabled 控制开关 (False时退化为纯PPO)
  5. 预训练 SVO 默认在强化学习阶段冻结; 在线推断模式由 config.svo 控制

梯度冲突修复说明:
  原版: Actor有Encoder_A, Critic有Encoder_B (各自独立)
  新版: Actor和Critic共享一个Encoder
  问题: actor_loss.backward() 释放计算图 → critic_loss.backward() 崩溃
  解决: loss = actor_loss + value_coef*critic_loss - entropy_coef*entropy
        单次 loss.backward() + 单次 optimizer.step()
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Tuple, Dict, Optional
from collections import deque

from src.models.encoder import HierarchicalTransformerEncoder, parse_observation
from src.models.svo_model import SVOObservedBetaModel


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


# ======================================================================== #
#  ActorHead: MLP-only (无内部Encoder)                                      #
# ======================================================================== #
# 
class ActorHead(nn.Module):
    """
    Actor策略头.

    输入: features (B, input_dim)  — 来自共享Encoder+aux拼接
    输出: mean (B, action_dim), std (B, action_dim)

    注意: 不包含Encoder, Encoder在PPOAgent层面共享.
    """

    def __init__(self, input_dim: int, action_dim: int, action_type: str = "continuous"):
        super().__init__()
        self.action_type = action_type
        self.action_dim = action_dim
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32),        nn.ReLU(),
        )
        self.mean_layer = nn.Linear(32, action_dim)
        if self.action_type == "continuous":
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.head(features)
        mean = self.mean_layer(x)
        if self.action_type == "discrete":
            return mean, None
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def get_action(self, features: torch.Tensor,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作 (tanh squashed Gaussian)."""
        mean, std = self.forward(features)

        if self.action_type == "discrete":
            dist = Categorical(logits=mean)
            action_id = torch.argmax(mean, dim=-1) if deterministic else dist.sample()
            action = nn.functional.one_hot(action_id, num_classes=self.action_dim).float()
            log_prob = dist.log_prob(action_id)
            return action, log_prob

        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(features.shape[0], device=features.device)
        else:
            dist = Normal(mean, std)
            action_raw = dist.rsample()
            action = torch.tanh(action_raw)

            # log_prob with tanh correction
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
            log_prob -= (2 * (np.log(2) - action_raw
                         - nn.functional.softplus(-2 * action_raw))).sum(dim=-1)

        return action, log_prob

    def evaluate(self, features: torch.Tensor,
                 action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估已有动作的log_prob和entropy."""
        mean, std = self.forward(features)
        if self.action_type == "discrete":
            dist = Categorical(logits=mean)
            if action.dim() > 1:
                action_id = torch.argmax(action, dim=-1)
            else:
                action_id = action.long()
            log_prob = dist.log_prob(action_id)
            entropy = dist.entropy()
            return log_prob, entropy

        dist = Normal(mean, std)

        action_raw = torch.atanh(torch.clamp(action, -0.999, 0.999))
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - action_raw
                     - nn.functional.softplus(-2 * action_raw))).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# ======================================================================== #
#  CriticHead: MLP-only                                                     #
# ======================================================================== #

class CriticHead(nn.Module):
    """
    Critic价值头.

    输入: features (B, input_dim)
    输出: value (B,)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 32),        nn.ReLU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)


# ======================================================================== #
#  RolloutBuffer (扩展版, 缓存 belief-game context)                         #
# ======================================================================== #

class RolloutBuffer:
    """
    PPO经验缓冲区, 扩展支持社会信念条件化交互数据缓存.

    SVO posterior、RSSM deterministic belief、社会信念条件化的周车预测响应轨迹和
    Level-1 BV 实际动作关系数据在 rollout 阶段计算/采集一次, 缓存在 buffer 中,
    PPO update时直接读取, 避免重复计算
    (SVO推断: GPU前向; response inference: 每步重算代价极大).

    内存开销: ~260 floats/step × 2048 steps ≈ 2MB (可忽略)
    """

    def __init__(self, buffer_size: int, state_dim: int, action_dim: int,
                 device: torch.device, config: Config = None,
                 use_svo_game: bool = True):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.full = False
        self.use_svo = use_svo_game
        # [CVaR] 主开关 (从 config 读, 全程不变)
        self.use_cvar = bool(config is not None and config.ppo.cvar_enabled)

        # 标准PPO数据
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # Belief-Game 扩展数据
        if use_svo_game and config is not None:
            N = config.encoder.num_neighbours
            H = config.svo.prediction_horizon
            prior_mu = config.svo.prior_mu       # [Fix Bug2] 从config取, 不硬编码
            prior_sigma = config.svo.prior_sigma
            self.svo_mu = np.full((buffer_size, N), prior_mu, dtype=np.float32)
            self.svo_sigma = np.full((buffer_size, N), prior_sigma, dtype=np.float32)
            self.svo_deter = np.zeros(
                (buffer_size, N, int(config.svo.rssm_hidden_dim)),
                dtype=np.float32,
            )
            self.pred_trajs = np.zeros((buffer_size, N, H, 5), dtype=np.float32)
            self.interact_mask = np.zeros((buffer_size, N), dtype=bool)
            # [Belief-Game/CVaR] step-wise game-conditioned urgency u_t.
            #   SVO posterior + Level-1 response evidence jointly produce u_t;
            #   CVaR uses it only to adjust the episode safety budget.
            #   Formal SPO keeps fixed epsilon; SVO-adaptive epsilon is an
            #   explicit ablation only.
            self.svo_budget_terms = np.zeros(buffer_size, dtype=np.float32)

        # [CVaR] 扩展数据
        if self.use_cvar:
            self.costs = np.zeros(buffer_size, dtype=np.float32)
            # 稠密分量 (w_safe*shortfall); episode 级约束对它取长度均值.
            self.cost_dense = np.zeros(buffer_size, dtype=np.float32)
            self.cost_values = np.zeros(buffer_size, dtype=np.float32)
            self.episode_ids = np.zeros(buffer_size, dtype=np.int64)
            # Filled by ``_cvar_dual_update`` after complete-episode tail
            # membership is known.  It includes the per-episode budget scale.
            self.cvar_sample_weights = np.zeros(buffer_size, dtype=np.float32)
            # Compatibility fallback: if SVO is off but CVaR is on,
            # _cvar_dual_update 仍要访问该数组 → 兜底分配为全 0 (等价于 d_e = d0)
            if not (use_svo_game and config is not None):
                self.svo_budget_terms = np.zeros(buffer_size, dtype=np.float32)

    def add(self, state, action, reward, value, log_prob, done,
            svo_mu=None, svo_sigma=None, svo_deter=None,
            pred_trajs=None, interact_mask=None,
            cost=0.0, cost_value=0.0, episode_id=0, svo_budget_term=0.0,
            cost_dense=0.0):
        """添加一步经验."""
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = done

        if self.use_svo and svo_mu is not None:
            self.svo_mu[i] = svo_mu
            self.svo_sigma[i] = svo_sigma
            if svo_deter is not None:
                self.svo_deter[i] = svo_deter
            self.pred_trajs[i] = pred_trajs
            self.interact_mask[i] = interact_mask
            # Compatibility field name; stores game-conditioned risk urgency.
            self.svo_budget_terms[i] = svo_budget_term

        if self.use_cvar:
            self.costs[i] = cost
            self.cost_dense[i] = cost_dense
            self.cost_values[i] = cost_value
            self.episode_ids[i] = episode_id
            # The compatibility urgency field is written with the belief-game
            # context above, not by the CVaR cost path.

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns(self, last_value: float, gamma: float,
                        gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """GAE计算returns和advantages (reward 路径)."""
        size = self.ptr
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:size]
        return returns, advantages

    def compute_cost_returns(self, last_cost_value: float, cost_gamma: float,
                             gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """[CVaR] GAE 计算 cost returns/advantages (cost 路径).

        和 compute_returns 完全同构, 只是用 self.costs/self.cost_values.
        """
        size = self.ptr
        advantages = np.zeros(size, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(size)):
            next_v = last_cost_value if t == size - 1 else self.cost_values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.costs[t] + cost_gamma * next_v * non_terminal - self.cost_values[t]
            last_gae = delta + cost_gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.cost_values[:size]
        return returns, advantages

    def get_data(self, returns: np.ndarray,
                 advantages: np.ndarray,
                 cost_returns: Optional[np.ndarray] = None,
                 cost_advantages: Optional[np.ndarray] = None) -> Dict[str, torch.Tensor]:
        """转换为torch tensors."""
        n = self.ptr
        data = {
            'states':     torch.FloatTensor(self.states[:n]).to(self.device),
            'actions':    torch.FloatTensor(self.actions[:n]).to(self.device),
            'log_probs':  torch.FloatTensor(self.log_probs[:n]).to(self.device),
            'returns':    torch.FloatTensor(returns).to(self.device),
            'advantages': torch.FloatTensor(advantages).to(self.device),
            'values':     torch.FloatTensor(self.values[:n]).to(self.device),  # [Fix] value clipping用
        }
        if self.use_svo:
            data['svo_mu']        = torch.FloatTensor(self.svo_mu[:n]).to(self.device)
            data['svo_sigma']     = torch.FloatTensor(self.svo_sigma[:n]).to(self.device)
            data['svo_deter']     = torch.FloatTensor(self.svo_deter[:n]).to(self.device)
            data['pred_trajs']    = torch.FloatTensor(self.pred_trajs[:n]).to(self.device)
            data['interact_mask'] = torch.BoolTensor(self.interact_mask[:n]).to(self.device)
            # [SPO] 给 SVO-adaptive ε 用; 标量 u_t per step ∈ [0, 1]
            data['svo_budget_terms'] = torch.FloatTensor(self.svo_budget_terms[:n]).to(self.device)
        if self.use_cvar and cost_returns is not None:
            data['cost_returns']    = torch.FloatTensor(cost_returns).to(self.device)
            data['cost_advantages'] = torch.FloatTensor(cost_advantages).to(self.device)
            data['cost_values']     = torch.FloatTensor(self.cost_values[:n]).to(self.device)
            data['cvar_sample_weights'] = torch.FloatTensor(
                self.cvar_sample_weights[:n]
            ).to(self.device)
        return data

    def reset(self):
        self.ptr = 0
        self.full = False


# ======================================================================== #
#  PPOAgent                                                                 #
# ======================================================================== #

class PPOAgent:
    """
    PPO智能体 (Belief-Game集成版)

    架构:
      shared_encoder (HierarchicalTransformerEncoder)
          ↓
      [Belief-Game Cross-Attention] ← posterior moments, deterministic belief,
                                      response trajectory, Level-1-action relation
          ↓
      actor_head (ActorHead) → action
      critic_head (CriticHead) → value

    训练协议:
      ppo_optimizer: encoder + actor_head + reward/cost critics
      observed Beta-RSSM: 独立预训练，RL 阶段严格冻结
    """

    # Agent setup, policy rollout and training integration omitted from this release.

    @staticmethod
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

    def _cvar_dual_update(self) -> Dict[str, float]:
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
        ppo_cfg = self.config.ppo
        n = self.buffer.ptr
        self.buffer.cvar_sample_weights[:n] = 0.0
        alpha = float(ppo_cfg.cvar_alpha)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f'cvar_alpha must be in (0, 1], got {alpha}')
        reference_history = list(self._cvar_episode_history)
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
            eid = int(self.buffer.episode_ids[i])
            pending = self._cvar_pending_episodes.setdefault(eid, {
                'cost': 0.0,
                'dense': 0.0,
                'urgency': [],
                'count': 0,
                'step': 0,
            })
            step_coefficient = episode_discount ** pending['step']
            step_cost = float(self.buffer.costs[i])
            pending['cost'] += step_cost * step_coefficient
            pending['dense'] += float(self.buffer.cost_dense[i]) * step_coefficient
            pending['urgency'].append(float(self.buffer.svo_budget_terms[i]))
            pending['count'] += 1
            pending['step'] += 1
            if bool(self.buffer.dones[i] > 0.5):
                risk = self._episode_risk(pending, ppo_cfg)
                item = {
                    'length': int(pending['count']),
                    **risk,
                }
                self._cvar_episode_history.append(item)
                rollout_risk[eid] = risk
                self._cvar_pending_episodes.pop(eid, None)
                completed_now += 1

        if len(self._cvar_episode_history) == 0:
            return {'cvar_hat_norm': 0.0, 'cvar_lambda': self.cvar_lambda,
                    'avg_budget': float(ppo_cfg.cvar_budget_base),
                    'avg_episode_cost': 0.0,
                    'avg_risk_urgency': 0.0,
                    'n_episodes_in_rollout': 0,
                    'n_episodes_in_cvar_window': 0,
                    'cvar_actor_threshold_norm': 0.0,
                    'cvar_weighted_step_fraction': 0.0}

        history = list(self._cvar_episode_history)
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
        for eid in np.unique(self.buffer.episode_ids[:n]):
            eid = int(eid)
            if eid not in rollout_risk and eid in self._cvar_pending_episodes:
                rollout_risk[eid] = self._episode_risk(
                    self._cvar_pending_episodes[eid], ppo_cfg
                )

        for i in range(n):
            risk = rollout_risk.get(int(self.buffer.episode_ids[i]))
            if risk is not None and risk['normalised_cost'] >= actor_threshold:
                self.buffer.cvar_sample_weights[i] = 1.0 / (
                    alpha * risk['budget']
                )

        # Do not repeatedly update lambda when a rollout contains no newly
        # completed episode; the window can still be logged for diagnostics.
        if completed_now > 0:
            violation = cvar_hat_norm - 1.0
            new_lambda = max(
                0.0,
                self.cvar_lambda
                + float(ppo_cfg.cvar_lambda_lr) * violation,
            )
            lambda_max = getattr(ppo_cfg, 'cvar_lambda_max', None)
            if lambda_max is not None:
                new_lambda = min(new_lambda, float(lambda_max))
            self.cvar_lambda = new_lambda

        return {
            'cvar_hat_norm':       float(cvar_hat_norm),
            'cvar_lambda':         float(self.cvar_lambda),
            'avg_budget':          float(d_arr.mean()),
            'avg_episode_cost':    float(C_arr.mean()),
            'avg_risk_urgency': float(U_arr.mean()),
            'n_episodes_in_rollout': int(completed_now),
            'n_episodes_in_cvar_window': int(M),
            'cvar_actor_threshold_norm': float(actor_threshold),
            'cvar_weighted_step_fraction': float(
                np.mean(self.buffer.cvar_sample_weights[:n] > 0.0)
            ) if n > 0 else 0.0,
        }
