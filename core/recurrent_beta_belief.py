"""Recurrent Beta belief network for continuous SVO inference."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence


@dataclass
class BeliefState:
    """One filtering state for one physical vehicle."""

    deter: torch.Tensor       # (B, D)
    stoch: torch.Tensor       # (B, C): Beta sample/mean or categorical vector


def _init_linear(module: nn.Module, gain: float = 1.0) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class RecurrentBetaBeliefNetwork(nn.Module):
    """Interaction-conditioned recurrent encoder with a bounded SVO belief.

    The implementation separates prediction from observation assimilation:

    * ``h_t`` is updated only from the previous stochastic state and the
      known ego-action context;
    * the prior ``p(z_t | h_t)`` cannot see the current interaction;
    * the posterior ``q(z_t | h_t, e_t)`` assimilates the current observation;
    * dynamics and representation KL terms use opposite stop-gradients.

    ``z_t`` is normalised SVO in ``(0, 1)``.  The recurrent filter always uses
    the posterior mean, so belief updates and the deterministic state remain
    stable.  Stochastic Beta samples are drawn only by the model's explicit
    risk/visualisation API and never injected into the training recurrence.
    """

    def __init__(
        self,
        obs_dim: int,
        deter_dim: int = 128,
        action_dim: int = 4,
        hidden_dim: int = 128,
        min_concentration: float = 2.0,
        max_concentration: float = 200.0,
        free_nats: float = 0.05,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.deter_dim = int(deter_dim)
        self.action_dim = int(action_dim)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.free_nats = float(free_nats)
        self.eps = 1e-5

        self.core_in = nn.Sequential(
            nn.Linear(1 + self.action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.core = nn.GRUCell(hidden_dim, self.deter_dim)
        # Observation-assimilation update for the filtered deterministic state.
        # The prior is computed before this update, preserving p(z_t|h_t^-),
        # while the h_t exposed to L5/decoder contains the latest observation.
        self.filter_in = nn.Sequential(
            nn.Linear(self.obs_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.filter_core = nn.GRUCell(hidden_dim, self.deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.post_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        _init_linear(self)
        nn.init.xavier_uniform_(self.prior_net[-1].weight, gain=0.1)
        nn.init.xavier_uniform_(self.post_net[-1].weight, gain=0.1)
        # mean=0.5, concentration≈2 gives an initially uninformative belief.
        with torch.no_grad():
            self.prior_net[-1].bias.copy_(torch.tensor([0.0, -5.0]))
            self.post_net[-1].bias.copy_(torch.tensor([0.0, -5.0]))

    def initial(self, batch_size: int, device: torch.device) -> BeliefState:
        return BeliefState(
            deter=torch.zeros(batch_size, self.deter_dim, device=device),
            stoch=torch.full((batch_size, 1), 0.5, device=device),
        )

    def _params(self, raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        mean = torch.sigmoid(raw[..., 0]).clamp(self.eps, 1.0 - self.eps)
        concentration = self.min_concentration + F.softplus(raw[..., 1])
        concentration = concentration.clamp(max=self.max_concentration)
        alpha = (mean * concentration).clamp_min(self.eps)
        beta = ((1.0 - mean) * concentration).clamp_min(self.eps)
        variance = (
            alpha * beta
            / ((alpha + beta).pow(2) * (alpha + beta + 1.0))
        ).clamp_min(1e-10)
        return {
            'alpha': alpha,
            'beta': beta,
            'mean': mean,
            'variance': variance,
            'std': variance.sqrt(),
            'concentration': concentration,
        }

    @staticmethod
    def _dist(params: Dict[str, torch.Tensor]) -> Beta:
        return Beta(params['alpha'], params['beta'])

    def observe_step(
        self,
        obs_embed: torch.Tensor,
        action_context: torch.Tensor,
        prev: Optional[BeliefState] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Tuple[BeliefState, Dict[str, torch.Tensor]]:
        batch_size = obs_embed.shape[0]
        if prev is None:
            prev = self.initial(batch_size, obs_embed.device)

        core_input = self.core_in(torch.cat([prev.stoch, action_context], dim=-1))
        pred_deter = self.core(core_input, prev.deter)
        prior = self._params(self.prior_net(pred_deter))
        post = self._params(
            self.post_net(torch.cat([pred_deter, obs_embed], dim=-1))
        )
        stoch = post['mean'].unsqueeze(-1)
        filter_input = self.filter_in(torch.cat([obs_embed, stoch], dim=-1))
        deter = self.filter_core(filter_input, pred_deter)

        if valid is not None:
            keep = valid.to(torch.bool).unsqueeze(-1)
            deter = torch.where(keep, deter, prev.deter)
            stoch = torch.where(keep, stoch, prev.stoch)
            scalar_keep = keep.squeeze(-1)
            for params in (prior, post):
                params['alpha'] = torch.where(
                    scalar_keep, params['alpha'], torch.ones_like(params['alpha'])
                )
                params['beta'] = torch.where(
                    scalar_keep, params['beta'], torch.ones_like(params['beta'])
                )
                params['mean'] = torch.where(
                    scalar_keep, params['mean'], torch.full_like(params['mean'], 0.5)
                )
                neutral_var = torch.full_like(params['variance'], 1.0 / 12.0)
                params['variance'] = torch.where(
                    scalar_keep, params['variance'], neutral_var
                )
                params['std'] = params['variance'].sqrt()
                params['concentration'] = params['alpha'] + params['beta']

        state = BeliefState(deter=deter, stoch=stoch)
        feat = {'deter': deter, 'stoch': stoch}
        for prefix, params in (('prior', prior), ('post', post)):
            for name, value in params.items():
                feat[f'{prefix}_{name}'] = value
        return state, feat

    def observe(
        self,
        obs_embed: torch.Tensor,
        action_context: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        initial: Optional[BeliefState] = None,
    ) -> Tuple[BeliefState, Dict[str, torch.Tensor]]:
        if obs_embed.ndim != 3:
            raise ValueError(f'obs_embed must be (B,S,D), got {obs_embed.shape}')
        if action_context.shape[:2] != obs_embed.shape[:2]:
            raise ValueError('action_context and obs_embed must share B,S dimensions')

        state = initial or self.initial(obs_embed.shape[0], obs_embed.device)
        keys = (
            'deter', 'stoch',
            'prior_alpha', 'prior_beta', 'prior_mean', 'prior_variance',
            'prior_std', 'prior_concentration',
            'post_alpha', 'post_beta', 'post_mean', 'post_variance',
            'post_std', 'post_concentration',
        )
        outputs = {key: [] for key in keys}
        for step in range(obs_embed.shape[1]):
            valid = None if valid_mask is None else valid_mask[:, step]
            state, feat = self.observe_step(
                obs_embed[:, step], action_context[:, step], state, valid
            )
            for key in keys:
                outputs[key].append(feat[key])
        return state, {
            key: torch.stack(values, dim=1) for key, values in outputs.items()
        }

    def kl_losses(
        self,
        post_alpha: torch.Tensor,
        post_beta: torch.Tensor,
        prior_alpha: torch.Tensor,
        prior_beta: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        post = Beta(post_alpha, post_beta)
        prior = Beta(prior_alpha, prior_beta)
        dyn = kl_divergence(
            Beta(post_alpha.detach(), post_beta.detach()), prior
        )
        rep = kl_divergence(
            post, Beta(prior_alpha.detach(), prior_beta.detach())
        )
        raw_dyn, raw_rep = dyn, rep
        if self.free_nats:
            dyn = torch.clamp(dyn, min=self.free_nats)
            rep = torch.clamp(rep, min=self.free_nats)

        if valid_mask is None:
            weight = torch.ones_like(dyn)
        else:
            weight = valid_mask.to(dyn.dtype)
        denom = weight.sum().clamp_min(1.0)
        reduce = lambda value: (value * weight).sum() / denom
        dyn_loss, rep_loss = reduce(dyn), reduce(rep)
        metrics = {
            'dyn_kl_raw': reduce(raw_dyn).detach(),
            'rep_kl_raw': reduce(raw_rep).detach(),
            'prior_entropy': reduce(prior.entropy()).detach(),
            'post_entropy': reduce(post.entropy()).detach(),
        }
        return dyn_loss, rep_loss, metrics
