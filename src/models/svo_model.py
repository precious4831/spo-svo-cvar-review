"""Continuous SVO inference from vehicle interaction histories.

Temporal attention encodes each actor's history. Relational attention combines
the ego and neighboring vehicle features before the recurrent Beta belief update.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence

from src.models.svo_belief import BeliefState, RecurrentBetaBeliefNetwork


INTENT_ANCHORS_DEG = [float(v) for v in range(0, 91, 5)]
NUM_INTENTS = len(INTENT_ANCHORS_DEG)


def uniform_intent(batch_size, device, num_intents=NUM_INTENTS):
    """Backward-compatible neutral state for the scalar Beta belief."""
    del num_intents
    return torch.full((batch_size, 1), 0.5, device=device)


def _safe_key_padding(mask: torch.Tensor) -> torch.Tensor:
    """Ensure every sequence has at least one unmasked key."""
    mask = mask.clone()
    all_invalid = ~mask.any(dim=-1)
    if all_invalid.any():
        mask[all_invalid, -1] = True
    return ~mask


class SVOInteractionTransformer(nn.Module):
    """Pairwise, mask-aware L1 temporal and L2 relational attention."""

    def __init__(self, feature_dim=5, hidden_dim=128, num_heads=4, num_layers=2,
                 dropout=0.1):
        super().__init__()
        if int(feature_dim) != 5:
            raise ValueError('SVOInteractionTransformer expects [x,y,yaw,vx,vy]')
        self.hidden_dim = int(hidden_dim)
        self.raw_feature_dim = int(feature_dim)
        dropout = float(dropout)
        # Normalised pose/velocity (5), instantaneous/0.2s/0.4s acceleration
        # (6), speed (1), and yaw rate (1).  Multi-scale derivatives suppress
        # one-frame CARLA physics noise while retaining response onset.
        self.motion_feature_dim = 13
        self.input_proj = nn.Linear(self.motion_feature_dim, hidden_dim)
        self.actor_type = nn.Embedding(2, hidden_dim)  # 0=ego, 1=target neighbour
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.rel_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.rel_norm = nn.LayerNorm(hidden_dim)
        self.relation_mlp = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _ego_frame(traj, ref):
        """Transform ``(..., T, 5)`` states into the latest ego frame."""
        rx = ref[..., 0:1].unsqueeze(-2)
        ry = ref[..., 1:2].unsqueeze(-2)
        ryaw = ref[..., 2:3].unsqueeze(-2)
        c, s = torch.cos(ryaw), torch.sin(ryaw)
        dx = traj[..., 0:1] - rx
        dy = traj[..., 1:2] - ry
        yaw = torch.atan2(
            torch.sin(traj[..., 2:3] - ryaw),
            torch.cos(traj[..., 2:3] - ryaw),
        )
        vx, vy = traj[..., 3:4], traj[..., 4:5]
        return torch.cat([
            dx * c + dy * s,
            -dx * s + dy * c,
            yaw,
            vx * c + vy * s,
            -vx * s + vy * c,
        ], dim=-1)

    @staticmethod
    def _masked_mean(x, mask):
        weight = mask.to(x.dtype).unsqueeze(-1)
        return (x * weight).sum(dim=-2) / weight.sum(dim=-2).clamp_min(1.0)

    @staticmethod
    def _lagged_rate(value, lag, dt):
        """Causal finite difference with a stable prefix for short histories."""
        result = torch.zeros_like(value)
        if value.shape[-2] <= 1:
            return result
        lag = max(1, min(int(lag), value.shape[-2] - 1))
        rate = (
            value[..., lag:, :] - value[..., :-lag, :]
        ) / max(float(lag) * float(dt), 1e-6)
        result[..., lag:, :] = rate
        result[..., :lag, :] = rate[..., :1, :]
        return result

    @staticmethod
    def _motion_features(relative_traj, dt=0.05):
        """Create scale-controlled kinematic features from a 5-D history."""
        pos_yaw_vel = torch.cat([
            relative_traj[..., 0:1] / 50.0,
            relative_traj[..., 1:2] / 10.0,
            relative_traj[..., 2:3] / math.pi,
            relative_traj[..., 3:5] / 30.0,
        ], dim=-1)
        velocity = relative_traj[..., 3:5]
        acceleration = SVOInteractionTransformer._lagged_rate(
            velocity, 1, dt
        )
        acceleration_short = SVOInteractionTransformer._lagged_rate(
            velocity, 4, dt
        )
        acceleration_long = SVOInteractionTransformer._lagged_rate(
            velocity, 8, dt
        )
        yaw_rate = torch.zeros_like(relative_traj[..., 2:3])
        if relative_traj.shape[-2] > 1:
            dyaw = torch.atan2(
                torch.sin(relative_traj[..., 1:, 2:3] -
                          relative_traj[..., :-1, 2:3]),
                torch.cos(relative_traj[..., 1:, 2:3] -
                          relative_traj[..., :-1, 2:3]),
            ) / max(float(dt), 1e-6)
            yaw_rate[..., 1:, :] = dyaw
            yaw_rate[..., 0, :] = dyaw[..., 0, :]
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True) / 30.0
        return torch.cat([
            pos_yaw_vel,
            acceleration.clamp(-10.0, 10.0) / 10.0,
            acceleration_short.clamp(-8.0, 8.0) / 8.0,
            acceleration_long.clamp(-6.0, 6.0) / 6.0,
            speed,
            yaw_rate.clamp(-2.0, 2.0) / 2.0,
        ], dim=-1)

    @staticmethod
    def _relation_features(ego_rel, npc_rel, dt=0.05):
        """Latest pair geometry, including an observable clipped TTC cue."""
        ego_last, npc_last = ego_rel[:, -1], npc_rel[:, -1]
        long_gap = -npc_last[:, 0]
        lateral_gap = -npc_last[:, 1]
        rel_velocity = ego_last[:, 3:5] - npc_last[:, 3:5]
        distance = torch.linalg.norm(npc_last[:, 0:2], dim=-1)
        closing = rel_velocity[:, 0]
        ttc = torch.full_like(long_gap, 10.0)
        approaching = (long_gap * closing < 0.0) & (closing.abs() > 1e-3)
        candidate_ttc = (-long_gap / closing.clamp(
            min=-1e6, max=1e6
        )).abs().clamp(0.0, 10.0)
        ttc = torch.where(approaching, candidate_ttc, ttc)
        npc_acc = SVOInteractionTransformer._lagged_rate(
            npc_rel[..., 3:5], 1, dt
        )[:, -1]
        npc_trend = SVOInteractionTransformer._lagged_rate(
            npc_rel[..., 3:5], 8, dt
        )[:, -1]
        relative_history = ego_rel[..., 3:5] - npc_rel[..., 3:5]
        relative_trend = SVOInteractionTransformer._lagged_rate(
            relative_history, 8, dt
        )[:, -1]
        return torch.stack([
            long_gap / 50.0,
            lateral_gap / 10.0,
            rel_velocity[:, 0] / 30.0,
            rel_velocity[:, 1] / 15.0,
            distance / 50.0,
            ttc / 10.0,
            npc_acc[:, 0].clamp(-10.0, 10.0) / 10.0,
            npc_acc[:, 1].clamp(-10.0, 10.0) / 10.0,
            npc_trend[:, 0].clamp(-6.0, 6.0) / 6.0,
            npc_trend[:, 1].clamp(-6.0, 6.0) / 6.0,
            relative_trend[:, 0].clamp(-6.0, 6.0) / 6.0,
            relative_trend[:, 1].clamp(-6.0, 6.0) / 6.0,
        ], dim=-1)

    def forward(self, ego_past, npc_past):
        single = ego_past.ndim == 3
        if single:
            ego_past, npc_past = ego_past.unsqueeze(1), npc_past.unsqueeze(1)
        if ego_past.ndim != 4 or npc_past.shape != ego_past.shape:
            raise ValueError(
                f'expected matching (B,S,T,5), got {ego_past.shape}, {npc_past.shape}'
            )

        batch, steps, history, feat = ego_past.shape
        flat_ego = ego_past.reshape(batch * steps, history, feat)
        flat_npc = npc_past.reshape(batch * steps, history, feat)
        ego_mask = (flat_ego.abs() > 1e-7).any(dim=-1)
        npc_mask = (flat_npc.abs() > 1e-7).any(dim=-1)
        ego_mask[:, -1] = True

        ref = flat_ego[:, -1]
        ego_rel = self._ego_frame(flat_ego, ref)
        npc_rel = self._ego_frame(flat_npc, ref)

        ego_features = self._motion_features(ego_rel)
        npc_features = self._motion_features(npc_rel)
        actors = torch.stack([ego_features, npc_features], dim=1)
        masks = torch.stack([ego_mask, npc_mask], dim=1)
        actors = actors.reshape(
            batch * steps * 2, history, self.motion_feature_dim
        )
        masks = masks.reshape(batch * steps * 2, history)
        kpm = _safe_key_padding(masks)

        actor_ids = torch.arange(2, device=actors.device).view(1, 2, 1)
        actor_ids = actor_ids.expand(batch * steps, 2, history).reshape(-1, history)
        encoded = self.input_proj(actors) + self.actor_type(actor_ids)
        encoded = self.temporal(encoded, src_key_padding_mask=kpm)
        pooled = self._masked_mean(encoded, masks)
        pooled = pooled.reshape(batch * steps, 2, self.hidden_dim)

        target = pooled[:, 1:2]
        related, _ = self.rel_attn(target, pooled, pooled)
        related = self.rel_norm(target + related).squeeze(1)
        pair_relation = self.relation_mlp(
            self._relation_features(ego_rel, npc_rel)
        )
        fused = self.output(torch.cat([
            pooled[:, 1], related, pair_relation,
        ], dim=-1))
        fused = fused.reshape(batch, steps, self.hidden_dim)
        return fused[:, 0] if single else fused


class TrajectoryDecoder(nn.Module):
    """Reconstruct target-vehicle futures from deterministic ``h_t``."""

    def __init__(self, latent_dim, hidden=128, traj_dim=5, pred_steps=10):
        super().__init__()
        self.pred_steps = int(pred_steps)
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.cell = nn.GRUCell(traj_dim + hidden, hidden)
        self.out_mu = nn.Linear(hidden, traj_dim)
        self.out_logvar = nn.Linear(hidden, traj_dim)

    def forward(self, latent, last_state, future_gt=None, teacher_forcing=0.0):
        context = self.latent_proj(latent)
        hidden = context
        current = last_state
        mus, logvars = [], []
        for step in range(self.pred_steps):
            hidden = self.cell(torch.cat([current, context], dim=-1), hidden)
            mu = self.out_mu(hidden)
            logvar = self.out_logvar(hidden).clamp(-6.0, 2.0)
            mus.append(mu)
            logvars.append(logvar)
            if (future_gt is not None and self.training and teacher_forcing > 0 and
                    torch.rand((), device=latent.device) < teacher_forcing):
                current = future_gt[:, step]
            else:
                current = mu
        return torch.stack(mus, dim=1), torch.stack(logvars, dim=1)


class SVOObservedBetaModel(nn.Module):
    """Continuous SVO filtering and response trajectory prediction."""

    def __init__(self, config):
        super().__init__()
        cfg = config.svo
        interaction_dim = int(getattr(cfg, 'interaction_dim', cfg.hidden_dim))
        self.beta = float(cfg.beta)
        self.tf_ratio = float(cfg.teacher_forcing_ratio)
        self.nll_weight = float(getattr(cfg, 'nll_loss_weight', 1.0))
        self.angle_weight = float(getattr(cfg, 'angle_loss_weight', 1.0))
        self.prior_angle_weight = float(
            getattr(cfg, 'prior_angle_loss_weight', 0.5)
        )
        self.prior_consistency_weight = float(
            getattr(cfg, 'prior_consistency_loss_weight', 0.1)
        )
        self.uninformative_weight = float(
            getattr(cfg, 'uninformative_loss_weight', 0.05)
        )
        self.temporal_weight = float(getattr(cfg, 'temporal_loss_weight', 0.05))
        self.response_weight = float(getattr(cfg, 'response_loss_weight', 1.0))
        self.evidence_power = float(getattr(cfg, 'evidence_weight_power', 2.0))
        self.evidence_threshold = float(getattr(cfg, 'evidence_threshold', 0.10))
        self.dyn_weight = float(getattr(cfg, 'rssm_dyn_weight', 0.5))
        self.rep_weight = float(getattr(cfg, 'rssm_rep_weight', 0.1))
        self.pred_steps = int(cfg.prediction_horizon)
        self.theta_min = float(getattr(cfg, 'theta_min_deg', 0.0))
        self.theta_max = float(getattr(cfg, 'theta_max_deg', 90.0))
        self.theta_span = self.theta_max - self.theta_min
        if self.theta_span <= 0:
            raise ValueError('theta_max_deg must be larger than theta_min_deg')
        # Retained as metadata for the categorical ablation only.
        self.register_buffer(
            'categorical_anchors',
            torch.linspace(
                self.theta_min, self.theta_max,
                int(getattr(cfg, 'categorical_num_classes', NUM_INTENTS)),
            ),
        )

        self.interaction_encoder = SVOInteractionTransformer(
            feature_dim=5,
            hidden_dim=interaction_dim,
            num_heads=int(getattr(cfg, 'interaction_heads', 4)),
            num_layers=int(getattr(cfg, 'interaction_layers', 2)),
            dropout=float(getattr(cfg, 'interaction_dropout', 0.1)),
        )
        # Keep the registered name ``rssm`` only to preserve legacy state_dict
        # keys. The canonical model name is RBBN.
        self.rssm = RecurrentBetaBeliefNetwork(
            obs_dim=interaction_dim,
            deter_dim=int(getattr(cfg, 'rssm_hidden_dim', cfg.hidden_dim)),
            action_dim=4,
            hidden_dim=int(cfg.hidden_dim),
            min_concentration=float(
                getattr(cfg, 'rssm_min_concentration', 2.0)
            ),
            max_concentration=float(
                getattr(cfg, 'rssm_max_concentration', 200.0)
            ),
            free_nats=float(getattr(cfg, 'rssm_free_nats', 0.05)),
        )
        self.decoder = TrajectoryDecoder(
            # h_t carries the representation used for prediction and RL.
            # The scalar Beta state is reserved for belief/risk reporting.
            latent_dim=self.belief_network.deter_dim,
            hidden=int(cfg.hidden_dim),
            pred_steps=self.pred_steps,
        )
        self.latent_deter_dim = self.belief_network.deter_dim
        self.latent_stoch_dim = 1
        # Kept outside state_dict so legacy checkpoints remain
        # strictly loadable. A train-only calibration fit may populate this
        # with [quadratic, linear, bias] coefficients in degree space.
        self.angle_calibration = (0.0, 1.0, 0.0)

        # Backward-compatible attribute names used by older diagnostics.
        self.traj_encoder = self.interaction_encoder
        self.rssm_cell = self.rssm

    @property
    def belief_network(self):
        """Canonical access to the Recurrent Beta Belief Network (RBBN)."""
        return self.rssm

    def init_states(self, batch_size, device):
        state = self.belief_network.initial(batch_size, device)
        return state.deter, state.stoch

    def init_temporal_state(self, batch_size, device):
        return self.belief_network.initial(batch_size, device).deter

    @staticmethod
    def _ensure_sequence(tensor):
        return tensor.unsqueeze(1) if tensor.ndim == 3 else tensor

    @staticmethod
    def _ego_action_context(ego_past):
        """Observed ego manoeuvre cue, normalised for belief recurrence."""
        last = ego_past[..., -1, :]
        prev = ego_past[..., -2, :] if ego_past.shape[-2] > 1 else last
        dv = last[..., 3:5] - prev[..., 3:5]
        dyaw = torch.atan2(
            torch.sin(last[..., 2] - prev[..., 2]),
            torch.cos(last[..., 2] - prev[..., 2]),
        ).unsqueeze(-1)
        speed = torch.linalg.norm(last[..., 3:5], dim=-1, keepdim=True)
        return torch.cat([dv / 5.0, dyaw / math.pi, speed / 30.0], dim=-1)

    @staticmethod
    def _reference(ego_past):
        ref = ego_past[..., -1, :]
        return (ref[..., 0:1].unsqueeze(-2), ref[..., 1:2].unsqueeze(-2),
                ref[..., 2:3].unsqueeze(-2))

    @staticmethod
    def _to_relative(traj, ego_past):
        rx, ry, ryaw = SVOObservedBetaModel._reference(ego_past)
        c, s = torch.cos(ryaw), torch.sin(ryaw)
        dx, dy = traj[..., 0:1] - rx, traj[..., 1:2] - ry
        yaw = torch.atan2(
            torch.sin(traj[..., 2:3] - ryaw),
            torch.cos(traj[..., 2:3] - ryaw),
        )
        vx, vy = traj[..., 3:4], traj[..., 4:5]
        return torch.cat([
            dx * c + dy * s, -dx * s + dy * c, yaw,
            vx * c + vy * s, -vx * s + vy * c,
        ], dim=-1)

    def _normalised_theta(self, theta):
        return ((theta - self.theta_min) / self.theta_span).clamp(1e-5, 1.0 - 1e-5)

    def _theta_from_unit(self, value):
        return self.theta_min + self.theta_span * value

    def set_angle_calibration(self, coefficients):
        coefficients = tuple(float(value) for value in coefficients)
        if len(coefficients) != 3:
            raise ValueError('angle calibration requires [quadratic, linear, bias]')
        self.angle_calibration = coefficients

    def get_angle_calibration(self):
        return list(self.angle_calibration)

    def _calibrate_theta(self, theta):
        quadratic, linear, bias = self.angle_calibration
        calibrated = quadratic * theta.pow(2) + linear * theta + bias
        return calibrated.clamp(self.theta_min, self.theta_max)

    def _beta_theta_stats(self, mean, std):
        raw_theta = self._theta_from_unit(mean)
        raw_sigma = self.theta_span * std
        calibrated_theta = self._calibrate_theta(raw_theta)
        quadratic, linear, _ = self.angle_calibration
        local_slope = (2.0 * quadratic * raw_theta + linear).abs()
        calibrated_sigma = (raw_sigma * local_slope).clamp_min(1e-4)
        return calibrated_theta, calibrated_sigma

    @staticmethod
    def _weighted_mean(value, weight):
        return (value * weight).sum() / weight.sum().clamp_min(1.0)


    def predict_sequence(self, npc_past, ego_past, valid_mask=None,
                         initial_state=None):
        npc_past = self._ensure_sequence(npc_past)
        ego_past = self._ensure_sequence(ego_past)
        if valid_mask is None:
            valid_mask = (npc_past.abs() > 1e-7).any(dim=-1).any(dim=-1)
        obs = self.interaction_encoder(ego_past, npc_past)
        action = self._ego_action_context(ego_past)
        final, feat = self.belief_network.observe(
            obs, action, valid_mask, initial_state
        )
        theta_post, sigma_post = self._beta_theta_stats(
            feat['post_mean'], feat['post_std']
        )
        theta_prior, sigma_prior = self._beta_theta_stats(
            feat['prior_mean'], feat['prior_std']
        )
        batch, steps = theta_post.shape
        last_rel = self._to_relative(
            npc_past[..., -1:, :], ego_past
        )[..., 0, :]
        latent = feat['deter']
        pred_mu, pred_logvar = self.decoder(
            latent.reshape(batch * steps, -1),
            last_rel.reshape(batch * steps, -1),
            future_gt=None,
            teacher_forcing=0.0,
        )
        pred_mu = pred_mu.reshape(batch, steps, self.pred_steps, -1)
        pred_logvar = pred_logvar.reshape(batch, steps, self.pred_steps, -1)
        return {
            **feat,
            'theta_post': theta_post,
            'sigma_post': sigma_post,
            'theta_prior': theta_prior,
            'sigma_prior': sigma_prior,
            'pred_mu': pred_mu,
            'pred_logvar': pred_logvar,
            'final_state': final,
        }

    @torch.no_grad()
    def infer_observed_latent(self, npc_traj, ego_traj,
                              h_prev=None, s_prev=None):
        npc_seq = self._ensure_sequence(npc_traj)
        ego_seq = self._ensure_sequence(ego_traj)
        initial = None
        if h_prev is not None and s_prev is not None:
            initial = BeliefState(h_prev, s_prev)
        result = self.predict_sequence(
            npc_seq, ego_seq, initial_state=initial
        )
        return {
            'theta_post': result['theta_post'][:, -1],
            'sigma_post': result['sigma_post'][:, -1],
            'theta_prior': result['theta_prior'][:, -1],
            'sigma_prior': result['sigma_prior'][:, -1],
            'alpha_post': result['post_alpha'][:, -1],
            'beta_post': result['post_beta'][:, -1],
            'alpha_prior': result['prior_alpha'][:, -1],
            'beta_prior': result['prior_beta'][:, -1],
            'mean_post_unit': result['post_mean'][:, -1],
            'mean_prior_unit': result['prior_mean'][:, -1],
            'deter': result['deter'][:, -1],
            'stoch': result['stoch'][:, -1],
            'pred_traj': result['pred_mu'][:, -1],
            'pred_logvar': result['pred_logvar'][:, -1],
            'state': result['final_state'],
        }

    @torch.no_grad()
    def infer_observed(self, npc_traj, ego_traj, h_prev=None, s_prev=None):
        out = self.infer_observed_latent(npc_traj, ego_traj, h_prev, s_prev)
        return (out['theta_post'], out['sigma_post'], out['theta_prior'],
                out['sigma_prior'], out['deter'], out['stoch'])

    @torch.no_grad()
    def infer_step_prior(self, npc_traj, ego_traj, h_prev, s_prev):
        out = self.infer_observed_latent(npc_traj, ego_traj, h_prev, s_prev)
        return (float(out['theta_prior'].item()), out['sigma_prior'],
                out['deter'], out['stoch'])

    @torch.no_grad()
    def infer_step(self, npc_traj, ego_traj, h_prev, theta_prev=None):
        if theta_prev is None:
            s_prev = uniform_intent(h_prev.shape[0], h_prev.device)
        else:
            s_prev = self._normalised_theta(theta_prev).reshape(-1, 1)
        out = self.infer_observed_latent(npc_traj, ego_traj, h_prev, s_prev)
        return out['theta_post'], out['sigma_post'], out['deter']

    @torch.no_grad()
    def infer(self, npc_traj, ego_traj):
        out = self.infer_observed_latent(npc_traj, ego_traj)
        return out['theta_post'], out['sigma_post']

    @torch.no_grad()
    def sample_theta(self, npc_traj, ego_traj, n_samples=1):
        out = self.infer_observed_latent(npc_traj, ego_traj)
        samples = Beta(out['alpha_post'], out['beta_post']).sample((n_samples,))
        samples = samples.transpose(0, 1)
        return self._calibrate_theta(self._theta_from_unit(samples))


# Legacy class alias.
SVOVariationalBIRL = SVOObservedBetaModel
