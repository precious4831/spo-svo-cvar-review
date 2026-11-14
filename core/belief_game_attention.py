import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_attention(attn_module, q, k, v, key_padding_mask=None):
    """
    安全的 attention 调用:
    如果某个 batch 的 key 全被 mask，则取消该 batch 的 mask 防止 NaN
    """
    if key_padding_mask is not None:
        all_masked = key_padding_mask.all(dim=-1)           # (B,)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
    out, w = attn_module(q, k, v, key_padding_mask=key_padding_mask)
    return out, w


class GameAwareCrossAttention(nn.Module):
    """
    融合社会信念后验、RSSM deterministic belief、社会信念条件化的周车预测响应轨迹，
    以及包含Level-1实际动作的车辆关系表征，形成ego的Belief-Game交互表示。

    输入:
        ego_enc      : (B, units)  L4输出的ego特征 (要求2D, 即num_modes=1)
        svo_mu       : (B, N)      SVO后验均值(度)
        svo_sigma    : (B, N)      SVO后验标准差(度)
        pred_trajs   : (B, N, H, 5) Beta-RSSM belief-conditioned predicted
                                    response trajectory
        interact_mask: (B, N)      交互mask, True=交互车辆

    输出:
        (B, units) 增强后的ego编码

    流程:
        1. SVO编码: [cos(μ), sin(μ), σ/σ_prior, log(σ)] → MLP → (B, N, units)
        2. 社会信念条件化预测响应轨迹编码: GRU(pred_trajs) → (B, N, units)
        3. 融合: posterior moments + deterministic belief + response trajectory
           + vehicle relation (including Level-1 actual action) → tokens
           (B, N, units)
        4. 置信度门控: tokens *= sigmoid(k - σ/τ)  低确信时衰减
        5. Cross-attention: ego_enc attend tokens (仅交互车辆)
        6. 残差连接: output = ego_enc + LayerNorm(attn_out)
    """

    # 门控超参 (控制置信度→门控值的映射)
    GATE_BIAS = 2.0     # 基础偏置, 高确信时gate≈sigmoid(2)=0.88
    GATE_TEMP = 10.0    # 温度, sigma/temp控制衰减速率

    def __init__(self, units=128, num_heads=2, prior_sigma=25.0,
                 deter_dim=128):
        super().__init__()
        self.prior_sigma = prior_sigma  # 用于归一化sigma输入

        # SVO角度 → embedding
        # 输入4维: [cos(mu_rad), sin(mu_rad), sigma/prior_sigma, log(sigma)]
        self.svo_proj = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, units),
        )

        # Deterministic RSSM belief and the per-actor L1/L2 relation
        # token are both retained; this prevents the policy from reducing the
        # social belief to a single angle.
        self.deter_proj = nn.Sequential(
            nn.Linear(deter_dim, units), nn.ReLU(), nn.Linear(units, units)
        )
        self.relation_proj = nn.Sequential(
            nn.Linear(units, units), nn.ReLU(), nn.Linear(units, units)
        )

        # Beta-RSSM belief-conditioned predicted response trajectory → embedding
        self.traj_gru = nn.GRU(5, units, batch_first=True)

        # 融合 posterior moments + h_t + response trajectory + actor relation
        self.fusion = nn.Linear(units * 4, units)

        # Cross-attention: ego attend NPC tokens
        self.cross_attn = nn.MultiheadAttention(
            units, num_heads, dropout=0, batch_first=True
        )
        self.norm = nn.LayerNorm(units)

    def forward(self, ego_enc, neighbor_rel, svo_mu, svo_sigma, svo_deter,
                pred_trajs, interact_mask):
        """
        ego_enc      : (B, units)
        svo_mu       : (B, N) degrees
        svo_sigma    : (B, N) degrees
        pred_trajs   : (B, N, H, 5) Beta-RSSM belief-conditioned predicted
                                    response trajectory
        interact_mask: (B, N) bool

        Returns: (B, units)
        """
        B, N = svo_mu.shape

        # --- 1. SVO角度编码 ---
        mu_rad = svo_mu * (np.pi / 180.0)
        svo_input = torch.stack([
            torch.cos(mu_rad),
            torch.sin(mu_rad),
            svo_sigma / self.prior_sigma,     # 归一化 (不再硬编码25.0)
            torch.log(svo_sigma + 1e-3),      # log尺度特征
        ], dim=-1)                             # (B, N, 4)
        svo_emb = self.svo_proj(svo_input)    # (B, N, units)

        # --- 2. belief-conditioned response trajectory encoding ---
        # reshape (B, N, H, 5) → (B*N, H, 5) 过GRU
        H = pred_trajs.shape[2]
        flat_trajs = pred_trajs.reshape(B * N, H, 5)
        _, h_traj = self.traj_gru(flat_trajs)              # (1, B*N, units)
        traj_emb = h_traj.squeeze(0).reshape(B, N, -1)     # (B, N, units)

        # --- 3. deterministic belief + current L1/L2 relation ---
        if svo_deter is None:
            deter_emb = torch.zeros_like(svo_emb)
        else:
            deter_emb = self.deter_proj(svo_deter)
        relation_emb = self.relation_proj(neighbor_rel)

        # --- 4. 融合 ---
        tokens = self.fusion(torch.cat(
            [svo_emb, deter_emb, traj_emb, relation_emb], dim=-1
        ))

        # --- 5. 置信度门控 ---
        # Keep a non-zero floor: high uncertainty is itself risk information
        # and must not make an actor disappear from the Belief-Game layer.
        gate = 0.35 + 0.65 * torch.sigmoid(
            self.GATE_BIAS - svo_sigma / self.GATE_TEMP
        ).unsqueeze(-1)                                    # (B, N, 1)
        tokens = tokens * gate

        # --- 6. Cross-attention ---
        q = ego_enc.unsqueeze(1)                           # (B, 1, units)

        # key_padding_mask: True = 忽略 → 非交互车辆忽略
        kpm = ~interact_mask if interact_mask is not None else None
        attn_out, _ = _safe_attention(
            self.cross_attn, q, tokens, tokens, key_padding_mask=kpm
        )                                                  # (B, 1, units)

        # --- 7. 残差连接 ---
        output = ego_enc + self.norm(F.relu(attn_out.squeeze(1)))
        if interact_mask is not None:
            output = torch.where(
                interact_mask.any(dim=-1, keepdim=True), output, ego_enc
            )
        return output
