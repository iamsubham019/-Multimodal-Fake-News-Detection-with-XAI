import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from config import CFG


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = query.size(0)

        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key.dim() == 2:
            key = key.unsqueeze(1)
        if value.dim() == 2:
            value = value.unsqueeze(1)

        Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.num_heads * self.head_dim)
        out = self.out_proj(out).squeeze(1)

        if return_weights:
            return out, attn_weights.mean(dim=1).squeeze(1)
        return out, None


class ModalityGate(nn.Module):
    def __init__(self, dim: int, num_modalities: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, *modalities: torch.Tensor) -> torch.Tensor:
        combined = torch.cat(modalities, dim=-1)
        weights = self.gate(combined)
        stacked = torch.stack(modalities, dim=1)
        gated = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return gated, weights


class CrossModalFusionModule(nn.Module):
    def __init__(self, use_text: bool = True, use_image: bool = True, use_social: bool = True):
        super().__init__()
        self.use_text = use_text
        self.use_image = use_image
        self.use_social = use_social

        dim = CFG.model.fusion_hidden_dim
        heads = CFG.model.fusion_num_heads
        dropout = CFG.model.fusion_dropout

        if use_text and use_image:
            self.text_img_attn = CrossModalAttention(dim, heads, dropout)
        if use_text and use_social:
            self.text_social_attn = CrossModalAttention(dim, heads, dropout)
        if use_image and use_social:
            self.img_social_attn = CrossModalAttention(dim, heads, dropout)

        active_modalities = sum([use_text, use_image, use_social])
        self.modality_gate = ModalityGate(dim, active_modalities)

        self.fusion_norm = nn.LayerNorm(dim)
        self.fusion_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

        self.inconsistency_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_emb: Optional[torch.Tensor] = None,
        image_emb: Optional[torch.Tensor] = None,
        social_emb: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Dict:
        attention_weights = {}
        active_modalities = []

        if self.use_text and text_emb is not None:
            active_modalities.append(text_emb)

        if self.use_image and image_emb is not None:
            if self.use_text and text_emb is not None:
                enriched_text, w = self.text_img_attn(
                    text_emb, image_emb, image_emb,
                    return_weights=return_attention_weights,
                )
                active_modalities[0] = enriched_text
                if w is not None:
                    attention_weights["text_attends_image"] = w
            active_modalities.append(image_emb)

        if self.use_social and social_emb is not None:
            if self.use_text and text_emb is not None:
                _, w = self.text_social_attn(
                    active_modalities[0], social_emb, social_emb,
                    return_weights=return_attention_weights,
                )
                if w is not None:
                    attention_weights["text_attends_social"] = w
            active_modalities.append(social_emb)

        gated_fusion, modality_weights = self.modality_gate(*active_modalities)

        fused = self.fusion_norm(gated_fusion)
        fused = fused + self.fusion_ffn(fused)
        fused = self.ffn_norm(fused)

        result = {"fused": fused, "modality_weights": modality_weights}

        if self.use_text and self.use_image and text_emb is not None and image_emb is not None:
            inconsistency = self.inconsistency_head(
                torch.cat([text_emb, image_emb], dim=-1)
            )
            result["inconsistency_score"] = inconsistency

        if return_attention_weights:
            result["attention_weights"] = attention_weights

        return result
