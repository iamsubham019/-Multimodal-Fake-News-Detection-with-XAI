import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from config import CFG
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from models.social_encoder import SocialGraphEncoder
from models.fusion import CrossModalFusionModule


class FakeNewsDetector(nn.Module):
    def __init__(
        self,
        use_text: bool = True,
        use_image: bool = True,
        use_social: bool = True,
    ):
        super().__init__()
        self.use_text = use_text
        self.use_image = use_image
        self.use_social = use_social

        if use_text:
            self.text_encoder = TextEncoder()
            self.text_encoder.freeze_backbone(num_layers_to_unfreeze=2)

        if use_image:
            self.image_encoder = ImageEncoder()
            self.image_encoder.freeze_backbone(num_layers_to_unfreeze=2)
            self.image_encoder.register_gradcam_hooks()

        if use_social:
            self.social_encoder = SocialGraphEncoder()

        self.fusion = CrossModalFusionModule(use_text, use_image, use_social)

        dim = CFG.model.fusion_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(dim, CFG.model.classifier_hidden_dim),
            nn.LayerNorm(CFG.model.classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(CFG.model.fusion_dropout),
            nn.Linear(CFG.model.classifier_hidden_dim, CFG.model.num_classes),
        )

        self.inconsistency_weight = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        graph_x=None,
        graph_edge_index=None,
        graph_num_nodes=None,
        return_attention_weights: bool = False,
        return_embeddings: bool = False,
    ) -> Dict:
        text_emb = image_emb = social_emb = None
        token_embs = patch_embs = None

        if self.use_text and input_ids is not None:
            text_emb, token_embs = self.text_encoder(
                input_ids, attention_mask,
                return_token_embeddings=return_embeddings,
            )

        if self.use_image and image is not None:
            image_emb, patch_embs = self.image_encoder(
                image,
                return_patch_embeddings=return_embeddings,
            )

        if self.use_social and graph_x is not None:
            social_emb = self.social_encoder(
                graph_x, graph_edge_index, graph_num_nodes,
            )

        fusion_out = self.fusion(
            text_emb=text_emb,
            image_emb=image_emb,
            social_emb=social_emb,
            return_attention_weights=return_attention_weights,
        )

        fused = fusion_out["fused"]
        logits = self.classifier(fused)

        result = {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "modality_weights": fusion_out.get("modality_weights"),
            "inconsistency_score": fusion_out.get("inconsistency_score"),
        }

        if return_attention_weights:
            result["attention_weights"] = fusion_out.get("attention_weights", {})

        if return_embeddings:
            result["text_embedding"] = text_emb
            result["image_embedding"] = image_emb
            result["social_embedding"] = social_emb
            result["fused_embedding"] = fused
            result["token_embeddings"] = token_embs
            result["patch_embeddings"] = patch_embs

        return result

    def get_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inconsistency_score: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss()(logits, labels)

        if inconsistency_score is not None:
            fake_mask = (labels == 1).float().unsqueeze(-1)
            inconsistency_loss = (
                fake_mask * (1 - inconsistency_score) +
                (1 - fake_mask) * inconsistency_score
            ).mean()
            total_loss = ce_loss + torch.sigmoid(self.inconsistency_weight) * inconsistency_loss
            return total_loss

        return ce_loss

    def predict(self, *args, **kwargs) -> Tuple[torch.Tensor, float]:
        self.eval()
        with torch.no_grad():
            out = self.forward(*args, **kwargs)
        pred = out["probabilities"].argmax(dim=-1)
        conf = out["probabilities"].max(dim=-1).values
        return pred, conf.item()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
