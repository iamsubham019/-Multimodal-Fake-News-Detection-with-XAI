import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Optional

from config import CFG


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(CFG.model.text_encoder)
        self.hidden_dim = CFG.model.text_hidden_dim

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(CFG.model.fusion_dropout),
            nn.Linear(self.hidden_dim, CFG.model.fusion_hidden_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_token_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        token_embeddings = outputs.last_hidden_state

        projected = self.projection(cls_embedding)

        if return_token_embeddings:
            return projected, token_embeddings

        return projected, None

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(CFG.model.text_encoder)

    def freeze_backbone(self, num_layers_to_unfreeze: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = False

        encoder_layers = self.backbone.encoder.layer
        for layer in encoder_layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.backbone.pooler.parameters():
            param.requires_grad = True
