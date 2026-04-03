import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPProcessor
from typing import Tuple, Optional

from config import CFG


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = CLIPVisionModel.from_pretrained(CFG.model.image_encoder)
        self.hidden_dim = CFG.model.image_hidden_dim

        clip_out_dim = self.backbone.config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(clip_out_dim, clip_out_dim),
            nn.LayerNorm(clip_out_dim),
            nn.GELU(),
            nn.Dropout(CFG.model.fusion_dropout),
            nn.Linear(clip_out_dim, CFG.model.fusion_hidden_dim),
        )

        self.gradcam_gradients = None
        self.gradcam_activations = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_patch_embeddings: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )

        cls_embedding = outputs.pooler_output
        patch_embeddings = outputs.last_hidden_state

        projected = self.projection(cls_embedding)

        if return_patch_embeddings:
            return projected, patch_embeddings

        return projected, None

    def register_gradcam_hooks(self):
        target_layer = self.backbone.vision_model.encoder.layers[-1]

        def save_gradient(grad):
            self.gradcam_gradients = grad

        def save_activation(module, input, output):
            self.gradcam_activations = output[0]
            if output[0].requires_grad:
                output[0].register_hook(save_gradient)

        target_layer.register_forward_hook(save_activation)

    def get_gradcam_map(self, image_size: int = 14) -> Optional[torch.Tensor]:
        if self.gradcam_gradients is None or self.gradcam_activations is None:
            return None

        gradients = self.gradcam_gradients
        activations = self.gradcam_activations

        weights = gradients.mean(dim=1, keepdim=True)
        cam = (weights * activations).sum(dim=-1)
        cam = torch.relu(cam)

        patch_tokens = cam[:, 1:]
        batch_size = patch_tokens.size(0)
        n_patches = patch_tokens.size(1)
        h = w = int(n_patches ** 0.5)

        cam_map = patch_tokens.view(batch_size, 1, h, w)
        cam_map = nn.functional.interpolate(
            cam_map, size=(image_size * 16, image_size * 16),
            mode="bilinear", align_corners=False,
        )

        cam_min = cam_map.flatten(2).min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        cam_max = cam_map.flatten(2).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
        cam_map = (cam_map - cam_min) / (cam_max - cam_min + 1e-8)

        return cam_map.squeeze(1)

    def freeze_backbone(self, num_layers_to_unfreeze: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = False

        encoder_layers = self.backbone.vision_model.encoder.layers
        for layer in encoder_layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
