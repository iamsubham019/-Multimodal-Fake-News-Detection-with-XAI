"""
Paper 2: Cross-Modal Semantic Inconsistency Detector

This module implements a contrastive CLIP-based detector that identifies when
an image is semantically inconsistent with the accompanying text — a strong
signal for out-of-context fake news.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from typing import List, Tuple, Dict, Optional
import numpy as np

from config import CFG


class CLIPInconsistencyDetector(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        clip_dim = self.clip.config.projection_dim

        for param in self.clip.parameters():
            param.requires_grad = False
        for layer in self.clip.text_model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        for layer in self.clip.vision_model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.inconsistency_head = nn.Sequential(
            nn.Linear(clip_dim * 3, clip_dim),
            nn.LayerNorm(clip_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(clip_dim, clip_dim // 2),
            nn.GELU(),
            nn.Linear(clip_dim // 2, 1),
        )

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict:
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        image_features = self.clip.get_image_features(pixel_values=pixel_values)

        text_norm = F.normalize(text_features, dim=-1)
        image_norm = F.normalize(image_features, dim=-1)

        cosine_sim = (text_norm * image_norm).sum(dim=-1, keepdim=True)
        diff = text_norm - image_norm
        combined = torch.cat([text_norm, image_norm, diff], dim=-1)

        inconsistency_logit = self.inconsistency_head(combined)
        inconsistency_score = torch.sigmoid(inconsistency_logit)

        result = {
            "inconsistency_score": inconsistency_score,
            "cosine_similarity": cosine_sim,
            "is_inconsistent": (inconsistency_score > 0.5).float(),
        }

        if return_embeddings:
            result["text_features"] = text_features
            result["image_features"] = image_features

        return result

    def get_contrastive_loss(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        text_norm = F.normalize(text_features, dim=-1)
        image_norm = F.normalize(image_features, dim=-1)
        temp = torch.clamp(self.temperature, 0.01, 0.5)

        logits_per_text = (text_norm @ image_norm.T) / temp
        logits_per_image = logits_per_text.T

        batch_size = text_features.size(0)
        paired_indices = torch.arange(batch_size, device=text_features.device)

        clip_loss = (
            F.cross_entropy(logits_per_text, paired_indices) +
            F.cross_entropy(logits_per_image, paired_indices)
        ) / 2

        cosine_sim = (text_norm * image_norm).sum(dim=-1)
        fake_mask = (labels == 1).float()
        consistency_loss = (
            fake_mask * F.relu(cosine_sim - 0.3) +
            (1 - fake_mask) * F.relu(0.7 - cosine_sim)
        ).mean()

        return clip_loss + 0.5 * consistency_loss

    def explain_inconsistency(
        self,
        text: str,
        image_tensor: torch.Tensor,
        processor: CLIPProcessor,
        top_k_patches: int = 5,
    ) -> Dict:
        self.eval()
        device = next(self.parameters()).device

        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        pixel_values = image_tensor.to(device)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(
                input_ids, attention_mask, pixel_values, return_embeddings=True
            )

        text_features = out["text_features"]
        image_features = out["image_features"]

        vision_outputs = self.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True)
        patch_features = vision_outputs.last_hidden_state[:, 1:, :]

        text_norm = F.normalize(text_features, dim=-1)
        patch_norm = F.normalize(patch_features, dim=-1)

        patch_similarities = (patch_norm * text_norm.unsqueeze(1)).sum(dim=-1)
        patch_similarities = patch_similarities[0].cpu().numpy()

        n_patches = patch_similarities.shape[0]
        h = w = int(n_patches ** 0.5)
        patch_map = patch_similarities.reshape(h, w)

        top_k_idx = np.argsort(patch_similarities.flatten())[-top_k_patches:]
        top_k_coords = [(idx // w, idx % w) for idx in top_k_idx]

        tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        tokens = [t for t in tokens if t not in ["<|startoftext|>", "<|endoftext|>", "!"]]

        return {
            "inconsistency_score": out["inconsistency_score"].item(),
            "cosine_similarity": out["cosine_similarity"].item(),
            "patch_similarity_map": patch_map,
            "top_inconsistent_patches": top_k_coords,
            "text_tokens": tokens,
            "verdict": "INCONSISTENT" if out["inconsistency_score"].item() > 0.5 else "CONSISTENT",
        }


class InconsistencyAwareDetector(nn.Module):
    def __init__(self, base_detector, inconsistency_detector: CLIPInconsistencyDetector):
        super().__init__()
        self.base_detector = base_detector
        self.inconsistency_detector = inconsistency_detector

        self.combination_gate = nn.Sequential(
            nn.Linear(CFG.model.fusion_hidden_dim + 1, CFG.model.fusion_hidden_dim),
            nn.GELU(),
            nn.Linear(CFG.model.fusion_hidden_dim, CFG.model.num_classes),
        )

    def forward(self, batch: Dict) -> Dict:
        device = next(self.parameters()).device

        base_out = self.base_detector(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            image=batch.get("image", torch.zeros(len(batch["input_ids"]), 3, 224, 224)).to(device),
            graph_x=batch.get("graph_x"),
            graph_edge_index=batch.get("graph_edge_index"),
            graph_num_nodes=batch.get("graph_num_nodes"),
            return_embeddings=True,
        )

        incons_out = self.inconsistency_detector(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch.get("image", torch.zeros(len(batch["input_ids"]), 3, 224, 224)).to(device),
        )

        fused_emb = base_out["fused_embedding"]
        incons_score = incons_out["inconsistency_score"]

        enhanced = torch.cat([fused_emb, incons_score], dim=-1)
        logits = self.combination_gate(enhanced)

        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "base_probabilities": base_out["probabilities"],
            "inconsistency_score": incons_score,
            "cosine_similarity": incons_out["cosine_similarity"],
        }
