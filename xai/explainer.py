import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Dict, List, Optional, Tuple
import shap
from captum.attr import LayerGradCam, LayerAttribution
from transformers import AutoTokenizer

from config import CFG, LABEL_MAP


class TextExplainer:
    def __init__(self, model, tokenizer: AutoTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def explain(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int = 1,
        n_background: int = 20,
    ) -> Dict:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        clean_tokens = [
            t.replace("Ġ", " ").replace("▁", " ").strip()
            for t in tokens
        ]

        token_scores = self._compute_gradient_attribution(
            input_ids, attention_mask, target_class
        )

        mask = attention_mask[0].bool().cpu().numpy()
        active_tokens = [t for t, m in zip(clean_tokens, mask) if m and t not in ["<s>", "</s>", "[CLS]", "[SEP]", "[PAD]"]]
        active_scores = token_scores[mask][:len(active_tokens)]

        active_scores = (active_scores - active_scores.min()) / (active_scores.max() - active_scores.min() + 1e-8)

        top_k = min(CFG.xai.top_k_tokens, len(active_tokens))
        top_indices = np.argsort(active_scores)[-top_k:][::-1]
        top_tokens = [(active_tokens[i], float(active_scores[i])) for i in top_indices]

        return {
            "tokens": active_tokens,
            "scores": active_scores.tolist(),
            "top_tokens": top_tokens,
        }

    def _compute_gradient_attribution(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        self.model.eval()
        embedding_layer = self.model.text_encoder.backbone.embeddings.word_embeddings

        embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

        outputs = self.model.text_encoder.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        projected = self.model.text_encoder.projection(cls)

        if hasattr(self.model, 'classifier'):
            dummy_fused = projected
            logits = self.model.classifier(dummy_fused)
        else:
            logits = projected

        score = logits[0, target_class] if logits.dim() > 1 and logits.size(-1) > 1 else logits[0]
        score.backward()

        gradients = embeddings.grad[0]
        scores = (gradients * embeddings[0].detach()).sum(dim=-1)
        return scores.detach().cpu().numpy()

    def visualize(
        self,
        explanation: Dict,
        title: str = "Text Attribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        tokens = explanation["tokens"][:30]
        scores = np.array(explanation["scores"][:30])

        fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.4), 3))

        colors = plt.cm.RdYlGn(scores)
        ax.bar(range(len(tokens)), scores, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Attribution Score")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


class ImageExplainer:
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    def explain(
        self,
        pixel_values: torch.Tensor,
        target_class: int = 1,
    ) -> Dict:
        pixel_values = pixel_values.to(self.device).requires_grad_(True)

        self.model.eval()
        batch_size = pixel_values.size(0)
        dummy_ids = torch.zeros(batch_size, 512, dtype=torch.long, device=pixel_values.device)
        dummy_mask = torch.ones(batch_size, 512, dtype=torch.long, device=pixel_values.device)
        dummy_x = [torch.zeros(1, 16) for _ in range(batch_size)]
        dummy_edge = [torch.zeros(2, 0, dtype=torch.long) for _ in range(batch_size)]
        dummy_nodes = [1 for _ in range(batch_size)]
        out = self.model(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            image=pixel_values,
            graph_x=dummy_x,
            graph_edge_index=dummy_edge,
            graph_num_nodes=dummy_nodes,
            return_embeddings=False,
        )
        logits = out["logits"]

        score = logits[0, target_class]
        score.backward()

        gradcam_map = self.model.image_encoder.get_gradcam_map()

        return {
            "gradcam_map": gradcam_map.cpu().detach().numpy() if gradcam_map is not None else None,
            "pixel_values": pixel_values.cpu().detach().numpy(),
        }

    def visualize(
        self,
        explanation: Dict,
        original_image: Optional[np.ndarray] = None,
        title: str = "GradCAM Image Attribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        if original_image is not None:
            img_display = original_image
        else:
            pv = explanation["pixel_values"][0]
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img_display = (pv.transpose(1, 2, 0) * std + mean).clip(0, 1)

        axes[0].imshow(img_display)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        if explanation["gradcam_map"] is not None:
            cam = explanation["gradcam_map"][0]
            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title("GradCAM Heatmap")
            axes[1].axis("off")

            heatmap = cm.jet(cam)[:, :, :3]
            overlay = 0.6 * img_display + 0.4 * heatmap
            overlay = overlay.clip(0, 1)
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")
        else:
            for ax in axes[1:]:
                ax.text(0.5, 0.5, "No image data", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")

        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


class FusionExplainer:
    def explain(self, modality_weights: torch.Tensor, use_social: bool = True) -> Dict:
        weights = modality_weights[0].cpu().detach().numpy()
        labels = []
        if True:
            labels.append("Text")
        if True:
            labels.append("Image")
        if use_social:
            labels.append("Social")
        labels = labels[:len(weights)]

        return {"modality_weights": dict(zip(labels, weights.tolist()))}

    def visualize(
        self,
        explanation: Dict,
        title: str = "Modality Contribution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        weights = explanation["modality_weights"]
        labels = list(weights.keys())
        values = list(weights.values())

        colors = ["#6366F1", "#22c55e", "#f59e0b"][:len(labels)]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2%}", ha="center", va="bottom", fontsize=11,
            )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Attention Weight")
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


class UnifiedExplainer:
    def __init__(self, model, tokenizer: AutoTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.text_explainer = TextExplainer(model, tokenizer, device)
        self.image_explainer = ImageExplainer(model, device)
        self.fusion_explainer = FusionExplainer()
        os.makedirs(CFG.xai.output_dir, exist_ok=True)

    def explain_prediction(
        self,
        batch: Dict,
        news_id: str = "sample",
    ) -> Dict:
        self.model.eval()

        with torch.no_grad():
            out = self.model(
             input_ids=batch.get("input_ids", torch.zeros(1, 512, dtype=torch.long)).to(self.device),
             attention_mask=batch.get("attention_mask", torch.ones(1, 512, dtype=torch.long)).to(self.device),
             image=None,
             graph_x=batch.get("graph_x") if batch.get("graph_x") is not None else [torch.zeros(1, 16)],
             graph_edge_index=batch.get("graph_edge_index") if batch.get("graph_edge_index") is not None else [torch.zeros(2, 0, dtype=torch.long)],
             graph_num_nodes=batch.get("graph_num_nodes") if batch.get("graph_num_nodes") is not None else [1],
             return_attention_weights=True,
             return_embeddings=False,
           )

        pred_class = out["probabilities"].argmax(dim=-1).item()
        confidence = out["probabilities"].max().item()

        explanation = {
            "prediction": LABEL_MAP[pred_class],
            "confidence": confidence,
            "probabilities": {
                "Real": out["probabilities"][0, 0].item(),
                "Fake": out["probabilities"][0, 1].item(),
            },
            "inconsistency_score": float(out["inconsistency_score"].item()) if out.get("inconsistency_score") is not None else 0.0,
        }

        if "input_ids" in batch:
            text_exp = self.text_explainer.explain(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                target_class=pred_class,
            )
            explanation["text"] = text_exp

        if self.model.use_image and "image" in batch:
            image_exp = self.image_explainer.explain(
                batch["image"].to(self.device),
                target_class=pred_class,
            )
            explanation["image"] = image_exp

        if out.get("modality_weights") is not None:
            fusion_exp = self.fusion_explainer.explain(
                out["modality_weights"],
                use_social=self.model.use_social,
            )
            explanation["fusion"] = fusion_exp

        return explanation

    def generate_report(
        self,
        explanation: Dict,
        news_id: str = "sample",
        save_dir: Optional[str] = None,
    ) -> str:
        save_dir = save_dir or CFG.xai.output_dir
        os.makedirs(save_dir, exist_ok=True)

        figures = {}

        if "text" in explanation:
            fig = self.text_explainer.visualize(
                explanation["text"],
                title=f"Text Attribution — Prediction: {explanation['prediction']}",
                save_path=os.path.join(save_dir, f"{news_id}_text_xai.png"),
            )
            figures["text"] = fig
            plt.close(fig)

        if "image" in explanation:
            fig = self.image_explainer.visualize(
                explanation["image"],
                title=f"GradCAM — Prediction: {explanation['prediction']}",
                save_path=os.path.join(save_dir, f"{news_id}_image_xai.png"),
            )
            figures["image"] = fig
            plt.close(fig)

        if "fusion" in explanation:
            fig = self.fusion_explainer.visualize(
                explanation["fusion"],
                title=f"Modality Weights — {explanation['prediction']} ({explanation['confidence']:.1%})",
                save_path=os.path.join(save_dir, f"{news_id}_fusion_xai.png"),
            )
            figures["fusion"] = fig
            plt.close(fig)

        report_path = os.path.join(save_dir, f"{news_id}_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"XAI Report — {news_id}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Prediction : {explanation['prediction']}\n")
            f.write(f"Confidence : {explanation['confidence']:.2%}\n")
            f.write(f"Real prob  : {explanation['probabilities']['Real']:.2%}\n")
            f.write(f"Fake prob  : {explanation['probabilities']['Fake']:.2%}\n")
            f.write(f"Text-Image Inconsistency Score: {explanation['inconsistency_score']:.4f}\n\n")

            if "text" in explanation:
                f.write("Top influential tokens:\n")
                for token, score in explanation["text"]["top_tokens"]:
                    f.write(f"  '{token}': {score:.4f}\n")
                f.write("\n")

            if "fusion" in explanation:
                f.write("Modality contributions:\n")
                for mod, w in explanation["fusion"]["modality_weights"].items():
                    f.write(f"  {mod}: {w:.2%}\n")

        return report_path
