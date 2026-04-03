import os
import io
import json
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms

from config import CFG, LABEL_MAP, LABEL_COLORS
from models.detector import FakeNewsDetector
from xai.explainer import UnifiedExplainer
from data.dataset import get_image_transform


st.set_page_config(
    page_title="Multimodal Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #6366F1; margin-bottom: 0.25rem; }
    .subtitle { font-size: 1rem; color: #6b7280; margin-bottom: 2rem; }
    .pred-real { background: #dcfce7; color: #166534; padding: 0.5rem 1.5rem; border-radius: 999px; font-size: 1.2rem; font-weight: 700; }
    .pred-fake { background: #fee2e2; color: #991b1b; padding: 0.5rem 1.5rem; border-radius: 999px; font-size: 1.2rem; font-weight: 700; }
    .metric-card { background: #f9fafb; border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid #e5e7eb; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #374151; margin: 1.5rem 0 0.75rem; border-bottom: 2px solid #6366F1; padding-bottom: 4px; }
    .token-highlight { display: inline-block; padding: 2px 6px; border-radius: 4px; margin: 2px; font-size: 0.85rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_explainer():
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model.text_encoder)

    model = FakeNewsDetector(use_text=True, use_image=False, use_social=True).to(device)

    checkpoint_path = CFG.app.model_checkpoint
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        st.sidebar.success(f"Model loaded (epoch {ckpt['epoch']})")
    else:
        st.sidebar.warning("No trained checkpoint found. Using untrained model (random predictions).")

    model.eval()
    explainer = UnifiedExplainer(model, tokenizer, device)
    return model, tokenizer, explainer, device


def tokenize_text(text: str, tokenizer):
    return tokenizer(
        text,
        max_length=CFG.data.max_text_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )


def prepare_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    transform = get_image_transform(train=False)
    tensor = transform(img).unsqueeze(0)
    return tensor, np.array(img.resize((224, 224))) / 255.0


def render_token_highlights(tokens, scores):
    html = "<div style='line-height:2.2;'>"
    for token, score in zip(tokens[:50], scores[:50]):
        if not token.strip():
            continue
        intensity = float(score)
        r = int(255 * intensity)
        g = int(200 * (1 - intensity))
        b = 50
        color = f"rgba({r},{g},{b},0.3)"
        html += f'<span class="token-highlight" style="background:{color};">{token}</span> '
    html += "</div>"
    return html


def render_modality_gauge(weights: dict):
    labels = list(weights.keys())
    values = list(weights.values())
    colors = ["#6366F1", "#22c55e", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(labels, values, color=colors[:len(labels)], width=0.4, edgecolor="white", linewidth=2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.0%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Weight", fontsize=10)
    ax.set_title("Modality Contribution", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    return fig


def main():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        show_attention = st.toggle("Show cross-modal attention weights", value=True)
        show_gradcam = st.toggle("Show GradCAM heatmap", value=True)
        st.markdown("---")
        st.markdown("### About this project")
        st.markdown("""
        **Multimodal Fake News Detection with XAI**

        This system analyzes:
        - 📝 Article **text** via RoBERTa
        - 🖼️ News **images** via CLIP-ViT
        - 🌐 **Social propagation** via GNN

        Explanations generated using:
        - SHAP / Gradient attribution (text)
        - GradCAM (images)
        - Cross-modal attention (fusion)
        """)

    st.markdown('<div class="main-title">🔍 Fake News Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multimodal detection with Explainable AI — Text · Image · Social Graph</div>', unsafe_allow_html=True)

    model, tokenizer, explainer, device = load_model_and_explainer()

    tab1, tab2, tab3 = st.tabs(["🔎 Analyze News", "📊 Explanation Dashboard", "📈 Model Info"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📝 News Article Text")
            sample_texts = {
                "Select an example...": "",
                "Suspicious (Sensational headline)": "BREAKING: Scientists discover that common household chemical CURES cancer but Big Pharma is HIDING it from you! Share before this gets deleted! The mainstream media won't cover this shocking revelation that doctors don't want you to know. Thousands have already been cured using this simple method.",
                "Credible (Factual tone)": "The Federal Reserve announced a 25 basis point interest rate increase on Wednesday, citing continued concerns about inflation. The decision was unanimous among voting members of the Federal Open Market Committee. The rate now stands at 5.5%, the highest level in over two decades. Officials signaled they may pause further increases to assess economic conditions.",
            }
            selected = st.selectbox("Load example:", list(sample_texts.keys()))
            text_input = st.text_area(
                "Paste or type news article text:",
                value=sample_texts[selected],
                height=200,
                placeholder="Enter the full news article text here...",
            )

        with col2:
            st.markdown("### 🖼️ Article Image (optional)")
            uploaded_image = st.file_uploader(
                "Upload news image",
                type=["jpg", "jpeg", "png"],
                help="Upload the image associated with the news article",
            )
            if uploaded_image:
                st.image(uploaded_image, use_container_width=True)

        analyze_btn = st.button("🚀 Analyze Article", type="primary", use_container_width=True)

        if analyze_btn and text_input.strip():
            with st.spinner("Analyzing article and generating explanations..."):
                encoding = tokenize_text(text_input, tokenizer)
                import torch as _torch
                batch = {
                 "input_ids": encoding["input_ids"],
                 "attention_mask": encoding["attention_mask"],
                 "graph_x": [_torch.zeros(1, 16)],
                 "graph_edge_index": [_torch.zeros(2, 0, dtype=_torch.long)],
                 "graph_num_nodes": [1],
                }

                image_array = None
                if uploaded_image and model.use_image:
                    img_tensor, image_array = prepare_image(uploaded_image)
                    batch["image"] = img_tensor
                else:
                    batch["image"] = torch.zeros(1, 3, 224, 224)

                explanation = explainer.explain_prediction(batch, news_id="user_input")

            st.markdown("---")
            pred = explanation["prediction"]
            conf = explanation["confidence"]

            result_col1, result_col2, result_col3, result_col4 = st.columns(4)

            css_class = "pred-fake" if pred == "Fake" else "pred-real"
            result_col1.markdown(f'<div class="metric-card"><div class="metric-value"><span class="{css_class}">{pred}</span></div><div class="metric-label">Prediction</div></div>', unsafe_allow_html=True)
            result_col2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#6366F1">{conf:.1%}</div><div class="metric-label">Confidence</div></div>', unsafe_allow_html=True)
            result_col3.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#22c55e">{explanation["probabilities"]["Real"]:.1%}</div><div class="metric-label">P(Real)</div></div>', unsafe_allow_html=True)
            result_col4.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ef4444">{explanation["probabilities"]["Fake"]:.1%}</div><div class="metric-label">P(Fake)</div></div>', unsafe_allow_html=True)

            incons = explanation.get("inconsistency_score", 0)
            if incons > 0.6:
                st.warning(f"⚠️ High text-image inconsistency detected: {incons:.2%} — the image may not match the article's claims.")
            elif incons > 0.3:
                st.info(f"ℹ️ Moderate text-image inconsistency: {incons:.2%}")

            st.session_state["explanation"] = explanation
            st.session_state["image_array"] = image_array
            st.success("Analysis complete! Switch to the **Explanation Dashboard** tab for full XAI details.")

        elif analyze_btn:
            st.error("Please enter some article text before analyzing.")

    with tab2:
        if "explanation" not in st.session_state:
            st.info("Run an analysis first in the **Analyze News** tab.")
        else:
            explanation = st.session_state["explanation"]
            image_array = st.session_state.get("image_array")

            st.markdown(f'<div class="section-header">📝 Text Attribution (What words triggered the decision?)</div>', unsafe_allow_html=True)
            if "text" in explanation:
                text_exp = explanation["text"]
                token_html = render_token_highlights(text_exp["tokens"], text_exp["scores"])
                st.markdown(token_html, unsafe_allow_html=True)

                st.markdown("**Top influential tokens:**")
                token_cols = st.columns(min(5, len(text_exp["top_tokens"])))
                for i, (token, score) in enumerate(text_exp["top_tokens"][:5]):
                    with token_cols[i]:
                        st.metric(f"#{i+1}", token, f"{score:.3f}")

            if show_attention and "fusion" in explanation:
                st.markdown('<div class="section-header">⚡ Modality Contribution (Which signal drove the prediction?)</div>', unsafe_allow_html=True)
                fig = render_modality_gauge(explanation["fusion"]["modality_weights"])
                st.pyplot(fig)
                plt.close(fig)

            if show_gradcam and "image" in explanation and explanation["image"]["gradcam_map"] is not None:
                st.markdown('<div class="section-header">🖼️ GradCAM Image Attribution (Where did the model look?)</div>', unsafe_allow_html=True)
                from xai.explainer import ImageExplainer
                img_exp = ImageExplainer(model, device)
                fig = img_exp.visualize(
                    explanation["image"],
                    original_image=image_array,
                    title="GradCAM Attribution",
                )
                st.pyplot(fig)
                plt.close(fig)

            st.markdown('<div class="section-header">📋 Full Explanation Report</div>', unsafe_allow_html=True)
            report_data = {
                "prediction": explanation["prediction"],
                "confidence": f"{explanation['confidence']:.2%}",
                "probabilities": explanation["probabilities"],
                "inconsistency_score": explanation.get("inconsistency_score", 0),
                "top_tokens": explanation.get("text", {}).get("top_tokens", []),
                "modality_weights": explanation.get("fusion", {}).get("modality_weights", {}),
            }
            st.json(report_data)

    with tab3:
        st.markdown("### Model Architecture Summary")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Text Stream**
            - Backbone: RoBERTa-base (125M params)
            - Output: 768-dim CLS embedding
            - Fine-tuning: Last 2 transformer layers

            **Image Stream**
            - Backbone: CLIP-ViT-Base/32
            - Output: 512-dim patch CLS embedding
            - Fine-tuning: Last 2 vision layers

            **Social Stream**
            - Architecture: GAT (2 layers, 4 heads)
            - Pooling: Mean + Max global pool
            - Output: 256-dim graph embedding
            """)
        with col_b:
            st.markdown("""
            **Fusion Module**
            - Cross-modal attention (8 heads)
            - Modality gating (learned weights)
            - FFN with GELU activation
            - Text-image inconsistency score

            **Classifier**
            - 2-layer MLP (512 → 256 → 2)
            - LayerNorm + Dropout (0.1)
            - Cross-entropy + inconsistency loss

            **XAI Methods**
            - Gradient attribution (text)
            - GradCAM via Captum (images)
            - Attention weights (fusion)
            """)

        log_path = os.path.join(CFG.train.log_dir, "training_results.json")
        if os.path.exists(log_path):
            with open(log_path) as f:
                training_log = json.load(f)
            tm = training_log.get("test_metrics", {})
            st.markdown("### Test Set Performance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{tm.get('accuracy', 0):.4f}")
            m2.metric("F1 Macro", f"{tm.get('f1_macro', 0):.4f}")
            m3.metric("F1 Fake", f"{tm.get('f1_fake', 0):.4f}")
            m4.metric("AUC-ROC", f"{tm.get('auc_roc', 0):.4f}")


if __name__ == "__main__":
    main()
