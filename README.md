# Multimodal Fake News Detection with Explainable AI



A tri-stream deep learning system that detects fake news by jointly analyzing **text**, **images**, and **social propagation graphs** — with full XAI explanations for every prediction.

---

## Project Structure

```
fakenews_xai/
├── config.py                        # All hyperparameters and settings
├── train.py                         # Training loop + ablation study
├── evaluate.py                      # Evaluation + visualization + XAI reports
├── app.py                           # Streamlit web demo
├── requirements.txt
│
├── data/
│   └── dataset.py                   # Dataset loading, preprocessing, DataLoaders
│
├── models/
│   ├── text_encoder.py              # RoBERTa text encoder
│   ├── image_encoder.py             # CLIP-ViT image encoder with GradCAM hooks
│   ├── social_encoder.py            # GAT-based social graph encoder
│   ├── fusion.py                    # Cross-modal attention fusion (core contribution)
│   ├── detector.py                  # Main FakeNewsDetector combining all streams
│   └── inconsistency_detector.py   # Paper 2: CLIP inconsistency module
│
├── xai/
│   └── explainer.py                 # Unified XAI: SHAP text, GradCAM image, attention fusion
│
└── outputs/
    ├── checkpoints/                 # Saved model weights
    ├── logs/                        # Training metrics JSON
    └── explanations/                # Per-sample XAI reports
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install -r requirements.txt
```

### 2. (Optional) Download FakeNewsNet dataset

```bash
git clone https://github.com/KaiDMML/FakeNewsNet.git
# Follow their instructions to download data
# Place in data/raw/fake/ and data/raw/real/
```

If no dataset is found, the system automatically generates a **synthetic dataset** so you can run and test the full pipeline immediately.

### 3. Train the model

```bash
python train.py
```

This will:
- Train the full tri-stream model for up to 20 epochs with early stopping
- Save the best checkpoint to `outputs/checkpoints/best_model.pt`
- Run a 5-configuration ablation study automatically
- Save all metrics to `outputs/logs/`

### 4. Evaluate + generate XAI reports

```bash
python evaluate.py
```

Generates:
- Confusion matrix
- ROC curve
- Training curves
- Ablation bar chart
- Per-sample XAI reports (text attribution + GradCAM + modality weights)

### 5. Run the web demo

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Model Architecture

```
Text (RoBERTa)  ──┐
                   ├──► Cross-Modal Attention Fusion ──► Classifier ──► Real/Fake + Explanation
Image (CLIP-ViT) ──┤
                   │
Social (GAT) ──────┘
                   │
                   └──► Text-Image Inconsistency Score  (Paper 2 signal)
```

### Key design decisions

**Why RoBERTa over BERT?**
RoBERTa uses dynamic masking during pretraining and removes the next-sentence prediction objective, consistently outperforming BERT on downstream classification tasks.

**Why CLIP-ViT over ResNet?**
CLIP's joint vision-language pretraining means image features are already semantically aligned with text — critical for cross-modal inconsistency detection.

**Why GAT over GCN?**
Graph Attention Networks learn to weight neighbor importance, which matters for propagation graphs where retweet source credibility varies.

**Why cross-modal attention instead of concatenation?**
Concatenation treats all modalities equally. Cross-modal attention lets the model learn WHICH modality to trust for each specific sample (a text-only fake news differs from an image-based one).

---

## XAI Methods

| Modality | Method | Library | Output |
|----------|--------|---------|--------|
| Text | Gradient × Embedding | Captum / custom | Per-token importance score |
| Image | GradCAM | Captum | Spatial heatmap over image |
| Fusion | Cross-modal attention | PyTorch | Per-modality contribution weights |
| Text-Image | CLIP cosine similarity | HuggingFace | Inconsistency score 0–1 |

---

## Paper 1: Full System

**Title idea:** "TriXFND: Cross-Modal Attention Fusion with Tri-Stream Explainable Fake News Detection"

**Unique contributions:**
1. Three-stream architecture with text + image + social propagation
2. Cross-modal attention fusion (learned per-sample, not fixed concatenation)
3. Auxiliary text-image inconsistency loss during training
4. Quantitative XAI evaluation using AOPC (Area Over the Perturbation Curve)
5. Ablation study proving each modality contributes

**Target venues:** IEEE Access, Expert Systems with Applications, Information Processing & Management

---

## Paper 2: Inconsistency Detection

**Title idea:** "Out-of-Context Image Detection: CLIP-Based Cross-Modal Inconsistency Scoring for Fake News with XAI"

**Unique contribution:**
Fake news frequently uses *real images* with *false captions* (out-of-context manipulation). This module uses CLIP's joint embedding space to compute a semantic inconsistency score, visualizes which image patches are most contradictory to the text, and uses this as an interpretable signal.

**Run Paper 2 module:**
```python
from models.inconsistency_detector import CLIPInconsistencyDetector
from transformers import CLIPProcessor

detector = CLIPInconsistencyDetector()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

result = detector.explain_inconsistency(
    text="Flood victims rescued in Mumbai",
    image_tensor=your_image_tensor,
    processor=processor,
)
print(result["verdict"])
print(f"Inconsistency: {result['inconsistency_score']:.4f}")
```

---

## Configuration

All settings live in `config.py`. Key parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `text_encoder` | `roberta-base` | HuggingFace model name |
| `image_encoder` | `openai/clip-vit-base-patch32` | CLIP variant |
| `fusion_num_heads` | `8` | Attention heads in fusion |
| `batch_size` | `16` | Reduce to 8 if OOM |
| `num_epochs` | `20` | With early stopping |
| `use_social` | `True` | Toggle social stream |

---

## Citation

If you use this code in your research paper, cite as:

```bibtex
@misc{yourname2025trixtnd,
  title={TriXFND: Cross-Modal Attention Fusion with Tri-Stream Explainable Fake News Detection},
  author={Your Name},
  year={2025},
  institution={Your Institution},
}
```

---

## License

MIT License. Free to use for academic research.
