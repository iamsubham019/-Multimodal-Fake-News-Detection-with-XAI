import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    dataset_name: str = "FakeNewsNet"
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    max_text_length: int = 512
    image_size: int = 224
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    max_graph_nodes: int = 100


@dataclass
class ModelConfig:
    text_encoder: str = "roberta-base"
    text_hidden_dim: int = 768
    image_encoder: str = "openai/clip-vit-base-patch32"
    image_hidden_dim: int = 512
    social_hidden_dim: int = 256
    social_gnn_layers: int = 2
    social_gnn_heads: int = 4
    fusion_hidden_dim: int = 512
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1
    num_classes: int = 2
    classifier_hidden_dim: int = 256


@dataclass
class TrainConfig:
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    early_stopping_patience: int = 5
    save_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    device: str = "cuda"
    seed: int = 42
    use_social: bool = True
    use_image: bool = False
    use_text: bool = True


@dataclass
class XAIConfig:
    shap_background_samples: int = 50
    shap_explanation_samples: int = 100
    gradcam_target_layer: str = "visual_projection"
    top_k_tokens: int = 10
    output_dir: str = "outputs/explanations"


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8501
    model_checkpoint: str = "outputs/checkpoints/best_model.pt"
    theme_color: str = "#6366F1"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    app: AppConfig = field(default_factory=AppConfig)
    project_name: str = "MultimodalFakeNewsXAI"
    version: str = "1.0.0"


CFG = Config()


LABEL_MAP = {0: "Real", 1: "Fake"}
LABEL_COLORS = {0: "#22c55e", 1: "#ef4444"}
