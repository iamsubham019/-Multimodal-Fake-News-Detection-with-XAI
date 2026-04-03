import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from torch_geometric.data import Data as GraphData
from typing import Optional, Tuple, Dict, List
import networkx as nx

from config import CFG


class NewsDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        tokenizer: AutoTokenizer,
        image_transform,
        use_image: bool = True,
        use_social: bool = True,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.use_image = use_image
        self.use_social = use_social

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = self.records[idx]

        text_encoding = self.tokenizer(
            record["text"],
            max_length=CFG.data.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(record["label"], dtype=torch.long),
            "news_id": record.get("id", str(idx)),
        }

        if self.use_image:
            image = self._load_image(record.get("image_path", None))
            item["image"] = image

        if self.use_social:
            graph = self._build_graph(record.get("propagation", None))
            item["graph_x"] = graph.x
            item["graph_edge_index"] = graph.edge_index
            item["graph_num_nodes"] = graph.num_nodes

        return item

    def _load_image(self, image_path: Optional[str]) -> torch.Tensor:
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                return self.image_transform(img)
            except Exception:
                pass
        dummy = torch.zeros(3, CFG.data.image_size, CFG.data.image_size)
        return dummy

    def _build_graph(self, propagation: Optional[Dict]) -> GraphData:
        if propagation is None:
            x = torch.zeros(1, 16)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            return GraphData(x=x, edge_index=edge_index, num_nodes=1)

        edges = propagation.get("edges", [])
        node_features = propagation.get("node_features", [])
        num_nodes = min(len(node_features), CFG.data.max_graph_nodes)

        if num_nodes == 0:
            x = torch.zeros(1, 16)
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            return GraphData(x=x, edge_index=edge_index, num_nodes=1)

        x = torch.tensor(node_features[:num_nodes], dtype=torch.float)
        if x.size(1) < 16:
            pad = torch.zeros(num_nodes, 16 - x.size(1))
            x = torch.cat([x, pad], dim=1)

        valid_edges = [
            [e[0], e[1]] for e in edges
            if e[0] < num_nodes and e[1] < num_nodes
        ]
        if valid_edges:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        return GraphData(x=x, edge_index=edge_index, num_nodes=num_nodes)


def get_image_transform(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((CFG.data.image_size + 32, CFG.data.image_size + 32)),
            transforms.RandomCrop(CFG.data.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    return transforms.Compose([
        transforms.Resize((CFG.data.image_size, CFG.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def load_fakenewsnet(data_dir: str) -> List[Dict]:
    records = []
    for split_label, label_id in [("fake", 1), ("real", 0)]:
        split_dir = os.path.join(data_dir, split_label)
        if not os.path.exists(split_dir):
            continue
        for news_id in os.listdir(split_dir):
            news_dir = os.path.join(split_dir, news_id)
            if not os.path.isdir(news_dir):
                continue

            news_json = os.path.join(news_dir, "news content.json")
            if not os.path.exists(news_json):
                continue

            with open(news_json, "r", encoding="utf-8") as f:
                content = json.load(f)

            text = content.get("text", "") or content.get("title", "")
            if len(text.strip()) < 20:
                continue

            image_path = None
            images_dir = os.path.join(news_dir, "images")
            if os.path.exists(images_dir):
                imgs = [
                    os.path.join(images_dir, f)
                    for f in os.listdir(images_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if imgs:
                    image_path = imgs[0]

            propagation = _load_propagation(news_dir)

            records.append({
                "id": f"{split_label}_{news_id}",
                "text": text[:2000],
                "image_path": image_path,
                "propagation": propagation,
                "label": label_id,
            })

    random.shuffle(records)
    return records


def _load_propagation(news_dir: str) -> Optional[Dict]:
    tweet_dir = os.path.join(news_dir, "tweets")
    if not os.path.exists(tweet_dir):
        return None

    tweet_files = [f for f in os.listdir(tweet_dir) if f.endswith(".json")][:CFG.data.max_graph_nodes]
    if not tweet_files:
        return None

    nodes = []
    edges = []

    for i, tf in enumerate(tweet_files):
        try:
            with open(os.path.join(tweet_dir, tf), "r") as f:
                tweet = json.load(f)
            followers = min(tweet.get("user", {}).get("followers_count", 0) / 1e6, 1.0)
            friends = min(tweet.get("user", {}).get("friends_count", 0) / 1e4, 1.0)
            verified = float(tweet.get("user", {}).get("verified", False))
            retweets = min(tweet.get("retweet_count", 0) / 1e4, 1.0)
            favorites = min(tweet.get("favorite_count", 0) / 1e4, 1.0)
            node_feat = [followers, friends, verified, retweets, favorites] + [0.0] * 11
            nodes.append(node_feat)
            if i > 0:
                edges.append([0, i])
        except Exception:
            continue

    if not nodes:
        return None

    return {"node_features": nodes, "edges": edges}


def generate_synthetic_dataset(n_samples: int = 1000) -> List[Dict]:
    fake_phrases = [
        "BREAKING: Scientists discover miracle cure hidden by government",
        "EXCLUSIVE: Celebrity secret exposed in leaked documents",
        "SHOCKING: Mainstream media won't tell you this truth",
        "REVEALED: What they don't want you to know about vaccines",
        "WARNING: New study shows alarming hidden dangers",
    ]
    real_phrases = [
        "Federal Reserve raises interest rates by 25 basis points",
        "New climate report published by international researchers",
        "City council votes to approve infrastructure spending bill",
        "University researchers publish findings in peer-reviewed journal",
        "Stock market closes higher after strong earnings reports",
    ]

    records = []
    for i in range(n_samples):
        label = random.randint(0, 1)
        phrases = fake_phrases if label == 1 else real_phrases
        base = random.choice(phrases)
        filler = " Additional context provides more detail about this developing story." * random.randint(1, 5)
        text = base + filler

        node_features = [[random.random() * 0.5 if label == 1 else random.random() for _ in range(16)]
                         for _ in range(random.randint(3, 20))]
        edges = [[0, j] for j in range(1, len(node_features))]

        records.append({
            "id": f"synthetic_{i}",
            "text": text,
            "image_path": None,
            "propagation": {"node_features": node_features, "edges": edges},
            "label": label,
        })

    random.shuffle(records)
    return records


def split_records(records: List[Dict]) -> Tuple[List, List, List]:
    n = len(records)
    n_train = int(n * CFG.data.train_split)
    n_val = int(n * CFG.data.val_split)
    return records[:n_train], records[n_train:n_train + n_val], records[n_train + n_val:]


def collate_fn(batch: List[Dict]) -> Dict:
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == "graph_edge_index":
            collated[key] = [item[key] for item in batch]
        elif key == "graph_x":
            collated[key] = [item[key] for item in batch]
        elif key == "graph_num_nodes":
            collated[key] = [item[key] for item in batch]
        elif key == "news_id":
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]

    return collated


def get_dataloaders(
    records_train: List[Dict],
    records_val: List[Dict],
    records_test: List[Dict],
    tokenizer,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = NewsDataset(
        records_train, tokenizer, get_image_transform(train=True),
        use_image=CFG.train.use_image, use_social=CFG.train.use_social,
    )
    val_ds = NewsDataset(
        records_val, tokenizer, get_image_transform(train=False),
        use_image=CFG.train.use_image, use_social=CFG.train.use_social,
    )
    test_ds = NewsDataset(
        records_test, tokenizer, get_image_transform(train=False),
        use_image=CFG.train.use_image, use_social=CFG.train.use_social,
    )

    train_loader = DataLoader(
        train_ds, batch_size=CFG.train.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.train.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG.train.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader
def load_fakenews_hf(n_samples: int = None) -> List[Dict]:
    from datasets import load_dataset
    print("Loading GonzaloA/fake_news from HuggingFace...")
    ds = load_dataset("GonzaloA/fake_news")

    records = []
    for split in ["train", "validation", "test"]:
        for item in ds[split]:
            text = str(item.get("text", "") or "").strip()
            title = str(item.get("title", "") or "").strip()
            if len(text) < 20:
                text = title
            if len(text) < 20:
                continue
            full_text = (title + " " + text).strip()[:2000]
            node_features = [
                [random.random() * 0.5 for _ in range(16)]
                for _ in range(random.randint(5, 25))
            ]
            edges = [[0, j] for j in range(1, len(node_features))]
            records.append({
                "id": f"hf_{split}_{len(records)}",
                "text": full_text,
                "image_path": None,
                "propagation": {"node_features": node_features, "edges": edges},
                "label": 1 - int(item["label"]),
            })

    random.shuffle(records)
    if n_samples:
        records = records[:n_samples]
    print(f"Loaded {len(records)} records from HuggingFace dataset")
    return records

