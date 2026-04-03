import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple

from config import CFG


class SocialGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        node_input_dim = 16

        self.input_proj = nn.Linear(node_input_dim, CFG.model.social_hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(CFG.model.social_gnn_layers):
            in_dim = CFG.model.social_hidden_dim
            out_dim = CFG.model.social_hidden_dim // CFG.model.social_gnn_heads
            self.gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=CFG.model.social_gnn_heads,
                    dropout=CFG.model.fusion_dropout,
                    concat=True,
                )
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(CFG.model.social_hidden_dim)
            for _ in range(CFG.model.social_gnn_layers)
        ])

        self.projection = nn.Sequential(
            nn.Linear(CFG.model.social_hidden_dim * 2, CFG.model.social_hidden_dim),
            nn.LayerNorm(CFG.model.social_hidden_dim),
            nn.GELU(),
            nn.Dropout(CFG.model.fusion_dropout),
            nn.Linear(CFG.model.social_hidden_dim, CFG.model.fusion_hidden_dim),
        )

    def forward(
        self,
        graph_x: List[torch.Tensor],
        graph_edge_index: List[torch.Tensor],
        graph_num_nodes: List[int],
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        data_list = []
        for x, edge_index, n in zip(graph_x, graph_edge_index, graph_num_nodes):
            data_list.append(Data(
                x=x.to(device),
                edge_index=edge_index.to(device),
                num_nodes=int(n),
            ))

        batch = Batch.from_data_list(data_list)

        h = F.gelu(self.input_proj(batch.x))

        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, batch.edge_index)
            h_new = F.dropout(h_new, p=CFG.model.fusion_dropout, training=self.training)
            h = norm(h_new + h if h.size(-1) == h_new.size(-1) else h_new)

        mean_pool = global_mean_pool(h, batch.batch)
        max_pool = global_max_pool(h, batch.batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)

        projected = self.projection(graph_embedding)

        return projected
