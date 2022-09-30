from typing import Optional

from torch import nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders import MLP
from tsl.nn.layers import PositionalEncoding


class PositionalEncoder(nn.Module):

    def __init__(self, in_channels, out_channels,
                 n_layers: int = 1,
                 n_nodes: Optional[int] = None):
        super(PositionalEncoder, self).__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.activation = nn.LeakyReLU()
        self.mlp = MLP(out_channels, out_channels, out_channels,
                       n_layers=n_layers, activation='relu')
        self.positional = PositionalEncoding(out_channels)
        if n_nodes is not None:
            self.node_emb = StaticGraphEmbedding(n_nodes, out_channels)
        else:
            self.register_parameter('node_emb', None)

    def forward(self, x, node_emb=None, node_index=None):
        if node_emb is None:
            node_emb = self.node_emb(token_index=node_index)
        # x: [b s c], node_emb: [n c] -> [b s n c]
        x = self.lin(x)
        x = self.activation(x.unsqueeze(-2) + node_emb)
        out = self.mlp(x)
        out = self.positional(out)
        return out
