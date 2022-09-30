from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import LayerNorm
from torch_geometric.nn import inits
from torch_geometric.typing import OptTensor
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders import MLP

from ..layers import PositionalEncoder, HierarchicalTemporalGraphAttention


class SPINHierarchicalModel(nn.Module):

    def __init__(self, input_size: int,
                 h_size: int,
                 z_size: int,
                 n_nodes: int,
                 z_heads: int = 1,
                 u_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 n_layers: int = 5,
                 eta: int = 3,
                 message_layers: int = 1,
                 reweight: Optional[str] = 'softmax',
                 update_z_cross: bool = True,
                 norm: bool = True,
                 spatial_aggr: str = 'add'):
        super(SPINHierarchicalModel, self).__init__()

        u_size = u_size or input_size
        output_size = output_size or input_size
        self.h_size = h_size
        self.z_size = z_size

        self.n_nodes = n_nodes
        self.z_heads = z_heads
        self.n_layers = n_layers
        self.eta = eta

        self.v = StaticGraphEmbedding(n_nodes, h_size)
        self.lin_v = nn.Linear(h_size, z_size, bias=False)
        self.z = nn.Parameter(torch.Tensor(1, z_heads, n_nodes, z_size))
        inits.uniform(z_size, self.z)
        self.z_norm = LayerNorm(z_size)

        self.u_enc = PositionalEncoder(in_channels=u_size,
                                       out_channels=h_size,
                                       n_layers=2)

        self.h_enc = MLP(input_size, h_size, n_layers=2)
        self.h_norm = LayerNorm(h_size)

        self.v1 = StaticGraphEmbedding(n_nodes, h_size)
        self.m1 = StaticGraphEmbedding(n_nodes, h_size)

        self.v2 = StaticGraphEmbedding(n_nodes, h_size)
        self.m2 = StaticGraphEmbedding(n_nodes, h_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(input_size, h_size)
            encoder = HierarchicalTemporalGraphAttention(
                h_size=h_size, z_size=z_size,
                msg_size=h_size,
                msg_layers=message_layers,
                reweight=reweight,
                mask_temporal=True,
                mask_spatial=l < eta,
                update_z_cross=update_z_cross,
                norm=norm,
                root_weight=True,
                aggr=spatial_aggr,
                dropout=0.0
            )
            readout = MLP(h_size, z_size, output_size,
                          n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def forward(self, x: Tensor, u: Tensor, mask: Tensor,
                edge_index: Tensor, edge_weight: OptTensor = None,
                node_index: OptTensor = None, target_nodes: OptTensor = None):
        if target_nodes is None:
            target_nodes = slice(None)
        if node_index is None:
            node_index = slice(None)

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #
        # Condition also embeddings Z on V.                                   #

        v_nodes = self.v(token_index=node_index)
        z = self.z[..., node_index, :] + self.lin_v(v_nodes)

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index, node_emb=v_nodes)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q. Then, condition H on two
        # different embeddings to distinguish valid values from masked ones.
        h = torch.where(mask.bool(), h + self.v1(), q + self.m1())
        # Normalize features
        h, z = self.h_norm(h), self.z_norm(z)

        imputations = []

        for l in range(self.n_layers):
            if l == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                h = torch.where(mask.bool(), h + self.v2(), h + self.m2())
            # Skip connection from input x
            h = h + self.x_skip[l](x) * mask
            # Masked Temporal GAT for encoding representation
            h, z = self.encoder[l](h, z, edge_index, mask=mask)
            target_readout = self.readout[l](h[..., target_nodes, :])
            imputations.append(target_readout)

        x_hat = imputations.pop(-1)

        return x_hat, imputations

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--h-size', type=int, tunable=True, default=32,
                        options=[16, 32])
        parser.opt_list('--z-size', type=int, tunable=True, default=32,
                        options=[32, 64, 128])
        parser.opt_list('--z-heads', type=int, tunable=True, default=2,
                        options=[1, 2, 4, 6])
        parser.add_argument('--u-size', type=int, default=None)
        parser.add_argument('--output-size', type=int, default=None)
        parser.opt_list('--encoder-layers', type=int, tunable=True, default=2,
                        options=[1, 2, 3, 4])
        parser.opt_list('--decoder-layers', type=int, tunable=True, default=2,
                        options=[1, 2, 3, 4])
        parser.add_argument('--message-layers', type=int, default=1)
        parser.opt_list('--reweight', type=str, tunable=True, default='softmax',
                        options=[None, 'softmax'])
        parser.add_argument('--update-z-cross', type=bool, default=True)
        parser.opt_list('--norm', type=bool, default=True, tunable=True,
                        options=[True, False])
        parser.opt_list('--spatial-aggr', type=str, tunable=True,
                        default='add', options=['add', 'softmax'])
        return parser
