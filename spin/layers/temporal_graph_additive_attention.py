from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from tsl.nn.layers.norm import LayerNorm

from .additive_attention import TemporalAdditiveAttention


class TemporalGraphAdditiveAttention(MessagePassing):
    def __init__(self, input_size: Union[int, Tuple[int, int]],
                 output_size: int,
                 msg_size: Optional[int] = None,
                 msg_layers: int = 1,
                 root_weight: bool = True,
                 reweight: Optional[str] = None,
                 temporal_self_attention: bool = True,
                 mask_temporal: bool = True,
                 mask_spatial: bool = True,
                 norm: bool = True,
                 dropout: float = 0.,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(TemporalGraphAdditiveAttention, self).__init__(node_dim=-2,
                                                             **kwargs)

        # store dimensions
        if isinstance(input_size, int):
            self.src_size = self.tgt_size = input_size
        else:
            self.src_size, self.tgt_size = input_size
        self.output_size = output_size
        self.msg_size = msg_size or self.output_size

        self.mask_temporal = mask_temporal
        self.mask_spatial = mask_spatial

        self.root_weight = root_weight
        self.dropout = dropout

        if temporal_self_attention:
            self.self_attention = TemporalAdditiveAttention(
                input_size=input_size,
                output_size=output_size,
                msg_size=msg_size,
                msg_layers=msg_layers,
                reweight=reweight,
                dropout=dropout,
                root_weight=False,
                norm=False
            )
        else:
            self.register_parameter('self_attention', None)

        self.cross_attention = TemporalAdditiveAttention(input_size=input_size,
                                                         output_size=output_size,
                                                         msg_size=msg_size,
                                                         msg_layers=msg_layers,
                                                         reweight=reweight,
                                                         dropout=dropout,
                                                         root_weight=False,
                                                         norm=False)

        if self.root_weight:
            self.lin_skip = Linear(self.tgt_size, self.output_size,
                                   bias_initializer='zeros')
        else:
            self.register_parameter('lin_skip', None)

        if norm:
            self.norm = LayerNorm(output_size)
        else:
            self.register_parameter('norm', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.cross_attention.reset_parameters()
        if self.self_attention is not None:
            self.self_attention.reset_parameters()
        if self.lin_skip is not None:
            self.lin_skip.reset_parameters()
        if self.norm is not None:
            self.norm.reset_parameters()

    def forward(self, x: OptPairTensor,
                edge_index: Adj, edge_weight: OptTensor = None,
                mask: OptTensor = None):
        # inputs: [batch, steps, nodes, channels]
        if isinstance(x, Tensor):
            x_src = x_tgt = x
        else:
            x_src, x_tgt = x
            x_tgt = x_tgt if x_tgt is not None else x_src

        n_src, n_tgt = x_src.size(-2), x_tgt.size(-2)

        # propagate query, key and value
        out = self.propagate(x=(x_src, x_tgt),
                             edge_index=edge_index, edge_weight=edge_weight,
                             mask=mask if self.mask_spatial else None,
                             size=(n_src, n_tgt))

        if self.self_attention is not None:
            s, l = x_src.size(1), x_tgt.size(1)
            if s == l:
                attn_mask = ~torch.eye(l, l, dtype=torch.bool,
                                       device=x_tgt.device)
            else:
                attn_mask = None
            temp = self.self_attention(x=(x_src, x_tgt),
                                       mask=mask if self.mask_temporal else None,
                                       temporal_mask=attn_mask)
            out = out + temp

        # skip connection
        if self.root_weight:
            out = out + self.lin_skip(x_tgt)

        if self.norm is not None:
            out = self.norm(out)

        return out

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_weight: OptTensor, mask_j: OptTensor) -> Tensor:
        # [batch, steps, edges, channels]

        out = self.cross_attention((x_j, x_i), mask=mask_j)

        if edge_weight is not None:
            out = out * edge_weight.view(-1, 1)
        return out
