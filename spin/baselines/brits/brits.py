import torch
from einops import rearrange
from torch import nn
from tsl.nn.functional import reverse_tensor

from .layers import RITS


class BRITS(nn.Module):

    def __init__(self, input_size: int, n_nodes: int, hidden_size: int = 64):
        super().__init__()
        self.n_nodes = n_nodes
        self.rits_fwd = RITS(input_size * n_nodes, hidden_size)
        self.rits_bwd = RITS(input_size * n_nodes, hidden_size)

    def forward(self, x, mask=None):
        # tsl shape to original shape: [b s n c] -> [b s c]
        x = rearrange(x, 'b s n c -> b s (n c)')
        mask = rearrange(mask, 'b s n c -> b s (n c)')
        # forward
        imp_fwd, pred_fwd = self.rits_fwd(x, mask)
        # backward
        x_bwd = reverse_tensor(x, dim=1)
        mask_bwd = reverse_tensor(mask, dim=1) if mask is not None else None
        imp_bwd, pred_bwd = self.rits_bwd(x_bwd, mask_bwd)
        imp_bwd, pred_bwd = reverse_tensor(imp_bwd, dim=1), \
                            [reverse_tensor(pb, dim=1) for pb in pred_bwd]
        # stack into shape = [batch, directions, steps, features]
        imputation = (imp_fwd + imp_bwd) / 2
        predictions = [imp_fwd, imp_bwd] + pred_fwd + pred_bwd

        imputation = rearrange(imputation, 'b s (n c) -> b s n c',
                               n=self.n_nodes)
        predictions = [rearrange(pred, 'b s (n c) -> b s n c', n=self.n_nodes)
                       for pred in predictions]

        return imputation, predictions

    @staticmethod
    def consistency_loss(imp_fwd, imp_bwd):
        loss = 0.1 * torch.abs(imp_fwd - imp_bwd).mean()
        return loss

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--hidden-size', type=int, default=64)
        return parser
