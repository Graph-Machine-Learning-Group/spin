import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn import inits

from .layers import EncoderLayer, PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, d_time, d_feature,
                 d_model, d_inner, n_head, d_k, d_v, dropout,
                 **kwargs):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs['input_with_mask']
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs['param_sharing_strategy']
        self.MIT = kwargs['MIT']

        if self.param_sharing_strategy == 'between_group':
            # For between_group, only need to create 1 group and
            # repeat n_groups times while forwarding
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head,
                             d_k, d_v, dropout, dropout, **kwargs)
                for _ in range(n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers and
            # repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head,
                             d_k, d_v, dropout, dropout, **kwargs)
                for _ in range(n_groups)
            ])

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def impute(self, x, mask, **kwargs):
        # tsl shape to original shape: [b s n c=1] -> [b s c]
        is_bsnc = False
        if x.ndim == 4:
            is_bsnc = True
            x, mask = x.squeeze(-1), mask.squeeze(-1)

        # 1st DMSA ############################################################

        # Cat mask (eventually)
        if self.input_with_mask:
            input_X = torch.cat([x, mask], dim=2)
        else:
            input_X = x
        input_X = self.embedding(input_X)
        enc_output = self.dropout(self.position_enc(input_X))

        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)
        # replace non-missing part with original data
        imputed_data = mask * x + (1 - mask) * learned_presentation

        if is_bsnc:
            imputed_data.unsqueeze_(-1), learned_presentation.unsqueeze_(-1)
        return imputed_data, learned_presentation


class SAITS(nn.Module):
    def __init__(self, input_size: int, window_size: int, n_nodes: int,
                 d_model: int = 256,
                 d_inner: int = 128,
                 n_head: int = 4,
                 d_k: int = None,  # or 64
                 d_v: int = 64,
                 n_groups: int = 2,
                 n_group_inner_layers: int = 1,
                 param_sharing_strategy: str = 'inner_group',
                 dropout: float = 0.1,
                 input_with_mask: bool = True,
                 diagonal_attention_mask: bool = True,
                 trainable_mask_token: bool = False):
        super().__init__()
        self.n_nodes = n_nodes
        self.input_size = input_size
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = input_with_mask
        self.param_sharing_strategy = param_sharing_strategy

        d_in = in_features = input_size * n_nodes
        if self.input_with_mask:
            d_in = 2 * d_in
        d_k = d_k or (d_model // n_head)  # from the appendix

        if trainable_mask_token:
            self.mask_token = nn.Parameter(torch.Tensor(1, 1, in_features))
            inits.uniform(in_features, self.mask_token)
        else:
            self.register_buffer('mask_token', torch.zeros(1, 1, in_features))

        if self.param_sharing_strategy == 'between_group':
            # For between_group, only need to create 1 group and
            # repeat n_groups times while forwarding
            n_layers = n_group_inner_layers
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers and
            # repeat n_group_inner_layers times in each group while forwarding
            n_layers = n_groups

        self.layer_stack_for_first_block = nn.ModuleList([
            EncoderLayer(d_time=window_size,
                         d_feature=d_in,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head,
                         d_k=d_k,
                         d_v=d_v,
                         dropout=dropout,
                         attn_dropout=0,
                         diagonal_attention_mask=diagonal_attention_mask)
            for _ in range(n_layers)
        ])
        self.layer_stack_for_second_block = nn.ModuleList([
            EncoderLayer(d_time=window_size,
                         d_feature=d_in,
                         d_model=d_model,
                         d_inner=d_inner,
                         n_head=n_head,
                         d_k=d_k,
                         d_v=d_v,
                         dropout=dropout,
                         attn_dropout=0,
                         diagonal_attention_mask=diagonal_attention_mask)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(p=dropout)
        self.position_enc = PositionalEncoding(d_model, n_position=window_size)
        # for operation on time dim
        self.embedding_1 = nn.Linear(d_in, d_model)
        self.reduce_dim_z = nn.Linear(d_model, in_features)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(d_in, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, in_features)
        self.reduce_dim_gamma = nn.Linear(in_features, in_features)
        # for delta decay factor
        self.weight_combine = nn.Linear(in_features + window_size, in_features)

    def forward(self, x: Tensor, mask: Tensor, **kwargs):
        # tsl shape to original shape: [b s n c] -> [b s c]
        x = rearrange(x, 'b s n c -> b s (n c)')
        mask = rearrange(mask, 'b s n c -> b s (n c)')
        # whiten missing values
        x = torch.where(mask.bool(), x, self.mask_token)

        # 1st DMSA ############################################################

        # Cat mask (eventually)
        if self.input_with_mask:
            x_in = torch.cat([x, mask], dim=-1)
        else:
            x_in = x
        x_in = self.embedding_1(x_in)
        z = self.dropout(self.position_enc(x_in))

        # Encode (deeply?)
        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    z, _ = encoder_layer(z)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    z, _ = encoder_layer(z)

        x_tilde_1 = self.reduce_dim_z(z)
        x_hat_1 = mask * x + (1 - mask) * x_tilde_1

        # 2nd DMSA ############################################################

        # Cat mask (eventually)
        if self.input_with_mask:
            x_in = torch.cat([x_hat_1, mask], dim=-1)
        else:
            x_in = x_hat_1
        x_in = self.embedding_2(x_in)
        z = self.position_enc(x_in)

        # Encode
        if self.param_sharing_strategy == 'between_group':
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    z, attn_weights = encoder_layer(z)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    z, attn_weights = encoder_layer(z)

        x_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(z)))

        # Average attention heads
        if attn_weights.size(1) > 1:
            attn_weights = attn_weights.mean(dim=1)

        weights = torch.cat([mask, attn_weights], dim=2)
        weights = F.sigmoid(self.weight_combine(weights))
        # combine x_tilde_1 and X_tilde_2
        # x_tilde_3 = (1 - weights) * x_tilde_2 + weights * x_tilde_1
        x_hat = torch.lerp(x_tilde_2, x_tilde_1, weights)
        # replace non-missing part with original data
        x_tilde = [x_tilde_1, x_tilde_2, x_hat]

        # restore original shape
        x_hat = rearrange(x_hat, 'b s (n c) -> b s n c', n=self.n_nodes)
        x_tilde = [rearrange(tens, 'b s (n c) -> b s n c', n=self.n_nodes)
                   for tens in x_tilde]

        return x_hat, x_tilde

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--d-model', type=int, default=256, tunable=True,
                        options=[64, 128, 256, 512, 1024])
        parser.opt_list('--d-inner', type=int, default=128, tunable=True,
                        options=[128, 256, 512, 1024, 2048, 4096])
        parser.opt_list('--n-head', type=int, default=4, tunable=True,
                        options=[2, 4, 8])
        parser.add_argument('--d-k', type=int, default=None)
        parser.opt_list('--d-v', type=int, default=64, tunable=True,
                        options=[64, 128, 256, 512])
        parser.add_argument('--dropout', type=float, default=0.1)
        #
        parser.opt_list('--n-groups', type=int, default=2, tunable=True,
                        options=[1, 2, 4, 6, 8])
        parser.add_argument('--n-group-inner-layers', type=int, default=1)
        parser.add_argument('--param-sharing-strategy', type=str,
                            default='inner_group')
        #
        parser.add_argument('--input-with-mask', type=bool, default=True)
        parser.add_argument('--diagonal-attention-mask', type=bool,
                            default=True)
        parser.add_argument('--trainable-mask-token', type=bool,
                            default=False)
        return parser
