from torch import nn
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders.mlp import MLP
from tsl.nn.blocks.encoders.transformer import SpatioTemporalTransformerLayer, \
    TransformerLayer
from tsl.nn.layers import PositionalEncoding
from tsl.utils.parser_utils import ArgParser, str_to_bool


class TransformerModel(nn.Module):
    r"""Spatiotemporal Transformer for multivariate time series imputation.

    Args:
        input_size (int): Input size.
        hidden_size (int): Dimension of the learned representations.
        output_size (int): Dimension of the output.
        ff_size (int): Units in the MLP after self attention.
        u_size (int): Dimension of the exogenous variables.
        n_heads (int, optional): Number of parallel attention heads.
        n_layers (int, optional): Number of layers.
        dropout (float, optional): Dropout probability.
        axis (str, optional): Dimension on which to apply attention to update
            the representations.
        activation (str, optional): Activation function.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 ff_size: int,
                 u_size: int,
                 n_heads: int = 1,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 condition_on_u: bool = True,
                 axis: str = 'both',
                 activation: str = 'elu'):
        super(TransformerModel, self).__init__()

        self.condition_on_u = condition_on_u
        if condition_on_u:
            self.u_enc = MLP(u_size, hidden_size, n_layers=2)
        self.h_enc = MLP(input_size, hidden_size, n_layers=2)

        self.mask_token = StaticGraphEmbedding(1, hidden_size)

        self.pe = PositionalEncoding(hidden_size)

        kwargs = dict(input_size=hidden_size,
                      hidden_size=hidden_size,
                      ff_size=ff_size,
                      n_heads=n_heads,
                      activation=activation,
                      causal=False,
                      dropout=dropout)

        if axis in ['steps', 'nodes']:
            transformer_layer = TransformerLayer
            kwargs['axis'] = axis
        elif axis == 'both':
            transformer_layer = SpatioTemporalTransformerLayer
        else:
            raise ValueError(f'"{axis}" is not a valid axis.')

        self.encoder = nn.ModuleList()
        self.readout = nn.ModuleList()
        for _ in range(n_layers):
            self.encoder.append(transformer_layer(**kwargs))
            self.readout.append(MLP(input_size=hidden_size,
                                    hidden_size=ff_size,
                                    output_size=output_size,
                                    n_layers=2,
                                    dropout=dropout))

    def forward(self, x, u, mask):
        # x: [batches steps nodes features]
        # u: [batches steps (nodes) features]
        x = x * mask

        h = self.h_enc(x)
        h = mask * h + (1 - mask) * self.mask_token()

        if self.condition_on_u:
            h = h + self.u_enc(u).unsqueeze(-2)

        h = self.pe(h)

        out = []
        for encoder, mlp in zip(self.encoder, self.readout):
            h = encoder(h)
            out.append(mlp(h))

        x_hat = out.pop(-1)
        return x_hat, out

    @staticmethod
    def add_model_specific_args(parser: ArgParser):
        parser.opt_list('--hidden-size', type=int, default=32, tunable=True,
                        options=[16, 32, 64, 128, 256])
        parser.opt_list('--ff-size', type=int, default=32, tunable=True,
                        options=[32, 64, 128, 256, 512, 1024])
        parser.opt_list('--n-layers', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--n-heads', type=int, default=1, tunable=True,
                        options=[1, 2, 3])
        parser.opt_list('--dropout', type=float, default=0., tunable=True,
                        options=[0., 0.1, 0.25, 0.5])
        parser.add_argument('--condition-on-u', type=str_to_bool, nargs='?',
                            const=True, default=True)
        parser.opt_list('--axis', type=str, default='both', tunable=True,
                        options=['steps', 'both'])
        return parser
