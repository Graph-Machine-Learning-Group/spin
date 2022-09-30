import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.W = nn.Parameter(torch.Tensor(input_size, input_size))
        self.b = nn.Parameter(torch.Tensor(input_size))

        m = 1 - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.shape[0])
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * self.m, self.b)
        return z_h


class TemporalDecay(nn.Module):
    def __init__(self, d_in, d_out, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.W = nn.Parameter(torch.Tensor(d_out, d_in))
        self.b = nn.Parameter(torch.Tensor(d_out))

        if self.diag:
            assert (d_in == d_out)
            m = torch.eye(d_in, d_in)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.shape[0])
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    @staticmethod
    def compute_delta(mask, freq=1):
        delta = torch.zeros_like(mask).float()
        one_step = torch.tensor(freq, dtype=delta.dtype, device=delta.device)
        for i in range(1, delta.shape[-2]):
            m = mask[..., i - 1, :]
            delta[..., i, :] = m * one_step + (1 - m) * torch.add(
                delta[..., i - 1, :], freq)
        return delta

    def forward(self, d):
        if self.diag:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class RITS(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size=64):
        super(RITS, self).__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        self.rnn_cell = nn.LSTMCell(2 * self.input_size, self.hidden_size)

        self.temp_decay_h = TemporalDecay(d_in=self.input_size,
                                          d_out=self.hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(d_in=self.input_size,
                                          d_out=self.input_size, diag=True)

        self.hist_reg = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg = FeatureRegression(self.input_size)

        self.weight_combine = nn.Linear(2 * self.input_size, self.input_size)

    def init_hidden_states(self, x):
        return torch.zeros((x.shape[0], self.hidden_size)).to(x.device)

    def forward(self, x, mask=None, delta=None):
        # x : [batch, steps, features]
        steps = x.shape[-2]

        if mask is None:
            mask = torch.ones_like(x, dtype=torch.uint8)
        if delta is None:
            delta = TemporalDecay.compute_delta(mask)

        # init rnn states
        h = self.init_hidden_states(x)
        c = self.init_hidden_states(x)

        imputation = []
        predictions = []
        for step in range(steps):
            d = delta[:, step, :]
            m = mask[:, step, :]
            x_s = x[:, step, :]

            gamma_h = self.temp_decay_h(d)

            # history prediction
            x_h = self.hist_reg(h)
            x_c = m * x_s + (1 - m) * x_h
            h = h * gamma_h

            # feature prediction
            z_h = self.feat_reg(x_c)

            # predictions combination
            gamma_x = self.temp_decay_x(d)
            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            alpha = torch.sigmoid(alpha)
            c_h = alpha * z_h + (1 - alpha) * x_h

            c_c = m * x_s + (1 - m) * c_h
            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))

            imputation.append(c_h)
            predictions.append(torch.stack((z_h, x_h), dim=0))

        # imputation -> [batch, steps, features]
        imputation = torch.stack(imputation, dim=-2)
        # predictions -> [predictions, batch, steps, features]
        predictions = torch.stack(predictions, dim=-2)

        return imputation, [*predictions]
