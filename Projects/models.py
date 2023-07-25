# This is an attempt to organize the code and allow for a bit clearner
# notebooks, but who really knows.

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy.linalg as la

from scipy.integrate import quadrature

# Training function slightly modified from PyTorch quickstart tutorial


def train(dataloader, model, device, lr=5e-3, T=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    size = len(dataloader.dataset)
    optimizer.zero_grad()
    model.train(True)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        # print("pred shape: ", pred.shape)
        loss = loss_fn(pred[:, -T:], y[:, -T:])

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


class RNN(nn.Module):

    def __init__(self, network_size=128, rank=1):

        super(RNN, self).__init__()
        self.network_size = network_size
        self.rank = rank

        self.m = nn.Parameter(torch.Tensor(network_size, rank))
        self.n = nn.Parameter(torch.Tensor(network_size, rank))
        self.wi = torch.Tensor(network_size)
        self.w = torch.Tensor(network_size, 1)
        self.x0 = torch.Tensor(network_size, 1)

        # Parameters for weight update formula
        self.tau = 100  # ms
        self.dt = 20  # ms

        # Activation function
        self.activation = nn.Tanh()

        with torch.no_grad():
            self.m.normal_(std=1)
            self.n.normal_(std=1)
            self.w.normal_(std=4)
            self.x0.zero_()
            self.wi.normal_(std=1)

    def forward(self, u, visible_activity=False):

        # print(u)
        if len(u.shape) == 1:
            u = u.unsqueeze(0)

        input_len = u.size(1)
        batch_size = u.size(0)

        x = torch.zeros(batch_size, self.network_size)
        z = torch.zeros(u.shape)

        r = self.activation(x)

        if visible_activity:
            unit_activity = torch.zeros(
                batch_size, input_len+1, self.network_size)
            unit_activity[:, 0, :] = x

        for i in range(input_len):
            delta_x = (
                -x
                + r.matmul(self.n).matmul(self.m.t()) / self.network_size
                + torch.outer(u[:, i], self.wi.squeeze())
            ) * (self.dt / self.tau)

            x = x + delta_x
            r = self.activation(x)
            if visible_activity:
                unit_activity[:, i+1, :] = r

            output = torch.matmul(r, self.w) / self.network_size
            z[:, i] = output.squeeze()

        if visible_activity:
            return z, unit_activity
        else:
            return z

    def get_mean_cov(self):
        """
        Returns the mean and covariance matrix of internal
        parameters in the form [m, n, wi, w]
        """
        m = self.m.detach().numpy()
        n = self.n.detach().numpy()
        wi = self.wi.detach().numpy()
        w = self.w.detach().numpy()
        vectors = [m, n, wi, w]

        mean = np.mean(vectors, axis=1)
        cov_matrix = np.cov(vectors)

        return torch.tensor(mean), torch.tensor(cov_matrix)


class FittedRankOneRNN(nn.Module):

    def __init__(self, mean, cov_mat, network_size=128):

        super(FittedRankOneRNN, self).__init__()
        self.network_size = network_size
        self.rank = 1
        self.mean = mean
        self.cov_mat = cov_mat

        mean = mean.numpy()
        cov_mat = cov_mat.numpy()

        params = np.random.multivariate_normal(
            mean, cov_mat, size=network_size)
        self.m = torch.tensor(params[:, 2], dtype=torch.float)
        self.n = torch.tensor(params[:, 1], dtype=torch.float)
        self.wi = torch.tensor(params[:, 0], dtype=torch.float)
        self.w = torch.tensor(params[:, 3], dtype=torch.float)
        self.x0 = torch.zeros(network_size, dtype=torch.float)

        # Parameters for weight update formula
        self.tau = 100  # ms
        self.dt = 20  # ms

        # Activation function
        self.activation = nn.Tanh()

    def forward(self, u, visible_activity=False):
        input_len = u.size(1)
        batch_size = u.size(0)
        x = self.x0
        z = torch.zeros(u.shape)

        r = self.activation(x)

        if visible_activity:
            unit_activity = torch.zeros(
                batch_size, input_len+1, self.network_size)
            unit_activity[:, 0, :] = x

        # unit rank rnn weight matrix J=mn^T/n
        # J = torch.matmul(self.m[:,None], self.n[None,:]) / self.network_size

        for i in range(input_len):
            delta_x = (
                -x
                + r.matmul(self.n[:, None]).matmul(self.m[:,
                                                          None].t()) / self.network_size
                + torch.matmul(u[:, i, None], self.wi[None, :])
            ) * (self.dt / self.tau)

            x = x + delta_x
            r = self.activation(x)
            if visible_activity:
                unit_activity[:, i+1, :] = x

            output = torch.matmul(self.activation(
                x), self.w) / self.network_size
            z[:, i] = output

        if visible_activity:
            return z, unit_activity
        else:
            return z


class OneDimEquivalent(nn.Module):

    def __init__(self, network_size=128, sig_m=1, sig_n=1, sig_I=1, sig_mn=1.4, sig_nI=2.6):

        super(OneDimEquivalent, self).__init__()
        self.network_size = network_size

        self.sig_m = sig_m
        self.sig_I = sig_I
        self.sig_mn = sig_mn
        self.sig_nI = sig_nI

        self.tau = 100  # ms
        self.dt = 20  # ms

        self.activation = np.tanh
        self.d_act = lambda x: 1 - np.tanh(x)**2

    def forward(self, u, visible_activity=False):

        k = 0
        v = 0

        u = u.detach().numpy().flatten()

        in_size = u.size

        k_hist = torch.zeros(in_size + 1)
        v_hist = torch.zeros(in_size + 1)

        k_hist[0] = k
        v_hist[0] = v

        a = 5

        print("idx, k, delta, gauss_int")

        for idx, in_val in enumerate(u):

            delta = ((self.sig_m**2)*(k**2) + (self.sig_I**2)*(in_val**2))**0.5

            def gauss_f(z): return self.d_act(delta*z)*np.exp(-(z**2)/2)
            gauss_int = quadrature(gauss_f, -a, a)
            gauss_int = gauss_int[0]
            gauss_int /= 2*np.pi

            sig_mn_hat = self.sig_mn * gauss_int
            sig_nI_hat = self.sig_nI * gauss_int

            dk_dt = (-k + sig_mn_hat*k + sig_nI_hat*v) * (self.dt / self.tau)
            k += dk_dt

            dv_dt = (-v + in_val) * (self.dt / self.tau)
            v += dv_dt

            # print(idx, k, delta, gauss_int)

            k_hist[idx+1] = k
            v_hist[idx+1] = v

        to_ret = torch.Tensor(self.activation(k_hist))

        if visible_activity:
            return self.activation(k), to_ret
        return self.activation(k)
