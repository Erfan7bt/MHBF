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
        m=self.m.detach().numpy()
        n=self.n.detach().numpy()
        wi=self.wi.detach().numpy()
        w=self.w.detach().numpy()
        packaged_vectors = np.zeros((2*self.rank+2, self.network_size))
        
        packaged_vectors[0:self.rank] = m.T
        packaged_vectors[self.rank:2*self.rank] = n.T
        packaged_vectors[2*self.rank] = wi.flatten()
        packaged_vectors[2*self.rank+1] = w.flatten()
        
        mean = np.mean(packaged_vectors, axis=1)
        cov_matrix = np.cov(packaged_vectors)
       
        return torch.tensor(mean), torch.tensor(cov_matrix)


class FittedRNN(nn.Module):

    def __init__(self, model):

        super(FittedRNN, self).__init__()

        self.network_size = model.network_size
        self.rank= model.rank
        mean, cov_mat = model.get_mean_cov()
        mean = mean.numpy()
        cov_mat = cov_mat.numpy()

        params = np.random.multivariate_normal(mean, cov_mat, size=self.network_size)
        self.m = torch.tensor(params[:,0:self.rank], dtype=torch.float)
        self.n = torch.tensor(params[:,self.rank:2*self.rank], dtype=torch.float)
        self.wi = torch.tensor(params[:,-2], dtype=torch.float)
        self.w = torch.tensor(params[:,-1], dtype=torch.float).unsqueeze(1)

        # Parameters for weight update formula
        self.tau = 100  # ms
        self.dt = 20  # ms

        # Activation function
        self.activation = nn.Tanh()

    def forward(self, u, visible_activity=False):

        # print(u)
        if len(u.shape) == 1:
            u = u.unsqueeze(0)

        input_len=u.size(1)
        batch_size=u.size(0)

        x = torch.zeros(batch_size, self.network_size)
        z = torch.zeros(u.shape)

        r = self.activation(x)

        if visible_activity:
            unit_activity = torch.zeros(batch_size, input_len+1, self.network_size)
            unit_activity[:,0,:] = x
        for i in range(input_len):
            delta_x = (
                -x
                + r.matmul(self.n).matmul(self.m.t()) / self.network_size
                + torch.outer(u[:,i], self.wi.squeeze())
           ) * (self.dt / self.tau)

            x = x + delta_x
            r = self.activation(x)
            if visible_activity:
                unit_activity[:,i+1,:] = x

            output = torch.matmul(r, self.w) / self.network_size
            z[:, i] = output.squeeze()

        if visible_activity:
            return z, unit_activity
        else:
            return z


class OneDimEquivalent(nn.Module):

    def __init__(self,model):

        super(OneDimEquivalent, self).__init__()
        
        self.rank= model.rank
        mean, cov_mat = model.get_mean_cov()
        cov_mat = cov_mat.detach().numpy()
        #[m, n, wi, w]

        self.sig_m = cov_mat[0,0]**0.5
        self.sig_I = cov_mat[2,2]**0.5
        self.sig_mn = cov_mat[0,1]
        self.sig_nI = cov_mat[1,2]
        self.sig_mw = cov_mat[0,3]
        self.sig_Iw = cov_mat[2,3]

        self.tau = 100  # ms
        self.dt = 20  # ms

        self.activation = np.tanh
        self.d_act = lambda x: 1 - np.tanh(x)**2

    def forward(self, u):

        k = 0
        v = 0
        z = 0

        u = u.detach().numpy().flatten()

        in_size = u.size

        k_hist = torch.zeros(in_size + 1)
        v_hist = torch.zeros(in_size + 1)
        z_hist = torch.zeros(in_size + 1)

        k_hist[0] = k
        v_hist[0] = v
        z_hist[0] = z

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
            sig_mw_hat = self.sig_mw * gauss_int
            # sig_Iw_hat = self.sig_Iw * gauss_int

            dk_dt = (-k + sig_mn_hat*k + sig_nI_hat*v) * (self.dt/self.tau)
            k += dk_dt

            dv_dt = (-v + in_val) * (self.dt / self.tau)
            v += dv_dt

            z= sig_mw_hat*k 
            # print(idx, k, delta, gauss_int)

            k_hist[idx+1] = k
            v_hist[idx+1] = v
            z_hist[idx+1] = z
        
        return torch.Tensor(z_hist)
    
class TwoDimEquivalent(nn.Module):
    def __init__(self, model):

        super(TwoDimEquivalent, self).__init__()

        self.rank = model.rank
        mean, cov_mat = model.get_mean_cov()

        cov_mat = cov_mat.detach().numpy()
        #[m, n, wi, w]

        self.sig_m1 = cov_mat[0, 0]**0.5
        self.sig_m2 = cov_mat[1, 1]**0.5
        self.sig_I = cov_mat[2, 2]**0.5
        self.sig_m1n1 = cov_mat[0, 2]
        self.sig_m2n2 = cov_mat[1, 3]
        self.sig_n1I = cov_mat[2, 4]
        self.sig_n2I = cov_mat[3, 4]
        self.sig_m1w = cov_mat[0, 5]
        self.sig_m2w = cov_mat[1, 5]
        self.sig_Iw = cov_mat[4, 5]


        self.tau = 100  # ms
        self.dt = 20  # ms

        self.activation = np.tanh
        self.d_act = lambda x: 1 - np.tanh(x)**2

    def forward(self, u):

        k1= 0
        k2 = 0
        v = 0
        z = 0

        u = u.detach().numpy().flatten()

        in_size = u.size

        k_hist = torch.zeros(in_size + 1, 2)
        v_hist = torch.zeros(in_size + 1)
        z_hist = torch.zeros(in_size + 1)

        k_hist[0,0] =k1
        k_hist[0,1] = k2 
        v_hist[0] = v
        z_hist[0] = z

        a = 5

        print("idx, k, delta, gauss_int")

        for idx, in_val in enumerate(u):

            delta = (
                (self.sig_m1**2)*(k1**2) +
                (self.sig_m2**2)*(k2**2)+ 
                (self.sig_I**2)*(in_val**2)
                )**0.5

            def gauss_f(z): return self.d_act(delta*z)*np.exp(-(z**2)/2)
            gauss_int = quadrature(gauss_f, -a, a)
            gauss_int = gauss_int[0]
            gauss_int /= 2*np.pi

            sig_m1n1_hat = self.sig_m1n1 * gauss_int
            sig_m2n2_hat = self.sig_m2n2 * gauss_int
            sig_n1I_hat = self.sig_n1I * gauss_int
            sig_n2I_hat = self.sig_n2I * gauss_int
            sig_m1w_hat = self.sig_m1w * gauss_int
            sig_m2w_hat = self.sig_m2w * gauss_int
            sig_Iw_hat = self.sig_Iw * gauss_int

            dk1_dt = (-k1 + sig_m1n1_hat*k1 + sig_n1I_hat*v) * (self.dt/self.tau)
            k1 += dk1_dt
            dk2_dt = (-k2 + sig_m2n2_hat*k2 + sig_n2I_hat*v) * (self.dt/self.tau)
            k2 += dk2_dt

            dv_dt = (-v + in_val) * (self.dt / self.tau)
            v += dv_dt

            z= sig_m1w_hat*k1 + sig_m2w_hat*k2 + sig_Iw_hat*v

            # print(idx, k, delta, gauss_int)

            k_hist[idx+1,0] = k1
            k_hist[idx+1,1] = k2
            v_hist[idx+1] = v
            z_hist[idx+1] = z

        return torch.Tensor(z_hist)
