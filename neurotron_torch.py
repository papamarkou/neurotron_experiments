# %%
import torch

import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset #, TensorDataset

# %%
class NeuroTron(nn.Module):
    def __init__(self, n, r, h, beta, theta, activation=nn.functional.relu, dtype=torch.float32):
        # n: number of input features
        # r: number of parameters
        # h: hidden layer width

        super().__init__()

        self.w = nn.Parameter(torch.empty(r, dtype=dtype), requires_grad=False)
        
        self.M = torch.randn(r, n, dtype=dtype)
        self.set_A(n, r, h, dtype=dtype)

        self.set_w()
        self.set_w_true(r, dtype=dtype)

        self.beta = beta
        self.theta = theta

        self.activation = activation

    def set_A(self, n, r, h, dtype=torch.float32):
        self.A = torch.empty(h-1, r, n, dtype=dtype)

        C = torch.randn(r, n, dtype=dtype)

        k = h // 2
        for i in range(-k, k+1):
            if i != 0:
                Z = self.M + i * C
                self.A[i, :, :] = Z

    def set_w(self):
        nn.init.constant_(self.w, 1.)
        # nn.init.uniform_(self.w)

    def set_w_true(self, r, dtype=torch.float32):
        self.w_true = torch.randn(r, dtype=dtype)

    def num_A(self):
        return self.A.shape[0]

    def f(self, x, w):
        postactivation = 0.
        for i in range(self.num_A()):
            preactivation = torch.matmul(torch.matmul(w, self.A[i, :, :]), x.t())
            postactivation += self.activation(preactivation)
        return postactivation / self.num_A()

    def attack(self, x):
        y_oracle = self.f(x, self.w_true)

        num_points = len(y_oracle)

        a = torch.bernoulli(torch.full([num_points], self.beta))
        xi = torch.distributions.uniform.Uniform(-self.theta, self.theta).sample([num_points])

        y_oracle += a * xi

        return y_oracle

    def forward(self, x):
        y_oracle = self.attack(x)

        output = self.f(x, self.w)

        return output, y_oracle

    def gradient(self, x, output, y_oracle):
        return torch.matmul(self.M, torch.matmul(output - y_oracle, x) / x.shape[1])

    def update_parameters(self, x, output, y_oracle, stepsize):
        self.w += stepsize * self.gradient(x, output, y_oracle)

    def weight_error(self):
        return torch.norm(self.w_true - self.w)

    def train(self, dataloader, stepsize, num_epochs):
        error = torch.empty(num_epochs, len(dataloader))

        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                output, y_oracle = self.forward(x)

                self.update_parameters(x, output, y_oracle, stepsize)

                error[epoch, batch_idx]  = self.weight_error()

        return error

# %%
neurotron = NeuroTron(n=8, r=20, h=10, beta=0.5, theta=1., dtype=torch.float64)

# %%
california_housing = fetch_california_housing(as_frame=True)

# %%
# california_housing.frame
# california_housing.data
# california_housing.target

# %%
x_train, x_test, y_train, y_test = train_test_split(california_housing.data, california_housing.target, test_size=0.25)

# %%
x_train = StandardScaler().fit_transform(x_train.to_numpy())
x_test = StandardScaler().fit_transform(x_test.to_numpy())

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
# train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
# train_dataset.tensors[0].shape, train_dataset.tensors[1].shape, len(train_dataset)

# %%
class XYDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'XYDataset'

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

# %%
train_dataset = XYDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# %%
error = neurotron.train(train_loader, 0.0001, 2)

# %%
plt.plot(torch.flatten(error))

# %%



