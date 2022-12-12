# %% [markdown]
# # Settings

# %%
import torch

import matplotlib.pyplot as plt
import torch.nn as nn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset #, TensorDataset

# %% [markdown]
# # The NeuroTron class

# %%
class NeuroTron(nn.Module):
    def __init__(self, n, r, h, activation=nn.functional.relu, w_init='const', dtype=torch.float32):
        """
        Arguments:
            n: number of input features
            r: number of parameters
            h: hidden layer width
            activation: activation function
        """

        super().__init__()

        self.w = nn.Parameter(torch.empty(r, dtype=dtype), requires_grad=False)
        
        self.M = torch.randn(r, n, dtype=dtype)
        self.set_A(n, r, h, dtype=dtype)

        self.set_w(w_init)

        self.activation = activation

    def set_A(self, n, r, h, dtype=torch.float32):
        self.A = torch.empty(h-1, r, n, dtype=dtype)

        C = torch.randn(r, n, dtype=dtype)

        k = h // 2
        for i in range(-k, k+1):
            if i != 0:
                Z = self.M + i * C
                self.A[i, :, :] = Z

    def set_w(self, init):
        if init == 'const':
            nn.init.constant_(self.w, 1.)
        elif init == 'unif':
            nn.init.uniform_(self.w)

    def num_A(self):
        return self.A.shape[0]

    def forward(self, x):
        postactivation = 0.
        for i in range(self.num_A()):
            preactivation = torch.matmul(torch.matmul(self.w, self.A[i, :, :]), x.t())
            postactivation += self.activation(preactivation)
        return postactivation / self.num_A()

    def gradient(self, x, output, y):
        return torch.matmul(self.M, torch.matmul(y - output, x) / x.shape[0])

    def update_parameters(self, x, output, y, stepsize):
        self.w.data.add_(stepsize * self.gradient(x, output, y))

    def train(self, dataloader, stepsize, num_epochs, loss):
        error = torch.empty(num_epochs, len(dataloader))

        for epoch in range(num_epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                output = self.forward(x)

                self.update_parameters(x, output, y, stepsize)

                error[epoch, batch_idx]  = loss(y, output)

        return error

# %% [markdown]
# # The PoisonedDataset class

# %%
class PoisonedDataset(Dataset):
    def __init__(self, x, y, beta, theta):
        self.x = x
        self.y = y
        self.beta = beta
        self.theta = theta

    def attack(self, y):
        a = torch.bernoulli(torch.full_like(y, self.beta))
        xi = torch.distributions.uniform.Uniform(torch.full_like(y, -self.theta), torch.full_like(y, self.theta)).sample()

        return y + a * xi

    def __repr__(self):
        return f'PoisonedDataset'

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.attack(self.y[i])

# %% [markdown]
# # California housing example

# %% [markdown]
# ## Prepare the data

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
train_dataset = PoisonedDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), 0.00, 0.01)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# %% [markdown]
# ## Instantiate NeuroTron class

# %%
neurotron = NeuroTron(n=8, r=6, h=10, dtype=torch.float64)

# %% [markdown]
# ## Training

# %%
error = neurotron.train(train_loader, 0.00001, 3, nn.MSELoss(reduction='mean'))

# %% [markdown]
# ## Plotting training and test loss

# %%
plt.plot(torch.flatten(error))

# %% [markdown]
# ## Printing dimensions of various tensors

# %%
x, y = next(iter(train_loader))

# %%
x.shape, x.shape[0], x.shape[1], y.shape

# %%
neurotron.w.shape, neurotron.A.shape, neurotron.M.shape

# %%
output = neurotron.forward(x)

# %%
output.shape

# %%
neurotron.w.shape, neurotron.A[0, :, :].shape, x.t().shape, x.shape

# %%
torch.matmul(neurotron.w, neurotron.A[0, :, :]).shape

# %%
torch.matmul(torch.matmul(neurotron.w, neurotron.A[0, :, :]), x.t()).shape

# %%


