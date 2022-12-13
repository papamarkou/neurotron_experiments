# %% [markdown]
# # Settings

# %%
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset

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
        self.A = torch.empty(h, r, n, dtype=dtype)

        C = torch.randn(r, n, dtype=dtype)

        k = h // 2
        i = 0

        for factor in range(-k, k+1):
            if factor != 0:
                Z = self.M + factor * C
                self.A[i, :, :] = Z
                i += 1

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

    def train(self, train_loader, stepsize, loss, test_loader=None, log_step=200):
        train_losses, test_losses = [], []

        for train_batch_idx, (train_data, train_targets) in enumerate(train_loader):
            train_output = self.forward(train_data)

            self.update_parameters(train_data, train_output, train_targets, stepsize)

            if (train_batch_idx % log_step == 0):
                train_losses.append(loss(train_targets, train_output))

                if (test_loader is not None):
                    test_loss = 0.

                    for test_batch_idx, (test_data, test_targets) in enumerate(test_loader):
                        test_output = self.forward(test_data)

                        test_loss += loss(test_targets, test_output)

                    test_loss /= len(test_loader)

                    test_losses.append(test_loss)

        if (test_loader is not None):
            test_losses = torch.stack(test_losses)

        return torch.stack(train_losses), test_losses

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
# # Standard normal example

# %% [markdown]
# ## Prepare the data

# %%
num_samples = 125000
num_features = 100

sampling_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(num_features, dtype=torch.float32), torch.eye(num_features, dtype=torch.float32)
)

normal_data = sampling_distribution.sample([num_samples])

normal_targets = torch.stack([sampling_distribution.log_prob(normal_data[i, :]).exp() for i in range(num_samples)], dim=0)

# print(normal_data.shape, normal_targets.shape)


# %%
x_train, x_test, y_train, y_test = train_test_split(normal_data, normal_targets, test_size=0.2)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
beta = 0.5
theta = 0.125

train_dataset = PoisonedDataset(x_train, y_train, beta=beta, theta=theta)
# test_dataset = PoisonedDataset(x_test, y_test, beta=beta, theta=theta)

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# %% [markdown]
# ## Instantiate NeuroTron class

# %%
neurotron = NeuroTron(n=num_features, r=25, h=10, dtype=torch.float32)

# %% [markdown]
# ## Training

# %%
num_epochs = 2

train_losses = []

verbose_msg = 'Train epoch {:' + str(len(str(num_epochs))) + '} of {:' + str(len(str(num_epochs))) +'}'

for epoch in range(num_epochs):
    print(verbose_msg.format(epoch+1, num_epochs))

    train_losses_in_epoch, _ = neurotron.train(
        train_loader, stepsize=0.0001, loss=nn.MSELoss(reduction='mean'), test_loader=None, log_step=10
    )

    train_losses.append(train_losses_in_epoch)

train_losses = torch.stack(train_losses, dim=0)

# %% [markdown]
# ## Plotting training and test loss

# %%
plt.plot(torch.flatten(train_losses), label="Train loss")
plt.legend(loc='upper right')

# %% [markdown]
# # California housing example

# %% [markdown]
# ## Prepare the data

# %%
california_housing = fetch_california_housing(as_frame=True)

# california_housing.frame
# california_housing.data
# california_housing.target

# %%
x_train, x_test, y_train, y_test = train_test_split(california_housing.data, california_housing.target, test_size=0.25)

# %%
x_train = StandardScaler().fit_transform(x_train.to_numpy(dtype=np.float32))
x_test = StandardScaler().fit_transform(x_test.to_numpy(dtype=np.float32))

y_train = y_train.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.float32)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# %%
beta = 0.
theta = 0.01

train_dataset = PoisonedDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), beta=beta, theta=theta)
# test_dataset = PoisonedDataset(torch.from_numpy(x_test), torch.from_numpy(y_test), beta=beta, theta=theta)

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# %% [markdown]
# ## Instantiate NeuroTron class

# %%
neurotron = NeuroTron(n=8, r=6, h=10, dtype=torch.float32)

# %% [markdown]
# ## Training

# %%
num_epochs = 2

train_losses = []

verbose_msg = 'Train epoch {:' + str(len(str(num_epochs))) + '} of {:' + str(len(str(num_epochs))) +'}'

for epoch in range(num_epochs):
    print(verbose_msg.format(epoch+1, num_epochs))

    train_losses_in_epoch, _ = neurotron.train(
        train_loader, stepsize=0.00001, loss=nn.MSELoss(reduction='mean'), test_loader=None, log_step=10
    )

    train_losses.append(train_losses_in_epoch)

train_losses = torch.stack(train_losses, dim=0)

# %% [markdown]
# ## Plotting training and test loss

# %%
plt.plot(torch.flatten(train_losses), label="Train loss")
plt.legend(loc='upper right')

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


