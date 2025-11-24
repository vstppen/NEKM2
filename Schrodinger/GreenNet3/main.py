# %% [markdown]
# ### In this file, we consider the PDE system $\mathcal{L} \mathbf{u} = \mathbf{f}$ with zero Dirichlet boundary condition, where
# $$
# \mathcal{L}=\left[\begin{array}{cc}
# 1 & -\lambda \Delta \\
# \lambda \Delta & 1
# \end{array}\right],
# \quad
# \mathbf{u}=\left[\begin{array}{c} u_1 \\ u_2 \end{array}\right]
# \quad
# \mathbf{f}=\left[\begin{array}{c} f_1 \\ f_2 \end{array}\right]
# $$

# %%
import torch
import torch.nn as nn
import numpy as np
import os

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_info.name}")
    print(f"GPU memory: {gpu_info.total_memory / 1024**2:.2f} MB")

# %%
from scipy import sparse
from scipy.sparse.linalg import spsolve

# %%
N = 41
n = N - 1
h = 1 / n 
m = n - 1
m2 = m * m

# %%
def generateData1(lambda_):
    id_m = np.identity(m)
    d1= np.identity(m)
    d1[0][1] = 1
    d1[m - 1][m - 2] = 1
    for i in range(0, m):
        d1[i][i] = -2
        if (i >= 1) and (i <= m - 2):
            d1[i][i - 1] = 1
            d1[i][i + 1] = 1
    D = np.kron(id_m, d1) + np.kron(d1, id_m)
    D = D / (h ** 2)

    L = np.zeros((2 * m2, 2 * m2))
    L[0: m2, 0: m2] = np.identity(m2)
    L[0: m2, m2: 2 * m2] = - lambda_ * D
    L[m2: 2 * m2, 0: m2] = lambda_ * D
    L[m2: 2 * m2, m2: 2 * m2] = np.identity(m2)

    L_sparse = sparse.csr_matrix(L)

    f = np.random.rand(5000, 2 * m2)
    u = np.zeros_like(f)
    for i in range(5000):
        u[i, :] = spsolve(L_sparse, f[i, :])

    lambda_np = lambda_ + np.zeros_like(f[:, 0].reshape((-1, 1)))

    return lambda_np, f, u

def generateData2(lambda_values):
    lambda_list = []
    f_list = []
    u_list = []

    for lambda_ in lambda_values:
        lambda_np, f, u = generateData1(lambda_)
        lambda_list.append(lambda_np)
        f_list.append(f)
        u_list.append(u)
    
    return np.concatenate(lambda_list, axis=0), np.concatenate(f_list, axis=0), np.concatenate(u_list, axis=0)

# %%
lambda__ = np.linspace(0.05, 0.1, 11)
lambda_values, f, u = generateData2(lambda__)
lambda_values, f, u = torch.tensor(lambda_values, dtype=torch.float32), torch.tensor(f, dtype=torch.float32), torch.tensor(u, dtype=torch.float32)
lambda_values.shape, f.shape, u.shape

# %%
import torch.nn.functional as F

class GreenFun(nn.Module):
    def __init__(self, N):     
        super(GreenFun, self).__init__()
        self.N = N
        self.lambda_layer = nn.Sequential(nn.Linear(1, N // 4), nn.ReLU(), nn.Linear(N // 4, N // 4), nn.ReLU(), nn.Linear(N // 4, N // 4), nn.ReLU(), nn.Linear(N // 4, N // 4), nn.ReLU(), nn.Linear(N // 4, N // 4))
        self.G_layer1 = nn.Sequential(nn.Linear(N, N // 4, bias = False))
        self.G_layer2 = nn.Sequential(nn.Linear(N // 4, N, bias = False))

    def forward(self, lambda_values,  f):   
        return self.G_layer2(self.lambda_layer(lambda_values) * self.G_layer1(f))

# %%
from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = TensorDataset(lambda_values, f, u)

# 定义训练集和测试集的大小
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# 将数据集按比例分成训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 获取训练集的数据
train_lambda = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
train_f = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
train_u = torch.stack([train_dataset[i][2] for i in range(len(train_dataset))])

# 获取测试集的数据
test_lambda = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
test_f = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
test_u = torch.stack([test_dataset[i][2] for i in range(len(test_dataset))])

train_lambda = train_lambda.to(device)
train_f = train_f.to(device)
train_u = train_u.to(device)
test_lambda = test_lambda.to(device)
test_f = test_f.to(device)
test_u = test_u.to(device)

# %%
class EarlyStopping:
    def __init__(self, patience, verbose, delta, path):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.8f} --> {val_loss:.8f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)

# %%
net = GreenFun(2 * m2).to(device)
criterion = nn.MSELoss()

import torch.optim.lr_scheduler as lr_scheduler
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 4500, factor=0.7)

early_stopping = EarlyStopping(patience = 60000, verbose=False, delta=1e-8, path='net.pth')
num_epochs = 200000

for epoch in range(num_epochs):
    
    net.train()

    optimizer.zero_grad()
    
    output = net(train_lambda, train_f)
    
    loss = criterion(output, train_u)
    loss.backward()
    optimizer.step()

    net.eval()

    with torch.no_grad():
        output_test = net(test_lambda, test_f)
        loss_test = criterion(output_test, test_u)

        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {loss.item():.8f} Testing Loss: {loss_test.item():.8f} || lr: {scheduler.get_last_lr()}")

    # 调整学习率
    scheduler.step(loss_test)

    early_stopping(loss_test, net)
    
    if early_stopping.early_stop:
        print("Early stopping!")
        break

# %%
net.load_state_dict(torch.load("net.pth", map_location = device))

# %%
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# %%
def computeErrors(u_exact, u_pre, printOrNot):
    error = u_exact - u_pre
    l2_norm_abs = np.linalg.norm(error, ord=2) / np.sqrt(error.size)
    max_norm_abs = np.max(np.abs(error))
    l2_norm_rel = np.linalg.norm(error, ord=2) / np.linalg.norm(u_exact, ord=2)
    max_norm_rel = np.max(np.abs(error)) / np.max(np.abs(u_exact))  
    
    l2_norm_rel_percent = l2_norm_rel * 100
    max_norm_rel_percent = max_norm_rel * 100
    
    if printOrNot == True:
        print(f"Absolute L2 Norm Error: {l2_norm_abs:.8f}")
        print(f"Absolute Max Norm Error: {max_norm_abs:.8f}")
        print(f"Relative L2 Norm Error: {l2_norm_rel_percent:.6f}%")
        print(f"Relative Max Norm Error: {max_norm_rel_percent:.6f}%")

# %%
def validation(lambda_, net):
    u1_exact = np.sin(np.pi * X) * Y * (1 - Y)
    u2_exact = X * (1 - X) * np.sin(np.pi * Y)
    laplace_u1 = -2 * np.sin(np.pi * X) - np.pi ** 2 * u1_exact
    laplace_u2 = -2 * np.sin(np.pi * Y) - np.pi ** 2 * u2_exact
    f1 = u1_exact - lambda_ * laplace_u2
    f2 = lambda_ * laplace_u1 + u2_exact

    f = np.concatenate([f1[1:-1, 1:-1].flatten(), f2[1:-1, 1:-1].flatten()])

    lambda_torch = (lambda_ + torch.zeros(1, 1)).to(device)
    f_torch = torch.tensor(f, dtype=torch.float32).view(1, -1).to(device)

    u_numerical_torch = net(lambda_torch, f_torch)
    u_numerical = u_numerical_torch.cpu().detach().numpy().flatten()

    u1_numerical = np.zeros_like(u1_exact)
    u2_numerical = np.zeros_like(u2_exact)
    u1_numerical[1:-1, 1:-1] = u_numerical[0: m2].reshape((m, m))
    u2_numerical[1:-1, 1:-1] = u_numerical[m2: 2 * m2].reshape((m, m))

    print("numerical result for u_1:")
    computeErrors(u1_exact, u1_numerical, True)

    print("numerical result for u_2:")
    computeErrors(u2_exact, u2_numerical, True)

# %%
validation(0.1, net)

# %%
validation(1/15, net)

# %%
validation(0.05, net)


