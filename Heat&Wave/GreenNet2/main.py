# %%
import torch
import torch.nn as nn
import numpy as np
import os

# %%
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu_info.name}")
    print(f"GPU memory: {gpu_info.total_memory / 1024**2:.2f} MB")

# %%
import torch.nn.functional as F

class GreenFun(nn.Module):
    def __init__(self, N, N_quadrature):     
        super(GreenFun, self).__init__()
        self.N = N
        self.N_quad = N_quadrature
        self.tau_layer = nn.Sequential(nn.Linear(1, (N_quadrature - 2) ** 2), nn.ReLU(), nn.Linear((N_quadrature - 2) ** 2, (N_quadrature - 2) ** 2), nn.ReLU(), nn.Linear((N_quadrature - 2) ** 2, (N_quadrature - 2) ** 2), nn.ReLU(), nn.Linear((N_quadrature - 2) ** 2, (N_quadrature - 2) ** 2))
        self.G_layer = nn.Sequential(nn.Linear(2, (N_quadrature - 2) ** 2), nn.ReLU(), nn.Linear((N_quadrature - 2) ** 2, (N_quadrature - 2) ** 2), nn.ReLU(), nn.Linear((N_quadrature - 2) ** 2, (N_quadrature - 2) ** 2), nn.ReLU(), nn.Linear((N_quadrature - 2) ** 2, (N_quadrature - 2) ** 2))

    def forward(self, f, x, tau):    # f: (batch_size, N_quad, N_quad), x: (N*N, 2), tau: (batch_size, 1)
        f = f[:, 1:-1, 1:-1].contiguous().view(-1, (self.N_quad - 2) ** 2)
        T = self.tau_layer(torch.sqrt(tau))
        G = self.G_layer(x.reshape(self.N, self.N, 2)[1:-1, 1:-1, :].reshape(-1, 2))     # G is (N, N_quad) with G(i, j) = G((x_i, y_i); (x_quad_j, y_quad_j))
        output = torch.matmul(f * T, G.t()) 
        output = output / self.N_quad 
        # print(output.shape)
        output = output.view(-1, self.N - 2, self.N - 2)
        return F.pad(output, (1, 1, 1, 1), mode='constant', value=0)


# %%
class MyNet(nn.Module):
    def __init__(self, N, N_quadrature):     
        super(MyNet, self).__init__()
        self.N = N
        self.N_quad = N_quadrature
        self.N2_ = (N - 2) ** 2
        self.N_quad2_ = (N - 2) ** 2
        self.f_layer1 = nn.Sequential(nn.Linear(self.N_quad2_, self.N2_), nn.ReLU(), nn.Linear(self.N2_, self.N2_), nn.ReLU(), nn.Linear(self.N2_, self.N2_))
        self.f_layer2 = nn.Sequential(nn.Linear(self.N_quad2_, self.N2_), nn.ReLU(), nn.Linear(self.N2_, self.N2_), nn.ReLU(), nn.Linear(self.N2_, self.N2_))

    def forward(self, f, tau):   # f: (batch_size, N_quad, N_quad), tau: (batch_size, 1)
        f = f[:, 1:-1, 1:-1].contiguous().view(-1, (self.N_quad - 2) ** 2)
        output = f + tau * self.f_layer1(f) + tau * tau * self.f_layer2(f)     
        output = output.view(-1, self.N - 2, self.N - 2)
        return F.pad(output, (1, 1, 1, 1), mode='constant', value=0) 

# %% [markdown]
# ### Modified Helmholtz equation
# $\begin{cases} u - \tau \Delta u  = f, & x \in \Omega \\ u(x) = 0, & x \in \partial \Omega \end{cases}$
# with $\Omega = [0, 1]^2$.
# 
# $\tau \in [1/256, 1/64]$

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random

def generate_gaussian_source_terms(num_samples, nx, ny, sigma):
    source_terms = []
    for _ in range(num_samples):
        f = np.random.randn(nx, ny)
        f = gaussian_filter(f, sigma=sigma)
        source_terms.append(f)
    return source_terms


def generate_functional_source_terms(num_samples, nx, ny):
    source_terms = []
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    for _ in range(num_samples // 4):
        a = np.random.uniform(0.5, 4.5)
        b = np.random.uniform(0.5, 4.5)
        c = np.random.uniform(0.5, 4.5)
        f = a * np.sin(b * np.pi * X) * np.sin(c * np.pi * Y)
        source_terms.append(f)

    for _ in range(num_samples // 4):
        a = np.random.uniform(0.5, 4.5)
        b = np.random.uniform(0.5, 4.5)
        c = np.random.uniform(0.5, 4.5)
        f = a * np.cos(b * np.pi * X) * np.cos(c * np.pi * Y)
        source_terms.append(f)

    for _ in range(num_samples // 4):
        a = np.random.uniform(0.5, 4.5)
        b = np.random.uniform(0.5, 4.5)
        c = np.random.uniform(0.5, 4.5)
        f = a * np.sin(b * np.pi * X) * np.cos(c * np.pi * Y)
        source_terms.append(f)

    for _ in range(num_samples - num_samples // 4 - num_samples // 4 - num_samples // 4):
        a = np.random.uniform(0.5, 4.5)
        b = np.random.uniform(0.5, 4.5)
        c = np.random.uniform(0.5, 4.5)
        f = a * np.cos(b * np.pi * X) * np.sin(c * np.pi * Y)
        source_terms.append(f)

    return source_terms

from scipy.fft import dstn, idstn
def solve_by_FFT2(tau, f, nx, ny, dx, dy):
    u = np.zeros((nx,ny))
    lam = np.zeros((nx-2,ny-2))
    for i in range(nx-2):
        for j in range(ny-2):
            lam[i,j] = 2 * (np.cos((i+1)*np.pi*dx)+np.cos((j+1)*np.pi*dy)-2) / (dx*dy)
    
    u[1:-1, 1:-1] = idstn(dstn(f[1:-1,1:-1],type=1)/(1 - tau * lam),type=1)
    
    return 1, u


def select(u, nx, ny, m):       # get low resolution data from high resolution data
    result = np.zeros((len(u), nx, ny))
    for l in range(len(u)):
        for i in range(nx):
            for j in range(ny):
                result[l, i, j] = u[l, m * i, m * j]

    return result


def generate_poisson_data1(tau, n, nx, ny):
    m = 2       # for getting higher resolution data
    dx, dy = 1 / (nx - 1), 1 / (ny - 1)
    nnx = m * (nx - 1) + 1
    nny = m * (ny - 1) + 1
    ddx, ddy = 1 / (nnx - 1), 1 / (nny - 1)
    
    source_terms1 = generate_gaussian_source_terms(n // 10, nx, ny, 1)
    source_terms2 = generate_gaussian_source_terms(n // 10, nx, ny, 2)
    source_terms3 = generate_gaussian_source_terms(n // 10, nx, ny, 3)
    source_terms4 = generate_gaussian_source_terms(n // 10, nx, ny, 4)
    source_terms5 = generate_gaussian_source_terms(n // 10, nx, ny, 5)
    source_terms6 = generate_functional_source_terms(n - 5 * (n // 10), nnx, nny)
    
    solutions1 = []
    solutions2 = []

    count = 0
    
    for f in source_terms1:
        success, u = solve_by_FFT2(tau, f, nx, ny, dx, dy)
        if success == 1:
            solutions1.append(u)

        count += 1
        if (count % 100 == 0):
            print("finish", count, "of", n)

    for f in source_terms2:
        success, u = solve_by_FFT2(tau, f, nx, ny, dx, dy)
        if success == 1:
            solutions1.append(u)

        count += 1
        if (count % 100 == 0):
            print("finish", count, "of", n)

    for f in source_terms3:
        success, u = solve_by_FFT2(tau, f, nx, ny, dx, dy)
        if success == 1:
            solutions1.append(u)

        count += 1
        if (count % 100 == 0):
            print("finish", count, "of", n)

    for f in source_terms4:
        success, u = solve_by_FFT2(tau, f, nx, ny, dx, dy)
        if success == 1:
            solutions1.append(u)

        count += 1
        if (count % 100 == 0):
            print("finish", count, "of", n)

    for f in source_terms5:
        success, u = solve_by_FFT2(tau, f, nx, ny, dx, dy)
        if success == 1:
            solutions1.append(u)

        count += 1
        if (count % 100 == 0):
            print("finish", count, "of", n)

    for f in source_terms6:
        success, u = solve_by_FFT2(tau, f, nnx, nny, ddx, ddy)
        if success == 1:
            solutions2.append(u)

        count += 1
        if (count % 100 == 0):
            print("finish", count, "of", n)

    a = np.array(source_terms1 + source_terms2 + source_terms3 + source_terms4 + source_terms5)
    b = select(np.array(source_terms6), nx, ny, m)
    c = np.array(solutions1)
    d = select(np.array(solutions2), nx, ny, m)
    
    return np.stack([a, b], axis=0), np.stack([c, d], axis=0)


def generate_poisson_data2(tau, n, nx, ny):
    f = []
    u = []
    ttau = []

    count = 1
    for t in tau:
        temp_tau = np.zeros((n, 1)) + t
        temp_f, temp_u = generate_poisson_data1(t, n, nx, ny)
        print("Finish", count, "of", len(tau), "tau's")
        f.append(temp_f)
        u.append(temp_u)
        ttau.append(temp_tau)

        count += 1

    return np.concatenate(f, axis=0), np.concatenate(u, axis=0), np.concatenate(ttau, axis=0)


# %%
N = 41
# tau = np.linspace(1/128, 1/32, 21)
# tau = np.linspace(0.75/128, 4.25/128, 15)
tau = np.linspace(1/8, 1/4, 16)
tau = tau ** 2 / 2
tau

# %%
if os.path.exists("f_np.npy") and os.path.exists("u_np.npy") and os.path.exists("tau_np.npy"):
    f_np, u_np, tau_np = np.load('f_np.npy'), np.load('u_np.npy'), np.load('tau_np.npy')
else:
    f_np, u_np, tau_np = generate_poisson_data2(tau, 2000, N, N)
    np.save('f_np.npy', f_np)
    np.save('u_np.npy', u_np)
    np.save('tau_np.npy', tau_np)

f, u, tau = torch.tensor(f_np, dtype=torch.float32).view(-1, N, N), torch.tensor(u_np, dtype=torch.float32).view(-1, N, N), torch.tensor(tau_np, dtype=torch.float32)

# %%
f.shape, u.shape, tau.shape

# %%
from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = TensorDataset(f, tau, u)

# 定义训练集和测试集的大小
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# 将数据集按比例分成训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 获取训练集的数据
train_f = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
train_tau = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
train_u = torch.stack([train_dataset[i][2] for i in range(len(train_dataset))])

# 获取测试集的数据
test_f = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
test_tau = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])
test_u = torch.stack([test_dataset[i][2] for i in range(len(test_dataset))])

train_f = train_f.to(device)
train_tau = train_tau.to(device)
train_u = train_u.to(device)
test_f = test_f.to(device)
test_tau = test_tau.to(device)
test_u = test_u.to(device)

# %%
x = torch.zeros(N*N, 2).to(device)
for i in range(N):
    for j in range(N):
        x[i*N+j, 0] = 1 * j / (N - 1)
        x[i*N+j, 1] = 1 * i / (N - 1)

# %%
f_validation1 = np.zeros((9, N, N))
U_validation1 = np.zeros((9, N, N))
tau_validation1 = np.array([1/128, 1/128, 1/128, 2/128, 2/128, 2/128, 4/128, 4/128, 4/128])

for k1 in range(3):
    for k2 in range(3):
        for i in range(N):
            for j in range(N):
                xx = 1 * j / (N - 1)
                yy = 1 * i / (N - 1)
                t = tau_validation1[k1 * 3 + k2]
                
                U_validation1[k1 * 3 + k2, i, j] = np.sin((k1 + 1) * np.pi * xx) * np.sin((k2 + 1) * np.pi * yy)
                f_validation1[k1 * 3 + k2, i, j] = - np.pi ** 2 * ((k1 + 1) * (k1 + 1) + (k2 + 1) * (k2 + 1)) * np.sin((k1 + 1) * np.pi * xx) * np.sin((k2 + 1) * np.pi * yy)
                f_validation1[k1 * 3 + k2, i, j] = U_validation1[k1 * 3 + k2, i, j] - t * f_validation1[k1 * 3 + k2, i, j]


U_validation1 = torch.tensor(U_validation1, dtype=torch.float32).to(device)
f_validation1 = torch.tensor(f_validation1, dtype=torch.float32).to(device)
tau_validation1 = torch.tensor(tau_validation1, dtype=torch.float32).view(-1, 1).to(device)

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
net = GreenFun(N, N).to(device)
# net = MyNet(N, N).to(device)
criterion = nn.MSELoss()

import torch.optim.lr_scheduler as lr_scheduler
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = 3000, factor=0.7, verbose=True)

early_stopping = EarlyStopping(patience = 50000, verbose=False, delta=1e-8, path='net.pth')
num_epochs = 500000

for epoch in range(num_epochs):
    
    net.train()

    optimizer.zero_grad()
    
    outputs = net(train_f, x, train_tau)
    # outputs = net(train_f, train_tau)
    
    # print(outputs.shape)
    loss = criterion(outputs, train_u)
    loss.backward()
    optimizer.step()

    net.eval()

    with torch.no_grad():
        outputs_test = net(test_f, x, test_tau)
        # outputs_test = net(test_f, test_tau)
        loss_test = criterion(outputs_test, test_u)

        if(epoch+1) % 100 == 0:
            outputs_validation1 = net(f_validation1, x, tau_validation1)
            # outputs_validation1 = net(f_validation1, tau_validation1)
            loss_validation1 = criterion(outputs_validation1, U_validation1)
            print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {loss.item():.8f} Testing Loss: {loss_test.item():.8f} Validation loss 1: {loss_validation1.item():.8f}")

    # 调整学习率
    scheduler.step(loss_test)

    early_stopping(loss_test, net)
    
    if early_stopping.early_stop:
        print("Early stopping!")
        break

# %%
net.load_state_dict(torch.load("net.pth", map_location = device))

tau_validation4 = np.array([0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019, 0.021, 0.023, 0.025, 0.027, 0.029, 0.031, 0.033, 0.035])
lll = len(tau_validation4)
f_validation4 = np.zeros((lll, N, N))
U_validation4 = np.zeros((lll, N, N))

for k in range(lll):
    for i in range(N):
        for j in range(N):
            xx = 1 * j / (N - 1)
            yy = 1 * i / (N - 1)

            t = tau_validation4[k]
        
            U_validation4[k, i, j] = xx * (1- xx) * yy * (1- yy) * np.exp(0.6 * xx + 0.8 * yy)
            f_validation4[k, i, j] = np.exp(0.6 * xx + 0.8 * yy) * (xx**2*yy**2 + 2.2*xx**2*yy + 1.4*xx*yy**2 + 0.4*xx**2 + 0.8*yy**2 - 4.6*xx*yy - 0.4*xx - 0.8*yy)
            f_validation4[k, i, j] = U_validation4[k, i, j] - t * f_validation4[k, i, j]
            

U_validation4 = torch.tensor(U_validation4, dtype=torch.float32).to(device)
f_validation4 = torch.tensor(f_validation4, dtype=torch.float32).to(device)
tau_validation4 = torch.tensor(tau_validation4, dtype=torch.float32).view(-1, 1).to(device)

outputs_validation4 = net(f_validation4, x, tau_validation4)
error = torch.abs(U_validation4 - outputs_validation4)

# 初始化存储范数的列表
norms = []

# 计算每一行的二范数和最大范数
for i in range(error.shape[0]):
    row = error[i, :]
    l2_norm = torch.norm(row, p=2).item() / np.sqrt(len(row))
    max_norm = torch.norm(row, p=float('inf')).item()
    l2_norm_relative = torch.norm(row, p=2).item() / torch.norm(U_validation4[i, :], p=2).item()
    max_norm_relative = torch.norm(row, p=float('inf')).item() / torch.norm(U_validation4[i, :], p=float('inf')).item()
    norms.append([round(tau_validation4[i].cpu().item(), 4), round(l2_norm, 8), round(max_norm, 8), round(l2_norm_relative, 8), round(max_norm_relative, 8)])

# 打印表格
flag = 0
print(f"{'tau':<10}{'L2 Norm':<15}{'Max Norm':<15}{'Relative L2 Norm':<20}{'Relative Max Norm'}")
for row in norms:
    if row[0] > 1/32 and flag == 0:
        print("--"*40)
        flag = 1
    print(f"{row[0]:<10}{row[1]:<15}{row[2]:<15}{row[3]:<20}{row[4]}")


