import sys
import os
sys.path.append(os.path.abspath('tests/test_synthetic_training/synthetic'))
sys.path.append(os.path.abspath('tests/test_synthetic_training'))

from dynamics import (bounce_roll_spin_forward,
                            bounce_roll_velocity_forward, 
                            bounce_slide_spin_forward, 
                            bounce_slide_velocity_forward, 
                            compute_alpha)
import torch
import numpy as np
from typing import Tuple
import torch.nn as nn





def generate_bounce_data(N: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = 0.22
    ez = 0.79
    vx = np.random.rand(N) * 10.0 - 5.0
    vy = np.random.rand(N)* 10.0 - 5.0
    vz = np.random.rand(N) * 5.0

    wx = np.random.rand(N) * 30.0*np.pi*2 - 15.0*np.pi*2
    wy = np.random.rand(N) * 30.0*np.pi*2 - 15.0*np.pi*2
    wz = np.random.rand(N) * 30.0*np.pi*2 - 15.0*np.pi*2

    v_new_N = []
    w_new_N = []
    for i in range(N):
        v = np.array([vx[i], vy[i], vz[i]])
        w = np.array([wx[i], wy[i], wz[i]])
        alpha = compute_alpha(v,w,[mu,ez])
        if alpha < 0.4:
            v_new = bounce_slide_velocity_forward(v,w,[mu,ez]).flatten()
            w_new = bounce_slide_spin_forward(v,w,[mu,ez]).flatten()
        else:
            v_new = bounce_roll_velocity_forward(v,w,[mu,ez]).flatten()
            w_new = bounce_roll_spin_forward(v,w,[mu,ez]).flatten()

        v_new_N.append(v_new)
        w_new_N.append(w_new)

    v_new_N = np.array(v_new_N)
    w_new_N = np.array(w_new_N)

    v_N = np.concatenate([vx.reshape(-1,1), vy.reshape(-1,1), vz.reshape(-1,1)], axis=1)
    w_N = np.concatenate([wx.reshape(-1,1), wy.reshape(-1,1), wz.reshape(-1,1)], axis=1)
    return v_N, w_N, v_new_N, w_new_N


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.linear_v = nn.Linear(4, 128)
        self.linear_w = nn.Linear(4, 128)
        self.dec = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        self.relu = nn.ReLU()

    def forward(self, v, w):
        v = torch.cat([v, torch.ones_like(v[:, :1])], dim=1)
        w = torch.cat([w, torch.ones_like(w[:, :1])], dim=1)

        h1 = self.relu(self.linear_v(v))
        h2 = self.relu(self.linear_w(w))
        # h = torch.cat([h1, h2], dim=1)
        h = h1 * h2

        h = self.dec(h)
        return h[:, :3], h[:, 3:]

class TestModel2(nn.Module):
    def __init__(self):
        super(TestModel2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(6, 64),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, v, w):
        v = v / 5.0
        w = w / (15.0*np.pi*2)

        x = torch.cat([v, w], dim=1)
        x = self.layer1(x)
        x = x + self.layer2(x)*x
        x = self.dec(x)

        v_new = x[:, :3] * 5.0
        w_new = x[:, 3:] * (15.0*np.pi*2)
        return v_new, w_new
    

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, v_N, w_N, v_new_N, w_new_N):
        self.v_N = v_N
        self.w_N = w_N
        self.v_new_N = v_new_N
        self.w_new_N = w_new_N

    def __len__(self):
        return len(self.v_N)

    def __getitem__(self, idx):
        return self.v_N[idx], self.w_N[idx], self.v_new_N[idx], self.w_new_N[idx]

def test_bounce_training():
    v_N, w_N, v_new_N, w_new_N = generate_bounce_data()
    v_N = torch.tensor(v_N, dtype=torch.float32)
    w_N = torch.tensor(w_N, dtype=torch.float32)
    v_new_N = torch.tensor(v_new_N, dtype=torch.float32)
    w_new_N = torch.tensor(w_new_N, dtype=torch.float32)

    v_N = v_N.to('cuda')
    w_N = w_N.to('cuda')
    v_new_N = v_new_N.to('cuda')
    w_new_N = w_new_N.to('cuda')

    model = TestModel2()
    model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = MyDataset(v_N, w_N, v_new_N, w_new_N)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(600):
        for v_N, w_N, v_new_N, w_new_N in dataloader: 
            optimizer.zero_grad()
            v_new_pred, w_new_pred = model(v_N, w_N)
            loss = torch.mean((v_new_pred - v_new_N)**2)*1000.0 + torch.mean((w_new_pred - w_new_N)**2)
            loss.backward()
            optimizer.step()
            print(f'loss: {loss.item()}')

    print('-------------------')
    for i in range(len(v_new_N)):
        v_new_pred, w_new_pred = model(v_N[i:i+1], w_N[i:i+1])
        print('v_new_pred:', v_new_pred.detach())
        print('v_new_N:', v_new_N[i])
        print('w_new_pred:', w_new_pred.detach())
        print('w_new_N:', w_new_N[i])
        print('-------------------')

if __name__ == '__main__':
    test_bounce_training()