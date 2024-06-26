import torch 
import torch.nn as nn


device = torch.device("cuda:2")


def generate_data():
    N = 400
    vx = torch.rand(N, 1) * 10.0 -5.0
    vy = torch.rand(N, 1) * 10.0 -5.0
    vz = torch.rand(N, 1) * 10.0 -5.0
    wx = torch.rand(N, 1) * 30.0 * torch.pi*2 - 15.0 * torch.pi * 2
    wy = torch.rand(N, 1) * 30.0 * torch.pi*2 - 15.0 * torch.pi * 2
    wz = torch.rand(N, 1) * 30.0 * torch.pi*2 - 15.0 * torch.pi * 2

    return torch.cat([vx, vy, vz, wx, wy, wz], dim=1)

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = generate_data().to(device)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    @classmethod
    def get_loader(cls, batch_size=16, shuffle=True):
        dataset = cls()
        # split dataset into train and test
        train_data, test_data = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        return train_loader, test_loader

def compute_gt(v,w, dt):
    cd = 0.1123
    cm = 0.015
    g = torch.tensor([0.0, 0.0, -9.81], device=v.device)
    acc = g + cd * v * torch.linalg.norm(v,dim=-1, keepdim=True) + cm * torch.cross(w, v)
    v_new = v + acc * dt
    return v_new

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
        self.layer3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )
        self.dec = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, v, w, dt):
        v = v / 5.0
        w = w / (15.0*torch.pi*2)

        x = torch.cat([v, w], dim=1)
        x = self.layer1(x)
        x = x + self.layer2(x)*x
        x = x + self.layer3(x)*x
        x = self.dec(x)
        return v * 5.0 + x * dt
    
class TestModel3(nn.Module):
    def __init__(self):
        super(TestModel3, self).__init__()
        self.layer_1 = nn.Sequential(nn.Linear(6, 64), nn.LeakyReLU())
        self.layer_2 = nn.Sequential(nn.Linear(6, 64), nn.LeakyReLU())

        self.dec = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(),nn.Linear(64,64), nn.LeakyReLU(), nn.Linear(64, 3))

    def forward(self, v, w, dt):
        v = v / 5.0
        w = w / (15.0*torch.pi*2)

        x = torch.cat([v, w], dim=1)
        x1 = self.layer_1(x)
        x2 = self.layer_2(x)
        x = x1 * x2
        x = self.dec(x)
        
        return v * 5.0 + x * dt

def batch_rotation_matrix(th):
    """
    th: (N, 3)
    rot_matrix: (N, 3, 3)
    """
    N = th.shape[0]
    cos_th = torch.cos(th)
    sin_th = torch.sin(th)
    zeros = torch.zeros_like(th[:, 0])
    ones = torch.ones_like(th[:, 0])
    rot_matrix = torch.stack([
        torch.stack([cos_th[:, 1]*cos_th[:, 2], -cos_th[:, 1]*sin_th[:, 2], sin_th[:, 1]], dim=1),
        torch.stack([sin_th[:, 0]*sin_th[:, 1]*cos_th[:, 2] + cos_th[:, 0]*sin_th[:, 2], -sin_th[:, 0]*sin_th[:, 1]*sin_th[:, 2] + cos_th[:, 0]*cos_th[:, 2], -sin_th[:, 0]*cos_th[:, 1]], dim=1),
        torch.stack([-cos_th[:, 0]*sin_th[:, 1]*cos_th[:, 2] + sin_th[:, 0]*sin_th[:, 2], cos_th[:, 0]*sin_th[:, 1]*sin_th[:, 2] + sin_th[:, 0]*cos_th[:, 2], cos_th[:, 0]*cos_th[:, 1]], dim=1)
    ], dim=1)
    return rot_matrix

def augment_data(data):
    v = data[:, :3]
    w = data[:, 3:]
    th = torch.rand_like(v) * 2 * torch.pi - torch.pi
    rot_matrix = batch_rotation_matrix(th)
    v_new = torch.einsum("ijk,ik->ij", rot_matrix, v)
    w_new = torch.einsum("ijk,ik->ij", rot_matrix, w)
    return torch.cat([v_new, w_new], dim=1)

def train_loop():
    train_loader, test_loader = Dataset.get_loader(batch_size=16, shuffle=True)
    model = TestModel2()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)
    criterion = nn.MSELoss()
    for epoch in range(500):
        model.train()
        for i, data in enumerate(train_loader):
            data = augment_data(data)
            v = data[:, :3]
            w = data[:, 3:]
            v_new_gt = compute_gt(v, w, 1.0)
            v_new = model(v, w, 1.0)
            loss = criterion(v_new, v_new_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 20 == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss {loss.item()}")

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            v = data[:, :3]
            w = data[:, 3:]
            v_new_gt = compute_gt(v, w,1.0)
            v_new = model(v, w, 1.0)
            
            v_all = torch.cat([v_new, v_new_gt], dim=1)
            print(v_all)
            criterion = nn.MSELoss()
            loss = criterion(v_new, v_new_gt)
            print(f"Test Loss: {loss.item()}")
            break

if __name__ == '__main__':
    train_loop()