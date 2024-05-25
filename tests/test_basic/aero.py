import torch
import torch.nn as nn
from cross import Cross
from norm import Norm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def aero_dynamics(p,v,w):
    g = torch.tensor([0.0, 0.0, -9.81], device=DEVICE)
    acc = g - 0.5 * v + 0.1 * torch.cross(w, v)
    return acc
   
def generate_dataset():
    p0 = torch.rand(3, device=DEVICE)
    v0 = torch.rand(3, device=DEVICE)*3.0
    # w0 = torch.rand(3, device=DEVICE)* 20.0
    w0 = torch.tensor([0.0, 20.0, 0.0], device=DEVICE)
    tspan = torch.linspace(0.0, 2.0, 100, device=DEVICE) + torch.rand(100, device=DEVICE)*1.0e-2
    tspan = tspan - tspan[0]

    p = [p0]
    for i in range(len(tspan)-1):
        dt = tspan[i+1] - tspan[i]
        acc = aero_dynamics(p0, v0, w0)
        p0 = p0 + v0 
        v0 = v0 + acc * dt
        p.append(p0)

    p = torch.stack(p)
    return tspan, p

def view_dataset():
    import matplotlib.pyplot as plt
    _, p = generate_dataset()
    p = p.detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p[:,0], p[:,1], p[:,2])
    plt.show()

def compute_v0(t,p):
    
    return (p[1] - p[0]) / (t[1] - t[0]) 

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross = Cross()
        self.norm = Norm()

    def forward(self, v, w):
        p = self.cross(v, w)
        p = self.norm(p)
        return p


def train():
    num_epochs = 100
    model = Model().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    w = 
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        t, p = generate_dataset()
        v = compute_v0(p,t)
        acc = model(v,w)
        v = v + acc
        loss = criterion(y, y_gt)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss: {loss.item()}")

if __name__ == '__main__':
    # generate_dataset()
    view_dataset()
