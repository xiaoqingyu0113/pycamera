import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
#import grucell from torch.nn.GRUCell   




# class SpatialGatingUnit(nn.Module):
#     def __init__(self, d_ffn, seq_len):
#         super().__init__()
#         self.norm = nn.LayerNorm(d_ffn)
#         self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
#         nn.init.constant_(self.spatial_proj.bias, 1.0)

#     def forward(self, x):
#         u, v = x.chunk(2, dim=-1)
#         v = self.norm(v)
#         v = self.spatial_proj(v)
#         out = u * v
#         return out
    



class MultiplyLayer32(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(6, hidden_size)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x3 = self.fc1(x)
        x4 = self.fc2(x)
        
        x = x3*x4
        x = self.fc(x)
        return x

class GRUCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(3,3)
        self.fc = nn.Linear(3,3)
    def forward(self, x, h):
        h = self.gru(x,h)
        y = self.fc(h)
        return y

class WidePerceptron1024(nn.Module):
    def __init__(self, hidden_size = 1024):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)
        )
    def forward(self, x):
        x = self.layer(x)
        return x

def generate_data(batch_size):
    return torch.randn(batch_size, 6, device='cuda')  *10



def compute_gt_cross(x):
    u = x[:, :3]
    v = x[:, 3:]
    return torch.linalg.cross(u, v)


def get_model(choose_model):
    model = getattr(sys.modules[__name__], choose_model)()
        
    model = model.cuda()  # Move model to GPU
    return model


def train(choose_model, epoch_num = 400):
    epoch_num = 400
    batch_size = 64
    model = get_model(choose_model)

    optim = torch.optim.Adam(model.parameters(), lr=5e-1)
    criteria = torch.nn.MSELoss()

    loss_container = []
    for epoch in range(epoch_num):
        optim.zero_grad()
        x =generate_data(batch_size)
        y = model(x)
        y_gt = compute_gt_cross(x)  
        loss = criteria(y, y_gt)
        loss.backward()
        optim.step()

        if epoch % 20 == 0:
            with torch.no_grad():
                x = generate_data(batch_size)
                y = model(x)
                y_gt = compute_gt_cross(x) 
                loss_valid = criteria(y, y_gt)
                loss_container.append(loss_valid.item())
    return loss_container

if __name__ == '__main__':
    
    test_model = ['MultiplyLayer32', 'WidePerceptron1024']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for model in test_model:
        loss_history = train(model)
        ax.plot(loss_history, label=model)
    ax.set_yscale('log')
    ax.legend()
    plt.show()

           
