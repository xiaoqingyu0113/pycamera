import torch
import torch.nn as nn

# class Cross(nn.Module):
#     def __init__(self, num_layers=2, hidden_size=128):
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.fc0 = nn.Linear(6, hidden_size)
#         self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])  # Use ModuleList
#         self.fc1 = nn.Linear(hidden_size, 3)
#         self.lm = nn.LayerNorm(hidden_size)

#     def forward(self, x):
#         x = self.relu(self.fc0(x))
#         for layer in self.layers:
#             x = self.relu(layer(x)) +x
#             x = self.lm(x)
            
#         x = self.fc1(x)
#         return x

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
    
# class Cross(nn.Module):
#     def __init__(self, num_layers=3, hidden_size=128):
#         super().__init__()
#         self.relu = nn.ReLU()
#         self.lm = nn.LayerNorm(6)
#         self.lm2 = nn.LayerNorm(hidden_size//2)
#         self.fc2 = nn.Linear(hidden_size//2, hidden_size//2)
#         self.fc0 = nn.Linear(6, hidden_size)
#         self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])  # Use ModuleList
#         self.fc3 = nn.Linear(hidden_size//2, 6)
#         # self.lm2 = nn.LayerNorm(hidden_size)
#         self.fc4 = nn.Linear(6,3)

#     def forward(self, x):
#         h = self.lm(x)
#         h = self.fc0(h)
#         h = self.relu(h)
#         u,v = h.chunk(2, dim=-1)
#         h = u * self.fc2(self.lm2(v))
#         h = self.fc3(h)
#         h = x + h
#         return self.fc4(h)


class Cross(nn.Module):
    def __init__(self, num_layers=3, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(6, hidden_size)
        self.fc3 = nn.Linear(6, hidden_size)
        self.fc4 = nn.Linear(6, hidden_size)
        self.fc = nn.Linear(hidden_size, 3)
        self.lm = nn.LayerNorm(6)

    def forward(self, x):
        x3 = self.fc1(x)
        x3 = x3/x3.norm(dim=-1, keepdim=True)
        x4 = self.fc2(x)
        x4 = x4/x4.norm(dim=-1, keepdim=True)

        x1 = self.fc3(x)
        x2 = self.fc4(x)

        x = x3*x4
        x = self.fc(x)
        return x
        
def generate_data():
    return torch.randn(batch_size, 6, device='cuda')  *10



def compute_cross(x):
    u = x[:, :3]/torch.linalg.norm(x[:, :3], dim=-1, keepdim=True)
    v = x[:, 3:]/torch.linalg.norm(x[:, :3], dim=-1, keepdim=True)
    return torch.linalg.cross(u, v)



epoch_num = 1000
batch_size = 64
model = Cross(num_layers=2, hidden_size=128)
model = model.cuda()  # Move model to GPU

optim = torch.optim.Adam(model.parameters(), lr=1e-1)
criteria = torch.nn.MSELoss()
for epoch in range(epoch_num):
    optim.zero_grad()
    x =generate_data()
    y = model(x)
    y_gt = compute_cross(x)  
    loss = criteria(y, y_gt)
    loss.backward()
    optim.step()

    if epoch % 100 == 0:
        print('-----------------')
        print(f"Epoch {epoch}, loss: {loss.item()}")
        with torch.no_grad():
            x = generate_data()
            y = model(x)
            y_gt = compute_cross(x) 

            print("y_gt: ", y_gt[0])
            print("y: ", y[0])


           
