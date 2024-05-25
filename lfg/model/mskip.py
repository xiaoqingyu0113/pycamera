import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations
import omegaconf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MSkipModel(nn.Module):
    def __init__(self, input_size, c_size, hidden_size, output_size):
        '''
        input_size: (his_len, feature_size)
        c_size: carry on feature size
        hidden_size: hidden size of the network
        output_size: output feature size 

        '''
        super().__init__()
        self.his_len = input_size[0]
        self.inp_feat_size = input_size[1]
        self.b_feat_size = 1
        self.c_size = c_size 
        self.h_size = hidden_size
        self.out_size = output_size

        self.mlp_0 = nn.Linear((self.his_len-1)*(self.inp_feat_size - self.b_feat_size) + self.c_size, self.h_size)
        self.mlp_g = nn.Linear(self.his_len - 1, self.h_size)
        self.mlp_a = nn.Linear(self.h_size, self.h_size)
        self.mlp_m = nn.Linear(self.h_size, self.h_size)
        self.mlp_y = nn.Linear(self.h_size, self.out_size)
        self.mlp_c = nn.Linear(self.h_size, self.c_size) 
        self.mlp_w = nn.Linear(3, self.c_size)
        
        self.gate = nn.Sigmoid()
        self.relu = nn.ReLU()

        nn.init.constant_(self.mlp_y.weight, 0.0)  # Set weights to 0 to stable the initial training
        nn.init.constant_(self.mlp_y.bias, 0.0)
        nn.init.constant_(self.mlp_g.weight, 0.0)
        nn.init.constant_(self.mlp_g.bias, 0.0)



    def forward(self, x, c, dt, w0=None):
        '''
            x: (his_len, feat_input_size): 
                
        '''
        v = x[0, :, :-1].reshape(-1)
        b = x[0, :, -1]

        if w0 is not None:
            c = self.mlp_w(w0)
        else:
            c = c

        
        h = self.mlp_0(torch.cat((v, c), dim=0))
        hn = h.max() - h.min()
        h = h/hn

        g = self.gate(self.mlp_g(b))
        h = h * g


        h = h + (self.mlp_a(h) + h*self.mlp_m(h))


        h = h*hn
        c_new=self.mlp_c(h)
        y_new = self.mlp_y(h)

        return y_new, c_new
    

def compute_velocities(stamped_positions):
    '''
    Compute the velocities from the stamped positions.

    Parameters:
        stamped_positions (torch.Tensor): A tensor of shape (N, 4) where each row is [t, x, y, z].
    Returns:
        torch.Tensor: A tensor of shape (N-1,3) containing [ vx, vy, vz] for each interval.
    '''

    # Calculate the differences in positions
    position_differences = stamped_positions[1:, 1:] - stamped_positions[:-1, 1:]

    # Calculate the differences in timestamps
    time_differences = stamped_positions[1:, 0] - stamped_positions[:-1, 0]

    # Prevent division by zero in case of same timestamp data
    time_differences = time_differences.clamp(min=1e-6)

    # Calculate velocities and incorporate timestamps
    velocities = position_differences / time_differences.unsqueeze(1)

    return velocities

def mskip_autoregr(model, data, camera_param_dict, fraction_est):
    # triangulation first
    for cm in camera_param_dict.values():
        cm.to_numpy()

    # stamped_positions N x [t, x, y ,z]
    stamped_positions = compute_stamped_triangulations(data.numpy(), camera_param_dict)
    stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)


    N = int(stamped_positions.shape[0] * fraction_est)
    his_len = model.his_len

    # Forward pass
    w0 = data[0, 6:9].float().to(DEVICE)
    c = None

    t_prev = stamped_positions[0,0]
    stamped_history = stamped_positions[:his_len,:]

    y = []

    for i in range(stamped_positions.shape[0]):
        t_curr = stamped_positions[i,0] # current time, step i
        dt = t_curr - t_prev

        if i < his_len:
            y.append(stamped_positions[i,1:])

        # predict the position, @ step i
        elif i == his_len:
            '''
            prepare x_input for the model.
            x_input is (1, his_len-1, 4), the last axis is [vx, vy, vz, z]
            '''
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            v, c = model(x_input, c, dt, w0=w0) # x is (3,)
            x = stamped_history[-1,1:] + v * dt
            stamped_history = stamped_positions[1:i+1,:]
            y.append(x)

        elif i < N:
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            
            v, c = model(x_input, c, dt)
            x = stamped_history[-1,1:] + v * dt
            stamped_history = stamped_positions[i-his_len+1:i+1,:]
            y.append(x)

        else:
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            v, c = model(x_input, c, dt)
            x = stamped_history[-1,1:] + v.squeeze(0) * dt
            stamped_x = torch.cat((torch.tensor([t_curr], device=DEVICE), x), dim=0).unsqueeze(0) # shape is (1, 4)
            stamped_history = torch.cat((stamped_history[1:,:],stamped_x), dim=0) # shape is (his_len, 4)
            y.append(x)

        t_prev = t_curr
        

    y = torch.stack(y, dim=0) # shape is (seq_len-1, 3)

    # back project to image
    for cm in camera_param_dict.values():
        cm.to_torch(device=DEVICE)
    cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]
    uv_pred = get_uv_from_3d(y, cam_id_list, camera_param_dict)
    
    return uv_pred

