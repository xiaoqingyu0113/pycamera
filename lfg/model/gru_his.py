import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations
import omegaconf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUHisModel(nn.Module):
    def __init__(self, input_size, his_len, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.his_len = his_len
        self.gru_cell = nn.GRUCell(input_size* (his_len-1), hidden_size)
        self.fc_out1 = nn.Linear(hidden_size, 32)
        self.fc_out2 = nn.Linear(32, output_size)
        nn.init.constant_(self.fc_out2.weight, 0)  # Set all weights to 0 to stable the initial training
        nn.init.constant_(self.fc_out2.bias, 0)  
        
        self.fc0 = nn.Linear(3, 32)
        self.fc01 = nn.Linear(32, hidden_size)
        self.relu = nn.ReLU()


    def forward(self, x, h, dt, w0=None):
        '''
            x: (batch_size, seq_len, input_size): 
                - input_size: [time_stamp, x,y,z]
        '''
        x = x.view(-1)

        if w0 is not None:
            h = self.fc01(self.relu(self.fc0(w0)))
        else:
            h = h
        h_new = h + self.gru_cell(x, h) * dt # newtonian hidden states
        v_new =  self.fc_out2(self.relu(self.fc_out1(h_new))) 


        return v_new, h_new
    

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

def gruhis_autoregr(model, data, camera_param_dict, fraction_est):
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
    h = None

    t_prev = stamped_positions[0,0]
    stamped_history = stamped_positions[:his_len,:]

    y = []

    for i in range(stamped_positions.shape[0]):
        t_curr = stamped_positions[i,0] # current time, step i
        # x = stamped_positions[i-1,1:] # previous position, step i-1
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
            v, h = model(x_input, h, dt, w0=w0) # x is (3,)
            x = stamped_history[-1,1:] + v * dt
            stamped_history = stamped_positions[1:i+1,:]
            y.append(x)

        elif i < N:
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            
            v, h = model(x_input, h, dt)
            x = stamped_history[-1,1:] + v * dt
            stamped_history = stamped_positions[i-his_len+1:i+1,:]
            y.append(x)

        else:
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            v, h = model(x_input, h, dt)
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

