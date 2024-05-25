import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations
import omegaconf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_param(m, val):
    torch.nn.init.uniform_(m.weight, -val, val)
    torch.nn.init.uniform_(m.bias, -val, val)


class PhysicsModel(nn.Module):
    def __init__(self, his_len, hidden_size):
        '''
        input_size: (his_len, feature_size)
        c_size: carry on feature size
        hidden_size: hidden size of the network
        output_size: output feature size 

        '''
        super().__init__()
        self.his_len = his_len
        self.hidden_size = hidden_size
        self.fc_v = nn.Linear((his_len-1)*3, 3)
        self.fc_1 = nn.Linear(3, 3)
        self.fc_2 = nn.Linear(3, 3)

        set_param(self.fc_v, 1e-6)
        set_param(self.fc_1, 1e-1)
        set_param(self.fc_2, 1e-1)

    def forward(self, x, w_in, dt):
        '''
            x: (b, his_len, feat_input_size): 
                
        '''
        v_in = x[0, :, :-1].reshape(-1)
        b_in = x[0, :, -1]

        v = self.fc_v(v_in)


        term1 = -self.fc_1(v)* torch.linalg.norm(v)*0.1196
        term2 =  torch.linalg.cross(v,w_in)*0.015
        
        # print('---------------------------------------')
        # print(x.detach().cpu().numpy())
        # print(v.detach().cpu().numpy(), w_in.detach().cpu().numpy())
        # print(term1.detach().cpu().numpy(), term2.detach().cpu().numpy())
        
        acc = term1 + term2 + torch.tensor([0,0,-9.8], device=DEVICE)

        dv = v + acc * dt
        dp = dv * dt
        
        w_out = w_in
        
        return dp, w_out
    


def compute_velocities(stamped_positions):
    '''
    Compute the velocities from the stamped positions.

    Parameters:
        stamped_positions (torch.Tensor): A tensor of shape (N, 4) where each row is [t, x, y, z].
    Returns:
        torch.Tensor: A tensor of shape (N-1,3) containing [ vx, vy, vz] for each interval.
    '''

    position_differences = stamped_positions[1:, 1:] - stamped_positions[:-1, 1:]
    time_differences = stamped_positions[1:, 0] - stamped_positions[:-1, 0]
    time_differences = time_differences.clamp(min=1e-6)
    velocities = position_differences / time_differences.unsqueeze(1)

    return velocities

def physics_autoregr(model, data, camera_param_dict, fraction_est):
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
    w = w0

    t_prev = stamped_positions[0,0]
    stamped_history = stamped_positions[:his_len,:]

    y = []

    for i in range(stamped_positions.shape[0]):
        # print('------------------------------------------\nstep:', i)
        # print('stamped_history:', stamped_history) 
        t_curr = stamped_positions[i,0] # current time, step i
        dt = t_curr - t_prev
        # if i > 60:
        #     raise
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
            dp, w = model(x_input, w, dt) 
            x = stamped_history[-1,1:] + dp
            stamped_history = stamped_positions[1:i+1,:]
            y.append(x)

        elif i < N:
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            
            dp, w = model(x_input, w, dt) 
            x = stamped_history[-1,1:] + dp
            stamped_history = stamped_positions[i-his_len+1:i+1,:]
            y.append(x)

        else:
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            dp, w = model(x_input, w, dt) 
            x = stamped_history[-1,1:] + dp
            stamped_x = torch.cat((torch.tensor([t_curr], device=DEVICE), x), dim=0).unsqueeze(0) # shape is (1, 4)
            stamped_history = torch.cat((stamped_history[1:,:],stamped_x), dim=0) # shape is (his_len, 4)
            y.append(x)

        t_prev = t_curr
        

    y = torch.stack(y, dim=0) # shape is (seq_len-1, 3)
    # print('y:', y.detach().cpu().numpy())
    # raise
    # raise
    # back project to image
    for cm in camera_param_dict.values():
        cm.to_torch(device=DEVICE)
    cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]
    uv_pred = get_uv_from_3d(y, cam_id_list, camera_param_dict)
    
    return uv_pred

if __name__ == '__main__':
    pass