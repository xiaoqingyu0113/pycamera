import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations, KalmanFilter
import omegaconf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_param(m, val):
    if isinstance(m, nn.Sequential):
        for mm in m:
            set_param(mm, val)
    elif isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -val, val)
        torch.nn.init.uniform_(m.bias, -val, val)


class PhysicsKFModel(nn.Module):
    def __init__(self):
        '''
        input_size: (his_len, feature_size)
        c_size: carry on feature size
        hidden_size: hidden size of the network
        output_size: output feature size 

        '''
        super().__init__()

        self.fc_v = nn.Sequential(nn.Linear(3, 3))
        self.fc_cr = nn.Sequential(nn.Linear(3, 3))
        self.fc_acc = nn.Linear(3*2, 3)

        set_param(self.fc_v, 1e-5)
        set_param(self.fc_cr, 1e-5)
        set_param(self.fc_acc, 1e-5)

    def forward(self, x, w_in, dt):
        '''
            x: (v0,b0): 
                
        '''
        v0 = x[:3]
        cr = torch.linalg.cross(v0, w_in)

        v = self.fc_v(v0)
        cr = self.fc_cr(cr)

        feat = torch.cat((v, cr))
        acc = self.fc_acc(feat)

        dv = v0 + acc * dt
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

def physicskf_autoregr(model, data, camera_param_dict, fraction_est):

    kf = KalmanFilter(device=DEVICE)

    # triangulation first
    for cm in camera_param_dict.values():
        cm.to_numpy()
        
    # stamped_positions N x [t, x, y ,z]
    stamped_positions = compute_stamped_triangulations(data.numpy(), camera_param_dict)
    stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)


    N = int(stamped_positions.shape[0] * fraction_est)
    # his_len = model.his_len

    # Forward pass
    w0 = data[0, 6:9].float().to(DEVICE)
    w = w0

    t_prev = stamped_positions[0,0]
    # stamped_history = stamped_positions[:his_len,:]

    y = []

    for i in range(stamped_positions.shape[0]):
        # print('------------------------------------------\nstep:', i)
        # print('stamped_history:', stamped_history) 
        t_curr = stamped_positions[i,0] # current time, step i
        dt = t_curr - t_prev

        if i < N:
            smooth_states = kf.smooth(stamped_positions[i,1:], dt)
            x0 = smooth_states[:3]
            y.append(x0)
        else:
            smooth_states = kf.smooth(y[-1], dt)
            x0 = smooth_states[:3]
            v0 = smooth_states[3:]
            x_input = torch.cat((v0, x0[2].unsqueeze(0)),dim=0)

            dp, w = model(x_input, w, dt) 
            x = x0 + dp

            y.append(x)

        t_prev = t_curr
        

    y = torch.stack(y, dim=0) # shape is (seq_len-1, 3)


    # back project to image
    for cm in camera_param_dict.values():
        cm.to_torch(device=DEVICE)
    cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]
    uv_pred = get_uv_from_3d(y, cam_id_list, camera_param_dict)
    
    return uv_pred

if __name__ == '__main__':
    pass