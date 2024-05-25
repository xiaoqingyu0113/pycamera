import torch
import numpy as np
import matplotlib.pyplot as plt
from train import get_data_loaders, get_camera_param_dict
import hydra
from lfg.util import get_uv_from_3d, compute_stamped_triangulations
from synthetic.dynamics import location_forward, velocity_forward
import torch.nn as nn

DEVICE = torch.device('cpu')

class Estimator(nn.Module):
    def __init__(self, his_len, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out
    
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
    stamped_positions = torch.from_numpy(stamped_positions).float()


    N = int(stamped_positions.shape[0] * fraction_est)
    his_len =5

    # Forward pass
    w0 = data[0, 6:9].float()
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
            
            v = velocities[-1,:]
            acc = torch.tensor([0.0, 0.0, -9.8]) - 0.1196 * v * torch.linalg.norm(v) + 0.015 * torch.linalg.cross(w, v)
            x = stamped_history[-1,1:] + v * dt + 0.5 * acc * dt * dt


            stamped_history = stamped_positions[1:i+1,:]
            y.append(x)

        elif i < N:
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            
            
            v = velocities[-1,:]

            acc = torch.tensor([0.0, 0.0, -9.8]) - 0.1196 * v * torch.linalg.norm(v) + 0.015 * torch.linalg.cross(w, v)
            x = stamped_history[-1,1:] + v * dt + 0.5 * acc * dt * dt
            stamped_history = stamped_positions[i-his_len+1:i+1,:]
            y.append(x)

        else:
            velocities = compute_velocities(stamped_history)
            x_input = torch.cat((velocities, stamped_history[1:,3:]), dim=1).unsqueeze(0) # x_input is (1, his_len-1, 4)
            
            
            v = velocities[-1,:]
            acc = torch.tensor([0.0, 0.0, -9.8]) - 0.1196 * v * torch.linalg.norm(v) + 0.015 * torch.linalg.cross(w, v)
            x = stamped_history[-1,1:] + v * dt + 0.5 * acc * dt * dt

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
    
    return uv_pred, y, stamped_positions




@hydra.main(version_base=None,config_path='../../conf', config_name='config')
def test(config):
    train_loader, test_loader = get_data_loaders(config)
    camera_param_dict = get_camera_param_dict(config)
    criterion = torch.nn.MSELoss()

    for i, data in enumerate(train_loader):
        
        data = data[0] # ignore the batch size
        uv_gt = data[1:, 4:6].float()

        uv_pred, y, stamped_positions = physics_autoregr(None, data, camera_param_dict, config.model.estimation_fraction)

        loss = criterion(uv_gt, uv_pred)
        print('loss:', loss.item())
    
        
        if i>0:
            break


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    y = y.detach().cpu().numpy()
    stamped_positions = stamped_positions.detach().cpu().numpy()
    ax.plot(stamped_positions[:,1], stamped_positions[:,2], stamped_positions[:,3], 'b')
    ax.plot(y[:,0], y[:,1], y[:,2], 'r')
    plt.show()
if __name__ == '__main__':
    test()