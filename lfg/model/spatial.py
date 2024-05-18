import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations
import omegaconf

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpatialModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.fc_out1 = nn.Linear(hidden_size//2, 64)
        self.fc_out2 = nn.Linear(64, output_size)
        nn.init.constant_(self.fc_out2.weight, 0)  # Set all weights to 0 to stable the initial training
        nn.init.constant_(self.fc_out2.bias, 0)  
        
        self.fc0 = nn.Linear(3, hidden_size)
        self.relu = nn.ReLU()


    def forward(self, x, h, dt, w0=None):
        '''
            x: (seq_len, input_size): 
                - input_size: [time_stamp, x,y,z]
            h: (hidden_size,)
        '''

        if w0 is not None:
            h = self.relu(self.fc0(w0)) # overwrite the hidden state with the initial value

        h_new = h + self.gru_cell(x, h) * dt # newtonian hidden states

        h_new1 = h_new[:h_new.shape[0]//2]
        h_new2 = h_new[h_new.shape[0]//2:]
        h_mul = h_new1 * h_new2

        x_new = x + self.fc_out2(self.relu(self.fc_out1(h_mul))) * dt


        return x_new, h_new
    

# def gru_loss(model, data, camera_param_dict, criterion, image_dim=None):
#     data = data[0] # ignore the batch size

#     uv_pred = gru_pass(model, data, camera_param_dict)
#     uv_gt = data[1:, 4:6].float().to(DEVICE)

#     if isinstance(image_dim, omegaconf.listconfig.ListConfig):
#         normalize = torch.tensor(image_dim,dtype=torch.float32, device=DEVICE)[None,:]
#         uv_gt = uv_gt / normalize
#         uv_pred = uv_pred / normalize

#     loss = criterion(uv_gt, uv_pred)

#     return loss


def spatial_autoregr(model, data, camera_param_dict, fraction_est):
    # triangulation first
    for cm in camera_param_dict.values():
        cm.to_numpy()
    stamped_positions = compute_stamped_triangulations(data.numpy(), camera_param_dict)
    stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)

    N = int(stamped_positions.shape[0] * fraction_est)

    # Forward pass
    w0 = data[0, 6:9].float().to(DEVICE)
    h = None

    t_prev = stamped_positions[0,0]
    x = stamped_positions[0,1:]
    y = [x]

    for i in range(1, stamped_positions.shape[0]):
        t_curr = stamped_positions[i,0] # current time, step i
        x = stamped_positions[i-1,1:] # previous position, step i-1
        dt = t_curr - t_prev
        # predict the position, @ step i
        if i == 1:
            x, h = model(x, h, dt, w0=w0)
        elif i < N:
            x, h = model(x, h, dt)
        else:
            x, h = model(y[-1], h, dt)

        t_prev = t_curr
        y.append(x)

    y = torch.stack(y, dim=0) # shape is (seq_len-1, 3)

    # back project to image
    for cm in camera_param_dict.values():
        cm.to_torch(device=DEVICE)
    cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]
    uv_pred = get_uv_from_3d(y, cam_id_list, camera_param_dict)
    
    return uv_pred

