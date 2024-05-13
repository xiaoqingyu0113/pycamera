import torch
import torch.nn as nn
from lfg.util import get_uv_from_3d, compute_stamped_triangulations

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUModel(nn.Module):
    '''
        GRU model
    '''
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cell = nn.GRUCell(input_size, hidden_size, num_layers)
        self.fc_out1 = nn.Linear(hidden_size, 32)
        self.fc_out2 = nn.Linear(32, output_size)
        # nn.init.constant_(self.fc_out2.weight, 0)  # Set all weights to 0
        # nn.init.constant_(self.fc_out2.bias, 0)  
        
        self.fc0 = nn.Linear(3, 32)
        self.fc01 = nn.Linear(32, hidden_size)
        self.relu = nn.ReLU()
        
    def forward(self, x, w0, pred_t):
        '''
            x: (batch_size, seq_len, input_size): 
                - input_size: [time_stamp, x,y,z]
        '''

        # print('x.shape' , x.shape)
        h = self.fc01(self.relu(self.fc0(w0)))
        tprev = x[0, 0, 0]
        y = [x[0, 0, 1:4]]

        # estimation
        for i in range(1, x.shape[1]):
            tcurr = x[0, i, 0]
            pos = x[0, i-1, 1:4]
            h = self.gru_cell(pos, h)
            vel = self.fc_out2(self.relu(self.fc_out1(h)))
            pos_pred = pos + vel * (tcurr - tprev)
            y.append(pos_pred)
            tprev = tcurr

        # prediction (autoregressive)
        for tcurr in pred_t:
            pos = y[-1]
            h = self.gru_cell(pos, h)
            vel = self.fc_out2(self.relu(self.fc_out1(h)))
            pos_pred = y[-1] + vel * (tcurr - tprev)

            y.append(pos_pred)

        # convert y list to tensor
        return torch.stack(y, dim=0)
    

def compute_gru_loss(model, data, camera_param_dict, criterion, fraction_est, image_dim=None):
    '''
        Compute the loss
    '''
    
    data = data[0] # ignore the batch size

    uv_pred = predict_result(model, data, camera_param_dict, fraction_est)
    uv_gt = data[1:, 4:6].float().to(DEVICE)

    if image_dim is not None:
        normalize = torch.tensor(image_dim,dtype=torch.float32, device=DEVICE)[None,:]
        uv_gt = uv_gt / normalize
        uv_pred = uv_pred / normalize
        
    loss = criterion(uv_gt, uv_pred)

    return loss

def predict_result(model, data, camera_param_dict, fraction_est, no_grad=False):
    '''
        Predict the result
        model: AnyModel
        data: [seq_len, input_size]
        camera_param_dict: Dict[str, CameraParam]
        fraction_est: float
        no_grad: False for training, True for testing
    '''
    # camera files 
    N_est = int(data.shape[0]*fraction_est)
    for cm in camera_param_dict.values():
        cm.to_numpy()
    stamped_positions = compute_stamped_triangulations(data[:N_est].numpy(), camera_param_dict)
    stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)
    w0 = data[0, 6:9].float().to(DEVICE)
    # Forward pass
    if no_grad:
        with torch.no_grad():
            y = model(stamped_positions[None,:,:],w0, data[N_est:,0]) # shape is (seq_len-1, 3)
    else:
        y = model(stamped_positions[None,:,:],w0, data[N_est:,0])
        
    # back project to image
    for cm in camera_param_dict.values():
        cm.to_torch(device=DEVICE)
    cam_id_list = [str(int(cam_id)) for cam_id in data[1:, 3]]
    uv_pred = get_uv_from_3d(y, cam_id_list, camera_param_dict)

    return uv_pred


 # test the model (OK)
    # data = next(iter(train_loader))[0] # ignore the batch size
    # print(data.shape)
    # data = train_loader.dataset[0]
    # print(data.shape)
    # for cm in camera_param_dict.values():
    #     cm.to_numpy()
    # stamped_positions = compute_stamped_triangulations(data[:25].numpy(), camera_param_dict)
    # stamped_positions = torch.from_numpy(stamped_positions).float().to(DEVICE)
    # w0 = data[0, 6:9].float().to(DEVICE)
    # y = model(stamped_positions[None,:,:],w0, data[25:,0])
    # print(y.shape)
    # raise
    # end test