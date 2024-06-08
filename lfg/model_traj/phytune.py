import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class PhyTune(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.param1 =  nn.Parameter(torch.rand(1,1)*1e-6)
        self.param2 =  nn.Parameter(torch.rand(1,1)*1e-6)
        self.param3 =  nn.Parameter(torch.rand(1,3)*1e-6)


    def forward(self, b, v, w, dt):
        '''
        input can be [b,3] or [b,1,3]
        '''
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        cross_vw = torch.linalg.cross(v, w)
        acc = self.param1 * norm_v*v + self.param2 * cross_vw + self.param3
        return v + acc*dt, w
    

def euler_updator(p, v, dt):
    return p + v*dt

def autoregr_PhyTune(data, model, est, cfg):
    '''
    data = [b, seq_len, 11]
    11: [traj_idx, time_stamp, p_xyz, v_xyz, w_xyz]
    '''
    tN = data[:,:, 1:2]
    pN = data[:,:, 2:5]
    p0 = pN[:, 0:1, :]
    v0 = data[:, 0:1, 5:8]
    w0 = data[:, 0:1, 8:11]

    if est is not None:
        p0, v0, w0 = est(data[:,:est.size,1:5], w0=w0)
    

    d_tN = torch.diff(tN, dim=1)
    
    pN_est = [p0]
    for i in range(1, data.shape[1]):
        dt = d_tN[:, i-1:i, :]
        b0 = p0[:,:,2:3]
        p0 = euler_updator(p0, v0, dt)
        v0, w0 = model(b0, v0, w0, dt)
        pN_est.append(p0)

    pN_est = torch.cat(pN_est, dim=1)
    return pN_est
