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

        self.mode_1_linear = nn.Linear(1, 1)
        self.mode_2_linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

        self.bc_linear = nn.Sequential(nn.Linear(6, 6),
                                        nn.ReLU(),
                                        nn.Linear(6, 6),
                        )

 

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.1, std=1e-3)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(self, b, v, w, dt):
        '''
        input can be [b,3] or [b,1,3]
        '''


        # check bounce
        # set ground truth bouncing condition
        condition1 = b < 0.0
        condition2 = v[..., 2:3] < 0.0
        condition = torch.logical_and(condition1, condition2)
        vw_bc = self.bc_linear(torch.cat([v, w], dim=-1))

        v =   torch.where(condition, vw_bc[..., :3], v)
        w =   torch.where(condition, vw_bc[..., 3:], w)

        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        cross_vw = torch.linalg.cross(v, w)
        acc_mode_1 = self.param1 * norm_v*v + self.param2 * cross_vw + self.param3


        v_new = v + acc_mode_1 * dt
        w_new = w
        

        # soft switch
        # gate1 = self.sigmoid(self.mode_1_linear(b)+ 10.0)
        # gate2 = 1.0 - gate1 # special case for softmax
        # v_new = gate1 *(v + acc_mode_1*dt) + gate2 * vw_mode_2[..., :3]
        # w_new = gate1 * w + gate2 * vw_mode_2[..., 3:]

        # hard switch
        # condition = gate1 > gate2
        # v_new = torch.where(condition, v + acc_mode_1 * dt, vw_mode_2[..., :3])
        # w_new = torch.where(condition, w, vw_mode_2[..., 3:])

        return v_new, w_new
    

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
