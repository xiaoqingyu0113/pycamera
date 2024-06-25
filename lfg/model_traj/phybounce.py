import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class BounceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(6, 64),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU()
        )

        self.dec = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, v, w):
        v = v / 3.0
        w = w / (30.0*torch.pi*2)

        x = torch.cat([v, w], dim=-1)
        x = self.layer1(x)
        x = x + self.layer2(x)*x
        x = self.dec(x)

        v_new = x[..., :3] * 3.0
        w_new = x[..., 3:] * (30.0*torch.pi*2)
        return v_new, w_new
    
class PhyBounce(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._init_random()
        
        self.bc_layer = BounceModel()

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if type(m) == nn.Linear:
            # random weight and bias with magnitude 1e-4
            torch.nn.init.normal_(m.weight, mean=0.0, std=1e-4)
            torch.nn.init.normal_(m.bias, mean=0.0, std=1e-4)

    
        

    def _init_gt(self):
        self.param1 =  nn.Parameter(torch.tensor([[0.1196]])) # C_d
        self.param2 =  nn.Parameter(torch.tensor([[0.015]])) # C_m
        self.param3 =  nn.Parameter(torch.tensor([[0.0,0.0,0.0]]))

    def _init_random(self):
        self.param1 =  nn.Parameter(torch.rand(1,1)*1e-4) # C_d
        self.param2 =  nn.Parameter(torch.rand(1,1)*1e-4) # C_m
        self.param3 =  nn.Parameter(torch.rand(1,3)*1e-4) # g

    
    def forward(self, b, v, w, dt):
        '''
        input can be [b,3] or [b,1,3]
        '''

        # check bounce
        condition1 = b < 0.0
        condition2 = v[..., 2:3] < 0.0
        condition = torch.logical_and(condition1, condition2)

        # v_bc, w_bc = self._predict_bounce(v, w)
        v_bc , w_bc= self.bc_layer(v,w)

        
        v =   torch.where(condition, v_bc, v)
        w =   torch.where(condition, w_bc, w)

        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        cross_vw = torch.linalg.cross(w, v)
        acc_mode_1 = -self.param1 * norm_v*v + self.param2 * cross_vw + self.param3 + torch.tensor([0.0, 0.0, -9.81], device=DEVICE).view(1,3)


        v_new = v + acc_mode_1 * dt
        w_new = w
        

        return v_new, w_new
    

def euler_updator(p, v, dt):
    return p + v*dt

def autoregr_PhyBounce(data, model, est, cfg):
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
