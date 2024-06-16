import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class MLayer(nn.Module):
    '''
    change the cross product to multiplication of hidden layers
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.param1 =  nn.Parameter(torch.rand(1,1)*1e-6)

        self.m_layer_1 = nn.Linear(8, self.hidden_size)
        self.m_layer_2 = nn.Linear(8, self.hidden_size)

        self.m_layer_dec = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size, 6))
        self.apply(self._init_weights)

        self._mu_v = torch.tensor([0.0, 3.0, 1.5], device=DEVICE)
        self._std_v = torch.tensor([0.5, 1.0, 1.0] , device=DEVICE)
        self._mu_w = torch.tensor([10.0, 0.0, 0.0], device=DEVICE)
        self._std_w = torch.tensor([10, 30, 10], device=DEVICE)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1e-6)
            if module.bias is not None:
                module.bias.data.zero_()


    def forward(self, b, v, w, dt):
        '''
        input can be [b,3] or [b,1,3]
        '''
        
        v = _normalize(v, self._mu_v, self._std_v)
        w = _normalize(w, self._mu_w, self._std_w)

        identity = torch.ones_like(b, device=DEVICE)
        x = torch.cat([identity, b, v, w], dim=-1)

    
        h1 = self.m_layer_1(x)
        h2 = self.m_layer_2(x)
        h = h1 * h2

        acc = self.m_layer_dec(h)

        v_new = _unnormalize(v + acc[...,:3] * dt, self._mu_v, self._std_v)
        w_new = _unnormalize(w + acc[..., 3:] * dt, self._mu_w, self._std_w)

        return v_new, w_new
    

def euler_updator(p, v, dt):
    return p + v*dt



def _normalize(x, mu, d):
    return (x - mu) / d
def _unnormalize(x, mu, d):
    return x * d + mu

def autoregr_MLayer(data, model, est, cfg):
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
