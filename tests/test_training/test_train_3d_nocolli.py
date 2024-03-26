import torch
import torch.nn as nn
import theseus as th

from lfg.graph import InvariantFactorGraph
import matplotlib.pyplot as plt
from mcf4pingpong.draw_util import set_axes_equal
from typing import Tuple, Sequence
from functools import partial
'''
This is a toy example for trajectory g = [0,0,-9.8]
    1. generate trajectory for a = g, t = [0,2] sec
    2. add random noise 
    3. train mlp([3,3]) to regress v0 + mlp *dt - v1
    4. compare mlp[3,3] with [0,0,9.8]

     p0 --MLP-- p1---MLP--p2
          |         |
         V0 --MLP-- V1
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_trajectory(visualize = False):
    def euler_int(carry, dt):
       p0, v0 = carry
       p1 = p0 + v0 * dt
       v1 =  v0 + torch.tensor([0.0,0.0,-9.8])*dt
       return (p1, v1), p1
    N=40
    v0 = torch.tensor([5.0,0.0,0.0])
    p0 = torch.tensor([0.0,0.0,0.0])
    t = torch.linspace(0.0,2.0,N)
    dts = torch.diff(t)
    
    pN = [p0]
    for dt in dts:
        (p0,v0), p0 = euler_int((p0,v0),dt)
        pN.append(p0)
    pN = torch.vstack(pN)
    noise_scale = 0.30  # Standard deviation of the noise
    noise = noise_scale *  torch.randn((N, 3))
    pN_noisy = pN + noise
    if visualize:
        fig= plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*pN_noisy.T)
        set_axes_equal(ax)
        plt.show()

    return t, pN_noisy, pN

def rotz(angles):
    angle = angles
    cos = torch.cos(angle)
    sin = torch.sin(angle)[0]
    
    return torch.tensor([[cos, -sin, 0.0],
                         [sin,  cos,  0.0],
                         [0.0,  0.0,   1.0]])



def pos_prior_errfn(optim_vars: Tuple[th.Variable,...], aux_vars: Tuple[th.Variable,...]):
    li, = optim_vars 
    li_prior, = aux_vars
    err = li.tensor[:,:3] - li_prior.tensor[:,:3]
    return err

def dynamic_forward(model: nn.Module, x:torch.Tensor, dt:torch.Tensor):
    p, th, v, w = x[:,:3], x[:,3:4], x[:,4:6], x[:, 6:7] # keep the batch

    # print(th)
    # print(rotz(th).shape)
    # print(torch.tensor([[v[0,0], 0.0, v[0,1]]]).shape)
    # raise
    pdot = rotz(th)@ torch.ones(3)#@torch.tensor([v[0,0], 0.0, v[0,1]])
    pdot = pdot[None,:]
    thdot = w
    vwdot = model(torch.cat((v,w),dim=1))

    print(pdot.shape)
    print(thdot.shape)
    print(vwdot.shape)
    x_dot = torch.cat((pdot,thdot, vwdot), dim=1)

    return x + x_dot * dt

def dynamic_errfn(model: nn.Module, optim_vars:Sequence[th.Variable], aux_vars:Sequence[th.Variable]):
    x1_var, x2_var = optim_vars
    dt_var, = aux_vars

    x1, x2 = x1_var.tensor, x2_var.tensor
    dt = dt_var.tensor

    x2_est = dynamic_forward(model, x1, dt)

    return x2_est - x2

def main():
    t, pN_noisy, pN_clean = generate_trajectory(visualize=False)
    objective = th.Objective()
    theseus_inputs = {}
    weight_l_prior = th.ScaleCostWeight(0.1)

    graph = InvariantFactorGraph()
    model = nn.Linear(3,3)

    for i, (ti, pi) in enumerate(zip(t, pN_noisy)):
        xi = th.Vector(7, name=f"x{i}")
        theseus_inputs.update({f'x{i}': torch.zeros(1,7)})
        
        xi_prior_tensor = torch.zeros((1,7))
        xi_prior_tensor[:, :3] = pi
        xi_prior = th.Variable(xi_prior_tensor, name=f"x{i}_prior")

        graph.add(th.AutoDiffCostFunction(
            (xi,), 
            pos_prior_errfn, 
            3, 
            aux_vars=(xi_prior,), 
            name=f"x{i}_prior_erfn", 
            cost_weight=weight_l_prior))
            
        if i>0:
            dt_prev = th.Variable(torch.tensor([[ti-t_prev]]),name=f'dt{i-1}') 
            theseus_inputs.update({f'dt{i-1}': torch.tensor([[ti-t_prev]])})

            graph.add(th.AutoDiffCostFunction(
                (xi_prev, xi), 
                partial(dynamic_errfn, model), 
                7, 
                aux_vars=(dt_prev,), 
                name=f"pose_between{i}"))
            
        xi_prev = xi
        t_prev = ti
    
    # for k,v in theseus_inputs.items():
    #     print(f"{k}:{v}")

    updated_inputs = graph.forward(theseus_inputs)
    

    pN_opt, vN_opt = [],[]
    for k,v in updated_inputs.items():
        if 'l' in k:
            pN_opt.append(v)
        elif 'v' in k:
            vN_opt.append(v)
    pN_opt = torch.vstack(pN_opt)

    fig= plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*pN_noisy.T)
    ax.scatter(*pN_opt.T)
    ax.plot(*pN_clean.numpy().T)
    set_axes_equal(ax)
    plt.show()

if __name__ == '__main__':
    main()
