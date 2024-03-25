import jax
import jax.numpy as jnp
from functools import partial
import time
import torch
from copy import deepcopy
from torch.profiler import profile, record_function, ProfilerActivity

def rk4( x0, y0, xf, n):
    vx, vy= x0, y0
    # Step size
    h = (xf - x0) / n
    # Iterative calculation
    for i in torch.arange(1, n + 1):
        k1 = h * ode(vx, vy)
        k2 = h * ode(vx + 0.5*h, vy + 0.5*k1)
        k3 = h * ode(vx + 0.5*h, vy + 0.5*k2)
        k4 = h * ode(vx + h, vy + k3)

        vx = x0 + i * h
        vy = vy+ (k1 + 2*k2 + 2*k3 + k4) / 6
    return vx, vy


def test_jax():
    jax.jit
    def ode(x,y):
        return jnp.array([y[1],y[2],y[3],y[0]])
    y0 = jax.random.normal(jax.random.PRNGKey(0), (4,))
    x0 = 0.0
    xf = 5.0

    rk4(ode, x0, y0, xf,100)

    start_time = time.time()
    vx, vy = rk4(ode, x0, y0, xf,100)
    jax_duration = time.time() - start_time
    print('jax time = ', jax_duration)
    print(vy)


@torch.jit.script
def ode(x,y):
    return torch.hstack([y[1],y[2],y[3],y[0]])

def test_torch():
    
    
    y0 = torch.randn(4,dtype=torch.float32).cuda()
    x0 = torch.tensor(0.0,dtype=torch.float32)
    xf = torch.tensor(5.0,dtype=torch.float32)
    n = torch.tensor(100,dtype=torch.int32)
    
    rk4_script = torch.jit.script(rk4)


    for _ in range(2):   
        vx, vy = rk4_script( x0, y0, xf,n)

    start_time = time.time()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("rk4_function"):     
    for _ in range(100):   
        vx, vy = rk4_script( x0, y0, xf,n)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    torch_duration = time.time() - start_time
    print('torch time = ', torch_duration)
    # print(vy)


if __name__ =='__main__':
    # test_jax()
    test_torch()