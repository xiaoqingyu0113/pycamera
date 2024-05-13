import numpy as np
from .dynamics import *

def predict_trajectory(p0, v0, w0, tspan, C_d=0.1196, C_m=0.015, mu = 0.22, ez = 0.79):
    # https://ieeexplore.ieee.org/document/5723394

    N = len(tspan)
    xN = np.zeros((N,9))
    
    xN[0] = np.array([p0,v0,w0]).flatten()

    for i in range(1, N):
        l = xN[i-1][:3]
        v = xN[i-1][3:6]
        w = xN[i-1][6:]
        t = tspan[i-1]
        t_now = tspan[i]

        if l[2]<0.0 and v[2] < 0.0: # bounce? Yes!
            al = compute_alpha(v,w,[mu,ez])
            if al < 0.4:
                v = bounce_slide_velocity_forward(v,w,[mu,ez]).flatten()
                w = bounce_slide_spin_forward(v,w,[mu,ez]).flatten()
            else:
                v = bounce_roll_velocity_forward(v,w,[mu,ez]).flatten()
                w = bounce_roll_spin_forward(v,w,[mu,ez]).flatten()
            l[2] = 0.0 # bounce reset
            xN[i,:3] = location_forward(l,v,t,t_now).flatten()
            xN[i,3:6] = velocity_forward(v,w,t,t_now,[C_d, C_m]).flatten()
            xN[i,6:] = w
        else: # no bounce
            xN[i,:3] = location_forward(l,v,t,t_now).flatten()
            xN[i,3:6] = velocity_forward(v,w,t,t_now,[C_d, C_m]).flatten()
            xN[i,6:] = w
    return xN
