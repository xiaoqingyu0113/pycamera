import numba
from numpy import *
@numba.jit(cache=True,nopython=True)
def location_forward(_Dummy_36, _Dummy_37, t1, t2):
    [l1_1, l1_2, l1_3] = _Dummy_36
    [v1_1, v1_2, v1_3] = _Dummy_37
    return array([[l1_1 + v1_1*(-t1 + t2)], [l1_2 + v1_2*(-t1 + t2)], [l1_3 + v1_3*(-t1 + t2)]])

@numba.jit(cache=True,nopython=True)
def velocity_forward(_Dummy_38, _Dummy_39, t1, t2, _Dummy_40):
    [v1_1, v1_2, v1_3] = _Dummy_38
    [w1_1, w1_2, w1_3] = _Dummy_39
    [C_d, C_m] = _Dummy_40
    return array([[v1_1 + (-t1 + t2)*(-C_d*v1_1*sqrt(v1_1**2 + v1_2**2 + v1_3**2) + C_m*(-v1_2*w1_3 + v1_3*w1_2))], [v1_2 + (-t1 + t2)*(-C_d*v1_2*sqrt(v1_1**2 + v1_2**2 + v1_3**2) + C_m*(v1_1*w1_3 - v1_3*w1_1))], [v1_3 + (-t1 + t2)*(-C_d*v1_3*sqrt(v1_1**2 + v1_2**2 + v1_3**2) + C_m*(-v1_1*w1_2 + v1_2*w1_1) - 9.81)]])

@numba.jit(cache=True,nopython=True)
def compute_alpha(_Dummy_41, _Dummy_42, _Dummy_43):
    [v1_1, v1_2, v1_3] = _Dummy_41
    [w1_1, w1_2, w1_3] = _Dummy_42
    [mu, ez] = _Dummy_43
    return mu*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6)

@numba.jit(cache=True,nopython=True)
def bounce_slide_velocity_forward(_Dummy_44, _Dummy_45, _Dummy_46):
    [v1_1, v1_2, v1_3] = _Dummy_44
    [w1_1, w1_2, w1_3] = _Dummy_45
    [mu, ez] = _Dummy_46
    return array([[0.02*mu*w1_2*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6) + v1_1*(-mu*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6) + 1.0)], [-0.02*mu*w1_1*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6) + v1_2*(-mu*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6) + 1.0)], [-ez*v1_3]])

@numba.jit(cache=True,nopython=True)
def bounce_slide_spin_forward(_Dummy_47, _Dummy_48, _Dummy_49):
    [v1_1, v1_2, v1_3] = _Dummy_47
    [w1_1, w1_2, w1_3] = _Dummy_48
    [mu, ez] = _Dummy_49
    return array([[-75.0*mu*v1_2*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6) + w1_1*(-1.5*mu*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6) + 1.0)], [75.0*mu*v1_1*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6) + w1_2*(-1.5*mu*(ez + 1)*abs(v1_3)/sqrt((v1_1 - 0.02*w1_2)**2.0 + (v1_2 + 0.02*w1_1)**2.0 + 1.0e-6) + 1.0)], [1.0*w1_3]])

@numba.jit(cache=True,nopython=True)
def bounce_roll_velocity_forward(_Dummy_50, _Dummy_51, _Dummy_52):
    [v1_1, v1_2, v1_3] = _Dummy_50
    [w1_1, w1_2, w1_3] = _Dummy_51
    [mu, ez] = _Dummy_52
    return array([[0.6*v1_1 + 0.008*w1_2], [0.6*v1_2 - 0.008*w1_1], [-ez*v1_3]])

@numba.jit(cache=True,nopython=True)
def bounce_roll_spin_forward(_Dummy_53, _Dummy_54, _Dummy_55):
    [v1_1, v1_2, v1_3] = _Dummy_53
    [w1_1, w1_2, w1_3] = _Dummy_54
    [mu, ez] = _Dummy_55
    return array([[-30.0*v1_2 + 0.4*w1_1], [30.0*v1_1 + 0.4*w1_2], [1.0*w1_3]])

