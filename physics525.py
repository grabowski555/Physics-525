#%%
import numpy as np

#simulation parameters
c = 0.5
va_over_c = 0.01
q = 0.028
m_i = 50
n_steps = 1000
dt = 300
q_over_m = q / m_i

def bilinear_interp(field:np.ndarray, x:np.ndarray, y:np.ndarray, Lx:float, Ly:float):
    nx,ny = field.shape
    dx = Lx / nx
    dy = Ly / ny

    x = np.mod(x,Lx)
    y = np.mod(y,Ly)

    ix = np.floor(x/dx).astype(int)
    iy = np.floor(y/dy).astype(int)
    ix1 = (ix + 1) % nx
    iy1 = (iy + 1) % ny

    F_00 = field[ix,iy]
    F_01 = field[ix,iy1]
    F_10 = field[ix1,iy]
    F_11 = field[ix1,iy1]

    fx = (x - ix * dx) / dx
    fy = (y - iy * dy) / dy

    return (
        (1 - fx)*(1 - fy)*F_00 
        + (1-fx) * fy * F_01
        + (1-fy) * fx * F_10
        + fy * fx * F_11
    )

def boris(bfield, efield, dt, vel_in, q_over_m ):
    # Non-relativistic
    half_dt = dt * 0.5
    v_minus = vel_in + q_over_m * efield * half_dt
    t_vec = q_over_m * half_dt * bfield
    t2 = np.sum(t_vec * t_vec,axis = 1)
    s = (2*t_vec) / (1 + t2)[:,None]
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_prime + np.cross(v_prime,s)
    return v_plus + q_over_m * efield * half_dt





    






    



