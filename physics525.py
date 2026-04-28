#%%
import numpy as np
import h5py

#simulation parameters
c = 0.5
va_over_c = 0.01
q = 0.028
m_i = 50
n_steps = 1000
dt = 300
q_over_m = q / m_i

filename = "low_res_060000.h5"
dx = 4
dy = 4
va_code = va_over_c * c

def loadin_fields(filename,dx,dy):
    with h5py.File(filename, "r") as f:
        fields = {
            "Ex": f["Ex"][...].astype(np.float64),
            "Ey": f["Ey"][...].astype(np.float64),
            "Ez": f["Ez"][...].astype(np.float64),
            "Bx": f["Bx"][...].astype(np.float64),
            "By": f["By"][...].astype(np.float64),
            "Bz": f["Bz"][...].astype(np.float64),
        }

    Nx, Ny = fields["Bx"].shape

    grid = {
        "Nx": Nx,
        "Ny": Ny,
        "dx": float(dx),
        "dy": float(dy),
        "Lx": float(Nx * dx),
        "Ly": float(Ny * dy)
    }
    return fields, grid

<<<<<<< ours
=======
fields, grid = loadin_fields(filename,dx,dy)

>>>>>>> theirs
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

def interp_fields(pos: np.ndarray, fields: dict, grid: dict):
    x = pos[:,0]
    y = pos[:,1]

    Lx = grid["Lx"]
    Ly = grid["Ly"]

    ex = bilinear_interp(fields["Ex"],x,y,Lx,Ly)
    ey = bilinear_interp(fields["Ey"],x,y,Lx,Ly)
    ez = bilinear_interp(fields["Ez"],x,y,Lx,Ly)
    bx = bilinear_interp(fields["Bx"],x,y,Lx,Ly)
    by = bilinear_interp(fields["By"],x,y,Lx,Ly)
    bz = bilinear_interp(fields["Bz"],x,y,Lx,Ly)

    e_field = np.stack([ex,ey,ez],axis = 1)
    b_field = np.stack([bx,by,bz],axis = 1)

    return e_field, b_field

def boris(bfield, efield, dt, vel_in, q_over_m):
        # Non-relativistic
    half_dt = dt * 0.5
    v_minus = vel_in + q_over_m * efield * half_dt
    t_vec = q_over_m * half_dt * bfield
    t2 = np.sum(t_vec * t_vec,axis = 1)
    s = (2*t_vec) / (1 + t2)[:,None]
    v_prime = v_minus + np.cross(v_minus, t_vec)
    v_plus = v_minus + np.cross(v_prime,s)
    return v_plus + q_over_m * efield * half_dt

def boris_pusher(pos0, vel0, q_over_m, dt, n_steps, fields, grid):
    n_particles = pos0.shape[0]

    Lx = grid["Lx"]
    Ly = grid["Ly"]

    pos_mod = pos0.copy()
    pos_unwrapped = pos0.copy()

    pos_hist = np.zeros((n_steps +1, n_particles, 3), dtype = np.float64)
    vel_hist = np.zeros((n_steps +1, n_particles, 3), dtype = np.float64)
    vel_half_hist = np.zeros((n_steps +1, n_particles, 3), dtype = np.float64)

    e0, b0 = interp_fields(pos_mod, fields, grid)
    vel_half = boris(b0, e0, 0.5*dt, vel0.copy(), q_over_m)
    vel_half_hist[0] = vel_half

    for n in range(n_steps):
        pos_unwrapped = pos_unwrapped + vel_half * dt

        #periodic
        pos_mod[:,0] = np.mod(pos_unwrapped[:,0], Lx)
        pos_mod[:,1] = np.mod(pos_unwrapped[:,1], Ly)
        pos_mod[:2] = pos_unwrapped[:,2]

        pos_hist[n+1] = pos_unwrapped

        #interpolate!!
        e_local, b_local = interp_fields(pos_mod, fields, grid)

        vel_half = boris(b_local, e_local, dt, vel_half, q_over_m)
        vel_half_hist[n+1] = vel_half

    for n in range(1, n_steps +1):
        vel_hist[n] = 0.5 * (vel_half_hist[n-1] + vel_half_hist[n])

    return pos_hist, vel_hist

# defining a basis relative to the mean magnetic field
def basis(b0_hat):
    tmp = np.array([1, 0, 0], dtype = np.float64)
    if np.allclose(np.abs(np.dot(tmp, b0_hat)), 1):
        tmp = np. array([0, 1, 0], dtype = np.float64)
    
    e1 = tmp - np.dot(tmp, b0_hat) * b0_hat
    e1_hat = e1 / np.linalg.norm(e1)
    e2_hat = np.cross(b0_hat, e1_hat)

    return e1_hat , e2_hat

def make_single_velocity(v_perp, b0_hat, phi=0.0, v_par=0.0):
    e1_hat, e2_hat = basis(b0_hat)
    vel = v_par * b0_hat + v_perp * (np.cos(phi) * e1_hat + np.sin(phi) * e2_hat)
    return vel[None, :]

def fft_k(field: np.ndarray, dx: float, dy: float, subtract_mean: bool = True):
    work = np.asarray(field, dtype = np.float64)
    if subtract_mean:
        work = work - work.mean()

    nx, ny = work.shape
    f_k = np.fft.fft2(work)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing = "ij")

    return f_k, kx_grid, ky_grid

def mean_B0(fields: dict):
    b0 = np.array (
        [fields["Bx"].mean(), fields["By"].mean(), fields["Bz"].mean()],
        dtype = np.float64
    )

    b0_mag = np.linalg.norm(b0)
    if b0_mag <= 0.0:
        raise ValueError("Mean mag field is zero")
    
    b0_hat = b0 / b0_mag

    b0_xy = np.array([b0_hat[0], b0_hat[1], 0], dtype = np.float64)
    b0_xy_mag = np.linalg.norm(b0_xy)
    if b0_xy_mag <= 0:
        raise ValueError("B0 has no x-y projection")
    
    b0_xy_hat = b0_xy / b0_xy_mag

    return b0, b0_hat, b0_xy_hat

<<<<<<< ours
=======
b0, b0_hat, b0_xy_hat = mean_B0(fields)

>>>>>>> theirs
def mag_power_spec(fields: dict, grid: dict):
    bx_k, kx_grid, ky_grid = fft_k(fields["Bx"], grid["dx"], grid["dy"], subtract_mean = True)
    by_k, ky_grid, ky_grid = fft_k(fields["By"], grid["dx"], grid["dy"], subtract_mean = True)
    bz_k, kz_grid, kz_grid = fft_k(fields["Bz"], grid["dx"], grid["dy"], subtract_mean = True)

    power = np.abs(bx_k)**2 + np.abs(by_k)**2 + np.abs(bz_k)**2

    return power, kx_grid, ky_grid

def angle_B0(kx_grid: np.ndarray, ky_grid: np.ndarray, b0_xy_hat: np.ndarray):
    kmag = np.hypot(kx_grid, ky_grid)

    k_dot_b = kx_grid * b0_xy_hat[0] + ky_grid * b0_xy_hat[1]

    cosang = np.zeros_like(kmag)
    nonzero = kmag > 0
    cosang[nonzero] = np.abs(k_dot_b[nonzero]) / kmag[nonzero]
    cosang = np.clip(cosang, 0, 1)

    return np.degrees(np.arccos(cosang))

def pick_k(
    power: np.ndarray,
    kx_grid: np.ndarray,
    ky_grid: np.ndarray,
    angles_deg: np.ndarray,
    target:str,
):
    kmag = np.hypot(kx_grid, ky_grid)
    valid = kmag > 0

    if target == "parallel":
        mask = valid & np.isclose(angles_deg, 0, atol=1e-9)
        if not np.any(mask):
            mask = valid & np.isclose(angles_deg, np.min(angles_deg[valid]), atol = 1e-9)

    elif target == "oblique":
        target_angle = 45
        tol = 15.0
        mask = valid & (np.abs(angles_deg - target_angle) <= tol)            
        if not np.any(mask):
            best = np.argmin(np.abs(angles_deg[valid] - target_angle))
            flat_valid = np.flatnonzero(valid)
            chosen_flat = flat_valid[best]
            mask = np.zeros_like(valid, dtype= bool)
            mask.flat[chosen_flat] = True

    else:
        raise ValueError("must be parallel or oblique")
        
    masked_power = np.where(mask, power, -np.inf)
    idx_flat = int(np.argmax(masked_power))
    ix, iy = np.unravel_index(idx_flat, power.shape)

    return {
        "ix": ix,
        "iy": iy,
        "kx": float(kx_grid[ix,iy]),
        "ky": float(ky_grid[ix,iy]),
        "kmag": float(kmag[ix,iy]),
        "lambda": float(2 * np.pi / kmag[ix,iy]),
        "power": float(power[ix,iy]),
        "angle_to_B0_deg": float(angles_deg[ix,iy]),
        "case": target,
    }

def omega(mode, c_wave = 1, b0_hat = None, use_parallel_dispersion = False):
    if use_parallel_dispersion and b0_hat is not None:
        k_vec = np.array([mode["kx"], mode["ky"],0],dtype = np.float64)
        return c_wave * float(np.dot(k_vec,b0_hat))
    return c_wave * mode["kmag"]

def make_xy_mesh(nx,ny,dx,dy):
    x = np.arange(nx, dtype = np.float64) * dx
    y = np.arange(ny, dtype = np.float64) * dy
    return np.meshgrid(x,y, indexing = "ij")

<<<<<<< ours
=======
power, kx_grid, ky_grid = mag_power_spec(fields, grid)
angles_deg = angle = angle_B0(kx_grid, ky_grid, b0_xy_hat)

>>>>>>> theirs
def project_coeff_k(coeff_vec, k_vec):
    k2 = float(np.dot(k_vec, k_vec))
    if k2 <= 0.0:
        return coeff_vec
    return coeff_vec - (np.dot(k_vec, coeff_vec)/k2) * k_vec

def project_coeff_axis_perp(coeff_vec, axis_vec):
    a2 = float(np.dot(axis_vec, axis_vec))
    if a2 <= 0.0:
        return coeff_vec
    return coeff_vec - (np.dot(axis_vec, coeff_vec) / a2) * axis_vec

# see derivation in notes
def parallel_e_from_b(coeff_b, k_vec, v_a):
    kmag = float(np.linalg.norm(k_vec))
    if kmag <= 0:
        return np.zeros(3, dtype = np.complex128)
    return -(v_a / kmag) * np.cross(k_vec, coeff_b)

def oblique_e_from_b(coeff_b, k_vec, v_a, b0_hat):
    kmag = float(np.linalg.norm(k_vec))
    if kmag <= 0:
        return np.zeros(3, dtype = np.complex128)
    
    k_dot_b0 = float(np.dot(k_vec, b0_hat))
    coeff_e_base = -(v_a * k_dot_b0 / (kmag * kmag)) * np.cross(k_vec, coeff_b)

    if np.abs(k_dot_b0) <= 1e-15:
        return coeff_e_base
    
    alpha = -np.dot(coeff_e_base, b0_hat) / k_dot_b0
    return coeff_e_base + alpha * k_vec.astype(np.complex128)

def reconstruct_using_coeff(
        coeff,
        nx,
        ny,
        grid,
        mode,
        t = 0,
        c_wave = 1,
        b0_hat = None,
        use_parallel_dispersion = False,
):
    xg, yg = make_xy_mesh(nx,ny, grid["dx"], grid["dy"])
    omega1 = omega(mode, c_wave, b0_hat = b0_hat, use_parallel_dispersion = use_parallel_dispersion)
    phase_arg = mode["kx"] * xg + mode["ky"] * yg - omega1*t

    ix, iy = mode["ix"], mode["iy"]
    ix_conj = (-ix) % nx
    iy_conj = (-iy) % ny
    factor = 1 if (ix_conj == ix and iy_conj == iy) else 2.0

    return factor * np.real(coeff * np.exp(1j * phase_arg))

def reconstruct_fields(
        fields,
        grid,
        mode,
        t = 0,
        c_wave = 1,
        add_mean_e_fields = False,
        add_mean_b_fields = True,
        use_alfven_e_parallel = False,
        use_alfven_e_oblique = False,
        b0_hat = None,
):
    reconstructed = {}

    nx, ny = fields["Bx"].shape
<<<<<<< ours
    k_vec = np.array([mode["kx"], mode["ky"], 0], dtyep = np.float64)
=======
    k_vec = np.array([mode["kx"], mode["ky"], 0], dtype = np.float64)
>>>>>>> theirs
    use_parallel_dispersion = use_alfven_e_oblique and (b0_hat is not None)

    bx_k, _,_ = fft_k(fields["Bx"], grid["dx"], grid["dy"], subtract_mean = True)
    by_k, _,_ = fft_k(fields["By"], grid["dx"], grid["dy"], subtract_mean = True)
    bz_k, _,_ = fft_k(fields["Bz"], grid["dx"], grid["dy"], subtract_mean = True)

    coeff_b = np.array(
        [
            bx_k[mode["ix"],mode["iy"] / (nx*ny)],
            by_k[mode["ix"],mode["iy"] / (nx*ny)],
            bz_k[mode["ix"],mode["iy"] / (nx*ny)],
        ],
        dtype = np.complex128
    )
    coeff_b = project_coeff_k(coeff_b, k_vec)

    if use_alfven_e_oblique and b0_hat is not None:
        coeff_e = oblique_e_from_b(coeff_b, k_vec, c_wave, b0_hat)

    elif use_alfven_e_parallel:
        if b0_hat is not None:
            coeff_b = project_coeff_axis_perp(coeff_b, b0_hat)
        coeff_e = parallel_e_from_b(coeff_b, k_vec, c_wave)

    else:
        ex_k, _, _ = fft_k(fields["Ex"], grid["dx"], grid["dy"], subtract_mean = True)
        ey_k, _, _ = fft_k(fields["Ey"], grid["dx"], grid["dy"], subtract_mean = True)
        ez_k, _, _ = fft_k(fields["Ez"], grid["dx"], grid["dy"], subtract_mean = True)

        coeff_e = np.array(
            [
            ex_k[mode["ix"],mode["iy"] / (nx*ny)],
            ey_k[mode["ix"],mode["iy"] / (nx*ny)],
            ez_k[mode["ix"],mode["iy"] / (nx*ny)],    
            ],
            dtype = np.complex128,
        )

    reconstructed["Bx"] = reconstruct_using_coeff(
        coeff_b[0], nx, ny, grid, mode, t=t, c_wave = c_wave,
        b0_hat = b0_hat if use_parallel_dispersion else None,
        use_parallel_dispersion = use_parallel_dispersion,
    )

    reconstructed["By"] = reconstruct_using_coeff(
        coeff_b[1], nx, ny, grid, mode, t=t, c_wave = c_wave,
        b0_hat = b0_hat if use_parallel_dispersion else None,
        use_parallel_dispersion = use_parallel_dispersion,
    )

    reconstructed["Bz"] = reconstruct_using_coeff(
        coeff_b[2], nx, ny, grid, mode, t=t, c_wave = c_wave,
        b0_hat = b0_hat if use_parallel_dispersion else None,
        use_parallel_dispersion = use_parallel_dispersion,
    )

    reconstructed["Ex"] = reconstruct_using_coeff(
        coeff_e[0], nx, ny, grid, mode, t=t, c_wave = c_wave,
        b0_hat = b0_hat if use_parallel_dispersion else None,
        use_parallel_dispersion = use_parallel_dispersion,
    )

    reconstructed["Ey"] = reconstruct_using_coeff(
        coeff_e[1], nx, ny, grid, mode, t=t, c_wave = c_wave,
        b0_hat = b0_hat if use_parallel_dispersion else None,
        use_parallel_dispersion = use_parallel_dispersion,
    )

    reconstructed["Ez"] = reconstruct_using_coeff(
        coeff_e[2], nx, ny, grid, mode, t=t, c_wave = c_wave,
        b0_hat = b0_hat if use_parallel_dispersion else None,
        use_parallel_dispersion = use_parallel_dispersion,
    )

    if add_mean_b_fields:
        reconstructed["Bx"] += fields["Bx"].mean()
        reconstructed["By"] += fields["By"].mean()
        reconstructed["Bz"] += fields["Bz"].mean()
    
    if add_mean_b_fields:
        reconstructed["Ex"] += fields["Ex"].mean()
        reconstructed["Ey"] += fields["Ey"].mean()
        reconstructed["Ez"] += fields["Ez"].mean()

    return reconstructed





    


    
    






