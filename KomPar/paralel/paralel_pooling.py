from multiprocessing import Pool
import vpython as vp
import numpy as np
import time

#fungsi simulasi SEBUAH proyektil supaya dapat dipanggil secara paralel
def simulate_projectile(args):
    mass, v0, params, dt, t_max = args
    def acceleration(x, v):
        g   = params['g']
        Cd  = params['Cd']
        area= params['area']
        vol = params['volume']
        v_mag = np.linalg.norm(v)

        if x[1] < 0:  # di dalam air
            rho = params['water_density']
            F_grav  = np.array([0, -mass*g, 0])
            F_buoy  = np.array([0, rho*vol*g, 0])
            if v_mag > 0:
                F_drag  = -0.5 * rho * Cd * area * v_mag * v
            else:
                F_drag  = np.zeros(3)
            F_curr  = params['current_force']
            F_net   = F_grav + F_buoy + F_drag + F_curr
        else:        # di udara
            rho     = params['air_density']
            F_grav  = np.array([0, -mass*g, 0])
            if v_mag > 0:
                F_drag  = -0.5 * rho * Cd * area * v_mag * v
            else:
                F_drag  = np.zeros(3)
            F_net   = F_grav + F_drag
        
        return F_net / mass

    def rk4_step(x, v):
        k1_x = dt * v
        k1_v = dt * acceleration(x, v)
        k2_x = dt * (v + 0.5*k1_v)
        k2_v = dt * acceleration(x + 0.5*k1_x, v + 0.5*k1_v)
        k3_x = dt * (v + 0.5*k2_v)
        k3_v = dt * acceleration(x + 0.5*k2_x, v + 0.5*k2_v)
        k4_x = dt * (v + k3_v)
        k4_v = dt * acceleration(x + k3_x, v + k3_v)

        x_next = x + (k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
        v_next = v + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6.0
        return x_next, v_next

    # inisialisasi
    steps = int(t_max / dt) + 1
    traj = np.zeros((steps, 3), dtype=float)
    x = np.zeros(3)
    v = v0.copy()
    for i in range(steps):
        traj[i] = x
        x, v = rk4_step(x, v)
    return traj



# MAIN PROGRAM
if __name__ == '__main__':
    # catat waktu mulai program
    start_time = time.perf_counter()
    print(f"Waktu mulai program: {start_time:.8f} detik")

    # parameter global
    g               = 9.8
    water_density   = 1000
    air_density     = 1.225
    Cd              = 0.47
    r               = 0.1
    volume          = (4/3)*np.pi*r**3
    area            = np.pi*r**2
    dt              = 0.001
    
    # parameter arus air
    current_force_magnitude = 5000
    current_azim_deg = 45  # sudut arah arus di bidang x-z
    azim_rad = np.radians(current_azim_deg)
    # vektor arah arus
    current_dir = np.array([np.cos(azim_rad), 0.0, np.sin(azim_rad)])
    current_force = current_force_magnitude * current_dir
    
    #===============================
    # testing simul
    t_max = 7.0
    number_of_proj = 40
    #===============================

    # state untuk tiap proyektil
    args_list = []
    for i in range(number_of_proj):
        mass     = 50 + i*50
        elev_rad = np.radians(10 + i*5)
        azim_rad = np.radians(i*30)
        power    = 2000 + i*400
        v0_mag   = power / mass
        v0 = np.array([
            v0_mag * np.cos(elev_rad) * np.cos(azim_rad),
            v0_mag * np.sin(elev_rad),
            v0_mag * np.cos(elev_rad) * np.sin(azim_rad)
        ])
        params = {
            'g': g,
            'water_density': water_density,
            'air_density'  : air_density,
            'volume'       : volume,
            'area'         : area,
            'Cd'           : Cd,
            'current_force': current_force
        }
        args_list.append((mass, v0, params, dt, t_max))

    # Pool untuk paralelisasi
    with Pool() as pool:
        all_trajectories = pool.map(simulate_projectile, args_list)


    # Visualisasi world
    direction_arrow_scale = 5
    scene = vp.canvas(title="Simulasi Proyektil (Paralel)", width=1920, height=1080, background=vp.color.white)
    vp.box(pos=vp.vector(0, -50, 0), size=vp.vector(100, 100, 100), color=vp.color.blue, opacity=0.2)
    vp.box(pos=vp.vector(0, 0, 0), size=vp.vector(100, 0.2, 100), color=vp.color.blue, opacity=0.5)

    # panah arus air
    arrow_spacing = 15
    for x in np.arange(-40, 50, arrow_spacing):
        for z in np.arange(-40, 50, arrow_spacing):
            vp.arrow(pos=vp.vector(x, 0.5, z), axis=vp.vector(*current_dir) * direction_arrow_scale,
                    shaftwidth=0.3, color=vp.color.cyan)

    # buat sphere dan simpan state trajektori
    spheres = []
    for traj in all_trajectories:
        color = vp.vector(np.random.rand(), np.random.rand(), np.random.rand())
        sph = vp.sphere(pos=vp.vector(*traj[0]), radius=r, color=color, make_trail=True)
        spheres.append((sph, traj))

    n_steps = all_trajectories[0].shape[0]
    for step in range(n_steps):
        vp.rate(1000)  # kecepatan update
        for sph, traj in spheres:
            sph.pos = vp.vector(*traj[step])

    # catat waktu selesai program
    end_time = time.perf_counter()
    print(f"Waktu selesai program: {end_time:.8f} detik")
    print(f"Total waktu eksekusi: {end_time - start_time:.8f} detik")