from multiprocessing import Pool
import vpython as vp
import numpy as np
import time
import matplotlib.pyplot as plt # Import Matplotlib

# Fungsi simulasi SEBUAH proyektil supaya dapat dipanggil secara paralel
def simulate_projectile(args):
    mass, v0, params, dt, t_max = args
    def acceleration(x, v):
        g   = params['g']
        Cd  = params['Cd']
        area= params['area']
        vol = params['volume']
        v_mag = np.linalg.norm(v)

        if x[1] < 0:  # di dalam air (y < 0)
            rho = params['water_density']
            F_grav  = np.array([0, -mass*g, 0])
            F_buoy  = np.array([0, rho*vol*g, 0])
            if v_mag > 0:
                F_drag  = -0.5 * rho * Cd * area * v_mag * v
            else:
                F_drag  = np.zeros(3)
            F_curr  = params['current_force']
            F_net   = F_grav + F_buoy + F_drag + F_curr
        else:        # di udara (y >= 0)
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
        # Hentikan simulasi jika proyektil keluar dari area relevan (misal jatuh terlalu jauh)
        if x[1] < -100 and i > 0: # Jika jatuh terlalu dalam di bawah air dan sudah bergerak
            traj = traj[:i] # Potong trajectory sampai titik ini
            break
        x, v = rk4_step(x, v)
    return traj


# MAIN PROGRAM
if __name__ == '__main__':
    # Catat waktu mulai program
    start_time = time.perf_counter()
    print(f"Waktu mulai program: {start_time:.8f} detik")

    # Parameter global
    g               = 9.8
    water_density   = 1000
    air_density     = 1.225
    Cd              = 0.47
    r               = 0.1
    volume          = (4/3)*np.pi*r**3
    area            = np.pi*r**2
    
    # Parameter arus air
    current_force_magnitude = 5000
    current_azim_deg = 45  # sudut arah arus di bidang x-z
    azim_rad = np.radians(current_azim_deg)
    # Vektor arah arus
    current_dir = np.array([np.cos(azim_rad), 0.0, np.sin(azim_rad)])
    current_force = current_force_magnitude * current_dir
    
    # === PARAMETER UNTUK PERBANDINGAN DT ===
    dt_values = [ 2, 1, 0.5, 0.001]
    t_max     = 7.0
    # Kita hanya akan mensimulasikan SATU proyektil untuk perbandingan dt agar visualisasi lebih jelas
    mass      = 100 
    elev_rad  = np.radians(45)
    azim_rad  = np.radians(0)
    power     = 5000
    v0_mag    = power / mass
    v0 = np.array([
        v0_mag * np.cos(elev_rad) * np.cos(azim_rad),
        v0_mag * np.sin(elev_rad),
        v0_mag * np.cos(elev_rad) * np.sin(azim_rad)
    ])
    # =======================================

    all_results_by_dt = {}

    # Lakukan simulasi untuk setiap nilai dt
    for dt_val in dt_values:
        print(f"\nSimulasi dengan dt = {dt_val}")
        
        # Parameter spesifik untuk simulasi ini
        params = {
            'g': g,
            'water_density': water_density,
            'air_density'  : air_density,
            'volume'       : volume,
            'area'         : area,
            'Cd'           : Cd,
            'current_force': current_force
        }
        
        # Siapkan argumen untuk fungsi simulate_projectile
        args_for_current_dt = [(mass, v0, params, dt_val, t_max)]

        # Pool untuk paralelisasi
        with Pool() as pool:
            # Karena hanya 1 proyektil, all_trajectories akan berisi satu trajectory
            current_dt_trajectory = pool.map(simulate_projectile, args_for_current_dt)[0] 
            all_results_by_dt[dt_val] = current_dt_trajectory
            print(f"Jumlah langkah simulasi untuk dt={dt_val}: {len(current_dt_trajectory)}")


    # --- Bagian Visualisasi VPython (3D) ---
    direction_arrow_scale = 5
    scene = vp.canvas(title="Simulasi Proyektil (Perbandingan dt - 3D)", width=1200, height=800, background=vp.color.white)
    
    # Lantai dan permukaan air
    vp.box(pos=vp.vector(0, -50, 0), size=vp.vector(200, 100, 200), color=vp.color.blue, opacity=0.2) # Area bawah air
    vp.box(pos=vp.vector(0, 0, 0), size=vp.vector(200, 0.2, 200), color=vp.color.blue, opacity=0.5) # Permukaan air

    # Panah arus air (opsional, bisa dihilangkan jika terlalu ramai)
    arrow_spacing = 30
    for x_arr in np.arange(-90, 100, arrow_spacing):
        for z_arr in np.arange(-90, 100, arrow_spacing):
            vp.arrow(pos=vp.vector(x_arr, 0.5, z_arr), axis=vp.vector(*current_dir) * direction_arrow_scale,
                    shaftwidth=0.3, color=vp.color.cyan, opacity=0.7)

    # Buat sphere dan simpan state trajektori
    spheres_data = []
    colors_vpython = [vp.color.red, vp.color.green, vp.color.blue, vp.color.orange, vp.color.purple, vp.color.black]
    color_index = 0

    for dt_val, traj in all_results_by_dt.items():
        current_color_vpython = colors_vpython[color_index % len(colors_vpython)] 
        color_index += 1

        # Membuat label untuk dt
        label_text = f"dt = {dt_val}"
        label = vp.label(pos=vp.vector(*traj[0]), text=label_text, xoffset=10, yoffset=20,
                         color=current_color_vpython, opacity=0, box=True, height=10) 

        # Membuat sphere dan trail
        sph = vp.sphere(pos=vp.vector(*traj[0]), radius=r, color=current_color_vpython, make_trail=True, retain=2000)
        
        spheres_data.append({'sphere': sph, 'trajectory': traj, 'label': label, 'dt_value': dt_val}) # Simpan dt_value juga

    # Animasi VPython
    max_steps = max(len(data['trajectory']) for data in spheres_data)
    animation_rate = 500

    print("\nMemulai animasi VPython...")
    for step in range(max_steps):
        vp.rate(animation_rate)
        for data in spheres_data:
            sph = data['sphere']
            traj = data['trajectory']
            label = data['label']

            if step < len(traj):
                new_pos = vp.vector(*traj[step])
                sph.pos = new_pos
                label.pos = new_pos + vp.vector(0, r + 0.5, 0) # Sesuaikan y-offset
                label.text = f"dt = {data['dt_value']}"
                label.opacity = 1
            else:
                sph.pos = vp.vector(*traj[-1])
                label.pos = vp.vector(*traj[-1]) + vp.vector(0, r + 0.5, 0)
                label.text = f"dt = {data['dt_value']}"


    # --- Bagian Visualisasi Matplotlib (2D) ---
    print("\nMembuat plot Matplotlib...")
    plt.figure(figsize=(12, 8)) # Ukuran figure Matplotlib
    plt.title('Perbandingan Lintasan Proyektil dengan Berbagai Nilai dt')
    plt.xlabel('Jarak Horizontal (m)')
    plt.ylabel('Ketinggian (m)')
    plt.grid(True)
    plt.axhline(0, color='blue', linestyle='--', linewidth=0.8, label='Permukaan Air') # Permukaan air

    colors_matplotlib = ['red', 'green', 'blue', 'orange', 'purple', 'black'] # Warna untuk Matplotlib
    color_idx_mpl = 0

    for dt_val, traj in all_results_by_dt.items():
        x_coords = traj[:, 0] # Ambil koordinat x (arah ke depan)
        y_coords = traj[:, 1] # Ambil koordinat y (ketinggian)
        z_coords = traj[:, 2] # Ambil koordinat z (arah samping)
        
        # Untuk plot 2D, kita bisa menggunakan jarak horizontal total (sqrt(x^2 + z^2))
        # atau hanya komponen x jika azimut tidak berubah banyak.
        # Mari kita pakai sqrt(x^2 + z^2) agar lebih umum untuk lintasan 3D yang diproyeksikan ke 2D.
        horizontal_distance = np.sqrt(x_coords**2 + z_coords**2)

        plt.plot(horizontal_distance, y_coords, 
                 label=f'dt = {dt_val}', 
                 color=colors_matplotlib[color_idx_mpl % len(colors_matplotlib)])
        
        # Tambahkan titik akhir lintasan untuk setiap dt
        plt.plot(horizontal_distance[-1], y_coords[-1], 'o', color=colors_matplotlib[color_idx_mpl % len(colors_matplotlib)])

        color_idx_mpl += 1

    plt.legend()
    plt.show() # Tampilkan plot Matplotlib

    # Catat waktu selesai program
    end_time = time.perf_counter()
    print(f"\nWaktu selesai program: {end_time:.8f} detik")
    print(f"Total waktu eksekusi: {end_time - start_time:.8f} detik")