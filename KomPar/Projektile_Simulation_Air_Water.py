import vpython as vp
import numpy as np

def acceleration(x, v, params):
    mass = params['mass']
    g = params['g']
    Cd = params['Cd']
    area = params['area']
    volume = params['volume']
    
    v_mag = np.linalg.norm(v)
    
    if x[1] < 0:
        # Jika di bawah permukaan air (y < 0): gunakan gaya di dalam air
        water_density = params['water_density']
        current_force = params['current_force']
        #gravitasi
        F_grav = np.array([0, -mass * g, 0])
        #buoyancy
        F_buoy = np.array([0, water_density * volume * g, 0])
        #drag di air
        if v_mag > 0:
            F_drag = -0.5 * water_density * Cd * area * v_mag * v
        else:
            # Jika kecepatan nol, drag diabaikan
            F_drag = np.zeros(3)
        #Gaya total
        F_net = F_grav + F_buoy + F_drag + current_force
        # F_net = F_grav + F_buoy + current_force
    else:
        #gaya di udara
        air_density = params['air_density']
        #gravitasi
        F_grav = np.array([0, -mass * g, 0])
        #drag di udara
        if v_mag > 0:
            F_drag = -0.5 * air_density * Cd * area * v_mag * v
        else:
            # Jika kecepatan nol, lupakan drag
            F_drag = np.zeros(3)
        
        F_net = F_grav + F_drag
        
    #a
    return F_net / mass


def rk4_step(x, v, dt, params):
    k1_x = dt * v
    k1_v = dt * acceleration(x, v, params)
    
    k2_x = dt * (v + 0.5 * k1_v)
    k2_v = dt * acceleration(x + 0.5 * k1_x, v + 0.5 * k1_v, params)
    
    k3_x = dt * (v + 0.5 * k2_v)
    k3_v = dt * acceleration(x + 0.5 * k2_x, v + 0.5 * k2_v, params)
    
    k4_x = dt * (v + k3_v)
    k4_v = dt * acceleration(x + k3_x, v + k3_v, params)
    
    x_next = x + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6.0
    v_next = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6.0
    
    return x_next, v_next


# Parameter 
g = 9.8                             # Percepatan gravitasi (m/s^2)
water_density = 1000                # Densitas air (kg/m^3)
r = 0.1                               # Radius proyektil bola (m)
volume = (4.0/3.0)*np.pi * r**3     # Volume bola (m^3)
area = np.pi * r**2                 # Luas penampang bola (m^2)
Cd = 0.47                           # Koefisien drag untuk bola
air_density = 1.225                 # Densitas udara (kg/m^3)



#USER INPUT
print("\nSIMULASI PERGERAKAN PROYEKTIL DI UDARA DAN DI DALAM AIR")
print("-------------------------------------------------------")
print("===> Informasi awal :")
print("Densitas air: %.0f kg/m^3" % water_density)
print("Densitas udara: %.3f kg/m^3" % air_density)
print("Percepatan gravitasi: %.1f m/s^2" % g)
print("Radius proyektil: %.0f m" % r)
print("Volume proyektil: %.0f m^3" % volume)
print("Luas penampang proyektil: %.0f m^2"% area)
print("Koefisien drag proyektil: %.2f \n" % Cd)


print("===> Masukkan parameter untuk proyektil:")
# initial_power = float(input("Gaya awal proyektil (Newton): "))
initial_power = 5000.0  # Gaya awal proyektil (N)
# elev_deg = float(input("Sudut elevasi (atas-bawah [0 ke depan, 90 ke atas]): "))
elev_deg = 45
# azimuth_deg = float(input("Sudut azimuth (kiri-kanan [90 ke kanan, 270 ke kiri]) dari belakang proyektil: "))
azimuth_deg = 0.0
# launch_height = float(input("Tinggi peluncuran (meter): "))
launch_height = 0.0 
# mass = float(input("Massa proyektil (kg): "))
mass=500
density_proj = mass / volume
print("Densitas proyektil: %.2f kg/m^3" % density_proj)


# print("\n===> Masukkan parameter untuk gaya arus air: ")
# current_magnitude = float(input("Magnitudo gaya arus air (Newton): "))
# current_elev_deg = float(input("Sudut elevasi arus air : "))
# current_azimuth_deg = float(input("Sudut azimuth arus air : "))
current_magnitude = 0.0
current_elev_deg = 0.0
current_azimuth_deg = 0.0



# deg to rad
elev_rad = np.radians(elev_deg)
azimuth_rad = np.radians(azimuth_deg)
current_elev_rad = np.radians(current_elev_deg)
current_azimuth_rad = np.radians(current_azimuth_deg)

# Hitung kecepatan awal (v0 = Newton power / mass) dan proyeksikan ke tiga sumbu
v0_magnitude = (initial_power*1) / mass #1 sebagai konstanta waktu bawah gaya yang diberikan selama 1 detik
# v0_magnitude = initial_power
#gaya awal proyektil sebagai vektor
v0 = np.array([
    v0_magnitude * np.cos(elev_rad) * np.cos(azimuth_rad),
    v0_magnitude * np.sin(elev_rad),  
    v0_magnitude * np.cos(elev_rad) * np.sin(azimuth_rad)
])
# gaya arus air konstan sebagai vektor
F_current = np.array([
    current_magnitude * np.cos(current_elev_rad) * np.cos(current_azimuth_rad),
    current_magnitude * np.sin(current_elev_rad),
    current_magnitude * np.cos(current_elev_rad) * np.sin(current_azimuth_rad)
])

#tinggi proyektil awal
x0 = np.array([0.0, launch_height, 0.0])
# Peringatan densitas proyektil lebih kecil dari air
if density_proj < water_density:
    print("WARNING: Densitas proyektil (%.2f kg/m^3) lebih kecil dari densitas air (%.0f kg/m^3)." % (density_proj, water_density))
    print("Saat masuk air, buoyancy akan sangat dominan jadi benda akan mengapung")

params = {
    'mass': mass,
    'g': g,
    'water_density': water_density,
    'volume': volume,
    'area': area,
    'Cd': Cd,
    'current_force': F_current,
    'air_density': air_density
}

# Setup visualisasi environment
scene = vp.canvas(title="Simulasi Pergerakan Proyektil", 
                  width=1920, 
                  height=1080, 
                  background=vp.color.white)
water_box = vp.box(pos=vp.vector(0, -50, 0), 
                   size=vp.vector(50, 100, 50), 
                   color=vp.color.blue, 
                   opacity=0.2)
water_surface = vp.box(pos=vp.vector(0, 0, 0), 
                       size=vp.vector(50, 0.2, 50), 
                       color=vp.color.blue, opacity=0.5)
proj = vp.sphere(pos=vp.vector(*x0), 
                 radius=r, 
                 color=vp.color.red, 
                 make_trail=True, 
                 trail_color=vp.color.white)
velocity_arrow = vp.arrow(pos=proj.pos,
                          axis=vp.vector(*(v0/np.linalg.norm(v0)*r*2)),
                          color=vp.color.yellow)

#sumbu pada titik awal dan proyektil
axis_length = 2.0
x_axis_cons = vp.arrow(pos=vp.vector(*x0),
                  axis=vp.vector(axis_length, 0, 0),
                  color=vp.color.red,
                  shaftwidth=0.05)
y_axis_cons = vp.arrow(pos=vp.vector(*x0),
                  axis=vp.vector(0, axis_length, 0),
                  color=vp.color.green,
                  shaftwidth=0.05)
z_axis_cons = vp.arrow(pos=vp.vector(*x0),
                  axis=vp.vector(0, 0, axis_length),
                  color=vp.color.blue,
                  shaftwidth=0.05)
x_axis = vp.arrow(pos=proj.pos,
                  axis=vp.vector(axis_length, 0, 0),
                  color=vp.color.red,
                  shaftwidth=0.05)
y_axis = vp.arrow(pos=proj.pos,
                  axis=vp.vector(0, axis_length, 0),
                  color=vp.color.green,
                  shaftwidth=0.05)
z_axis = vp.arrow(pos=proj.pos,
                  axis=vp.vector(0, 0, axis_length),
                  color=vp.color.blue,
                  shaftwidth=0.05)

#initial conditions
x = x0
v = v0
dt = 0.0001  # Time step
t = 0.0

# MAIN LOOP
maks_height = 0
max_reach = 0
kesentuh= False
t_undawatar = 0.0
jarak_underwater = 0.0

while True:
    vp.rate(10000)
    
    # Animasi osilasi permukaan air (kalo mau)
    # water_surface.pos.y = 0.05 * np.sin(2 * t)
    
    #RK 4 untuk posisi dan kecepatan
    x, v = rk4_step(x, v, dt, params)
    
    proj.pos = vp.vector(*x)
    if np.linalg.norm(v) > 0:
        velocity_arrow.axis = vp.vector(*(v/np.linalg.norm(v)*r*2))
    velocity_arrow.pos = proj.pos
    
    # Update sumbu supaya selalu dipusat proyektil
    x_axis.pos = proj.pos
    y_axis.pos = proj.pos
    z_axis.pos = proj.pos
