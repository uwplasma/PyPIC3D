import numpy as np

kb = 1.380649e-23
mu = 1.25663706e-6
eps = 8.854e-12
me  = 9.10938356e-31
mi  = 1 * me   #1.626e-27
c  = 2.99792458e8
q  = 1.602e-19
# fundamental constants

vth_e = 0.05 * c
vth_i = vth_e
# calculate the thermal velocities

N_particles = 100000
N_background = int(0.3 * N_particles)
# number of particles

n0 = 1e22
nb = 0.3 * n0
# background density and peak density

wpi = q * np.sqrt( n0 / mi / eps)
# ion plasma frequency

di = c / wpi
# ion inertial length

x_wind = 15 * di
z_wind = 15 * di
# dimensions of the simulation box

weight = n0 / N_particles * x_wind * z_wind
# weight of each macroparticle

debye_length = np.sqrt( eps * me * vth_e**2 / n0 / q**2 )
# debye length of electrons

nx = 500
nz = 500
# number of grid points

dx = x_wind / nx
dz = z_wind / nz
# spatial resolution

ds_per_debye = debye_length / dx
# number of grid points per debye length

x = np.linspace(-x_wind/2, x_wind/2, nx, dtype=float)
z = np.linspace(-z_wind/2, z_wind/2, nz, dtype=float)
X, Z = np.meshgrid(x, z, indexing='ij')
# define the grid

lam = 0.5*di
# wavelength of the current sheet

B0 = 0.2
# magnetic field strength

Bx = B0 * np.tanh(  Z / lam  )
Bz = np.zeros_like(Bx)
# components of the magnetic field

psi_0 = 100 * B0
print(f"psi_0: {psi_0}")
print(f"psi_0/B0: {psi_0/B0}")
Bx += psi_0 * np.cos(2 * np.pi * X / x_wind, dtype=float) * np.sin(np.pi * Z / z_wind, dtype=float)
Bz += -psi_0 * np.sin(2 * np.pi * X / x_wind, dtype=float) * np.cos(np.pi * Z / z_wind, dtype=float)
# perturbation to the magnetic field

electron_x = np.random.uniform(-x_wind / 2, x_wind / 2, N_particles)
electron_y = np.zeros_like(electron_x, dtype=float)
electron_z = 1 / np.cosh(np.linspace(-z_wind / 2, z_wind / 2, N_particles, dtype=float) / lam )**2
# initial positions of electrons

ion_x = np.random.uniform(-x_wind / 2, x_wind / 2, N_particles)
ion_y = np.zeros_like(ion_x)
ion_z = 1 / np.cosh(np.linspace(-z_wind / 2, z_wind / 2, N_particles, dtype=float) / lam )**2
# initial positions of ions

ev_x = np.random.normal(0, vth_e, N_particles)
ev_y = np.random.normal(0, vth_e, N_particles)
ev_z = np.random.normal(0, vth_e, N_particles)
# maxwellian for electron velocities

iv_x = np.random.normal(0, vth_i, N_particles)
iv_y = np.random.normal(0, vth_i, N_particles)
iv_z = np.random.normal(0, vth_i, N_particles)
# maxwellian for ion velocities

def U(z):
    u0 = -c*B0/(4*np.pi*q*lam)
    zbar = z / lam
    return u0 * 1 / np.cosh( zbar )**2 / ( n0 * np.cosh(zbar )**2 + nb )
# function to calculate the drift velocity

ev_y += U(electron_z)
# add the drift velocity to the electron y-velocity

Bx = np.expand_dims(Bx, axis=1)
Bz = np.expand_dims(Bz, axis=1)
# Expand dimensions to match fields shape

np.save('Bx.npy', Bx)
np.save('Bz.npy', Bz)
np.save('electron_x.npy', electron_x)
np.save('electron_y.npy', electron_y)
np.save('electron_z.npy', electron_z)
np.save('ion_x.npy', ion_x)
np.save('ion_y.npy', ion_y)
np.save('ion_z.npy', ion_z)
np.save('electron_vx.npy', ev_x)
np.save('electron_vy.npy', ev_y)
np.save('electron_vz.npy', ev_z)
np.save('ion_vx.npy', iv_x)
np.save('ion_vy.npy', iv_y)
np.save('ion_vz.npy', iv_z)
# save npy arrays

print("Initial conditions generated and saved.")
print(f"Number of particles: {N_particles}")
print(f"Grid size: {nx} x {nz}")
print(f"Box dimensions: {x_wind} m x {z_wind} m")
print(f"Ion inertial length: {di} m")
print(f"Magnetic field strength: {B0} T")
print(f"me: {me}, mi: {mi}")
print(f"Thermal velocities: vth_e = {vth_e} m/s, vth_i = {vth_i} m/s")
print(f"Thermal velocity ratio (vth_i/vth_e): {vth_i/vth_e}")
print(f"Thermal velocity ratio (vth/c): vth_e/c: {vth_e/c}, vth_i/c: {vth_i/c}")
print(f"Peak Particles: {N_particles}")
print(f"Background Particles: {N_background}")
print(f"Weight per particle: {weight} m^-3")
print(f"ds per debye: {ds_per_debye}")
print(f"Average Electron Velocity: {np.mean(np.sqrt( ev_x**2 + ev_y**2 + ev_z**2 ) ) / c} c")
print(f"Average Ion Velocity: {np.mean(np.sqrt( iv_x**2 + iv_y**2 + iv_z**2 ) ) / c} c")
