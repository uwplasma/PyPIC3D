import numpy as np

x_wind = 1
y_wind = 1
nx = 400
ny = 400
dx = x_wind / nx
dy = y_wind / ny
# spatial resolution

x = np.arange(-x_wind / 2, x_wind / 2, dx)
y = np.arange(-y_wind / 2, y_wind / 2, dy)
# X, Y = np.meshgrid(x, y)
# define the grid


#### YEE GRID ####
x_ = np.arange(-x_wind / 2 + dx/2, x_wind / 2 + dx/2, dx)
y_ = np.arange(-y_wind / 2 + dy/2, y_wind / 2 + dy/2, dy)
# define the staggered grid for the Yee solver (shifted by one half cell size to account for ghost cells)

Bx_X, Bx_Y = np.meshgrid(x_, y)
By_X, By_Y = np.meshgrid(x, y_)
# create the staggered grids for the magnetic field components

B0 = 0.01 #1/ np.sqrt(4 * np.pi)
# magnetic field strength

C  = 2.9e8
V0 = 0.3*C
# velocity magnitude

Bx = -B0 * np.sin(2 * np.pi * Bx_Y / y_wind)
By = B0 * np.sin(4 * np.pi * By_X / x_wind)
# components of the magnetic field

N_particles = 16000000
N_ions      = 16000000
# number of particles

electron_x = np.random.uniform(-x_wind / 2, x_wind / 2, N_particles)
electron_y = np.random.uniform(-y_wind / 2, y_wind / 2, N_particles)
electron_z = np.zeros(N_particles)
# initial positions of electrons

ion_x = np.random.uniform(-x_wind / 2, x_wind / 2, N_ions)
ion_y = np.random.uniform(-y_wind / 2, y_wind / 2, N_ions)
ion_z = np.zeros(N_ions)
# initial positions of ions

electron_vx = -V0 *  np.sin(2* np.pi * (electron_y + y_wind/2) / y_wind)
electron_vy = V0  *  np.sin(2* np.pi * (electron_x + x_wind/2) / x_wind)
electron_vz = np.zeros(N_particles)
# initial velocities of electrons

ion_vx = -V0 *  np.sin(2* np.pi * (ion_y + y_wind/2) / y_wind)
ion_vy = V0 *  np.sin(2* np.pi * (ion_x + x_wind/2) / x_wind)
ion_vz = np.zeros(N_ions)
# initial velocities of electrons and ions

Bx = np.expand_dims(Bx, axis=-1)
By = np.expand_dims(By, axis=-1)
# Expand dimensions to match fields shape

np.save('Bx.npy', Bx)
np.save('By.npy', By)
np.save('electron_x.npy', electron_x)
np.save('electron_y.npy', electron_y)
np.save('electron_z.npy', electron_z)
np.save('ion_x.npy', ion_x)
np.save('ion_y.npy', ion_y)
np.save('ion_z.npy', ion_z)
np.save('electron_vx.npy', electron_vx)
np.save('electron_vy.npy', electron_vy)
np.save('ion_vx.npy', ion_vx)
np.save('ion_vy.npy', ion_vy)
np.save('electron_vz.npy', electron_vz)
np.save('ion_vz.npy', ion_vz)
# save npy arrays
