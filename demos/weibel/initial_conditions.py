import numpy as np
# load external libraries

x_wind = 1
y_wind = 1
nx = 50
ny = 50
dx = x_wind / nx
dy = y_wind / ny
# spatial resolution

x = np.arange(-x_wind / 2, x_wind / 2, dx)
y = np.arange(-y_wind / 2, y_wind / 2, dy)
X, Y = np.meshgrid(x, y)
# define the grid

N_electrons = 20000
N_ions      = 20000
# number of particles

electron_vx = np.random.choice([1e8, -1e8], size=N_electrons) +  np.random.normal(0, 5996000.0, N_electrons)
# initial vx of electrons

electron_z = np.zeros(N_electrons)
# initial positions of electrons

ion_x = np.random.uniform(-x_wind / 2, x_wind / 2, N_ions)
ion_y = np.random.uniform(-y_wind / 2, y_wind / 2, N_ions)
ion_z = np.zeros(N_ions)
# initial positions of ions

np.save('electron_vx.npy', electron_vx)
np.save('electron_z.npy', electron_z)
np.save('ion_x.npy', ion_x)
np.save('ion_y.npy', ion_y)
np.save('ion_z.npy', ion_z)
# save npy arrays