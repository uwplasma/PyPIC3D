import jax.numpy as jnp

################### Spatial Parameters ##########################
Nx = 200
Ny = 200
Nz = 200

x_wind = 1e-1
y_wind = 1e-1
z_wind = 1e-1
##################################################################

potential_wall = 1e-1

Ey = jnp.zeros((Nx, Ny, Nz))
Ez = jnp.zeros((Nx, Ny, Nz))
# no potential wall along y or z
Ex = jnp.zeros((Nx, Ny, Nz))
# initialize as zeros

xmin = -0.05e-1
xmax = 0.05e-1

ximin = int( (xmin + 0.5*x_wind) / x_wind )
ximax = int( (xmax + 0.5*x_wind) / x_wind )

Ex = Ex.at[ximin:ximax, :, :] .set(potential_wall)


jnp.save('barrier_x.npy', Ex)
jnp.save('barrier_y.npy', Ey)
jnp.save('barrier_z.npy', Ez)