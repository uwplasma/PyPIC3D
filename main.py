# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities

import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
# Importing relevant libraries

from plotting import plot_fields, plot_positions, plot_rho, cond_decorator
from particle import initial_particles, update_position
from efield import solve_poisson, laplacian, update_rho, compute_pe
from bfield import boris, curlx, curly, curlz, update_B
# import code from other files

jax.config.update('jax_platform_name', 'cpu')
# set Jax to use CPUs

############################ SETTINGS #####################################################################
save_data = True
plotfields = True
plotpositions = True
# booleans for plotting/saving data

benchmark = False
# still need to implement benchmarking

verbose   = True
# booleans for debugging

############################ INITIALIZE EVERYTHING #######################################################
# I am starting by simulating a hydrogen plasma
print("Initializing Simulation...")

eps = 8.854e-12
# permitivity of freespace
C = 3e8 # m/s
# Speed of light
eps = 8.854e-12
# permitivity of freespace
C = 3e8 # m/s
# Speed of light
kb = 1.380649e-23 # J/K
# Boltzmann's constant
me = 9.1093837e-31 # Kg
# mass of the electron
mi = 1.67e-23 # Kg
# mass of the ion
q_e = 1.602e-19
# charge of electron
q_i = -1.602e-19
# charge of ion
Te = 100 # K
# electron temperature
Ti = 100 # K
# ion temperature
# assuming an isothermal plasma for now

N_electrons = 100
N_ions      = 100
# specify the number of electrons and ions in the plasma

Nx = 50
Ny = 50
Nz = 50
# specify the number of array spacings in x, y, and z
x_wind = 1e-2
y_wind = 1e-2
z_wind = 1e-2
# specify the size of the spatial window in meters

dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
# compute the spatial resolution
print(f'Dx: {dx}')
print(f'Dy: {dy}')
print(f'Dz: {dz}')

################ Courant Condition #############################################################################
courant_number = 1
dt = courant_number / (  C * ( (1/dx) + (1/dy) + (1/dz) )   )
# calculate spatial resolution using courant condition

t_wind = 1e-9
# time window for simultion
Nt     = int( t_wind / dt )
# Nt for resolution


print(f'time window: {t_wind}')
print(f'Nt:          {Nt}')
print(f'dt:          {dt}')

plot_freq = 1
# how often to plot the data

Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
# initialize the electric field arrays as 0
Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
# initialize the magnetic field arrays as 0

key1 = random.key(4353)
key2 = random.key(1043)
electron_x, electron_y, electron_z, ev_x, ev_y, ev_z  = initial_particles(N_electrons, x_wind, y_wind, z_wind, me, Te, kb, key1)
ion_x, ion_y, ion_z, iv_x, iv_y, iv_z                 = initial_particles(N_ions, x_wind, y_wind, z_wind, mi, Ti, kb, key2)
# initialize the positions and velocities of the electrons and ions in the plasma.
# eventually, I need to update the initialization to use a more accurate position and velocity distribution.

M = None
# set poisson solver precondition to None


for t in range(30):
    rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # reset value of charge density

    if t % plot_freq == 0:
    ############## PLOTTING ###################################################################
        if plotpositions:
            particlesx = jnp.concatenate([electron_x, ion_x])
            particlesy = jnp.concatenate([electron_y, ion_y])
            particlesz = jnp.concatenate([electron_z, ion_z])
            plot_positions( particlesx, particlesy, particlesz, t, x_wind, y_wind, z_wind)
        if plotfields:
            plot_fields(Ex, Ey, Ez, t, "E", dx, dy, dz)
            plot_fields(Bx, By, Bz, t, "B", dx, dy, dz)
            plot_rho(rho, t, "rho", dx, dy, dz)
    # plot the particles and save as png file

    ############### SOLVE E FIELD ######################################################################################
    print(f'Time: {t*dt} s')
    rho = update_rho(N_electrons, electron_x, electron_y, electron_z, dx, dy, dz, q_e, rho)
    rho = update_rho(N_ions, ion_x, ion_y, ion_z, dx, dy, dz, q_i, rho)
    # update the charge density field

    if verbose: print(f"Calculating Charge Density, Max Value: {jnp.max(rho)}")
    # print the maximum value of the charge density

    if t == 0:
        phi = solve_poisson(rho, eps, dx, dy, dz, phi=rho, M=None)
    else:
        phi = solve_poisson(rho, eps, dx, dy, dz, phi=phi, M=M)

    if verbose: print(f"Calculating Electric Potential, Max Value: {jnp.max(phi)}")
    # print the maximum value of the electric potential
    if verbose: print( f'Poisson Relative Percent Difference: {compute_pe(phi, rho, eps, dx, dy, dz)}%')
    # Use conjugated gradients to calculate the electric potential from the charge density

    E_fields = jnp.gradient(phi)
    Ex       = -1 * E_fields[0]
    Ey       = -1 * E_fields[1]
    Ez       = -1 * E_fields[2]
    # Calculate the E field using the gradient of the potential

    if verbose: print(f"Calculating Electric Field, Max Value: {jnp.max(Ex)}")
    # print the maximum value of the electric field

    ############### UPDATE ELECTRONS ##########################################################################################
    ev_x, ev_y, ev_z = boris(q_e, Ex, Ey, Ez, Bx, By, Bz, electron_x, \
                             electron_y, electron_z, ev_x, ev_y, ev_z, dt, me)
    # implement the boris push algorithm to solve for new electron velocities

    if verbose: print(f"Calculating Electron Velocities, Max Value: {jnp.max(ev_x)}")
    # print the maximum value of the electron velocities

    electron_x, electron_y, electron_z = update_position(electron_x, electron_y, electron_z, ev_x, ev_y, ev_z, dt)
    # Update the positions of the particles

    if verbose: print(f"Calculating Electron Positions, Max Value: {jnp.max(electron_x)}")
    # print the maximum value of the electron positions

    ############### UPDATE IONS ################################################################################################
    iv_x, iv_y, iv_z = boris(q_i, Ex, Ey, Ez, Bx, By, Bz, ion_x, \
                             ion_y, ion_z, iv_x, iv_y, iv_z, dt, mi)
    # use boris push for ion velocities

    if verbose: print(f"Calculating Ion Velocities, Max Value: {jnp.max(iv_x)}")
    # print the maximum value of the ion velocities

    ion_x, ion_y, ion_z  = update_position(ion_x, ion_y, ion_z, iv_x, iv_y, iv_z, dt)
    # Update the positions of the particles
    if verbose: print(f"Calculating Ion Positions, Max Value: {jnp.max(ion_x)}")
    # print the maximum value of the ion positions

    ################ MAGNETIC FIELD UPDATE #######################################################################
    Bx, By, Bz = update_B(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt)
    # update the magnetic field using the curl of the electric field
    if verbose: print(f"Calculating Magnetic Field, Max Value: {jnp.max(Bx)}")
    # print the maximum value of the magnetic field

    if save_data:
        if t % plot_freq == 0:
            jnp.save(f'data/rho/rho_{t:}', rho)
            jnp.save(f'data/phi/phi_{t:}', phi)
    # save the data for the charge density and potential