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

from plotting import plot_fields, plot_positions
from particle import initial_particles, update_position
from efield import solve_poisson, laplacian, compute_rho
from bfield import boris, curlx, curly, curlz, update_B
# import code from other files


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
x_wind = 0.5e-2
y_wind = 0.5e-2
z_wind = 0.5e-2
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

plot_freq = 3
# plot data every 5 timesteps

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

average_rho            = []
average_poisson        = []
average_E              = []
average_electron_Force = []
average_ion_Force      = []
average_e_update       = []
average_ion_update     = []
average_plot           = []
# create lists for average times

M = None
# set poisson solver precondition to None


for t in range(20):
    if t % plot_freq == 0:
    ############## PLOTTING ###################################################################   
        start = time.time()
        particlesx = jnp.concatenate([electron_x, ion_x])
        particlesy = jnp.concatenate([electron_y, ion_y])
        particlesz = jnp.concatenate([electron_z, ion_z])
        plot_positions( particlesx, particlesy, particlesz, t, x_wind, y_wind, z_wind)
        plot_fields(Ex, Ey, Ez, t, "E", dx, dy, dz)
        plot_fields(Bx, By, Bz, t, "B", dx, dy, dz)
        end  = time.time()
        print(f'Time Spent on Plotting: {end-start} s')
        average_plot.append(end-start)
    # plot the particles and save as png file

    ############### SOLVE E FIELD ######################################################################################
    print(f'Time: {t*dt} s')
    print("Solving Electric Field...")
    start = time.time()
    rho = jax.numpy.zeros(shape = (Nx, Ny, Nz))
    rho    = compute_rho(rho, electron_x, electron_y, electron_z, ion_x, ion_y, ion_z, dx, dy, dz, q_i, q_e)
    end   = time.time()
    print(f"Time Spent on Rho: {end-start} s")
    average_rho.append(end-start)
    #print( f'Max Value of Rho: {jnp.max(rho)}' )
    # compute the charge density of the plasma
    start = time.time()
    phi = solve_poisson(rho, eps, dx, dy, dz, M)
    end   = time.time()
    #print(f"Time Spent on Phi: {end-start} s")
    average_poisson.append(end-start)
    #print( f'Max Value of Phi: {jnp.max(phi)}' )
    print( f'Max Laplacian of Phi: {jnp.max(laplacian(phi, dx, dy, dz))}')
    print( f'Max Charge Density: {jnp.max(rho)}' )
    print( f'Poisson Error: {jnp.max( laplacian(phi, dx, dy, dz) - (1/eps)*rho )}' )
    # Use conjugated gradients to calculate the electric potential from the charge density

    start = time.time()
    E_fields = jnp.gradient(phi)
    Ex       = -1 * E_fields[0]
    Ey       = -1 * E_fields[1]
    Ez       = -1 * E_fields[2]
    end = time.time()
    #print(f'Time Spent Calculating E: {end-start} s')
    average_E.append(end-start)
    # Calculate the E field using the gradient of the potential

    ################ MAGNETIC FIELD UPDATE #######################################################################
    Bx, By, Bz = update_B(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt)

    ############### UPDATE ELECTRONS ##########################################################################################
    print("Updating Electrons...")
    start = time.time()
    ev_x, ev_y, ev_z = boris(q_e, Ex, Ey, Ez, Bx, By, Bz, electron_x, \
                             electron_y, electron_z, ev_x, ev_y, ev_z, dt, me)
    # implement the boris push algorithm to solve for new electron velocities

    electron_x, electron_y, electron_z = update_position(electron_x, electron_y, electron_z, ev_x, ev_y, ev_z, dt)
    # Update the positions of the particles
    end   = time.time()
    print(f'Time Spent on Updating Electrons: {end-start} s')
    average_e_update.append(end-start)

    ############### UPDATE IONS ################################################################################################
    print("Updating Ions...")

    start = time.time()
    iv_x, iv_y, iv_z = boris(q_i, Ex, Ey, Ez, Bx, By, Bz, ion_x, \
                             ion_y, ion_z, iv_x, iv_y, iv_z, dt, mi)
    # use boris push for ion velocities
    ion_x, ion_y, ion_z  = update_position(ion_x, ion_y, ion_z, iv_x, iv_y, iv_z, dt)
    end   = time.time()
    print(f"Time Spent on Updating Ions: {end-start} s")
    average_ion_update.append(end-start)
    # Update the positions of the particles


print(f"Average Rho: {np.mean(average_rho[1:])} s")
print(f"Average Poisson: {np.mean(average_poisson[1:])} s")
print(f"Average E: {np.mean(average_E[1:])} s")
print(f"Average Electron Update: {np.mean(average_e_update[1:])} s")
print(f"Average Ion Update: {np.mean(average_ion_update[1:])} s")
print(f"Average Plotting: {np.mean(average_plot[1:])} s")

totaltime = np.mean(average_rho[1:]) + np.mean(average_poisson[1:]) + np.mean(average_E[1:]) + np.mean(average_e_update[1:]) + np.mean(average_ion_update[1:]) + np.mean(average_plot[1:])
print(f'Average Time Per Step {totaltime} s')

totaljittime = np.mean(average_rho[0]) + np.mean(average_poisson[0]) + np.mean(average_E[0]) + np.mean(average_e_update[0]) + np.mean(average_ion_update[0]) + np.mean(average_plot[0])
print(f'JIT Compile Time: {totaljittime} s')