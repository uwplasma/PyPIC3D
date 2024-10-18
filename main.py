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
import equinox as eqx
# Importing relevant libraries

from plotting import plot_fields, plot_positions, plot_rho
from plotting import plot_velocities, plot_velocity_histogram, plot_KE
from plotting import plot_probe, plot_fft, phase_space, multi_phase_space
from particle import initial_particles, update_position, total_KE, total_momentum
from particle import cold_start_init
from fields import boris, update_B, calculateE, initialize_fields, probe
from fields import magnitude_probe,  freq_probe, freq, spectralBsolve, totalfield_energy, spectralEsolve
from fields import update_E
from model import PoissonPrecondition
# import code from other files

jax.config.update('jax_platform_name', 'cpu')
# set Jax to use CPUs

############################# GPUS   #######################################################################
GPUs = False
# booleans for using GPUs to solve Poisson's equation

############################## Neural Network Preconditioner ################################################
NN = False
model_name = "Preconditioner.eqx"
# booleans for using a neural network to precondition Poisson's equation solver

############################ SETTINGS #####################################################################
save_data = False
plotfields = True
plotpositions = False
plotvelocities = False
plotKE = False
plotEnergy = False
plasmaFreq = False
phaseSpace = True
# booleans for plotting/saving data

benchmark = False
# still need to implement benchmarking

verbose   = False
# booleans for debugging

electrostatic = True
cold_start = False
# start all the particles at the same place (rho = 0)

if benchmark: jax.profiler.start_trace("/home/christopherwoolford/Documents/PyPIC3D/tensorboard")
# start the profiler using tensorboard

############################ INITIALIZE EVERYTHING #######################################################
# I am starting by simulating a hydrogen plasma
print("Initializing Simulation...")
start = time.time()

############################# SIMULATION PARAMETERS ########################################################
bc = "spectral"
# boundary conditions: periodic, dirichlet, neumann, spectral

eps = 8.854e-12
# permitivity of freespace
C = 3e8 # m/s
# Speed of light
kb = 1.380649e-23 # J/K
# Boltzmann's constant
me = 9.1093837e-31 # Kg
# mass of the electron
mi = 1.67e-27 # Kg
# mass of the ion
q_e = -1.602e-19
# charge of electron
q_i = 1.602e-19
# charge of ion
Te = 233000 # K
# electron temperature
Ti = 233000 # K
# ion temperature
# assuming an isothermal plasma for now

N_electrons = 5000
N_ions      = 5000
# specify the number of electrons and ions in the plasma

Nx = 30
Ny = 30
Nz = 30
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
dt = dt/10
# dt = courant_number * min(dx, dy, dz) / (C)
# calculate spatial resolution using courant condition

ne = N_electrons / (x_wind*y_wind*z_wind)
theoretical_freq = jnp.sqrt(  ne * q_e**2  / (eps*me)  )
# calculate the expected plasma frequency from analytical theory, w = sqrt( ne^2 / (eps * me) )

print(f"Theoretical Plasma Frequency: {theoretical_freq} Hz")

if theoretical_freq * dt > 2.0:
    print(f"# of Electrons is Low and may introduce numerical stability")
    print(f"In order to correct this, # of Electrons needs to be at least { (2/dt)**2 * (me*eps/q_e**2) } for this spatial resolution")


debye_length = jnp.sqrt( eps * kb * Te / (ne * q_e**2) )
# calculate the debye length of the plasma
print(f"Debye Length: {debye_length} m")

if debye_length < dx:
    print(f"Debye Length is less than the spatial resolution, this may introduce numerical instability")



t_wind = 0.25e-10
# time window for simultion
Nt     = int( t_wind / dt )
# Nt for resolution

print(f'time window: {t_wind}')
print(f'Nt:          {Nt}')
print(f'dt:          {dt}')

plot_freq = 100
# how often to plot the data

Ex, Ey, Ez, Bx, By, Bz, phi, rho = initialize_fields(Nx, Ny, Nz)
# initialize the electric and magnetic fields

key1 = random.key(4353)
key2 = random.key(1043)
key3 = random.key(1234)
key4 = random.key(2345)
key5 = random.key(3456)
# random keys for initializing the particles
if cold_start:
    electron_x, electron_y, electron_z, ev_x, ev_y, ev_z = cold_start_init(0, N_electrons, x_wind, y_wind, z_wind, me, Te, kb, key1, key2, key3)
    ion_x, ion_y, ion_z, iv_x, iv_y, iv_z                 = cold_start_init(0, N_ions, x_wind, y_wind, z_wind, mi, Ti, kb, key3, key4, key5)
else:
    electron_x, electron_y, electron_z, ev_x, ev_y, ev_z  = initial_particles(N_electrons, x_wind, y_wind, z_wind, me, Te, kb, key1, key2, key3)
    ion_x, ion_y, ion_z, iv_x, iv_y, iv_z                 = initial_particles(N_ions, x_wind, y_wind, z_wind, mi, Ti, kb, key3, key4, key5)
# initialize the positions and velocities of the electrons and ions in the plasma.
# eventually, I need to update the initialization to use a more accurate position and velocity distribution.

#################################### Two Stream Instability #####################################################
print(f"Thermal Velocity: {jnp.sqrt(2*kb*Te/me)}")

alternating_ones = (-1)**jnp.array(range(0,N_electrons))
v0=1.5*2657603.0
ev_x = v0*alternating_ones

# ev_x *= ( 1 + 0.1*jnp.sin(6*jnp.pi * electron_x / x_wind) )
# add perturbation to the electron velocities

iv_x = 0
iv_y = 0
iv_z = 0
# initialize the ion velocities to zero
##################################### Neural Network Preconditioner ################################################

key = jax.random.PRNGKey(0)
# define the random key
if NN:
    model = PoissonPrecondition( Nx=Nx, Ny=Ny, Nz=Nz, hidden_dim=3000, key=key)
    # define the model
    model = eqx.tree_deserialise_leaves(model_name, model)
    # load the model from file

#################################### MAIN LOOP #####################################################################


M = None
# set poisson solver precondition to None for now

if plotfields: Eprobe = []
if plotfields: averageE = []
if plotKE:
    KE = []
    KE_time = []
if plasmaFreq: freqs = []
if plotEnergy: total_energy = []
if plotEnergy: total_p      = []

if not electrostatic:
        Ex, Ey, Ez, phi, rho = calculateE(N_electrons, electron_x, electron_y, electron_z, \
            N_ions, ion_x, ion_y, ion_z,                                               \
            dx, dy, dz, q_e, q_i, rho, eps, phi, 0, M, Nx, Ny, Nz, x_wind, y_wind, z_wind, bc, verbose, GPUs)
    
for t in range(Nt):
    print(f'Iteration {t}, Time: {t*dt} s')

    ############### SOLVE E FIELD ############################################################################################

    if NN:
        if t == 0:
            M = None
        else:
            M = model(phi, rho)
    # solve for the preconditioner using the neural network
    if electrostatic:
        Ex, Ey, Ez, phi, rho = calculateE(N_electrons, electron_x, electron_y, electron_z, \
                N_ions, ion_x, ion_y, ion_z,                                               \
                dx, dy, dz, q_e, q_i, rho, eps, phi, t, M, Nx, Ny, Nz, x_wind, y_wind, z_wind, bc, verbose, GPUs)
        
        if verbose: print(f"Calculating Electric Field, Max Value: {jnp.max(Ex)}")
        # print the maximum value of the electric field

    ############### UPDATE ELECTRONS ##########################################################################################
    if GPUs:
        with jax.default_device(jax.devices('gpu')[0]):
            ev_x, ev_y, ev_z = boris(q_e, Ex, Ey, Ez, Bx, By, Bz, electron_x, \
                        electron_y, electron_z, ev_x, ev_y, ev_z, dt, me)
    else:
        ev_x, ev_y, ev_z = boris(q_e, Ex, Ey, Ez, Bx, By, Bz, electron_x, \
                            electron_y, electron_z, ev_x, ev_y, ev_z, dt, me)
    # implement the boris push algorithm to solve for new electron velocities

    if verbose: print(f"Calculating Electron Velocities, Max Value: {jnp.max(ev_x)}")
    # print the maximum value of the electron velocities

    electron_x, electron_y, electron_z = update_position(electron_x, electron_y, electron_z, ev_x, ev_y, ev_z, dt, x_wind, y_wind, z_wind)
    # Update the positions of the particles

    if verbose: print(f"Calculating Electron Positions, Max Value: {jnp.max(electron_x)}")
    # print the maximum value of the electron positions

    ############### UPDATE IONS ################################################################################################
    if N_ions > 0:
        if GPUs:
            with jax.default_device(jax.devices('gpu')[0]):
                iv_x, iv_y, iv_z = boris(q_i, Ex, Ey, Ez, Bx, By, Bz, ion_x, \
                            ion_y, ion_z, iv_x, iv_y, iv_z, dt, mi)
        else:
            iv_x, iv_y, iv_z = boris(q_i, Ex, Ey, Ez, Bx, By, Bz, ion_x, \
                                    ion_y, ion_z, iv_x, iv_y, iv_z, dt, mi)
        # use boris push for ion velocities

        if verbose: print(f"Calculating Ion Velocities, Max Value: {jnp.max(iv_x)}")
        # print the maximum value of the ion velocities

        ion_x, ion_y, ion_z  = update_position(ion_x, ion_y, ion_z, iv_x, iv_y, iv_z, dt, x_wind, y_wind, z_wind)
        # Update the positions of the particles
        if verbose: print(f"Calculating Ion Positions, Max Value: {jnp.max(ion_x)}")
        # print the maximum value of the ion positions

    ################ MAGNETIC FIELD UPDATE #######################################################################
    if not electrostatic:
        if bc == "spectral":
            Bx, By, Bz = spectralBsolve(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt)
        else:
            Bx, By, Bz = update_B(Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt)
        # update the magnetic field using the curl of the electric field
        if verbose: print(f"Calculating Magnetic Field, Max Value: {jnp.max(Bx)}")
        # print the maximum value of the magnetic field

        if bc == "spectral":
            Ex, Ey, Ez = spectralEsolve(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, dt, C)
        else:
            Ex, Ex, Ez = update_E(Ex, Ey, Ez, Bx, By, Bz, dx, dy, dz, dt, C)


    if t % plot_freq == 0:
    ############## PLOTTING ###################################################################
        if plotEnergy:
            mu = 1.2566370613e-6
            # permeability of free space
            total_energy.append(  \
                totalfield_energy(Ex, Ey, Ez, Bx, By, Bz, mu, eps) + \
                total_KE(me, ev_x, ev_y, ev_z) + total_KE(mi, iv_x, iv_y, iv_z) )
            total_p.append( total_momentum(me, ev_x, ev_y, ev_z) + total_momentum(mi, iv_x, iv_y, iv_z) )
        if plotpositions:
            particlesx = jnp.concatenate([electron_x, ion_x])
            particlesy = jnp.concatenate([electron_y, ion_y])
            particlesz = jnp.concatenate([electron_z, ion_z])
            plot_positions( particlesx, particlesy, particlesz, t, x_wind, y_wind, z_wind)
        if plotvelocities:
            if not plotpositions:
                particlesx = jnp.concatenate([electron_x, ion_x])
                particlesy = jnp.concatenate([electron_y, ion_y])
                particlesz = jnp.concatenate([electron_z, ion_z])
                # get the particle positions if they haven't been computed
            velocitiesx = jnp.concatenate([ev_x, iv_x])
            velocitiesy = jnp.concatenate([ev_y, iv_y])
            velocitiesz = jnp.concatenate([ev_z, iv_z])
            plot_velocities( particlesx, particlesy, particlesz, velocitiesx, velocitiesy, velocitiesz, t, x_wind, y_wind, z_wind)
            plot_velocity_histogram(velocitiesx, velocitiesy, velocitiesz, t, nbins=100)
        if plotfields:
            plot_fields(Ex, Ey, Ez, t, "E", dx, dy, dz)
            plot_fields(Bx, By, Bz, t, "B", dx, dy, dz)
            plot_rho(rho, t, "rho", dx, dy, dz)
            Eprobe.append(magnitude_probe(Ex, Ey, Ez, int(Nx/2), int(Ny/2), int(Nz/2)))
            averageE.append(  jnp.mean( jnp.sqrt( Ex**2 + Ey**2 + Ez**2 ) )   )
        # plot the particles and save as png file
        if plotKE:
            KE.append(total_KE(me, ev_x, ev_y, ev_z) + total_KE(mi, iv_x, iv_y, iv_z))
            KE_time.append(t*dt)
        # calculate the total kinetic energy of the particles
        if plasmaFreq:
            n = jnp.zeros((Nx, Ny, Nz))
            current_freq = freq( n, N_electrons, electron_x, electron_y, electron_z, Nx, Ny, Nz, dx, dy, dz) 
            # calculate the current plasma frequency at time t
            freqs.append( current_freq )
            if verbose: print(f"Plasma Frequency: {current_freq} Hz")
        # calculate the plasma frequency
        if save_data:
            jnp.save(f'data/rho/rho_{t:09}', rho)
            jnp.save(f'data/phi/phi_{t:09}', phi)
        # save the data for the charge density and potential
        if phaseSpace:
            phase_space(electron_x, ev_x, t, "Electronx")
            # phase_space(electron_y, ev_y, t, "Electrony")
            # phase_space(electron_z, ev_z, t, "Electronz")
            phase_space(ion_x, iv_x, t, "Ionx")
            # phase_space(ion_y, iv_y, t, "Iony")
            # phase_space(ion_z, iv_z, t, "Ionz")

            multi_phase_space(electron_x, ion_x, ev_x, iv_x, t, "Electrons", "Ions", "x", x_wind)
            multi_phase_space(electron_y, ion_y, ev_y, iv_y, t, "Electrons", "Ions", "y", y_wind)
            multi_phase_space(electron_z, ion_z, ev_z, iv_z, t, "Electrons", "Ions", "z", z_wind)
        # save the phase space data
if plotfields:
    plot_probe(Eprobe, "Electric Field", "ElectricField")
    efield_freq = plot_fft(Eprobe, dt*plot_freq, "FFT of Electric Field", "E_FFT")
    plot_probe(averageE, "Electric Field", "AvgElectricField")
    print(f'Electric Field Frequency: {efield_freq}')
    # plot the electric field probe
if plotKE:
    plot_KE(KE, KE_time)
    ke_freq = plot_fft(KE, dt*plot_freq, "FFT of Kinetic Energy", "KE_FFT")
    print(f'KE Frequency: {ke_freq}')
    # plot the total kinetic energy of the particles
if plasmaFreq:
    plot_probe(freqs, "Plasma Frequency", "PlasmaFrequency")
    # plot the plasma frequency
    average_freq = jnp.mean( jnp.asarray(freqs[ int(len(freqs)/4):int(3*len(freqs)/4)  ] ) )
    print(f'Average Plasma Frequency: {average_freq}')
if plotEnergy:
    plot_probe(total_energy, "Total Energy", "TotalEnergy")
    #energy_freq = plot_fft(total_energy, dt*plot_freq, "FFT of Total Energy", "Energy_FFT")
    #print(f'Energy Frequency: {energy_freq}')
    # plot the total energy of the system

    plot_probe(total_p, "Total Momentum", "TotalMomentum")
    #momentum_freq = plot_fft(total_p, dt*plot_freq, "FFT of Total Momentum", "Momentum_FFT")
    #print(f'Momentum Frequency: {momentum_freq}')
    # plot the total momentum of the system

end = time.time()
print(f"Total Simulation Time: {end-start}")

if benchmark: jax.profiler.stop_trace()
# stop the profiler and save the data to tensorboard