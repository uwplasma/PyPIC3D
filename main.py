# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities

import time
import jax
from jax import random
import jax.numpy as jnp
import equinox as eqx
import os, sys
# Importing relevant libraries

from src.plotting import (
    plot_fields, plot_positions, plot_rho, plot_velocities,
    plot_velocity_histogram, plot_KE, plot_probe, plot_fft,
    phase_space, multi_phase_space, particles_phase_space,
    number_density, totalfield_energy, probe, freq,
    magnitude_probe, write_probe
)
from src.particle import (
    initial_particles, update_position, total_KE, total_momentum,
    cold_start_init, particle_species
)
from src.fields import (
    calculateE
)

from src.spectral import (
    spectralBsolve, spectralEsolve
)

from src.fdtd import (
    update_B, update_E
)

from src.autodiff import (
    autodiff_update_B, autodiff_update_E
)


from src.utils import (
    plasma_frequency, courant_condition,
    debye_length, update_parameters_from_toml, dump_parameters_to_toml,
    load_particles_from_toml, use_gpu_if_set, precondition, build_grid
)

from src.errors import (
    compute_electric_divergence_error, compute_magnetic_divergence_error
)

from src.defaults import (
    default_parameters
)

from src.charge_conservation import (
    current_correction, marder_correction
)

from src.model import (
    PoissonPrecondition
)
# Importing functions from other files

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
config_file = "config.toml"
# path to the configuration file

plotting_parameters, simulation_parameters = default_parameters()
# load the default parameters

if os.path.exists(config_file):
    simulation_parameters, plotting_parameters = update_parameters_from_toml(config_file, simulation_parameters, plotting_parameters)


cold_start = False
# start all the particles at the same place (rho = 0)

############################# SIMULATION PARAMETERS ########################################################

save_data = plotting_parameters["save_data"]
plotfields = plotting_parameters["plotfields"]
plotpositions = plotting_parameters["plotpositions"]
plotvelocities = plotting_parameters["plotvelocities"]
plotKE = plotting_parameters["plotKE"]
plotEnergy = plotting_parameters["plotEnergy"]
plasmaFreq = plotting_parameters["plasmaFreq"]
phaseSpace = plotting_parameters["phaseSpace"]
plot_errors = plotting_parameters["plot_errors"]
plot_freq = plotting_parameters["plotting_interval"]
# booleans for plotting/saving data

name = simulation_parameters["name"]
solver = simulation_parameters["solver"]
bc = simulation_parameters["bc"]
eps = simulation_parameters["eps"]
mu = simulation_parameters["mu"]
C = simulation_parameters["C"]
kb = simulation_parameters["kb"]
me = simulation_parameters["me"]
mi = simulation_parameters["mi"]
q_e = simulation_parameters["q_e"]
q_i = simulation_parameters["q_i"]
Te = simulation_parameters["Te"]
Ti = simulation_parameters["Ti"]
N_electrons = simulation_parameters["N_electrons"]
N_ions = simulation_parameters["N_ions"]
Nx = simulation_parameters["Nx"]
Ny = simulation_parameters["Ny"]
Nz = simulation_parameters["Nz"]
x_wind = simulation_parameters["x_wind"]
y_wind = simulation_parameters["y_wind"]
z_wind = simulation_parameters["z_wind"]
t_wind = simulation_parameters["t_wind"]
electrostatic = simulation_parameters["electrostatic"]
benchmark = simulation_parameters["benchmark"]
verbose = simulation_parameters["verbose"]
# set the simulation parameters

if benchmark: jax.profiler.start_trace("/home/christopherwoolford/Documents/PyPIC3D/tensorboard")
# start the profiler using tensorboard

if solver == 'spectral':
    from src.spectral import particle_push
    from src.fields import initialize_fields
elif solver == 'fdtd':
    from src.fdtd import particle_push
    from src.fields import initialize_fields
elif solver == 'autodiff':
    from src.autodiff import particle_push, initialize_fields
# set the particle push method

print(f"Initializing Simulation: {name}\n")
start = time.time()
# start the timer

dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
# compute the spatial resolution

world = {'dx': dx, 'dy': dy, 'dz': dz, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'x_wind': x_wind, 'y_wind': y_wind, 'z_wind': z_wind}

################ Courant Condition #############################################################################
courant_number = 1
dt = courant_condition(courant_number, world, simulation_parameters)
# calculate spatial resolution using courant condition

theoretical_freq = plasma_frequency(N_electrons, world, eps, me, q_e)
# calculate the expected plasma frequency from analytical theory, w = sqrt( ne^2 / (eps * me) )

if theoretical_freq * dt > 2.0:
    print(f"# of Electrons is Low and may introduce numerical stability")
    print(f"In order to correct this, # of Electrons needs to be at least { (2/dt)**2 * (me*eps/q_e**2) } for this spatial resolution")

debye = debye_length(eps, Te, N_electrons, world, q_e, kb)
# calculate the debye length of the plasma


if debye < dx:
    print(f"Debye Length is less than the spatial resolution, this may introduce numerical instability")

Nt     = int( t_wind / dt )
# Nt for resolution


print(f'time window: {t_wind}')
print(f'x window: {x_wind}')
print(f'y window: {y_wind}')
print(f'z window: {z_wind}')
print(f"\nResolution")
print(f'dx: {dx}')
print(f'dy: {dy}')
print(f'dz: {dz}')
print(f'dt:          {dt}')
print(f'Nt:          {Nt}\n')

################################### INITIALIZE PARTICLES ########################################################

particles = load_particles_from_toml("config.toml", simulation_parameters, dx, dy, dz)

if solver == 'autodiff':
    Ex, Ey, Ez, Bx, By, Bz = initialize_fields(particles, (1/(4*jnp.pi*eps)))
    rho = jnp.zeros((Nx, Ny, Nz))
    phi = jnp.zeros((Nx, Ny, Nz))
# initialize the electric and magnetic fields
else:
    Ex, Ey, Ez, Bx, By, Bz, phi, rho = initialize_fields(world)
# initialize the electric and magnetic fields

key1 = random.key(4353)
key2 = random.key(1043)
key3 = random.key(1234)
key4 = random.key(2345)
key5 = random.key(3456)
# random keys for initializing the particles
# if cold_start:
#     electron_x, electron_y, electron_z, ev_x, ev_y, ev_z = cold_start_init(0, N_electrons, x_wind, y_wind, z_wind, me, Te, kb, key1, key2, key3)
#     ion_x, ion_y, ion_z, iv_x, iv_y, iv_z                 = cold_start_init(0, N_ions, x_wind, y_wind, z_wind, mi, Ti, kb, key3, key4, key5)
# else:
#     electron_x, electron_y, electron_z, ev_x, ev_y, ev_z  = initial_particles(N_electrons, x_wind, y_wind, z_wind, me, Te, kb, key1, key2, key3)
#     ion_x, ion_y, ion_z, iv_x, iv_y, iv_z                 = initial_particles(N_ions, x_wind, y_wind, z_wind, mi, Ti, kb, key3, key4, key5)
# # initialize the positions and velocities of the electrons and ions in the plasma.
# # eventually, I need to update the initialization to use a more accurate position and velocity distribution.

# electron_masses = jnp.ones(N_electrons) * me
# ion_masses = jnp.ones(N_ions) * mi
# electrons = particle_species("electrons", N_electrons, q_e, electron_masses, ev_x, ev_y, ev_z, electron_x, electron_y, electron_z)
# ions = particle_species("ions", N_ions, q_i, ion_masses, iv_x, iv_y, iv_z, ion_x, ion_y, ion_z)
# particles = [electrons, ions]
# # create the particle species

#################################### Two Stream Instability #####################################################
N_particles = particles[0].get_number_of_particles()
electron_x, electron_y, electron_z = particles[0].get_position()
alternating_ones = (-1)**jnp.array(range(0,N_particles))
v0=1.5*2657603.0
ev_x = v0*alternating_ones
ev_x *= ( 1 + 0.1*jnp.sin(6*jnp.pi * electron_x / x_wind) )
ev_y = jnp.zeros(N_particles)
ev_z = jnp.zeros(N_particles)
particles[0].set_velocity(ev_x, ev_y, ev_z)

# # add perturbation to the electron velocities

# iv_x = jnp.zeros(N_ions)
# iv_y = jnp.zeros(N_ions)
# iv_z = jnp.zeros(N_ions)
# # initialize the ion velocities to zero

# particles[0].set_velocity(ev_x, ev_y, ev_z)
# particles[1].set_velocity(iv_x, iv_y, iv_z)
# # update the velocities of the particles
##################################### Neural Network Preconditioner ################################################

key = jax.random.PRNGKey(0)
# define the random key
M = None
if NN:
    model = PoissonPrecondition( Nx=Nx, Ny=Ny, Nz=Nz, hidden_dim=3000, key=key)
    # define the model
    model = eqx.tree_deserialise_leaves(model_name, model)
    # load the model from file
else: model = None

#################################### MAIN LOOP ####################################################################

if plotfields: Eprobe = []
if plotfields: averageE = []
if plotKE:
    KE = []
    KE_time = []
if plasmaFreq: freqs = []
if plotEnergy: total_energy = []
if plotEnergy: total_p      = []
if plot_errors: div_error_E, div_error_B = [], []

if not electrostatic:
        Ex, Ey, Ez, phi, rho = calculateE(world, particles, rho, eps, phi, M, 0, solver, bc, verbose, GPUs)

grid, staggered_grid = build_grid(world)
# build the grid for the fields

print(f"Theoretical Plasma Frequency: {theoretical_freq} Hz")
print(f"Debye Length: {debye} m")
print(f"Thermal Velocity: {jnp.sqrt(2*kb*Te/me)}\n")

# avg_jx = []
# avg_jy = []
# avg_jz = []

perturbation_period = 20*dt # starting with 5 dt's for now
velocity_perturbation = 0e9 # m/s
# perturbation velocity
def perturb_function(t, dt, perturbation_period, velocity_perturbation):
    perturbation = velocity_perturbation * jnp.sin(2 * jnp.pi * t * dt / perturbation_period)
    return perturbation
# function to calculate the perturbation

Jx = jnp.zeros((Nx, Ny, Nz))
Jy = jnp.zeros((Nx, Ny, Nz))
Jz = jnp.zeros((Nx, Ny, Nz))



#avg_z = []
p = []

# By = jnp.ones((Nx, Ny, Nz)) * 1
# vx, vy, vz = particles[0].get_velocity()
# #vx += perturb_function(t, dt, perturbation_period, velocity_perturbation)
# vy = jnp.zeros((N_electrons))
# vz = jnp.zeros((N_electrons))
# vx = jnp.ones((N_electrons)) * 0.75e6
# particles[0].set_velocity(vx, vy, vz)

for t in range(Nt):
    print(f'Iteration {t}, Time: {t*dt} s')

    # vx, vy, vz = particles[0].get_velocity()
    # vx += perturb_function(t, dt, perturbation_period, velocity_perturbation)
    # particles[0].set_velocity(vx, vy, vz)

    ############### SOLVE E FIELD ############################################################################################
    M = precondition(NN, phi, rho, model)
    # solve for the preconditioner using the neural network
    if electrostatic:
        Ex, Ey, Ez, phi, rho = calculateE(world, particles, rho, eps, phi, M, t, solver, bc, verbose, GPUs)
        if verbose: print(f"Calculating Electric Field, Max Value: {jnp.max(jnp.sqrt(Ex**2 + Ey**2 + Ez**2))}")
        # print the maximum value of the electric field

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):
        if verbose: print(f'Updating {particles[i].get_name()}')
        particles[i] = particle_push(particles[i], Ex, Ey, Ez, Bx, By, Bz, grid, staggered_grid, dt, GPUs)
        # use boris push for particle velocities
        if verbose: print(f"Calculating {particles[i].get_name()} Velocities, Mean Value: {jnp.mean(jnp.abs(particles[i].get_velocity()[0]))}")
        particles[i].update_position(dt, x_wind, y_wind, z_wind)
        if verbose: print(f"Calculating {particles[i].get_name()} Positions, Mean Value: {jnp.mean(jnp.abs(particles[i].get_position()[0]))}")

    ################ FIELD UPDATE #######################################################################
    if not electrostatic:
        if solver == "spectral" or solver == "fdtd":
            Jx, Jy, Jz = current_correction(particles, Nx, Ny, Nz)
        # calculate the corrections for charge conservation using villasenor buneamn 1991
        if solver == "spectral":
            Bx, By, Bz = spectralBsolve(grid, staggered_grid, Bx, By, Bz, Ex, Ey, Ez, world, dt)
        elif solver == "fdtd":
            Bx, By, Bz = update_B(grid, staggered_grid, Bx, By, Bz, Ex, Ey, Ez, dx, dy, dz, dt, bc)
        elif solver == "autodiff":
            Bx, By, Bz = autodiff_update_B(Bx, By, Bz, Ex, Ey, Ez, dt)
        # update the magnetic field using the curl of the electric field
        if verbose: print(f"Calculating Magnetic Field, Max Value: {jnp.max(jnp.sqrt(Bx**2 + By**2 + Bz**2))}")
        # print the maximum value of the magnetic field

        if solver == "spectral":
            Ex, Ey, Ez = spectralEsolve(grid, staggered_grid, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, world, dt, C, mu)
        elif solver == "fdtd":
            Ex, Ex, Ez = update_E(grid, staggered_grid, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, dx, dy, dz, dt, C, eps, bc)
        elif solver == 'autodiff':
            Ex, Ey, Ez = autodiff_update_E(Ex, Ey, Ez, Bx, By, Bz, dt, C)
        # update the electric field using the curl of the magnetic field

    ################## PLOTTING ########################################################################################

    if t % plot_freq == 0:
        write_probe(jnp.mean(jnp.sqrt(Ex**2 + Ey**2 + Ez**2)), t*dt, "avg_E.txt")
        #avg_z.append(particles[0].get_position()[2])
        p0 = 0
        for particle in particles:
            p0 += particle.momentum()
        p.append(p0)
        # avg_jx.append(jnp.mean(Jx))
        # avg_jy.append(jnp.mean(Jy))
        # avg_jz.append(jnp.mean(Jz))
        # calculate the average current density
        if plotpositions:
            plot_positions(particles, t, x_wind, y_wind, z_wind)
        if plotvelocities:
            plot_velocities(particles, t, x_wind, y_wind, z_wind)
        if phaseSpace:
            particles_phase_space(particles, t, "Particles")
        if plotfields:
            plot_fields(Ex, Ey, Ez, t, "E", dx, dy, dz)
            plot_fields(Bx, By, Bz, t, "B", dx, dy, dz)
            plot_rho(rho, t, "rho", dx, dy, dz)
            Eprobe.append(magnitude_probe(Ex, Ey, Ez, int(Nx/2), int(Ny/2), int(Nz/2)))
            averageE.append(  jnp.mean( jnp.sqrt( Ex**2 + Ey**2 + Ez**2 ) )   )
        if plotKE:
            ke = 0
            for particle in particles:
                ke += particle.kinetic_energy()
            KE.append(ke)
            KE_time.append(t*dt)
        if plot_errors:
            div_error_E.append(compute_electric_divergence_error(Ex, Ey, Ez, rho, eps, dx, dy, dz, solver, bc))
            div_error_B.append(compute_magnetic_divergence_error(Bx, By, Bz, dx, dy, dz, solver, bc))

#     if t % plot_freq == 0:
#     ############## PLOTTING ###################################################################
#         if plotEnergy:
#             mu = 1.2566370613e-6
#             # permeability of free space
#             total_energy.append(  \
#                 totalfield_energy(Ex, Ey, Ez, Bx, By, Bz, mu, eps) + \
#                 total_KE(me, ev_x, ev_y, ev_z) + total_KE(mi, iv_x, iv_y, iv_z) )
#             total_p.append( total_momentum(me, ev_x, ev_y, ev_z) + total_momentum(mi, iv_x, iv_y, iv_z) )
#         if plotpositions:
#             particlesx = jnp.concatenate([electron_x, ion_x])
#             particlesy = jnp.concatenate([electron_y, ion_y])
#             particlesz = jnp.concatenate([electron_z, ion_z])
#             plot_positions( particlesx, particlesy, particlesz, t, x_wind, y_wind, z_wind)
#         if plotvelocities:
#             if not plotpositions:
#                 particlesx = jnp.concatenate([electron_x, ion_x])
#                 particlesy = jnp.concatenate([electron_y, ion_y])
#                 particlesz = jnp.concatenate([electron_z, ion_z])
#                 # get the particle positions if they haven't been computed
#             velocitiesx = jnp.concatenate([ev_x, iv_x])
#             velocitiesy = jnp.concatenate([ev_y, iv_y])
#             velocitiesz = jnp.concatenate([ev_z, iv_z])
#             plot_velocities( particlesx, particlesy, particlesz, velocitiesx, velocitiesy, velocitiesz, t, x_wind, y_wind, z_wind)
#             plot_velocity_histogram(velocitiesx, velocitiesy, velocitiesz, t, nbins=100)
#         if plotfields:
#             plot_fields(Ex, Ey, Ez, t, "E", dx, dy, dz)
#             plot_fields(Bx, By, Bz, t, "B", dx, dy, dz)
#             plot_rho(rho, t, "rho", dx, dy, dz)
#             Eprobe.append(magnitude_probe(Ex, Ey, Ez, int(Nx/2), int(Ny/2), int(Nz/2)))
#             averageE.append(  jnp.mean( jnp.sqrt( Ex**2 + Ey**2 + Ez**2 ) )   )
#         # plot the particles and save as png file
#         if plotKE:
#             KE.append(total_KE(me, ev_x, ev_y, ev_z) + total_KE(mi, iv_x, iv_y, iv_z))
#             KE_time.append(t*dt)
#         # calculate the total kinetic energy of the particles
#         if plasmaFreq:
#             n = jnp.zeros((Nx, Ny, Nz))
#             current_freq = freq( n, N_electrons, electron_x, electron_y, electron_z, Nx, Ny, Nz, dx, dy, dz) 
#             # calculate the current plasma frequency at time t
#             freqs.append( current_freq )
#             if verbose: print(f"Plasma Frequency: {current_freq} Hz")
#         # calculate the plasma frequency
#         if save_data:
#             jnp.save(f'data/rho/rho_{t:09}', rho)
#             jnp.save(f'data/phi/phi_{t:09}', phi)
#         # save the data for the charge density and potential
#         if phaseSpace:
#             phase_space(electron_x, ev_x, t, "Electronx")
#             phase_space(ion_x, iv_x, t, "Ionx")
#             multi_phase_space(electron_x, ion_x, ev_x, iv_x, t, "Electrons", "Ions", "x", x_wind)
#             multi_phase_space(electron_y, ion_y, ev_y, iv_y, t, "Electrons", "Ions", "y", y_wind)
#             multi_phase_space(electron_z, ion_z, ev_z, iv_z, t, "Electrons", "Ions", "z", z_wind)
#         # save the phase space data
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
# if plasmaFreq:
#     plot_probe(freqs, "Plasma Frequency", "PlasmaFrequency")
#     # plot the plasma frequency
#     average_freq = jnp.mean( jnp.asarray(freqs[ int(len(freqs)/4):int(3*len(freqs)/4)  ] ) )
#     print(f'Average Plasma Frequency: {average_freq}')
# if plotEnergy:
#     plot_probe(total_energy, "Total Energy", "TotalEnergy")
#     # plot the total energy of the system
#     plot_probe(total_p, "Total Momentum", "TotalMomentum")
#     # plot the total momentum of the system
# plot_probe(avg_jx, "Average Jx", "AverageJx")
# plot_probe(avg_jy, "Average Jy", "AverageJy")
# plot_probe(avg_jz, "Average Jz", "AverageJz")
# plot the average current density

#plot_probe(avg_z, "Average Z Position", "AverageZ")
plot_probe(p, "Total Momentum", "TotalMomentum")

if plot_errors:
    plot_probe(div_error_E, "Divergence Error of E Field", f"div_error_E")
    plot_probe(div_error_B, "Divergence Error of B Field", f"div_error_B")
end = time.time()
# end the timer
duration = end - start
# calculate the total simulation time

simulation_stats = {"total_time": duration, "total_iterations": Nt, "time_per_iteration": duration/Nt, "debye_length": debye.item(), \
    "plasma_frequency": theoretical_freq.item(), 'thermal_velocity': jnp.sqrt(2*kb*Te/me).item()}

dump_parameters_to_toml(simulation_stats, simulation_parameters, plotting_parameters)
# save the parameters to an output file

print(f"\nSimulation Complete")
print(f"Total Simulation Time: {duration} s")

if benchmark: jax.profiler.stop_trace()
# stop the profiler and save the data to tensorboard