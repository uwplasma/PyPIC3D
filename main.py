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
import matplotlib.pyplot as plt
# Importing relevant libraries


from PyPIC3D.plotting import (
    plot_fields, plot_positions, plot_rho,
    plot_KE, plot_probe, plot_fft,
    particles_phase_space, write_probe,
    dominant_modes, plot_dominant_modes, magnitude_probe,
    save_vector_field_as_vtk
)
from PyPIC3D.particle import (
    initial_particles, update_position, total_KE, total_momentum,
    cold_start_init, particle_species
)
from PyPIC3D.fields import (
    calculateE
)

from PyPIC3D.spectral import (
    spectralBsolve, spectralEsolve, spectral_divergence_correction
)

from PyPIC3D.fdtd import (
    update_B, update_E, fdtd_current_correction
)

from PyPIC3D.autodiff import (
    autodiff_update_B, autodiff_update_E
)


from PyPIC3D.utils import (
    plasma_frequency, courant_condition,
    debye_length, update_parameters_from_toml, dump_parameters_to_toml,
    load_particles_from_toml, use_gpu_if_set, precondition, build_grid
)

from PyPIC3D.errors import (
    compute_electric_divergence_error, compute_magnetic_divergence_error
)

from PyPIC3D.defaults import (
    default_parameters
)


from PyPIC3D.model import (
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

plotting_parameters, simulation_parameters, constants = default_parameters()
# load the default parameters

if os.path.exists(config_file):
    simulation_parameters, plotting_parameters, constants = update_parameters_from_toml(config_file, simulation_parameters, plotting_parameters, constants)


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
plot_dispersion = plotting_parameters["plot_dispersion"]
plot_freq = plotting_parameters["plotting_interval"]
# booleans for plotting/saving data

name = simulation_parameters["name"]
solver = simulation_parameters["solver"]
bc = simulation_parameters["bc"]
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
    from PyPIC3D.spectral import particle_push
    from PyPIC3D.fields import initialize_fields
elif solver == 'fdtd':
    from PyPIC3D.fdtd import particle_push
    from PyPIC3D.fields import initialize_fields
elif solver == 'autodiff':
    from PyPIC3D.autodiff import particle_push, initialize_fields
# set the particle push method

print(f"Initializing Simulation: {name}\n")
start = time.time()
# start the timer

dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
# compute the spatial resolution

world = {'dx': dx, 'dy': dy, 'dz': dz, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'x_wind': x_wind, 'y_wind': y_wind, 'z_wind': z_wind}

################ Courant Condition #############################################################################
courant_number = 1 #/100
dt = courant_condition(courant_number, world, simulation_parameters, constants)
# calculate spatial resolution using courant condition
dt = dt*200
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

particles = load_particles_from_toml("config.toml", simulation_parameters, world, constants)


theoretical_freq = plasma_frequency(particles[0], world, constants)
# calculate the expected plasma frequency from analytical theory, w = sqrt( ne^2 / (eps * me) )

if theoretical_freq * dt > 2.0:
    print(f"# of Electrons is Low and may introduce numerical stability")
    print(f"In order to correct this, # of Electrons needs to be at least { (2/dt)**2 * (me*eps/q_e**2) } for this spatial resolution")

debye = debye_length(particles[0], world, constants)
# calculate the debye length of the plasma


if debye < dx:
    print(f"Debye Length is less than the spatial resolution, this may introduce numerical instability")



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


#################################### Two Stream Instability #####################################################
N_particles = particles[0].get_number_of_particles()
Te = particles[0].get_temperature()
me = particles[0].get_mass()
electron_x, electron_y, electron_z = particles[0].get_position()
ev_x, ev_y, ev_z = particles[0].get_velocity()
alternating_ones = (-1)**jnp.array(range(0,N_particles))
relative_drift_velocity = 0.5*jnp.sqrt(3*constants['kb']*Te/me)
perturbation = relative_drift_velocity*alternating_ones
#perturbation *= (1 + 0.1*jnp.sin(2*jnp.pi*electron_x/x_wind))
ev_x = perturbation
ev_y = jnp.zeros(N_particles)
ev_z = jnp.zeros(N_particles)
particles[0].set_velocity(ev_x, ev_y, ev_z)

electron_y = jnp.zeros(N_particles)# jnp.ones(N_particles) * y_wind/4*alternating_ones
electron_z = jnp.zeros(N_particles)
particles[0].set_position(electron_x, electron_y, electron_z)
# put electrons with opposite velocities in the same position along y

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
plot_t = []
min_v = []
if plotfields: Eprobe = []
if plotfields: averageE = []
if plotKE:
    KE = []
    KE_time = []
if plasmaFreq: freqs = []
if plotEnergy: total_energy = []
if plotEnergy: total_p      = []
if plot_errors: div_error_E, div_error_B = [], []
if plot_dispersion: kz = []
if not electrostatic:
        Ex, Ey, Ez, phi, rho = calculateE(world, particles, constants, rho, phi, M, 0, solver, bc, verbose, GPUs)

grid, staggered_grid = build_grid(world)
# build the grid for the fields

print(f"Theoretical Plasma Frequency: {theoretical_freq} Hz")
print(f"Debye Length: {debye} m")
print(f"Thermal Velocity: {jnp.sqrt(3*constants['kb']*Te/me)}\n")


Jx = jnp.zeros((Nx, Ny, Nz))
Jy = jnp.zeros((Nx, Ny, Nz))
Jz = jnp.zeros((Nx, Ny, Nz))



p = []

for t in range(Nt):
    print(f'Iteration {t}, Time: {t*dt} s')

    ################## PLOTTING ########################################################################################

    if t % plot_freq == 0:
        vx, vy, vz = particles[0].get_velocity()
        jnp.save(f'data/vx/vx_{t:09}', vx)
        jnp.save(f'data/vy/vy_{t:09}', vy)
        jnp.save(f'data/vz/vz_{t:09}', vz)
        # save the velocity data
        plot_t.append(t*dt)

        p0 = 0
        for particle in particles:
            p0 += particle.momentum()
        p.append(p0)

        if plotpositions:
            plot_positions(particles, t, x_wind, y_wind, z_wind)
        if plotvelocities:
            plot_velocities(particles, t, x_wind, y_wind, z_wind)
        if phaseSpace:
            particles_phase_space([particles[0]], t, "Particles")
        if plotfields:
            save_vector_field_as_vtk(Ex, Ey, Ez, grid, f"plots/fields/E_{t:09}.vtr")
            save_vector_field_as_vtk(Bx, By, Bz, staggered_grid, f"plots/fields/B_{t:09}.vtr")
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


    ############### SOLVE E FIELD ############################################################################################
    M = precondition(NN, phi, rho, model)
    # solve for the preconditioner using the neural network
    if electrostatic:
        Ex, Ey, Ez, phi, rho = calculateE(world, particles, constants, rho, phi, M, t, solver, bc, verbose, GPUs)
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
        if solver == "fdtd":
            Jx, Jy, Jz = fdtd_current_correction(particles, Nx, Ny, Nz)
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
            Ex, Ey, Ez = spectral_divergence_correction(Ex, Ey, Ez, rho, dx, dy, dz, dt, constants)
            Ex, Ey, Ez = spectralEsolve(grid, staggered_grid, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, world, dt, constants)
        elif solver == "fdtd":
            Ex, Ex, Ez = update_E(grid, staggered_grid, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, dx, dy, dz, dt, C, eps, bc)
        elif solver == 'autodiff':
            Ex, Ey, Ez = autodiff_update_E(Ex, Ey, Ez, Bx, By, Bz, dt, C)
        # update the electric field using the curl of the magnetic field

if plotfields:
    plot_probe(Eprobe, plot_t, "Electric Field", "ElectricField")
    plot_probe(averageE, plot_t, "Electric Field", "AvgElectricField")
    # plot the electric field probe
if plotKE:
    plot_KE(KE, KE_time)
    ke_freq = plot_fft(KE, dt*plot_freq, "FFT of Kinetic Energy", "KE_FFT")
    print(f'KE Frequency: {ke_freq}')
    # plot the total kinetic energy of the particles

if plot_dispersion:
    plot_dominant_modes(jnp.asarray(kz), plot_t, "Dominant Modes over Time", "Modes")
    # plot the dispersion relation

plot_probe(p, plot_t, "Total Momentum", "TotalMomentum")

if plot_errors:
    plot_probe(div_error_E, plot_t, "Divergence Error of E Field", f"div_error_E")
    plot_probe(div_error_B, plot_t, "Divergence Error of B Field", f"div_error_B")
end = time.time()
# end the timer
duration = end - start
# calculate the total simulation time

simulation_stats = {"total_time": duration, "total_iterations": Nt, "time_per_iteration": duration/Nt, "debye_length": debye.item(), \
    "plasma_frequency": theoretical_freq.item(), 'thermal_velocity': jnp.sqrt(2*constants['kb']*Te/me).item()}

dump_parameters_to_toml(simulation_stats, simulation_parameters, plotting_parameters, constants, particles)
# save the parameters to an output file

print(f"\nSimulation Complete")
print(f"Total Simulation Time: {duration} s")

if benchmark: jax.profiler.stop_trace()
# stop the profiler and save the data to tensorboard