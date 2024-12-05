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
import functools
import argparse
# Importing relevant libraries


from PyPIC3D.plotting import (
    plot_fields, plot_positions, plot_rho,
    plot_KE, plot_probe, plot_fft,
    particles_phase_space, write_probe,
    dominant_modes, plot_dominant_modes, magnitude_probe,
    save_vector_field_as_vtk
)
from PyPIC3D.particle import (
    initial_particles, total_KE, total_momentum,
    particle_species
)
from PyPIC3D.fields import (
    calculateE, update_B, update_E, initialize_fields
)

from PyPIC3D.J import (
    VB_correction
)

from PyPIC3D.pstd import (
    spectral_curl, initialize_magnetic_field
)

from PyPIC3D.fdtd import (
    centered_finite_difference_curl
)


from PyPIC3D.utils import (
    plasma_frequency, courant_condition,
    debye_length, update_parameters_from_toml, dump_parameters_to_toml,
    load_particles_from_toml, use_gpu_if_set, precondition, build_coallocated_grid,
    build_yee_grid, convert_to_jax_compatible, load_external_fields_from_toml,
    check_stability, print_stats
)

from PyPIC3D.errors import (
    compute_electric_divergence_error, compute_magnetic_divergence_error
)

from PyPIC3D.defaults import (
    default_parameters
)

from PyPIC3D.boris import (
    particle_push
)

from PyPIC3D.model import (
    PoissonPrecondition
)
# Importing functions from other files

jax.config.update("jax_enable_x64", True)
# set Jax to use 64 bit precision
jax.config.update("jax_debug_nans", True)
# debugging for nans

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
parser = argparse.ArgumentParser(description="3D PIC code using Jax")
parser.add_argument('--config', type=str, default="config.toml", help='Path to the configuration file')
args = parser.parse_args()
# argument parser for the configuration file

config_file = args.config
# path to the configuration file

plotting_parameters, simulation_parameters, constants = default_parameters()
# load the default parameters

if os.path.exists(config_file):
    simulation_parameters, plotting_parameters, constants = update_parameters_from_toml(config_file, simulation_parameters, plotting_parameters, constants)

############################# SIMULATION PARAMETERS ########################################################
# Update locals with simulation parameters
for key, value in simulation_parameters.items():
    locals()[key] = value

# Update locals with plotting parameters
for key, value in plotting_parameters.items():
    locals()[key] = value

if benchmark: jax.profiler.start_trace("/home/christopherwoolford/Documents/PyPIC3D/tensorboard")
# start the profiler using tensorboard

print(f"Initializing Simulation: {name}\n")
start = time.time()
# start the timer

dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
# compute the spatial resolution

################ Courant Condition #############################################################################
courant_number = 1
dt = courant_condition(courant_number, dx, dy, dz, simulation_parameters, constants)
# calculate spatial resolution using courant condition
Nt     = int( t_wind / dt )
# Nt for resolution

world = {'dt': dt, 'Nt': Nt, 'dx': dx, 'dy': dy, 'dz': dz, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'x_wind': x_wind, 'y_wind': y_wind, 'z_wind': z_wind}
# set the simulation world parameters

world = convert_to_jax_compatible(world)
constants = convert_to_jax_compatible(constants)
# convert the world parameters to jax compatible format

print_stats(world)
################################### INITIALIZE PARTICLES AND FIELDS ########################################################

particles = load_particles_from_toml(config_file, simulation_parameters, world, constants)
# load the particles from the configuration file

theoretical_freq, debye, thermal_velocity = check_stability(world, constants, particles[0], dt)
# check the stability of the simulation

Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, phi, rho = initialize_fields(world)
# initialize the electric and magnetic fields

Ex, Ey, Ez, Bx, By, Bz = load_external_fields_from_toml([Ex, Ey, Ez, Bx, By, Bz], config_file)
# add any external fields to the simulation

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
if plotfields: averageB = []
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

E_grid, B_grid = build_yee_grid(world)
# build the grid for the fields



# if solver == "spectral" and not electrostatic:
#     Bx, By, Bz = initialize_magnetic_field(particles, E_grid, B_grid, world, constants, GPUs)
#     # initialize the magnetic field using the curl of the electric field

p = []

if solver == "spectral":
    curl_func = functools.partial(spectral_curl, world=world)
elif solver == "fdtd":
    curl_func = functools.partial(centered_finite_difference_curl, dx=dx, dy=dy, dz=dz, bc=bc)

if plotfields:
    Jx_probe, Jy_probe, Jz_probe = [], [], []

avg_x = []
avg_z = []
avg_y = []

eta1, eta2 = [], []
zeta1, zeta2 = [], []
xi1, xi2 = [], []


for i, particle in enumerate(particles):
    print(f"Initial positions of particle species {i}: {particle.get_position()}")
    print(f"Initial velocities of particle species {i}: {particle.get_velocity()}")

print(f"Mean Electric Field (Ex): {jnp.mean(Ex)}")
print(f"Mean Electric Field (Ey): {jnp.mean(Ey)}")
print(f"Mean Electric Field (Ez): {jnp.mean(Ez)}")
print(f"Mean Magnetic Field (Bx): {jnp.mean(Bx)}")
print(f"Mean Magnetic Field (By): {jnp.mean(By)}")
print(f"Mean Magnetic Field (Bz): {jnp.mean(Bz)}")


# avg_vz = []
for t in range(Nt):
    print(f'Iteration {t}, Time: {t*dt} s')
    ################## PLOTTING ########################################################################################

    if t % plotting_interval == 0:
        avg_x.append(jnp.mean(particles[0].get_position()[0]))
        avg_y.append(jnp.mean(particles[0].get_position()[1]))
        avg_z.append(jnp.mean(particles[0].get_position()[2]))
        print(f"Mean E magnitude: {jnp.mean(jnp.sqrt(Ex**2 + Ey**2 + Ez**2))}")
        print(f"Mean B magnitude: {jnp.mean(jnp.sqrt(Bx**2 + By**2 + Bz**2))}")
        plot_t.append(t*dt)

        p0 = 0
        for particle in particles:
            p0 += particle.momentum()
        p.append(p0)

        if plotpositions:
            plot_positions(particles, t, x_wind, y_wind, z_wind)
        # if plotvelocities:
        #     plot_velocities(particles, t, x_wind, y_wind, z_wind)
        if phaseSpace:
            particles_phase_space([particles[0]], world, t, "Particles")
        if plotfields:
            save_vector_field_as_vtk(Ex, Ey, Ez, E_grid, f"plots/fields/E_{t:09}.vtr")
            save_vector_field_as_vtk(Bx, By, Bz, B_grid, f"plots/fields/B_{t:09}.vtr")
            plot_rho(rho, t, "rho", dx, dy, dz)
            Eprobe.append(magnitude_probe(Ex, Ey, Ez, int(Nx/2), int(Ny/2), int(Nz/2)))
            averageE.append(  jnp.mean( jnp.sqrt( Ex**2 + Ey**2 + Ez**2 ) )   )
            averageB.append(  jnp.mean( jnp.sqrt( Bx**2 + By**2 + Bz**2 ) )   )
        if plotKE:
            ke = 0
            for particle in particles:
                ke += particle.kinetic_energy()
            KE.append(ke)
            KE_time.append(t*dt)
        if plot_errors:
            div_error_E.append(compute_electric_divergence_error(Ex, Ey, Ez, rho, constants, world, solver, bc))
            div_error_B.append(compute_magnetic_divergence_error(Bx, By, Bz, world, solver, bc))

        if plotfields:
            Jx_probe.append(jnp.mean(Jx))
            Jy_probe.append(jnp.mean(Jy))
            Jz_probe.append(jnp.mean(Jz))
            print(f"Mean Jx: {jnp.mean(Jx)}")
            print(f"Mean Jy: {jnp.mean(Jy)}")
            print(f"Mean Jz: {jnp.mean(Jz)}")
            plt.title(f'E at t={t*dt:.2e}s')
            Emag = jnp.sqrt(Ex**2 + Ey**2 + Ez**2)
            plt.imshow(Emag[:, :, int(Nz/2)], origin='lower', extent=[0, x_wind, 0, y_wind])
            plt.colorbar(label='E')
            plt.tight_layout()
            plt.savefig(f'plots/E_slice/E_slice_{t:09}.png')
            plt.close()

            plt.title(f'Charge Density at t={t*dt:.2e}s')
            plt.imshow(rho[:, :, int(Nz/2)], origin='lower', extent=[0, x_wind, 0, y_wind])
            plt.colorbar(label='Charge Density')
            plt.tight_layout()
            plt.savefig(f'plots/rho_slice/rho_slice_{t:09}.png')
            plt.close()


    ############### SOLVE E FIELD ############################################################################################
    M = precondition(NN, phi, rho, model)
    # solve for the preconditioner using the neural network
    if electrostatic:
        Ex, Ey, Ez, phi, rho = calculateE(world, particles, constants, rho, phi, M, t, solver, bc, verbose, GPUs)
        if verbose: print(f"Calculating Electric Field, Max Value: {jnp.max(jnp.sqrt(Ex**2 + Ey**2 + Ez**2))}")
        # print the maximum value of the electric field

    ################ PARTICLE PUSH ########################################################################################
    for i in range(len(particles)):
        if particles[i].get_number_of_particles() > 0:
            if verbose: print(f'Updating {particles[i].get_name()}')
            particles[i] = particle_push(particles[i], Ex, Ey, Ez, Bx, By, Bz, E_grid, B_grid, dt, GPUs)
            # use boris push for particle velocities
            if verbose: print(f"Calculating {particles[i].get_name()} Velocities, Mean Value: {jnp.mean(jnp.abs(particles[i].get_velocity()[0]))}")
            particles[i].update_position(dt, x_wind, y_wind, z_wind)
            if verbose: print(f"Calculating {particles[i].get_name()} Positions, Mean Value: {jnp.mean(jnp.abs(particles[i].get_position()[0]))}")
            # update the particle positions
    ################ FIELD UPDATE ################################################################################################
    if not electrostatic:
        Jx, Jy, Jz = VB_correction(particles, Nx, Ny, Nz)
        # calculate the corrections for charge conservation using villasenor buneamn 1991
        Ex, Ey, Ez = update_E(E_grid, B_grid, (Ex, Ey, Ez), (Bx, By, Bz), (Jx, Jy, Jz), world, constants, curl_func)
        # update the electric field using the curl of the magnetic field
        Bx, By, Bz = update_B(E_grid, B_grid, (Bx, By, Bz), (Ex, Ey, Ez), world, constants, curl_func)
        # update the magnetic field using the curl of the electric field


if plotfields:
    plot_probe(Eprobe, plot_t, "Electric Field", "ElectricField")
    plot_probe(averageE, plot_t, "Electric Field", "AvgElectricField")
    plot_probe(averageB, plot_t, "Magnetic Field", "AvgMagneticField")
    plot_probe(Jx_probe, plot_t, "Jx", "Jx")
    plot_probe(Jy_probe, plot_t, "Jy", "Jy")
    plot_probe(Jz_probe, plot_t, "Jz", "Jz")
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

plot_probe(avg_x, plot_t, "Average Electron X Position", "AverageX")
plot_probe(avg_y, plot_t, "Average Electron Y Position", "AverageY")
plot_probe(avg_z, plot_t, "Average Electron Z Position", "AverageZ")

if plot_errors:
    plot_probe(div_error_E, plot_t, "Divergence Error of E Field", f"div_error_E")
    plot_probe(div_error_B, plot_t, "Divergence Error of B Field", f"div_error_B")
end = time.time()
# end the timer
duration = end - start
# calculate the total simulation time

simulation_stats = {"total_time": duration, "total_iterations": Nt, "time_per_iteration": duration/Nt, "debye_length": debye.item(), \
    "plasma_frequency": theoretical_freq.item(), 'thermal_velocity': thermal_velocity.item()}

dump_parameters_to_toml(simulation_stats, simulation_parameters, plotting_parameters, constants, particles)
# save the parameters to an output file

print(f"\nSimulation Complete")
print(f"Total Simulation Time: {duration} s")

if benchmark: jax.profiler.stop_trace()
# stop the profiler and save the data to tensorboard