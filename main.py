# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities


########################################## IMPORT LIBRARIES #############################################
import time
import jax
import jax.numpy as jnp
import argparse
from functools import partial
from tqdm import tqdm
# Importing relevant libraries

from PyPIC3D.plotting import (
    plotter
)

from PyPIC3D.utils import (
    dump_parameters_to_toml
)

from PyPIC3D.initialization import (
    initialize_simulation
)

from PyPIC3D.evolve import (
    time_loop
)

from PyPIC3D.autodiff import (
    kinetic_energy_grad
)

from PyPIC3D.plotting import (
    write_data
)
# Importing functions from the PyPIC3D package
############################################################################################################

###################### JAX SETTINGS ########################################################################
jax.config.update("jax_enable_x64", True)
# set Jax to use 64 bit precision
jax.config.update("jax_debug_nans", True)
# debugging for nans

jax.config.update('jax_platform_name', 'cpu')
# set Jax to use CPUs
############################################################################################################

############################ ARG PARSER ####################################################################
parser = argparse.ArgumentParser(description="3D PIC code using Jax")
parser.add_argument('--config', type=str, default="config.toml", help='Path to the configuration file')
args = parser.parse_args()
# argument parser for the configuration file
config_file = args.config
# path to the configuration file

############################################################################################################

##################################### INITIALIZE SIMULATION ################################################

particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, phi, rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, M, solver, bc, electrostatic, verbose, GPUs, start, Nt, debye, theoretical_freq, thermal_velocity, curl_func = initialize_simulation(config_file)
# initialize the simulation


loop = partial(time_loop, E_grid=E_grid, B_grid=B_grid, world=world, constants=constants, plotting_parameters=plotting_parameters, \
                   curl_func=curl_func, M=M, solver=solver, bc=bc, electrostatic=electrostatic, verbose=verbose, GPUs=GPUs)
# partial function for the time loop

############################################################################################################



# N_particles = particles[0].get_number_of_particles()
# Te = particles[0].get_temperature()
# me = particles[0].get_mass()
# x_wind, y_wind, z_wind = world['x_wind'], world['y_wind'], world['z_wind']
# electron_x, electron_y, electron_z = particles[0].get_position()
# ev_x, ev_y, ev_z = particles[0].get_velocity()
# alternating_ones = (-1)**jnp.array(range(0,N_particles))
# relative_drift_velocity = 0.75*jnp.sqrt(3*constants['kb']*Te/me)
# perturbation = relative_drift_velocity*alternating_ones
# perturbation *= (1 + 0.1*jnp.sin(2*jnp.pi*electron_x/x_wind))
# ev_x = perturbation
# ev_y = jnp.zeros(N_particles)
# ev_z = jnp.zeros(N_particles)
# particles[0].set_velocity(ev_x, ev_y, ev_z)

# #electron_x = jnp.zeros(N_particles)
# electron_y = jnp.zeros(N_particles)# jnp.ones(N_particles) * y_wind/4*alternating_ones
# electron_z = jnp.zeros(N_particles)
# particles[0].set_position(electron_x, electron_y, electron_z)
# #put electrons with opposite velocities in the same position along y




###################################################### SIMULATION LOOP #####################################
start = time.time()
# start the timer

for t in tqdm(range(Nt)):
    particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi = loop(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi)
    # time loop to update the particles and fields
    plotter(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, M, solver, bc, electrostatic, verbose, GPUs)
    # save the data
    grad1, grad2, grad3 = kinetic_energy_grad(particles[0])
    write_data('data/dKE_dvx.txt', t*world['dt'], jnp.mean(grad1))
    write_data('data/dKE_dvy.txt', t*world['dt'], jnp.mean(grad2))
    write_data('data/dKE_dvz.txt', t*world['dt'], jnp.mean(grad3))
    # write the gradient of the kinetic energy with respect to the velocity of the electrons



end = time.time()
# end the timer
#############################################################################################################

####################################### DUMP PARAMETERS TO TOML #############################################

duration = end - start
# calculate the total simulation time

simulation_stats = {"total_time": duration, "total_iterations": Nt, "time_per_iteration": duration/Nt, "debye_length": debye.item(), \
    "plasma_frequency": theoretical_freq.item(), 'thermal_velocity': thermal_velocity.item()}

dump_parameters_to_toml(simulation_stats, simulation_parameters, plotting_parameters, constants, particles)
# save the parameters to an output file

print(f"\nSimulation Complete")
print(f"Total Simulation Time: {duration} s")
print(f"Time Per Iteration: {duration/Nt} s")
###############################################################################################################