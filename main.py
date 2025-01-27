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
import numpy as np
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
#jax.config.update("jax_disable_jit", True)
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

particles, Ex, Ey, Ez, Ex_ext, Ey_ext, Ez_ext, Bx, By, Bz, Bx_ext, By_ext, Bz_ext, Jx, Jy, Jz, \
    phi, rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, \
        plasma_parameters, M, solver, bc, electrostatic, verbose, GPUs, start, Nt, curl_func, \
            pecs, lasers, surfaces = initialize_simulation(config_file)
# initialize the simulation


loop = partial(time_loop, Ex_ext=Ex_ext, Ey_ext=Ey_ext, Ez_ext=Ez_ext, Bx_ext=Bx_ext, By_ext=By_ext, Bz_ext=Bz_ext, E_grid=E_grid, \
    B_grid=B_grid, world=world, constants=constants, pecs=pecs, lasers=lasers, surfaces=surfaces,                                                   \
        curl_func=curl_func, M=M, solver=solver, bc=bc, electrostatic=electrostatic, verbose=verbose, GPUs=GPUs)
# partial function for the time loop

############################################################################################################


###################################################### SIMULATION LOOP #####################################
start = time.time()
# start the timer

# jax.profiler.start_trace("/tmp/tensorboard")

for t in tqdm(range(Nt)):
    particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi = loop(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi)
    # time loop to update the particles and fields
    plotter(t, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, simulation_parameters, solver, bc)
    # plot the data

# jax.profiler.stop_trace()
# # stop the trace

end = time.time()
# end the timer
#############################################################################################################

####################################### DUMP PARAMETERS TO TOML #############################################

duration = end - start
# calculate the total simulation time

simulation_stats = {
    "total_time": duration,
    "total_iterations": Nt,
    "time_per_iteration": duration / Nt
}

dump_parameters_to_toml(simulation_stats, simulation_parameters, plasma_parameters, plotting_parameters, constants, particles)
# save the parameters to an output file

print(f"\nSimulation Complete")
print(f"Total Simulation Time: {duration} s")
print(f"Time Per Iteration: {duration/Nt} s")
###############################################################################################################