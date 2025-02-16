# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities

########################################## IMPORT LIBRARIES #############################################
import time
import jax
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm
import numpy as np
import toml
# Importing relevant libraries

from PyPIC3D.plotting import (
    plotter
)

from PyPIC3D.utils import (
    dump_parameters_to_toml, load_config_file
)

from PyPIC3D.initialization import (
    initialize_simulation
)

from PyPIC3D.plotting import (
    write_data
)
# Importing functions from the PyPIC3D package
############################################################################################################


def run_PyPIC3D(config_file):
    ##################################### INITIALIZE SIMULATION ################################################

    loop, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, \
        phi, rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, \
            plasma_parameters, M, solver, bc, electrostatic, verbose, GPUs, Nt, curl_func, \
                pecs, lasers, surfaces = initialize_simulation(toml_file)
    # initialize the simulation

    loop = jax.jit(loop)
    # jit the loop function
    ############################################################################################################


    ###################################################### SIMULATION LOOP #####################################
    start = time.time()
    # start the timer

    # jax.profiler.start_trace("/tmp/tensorboard")

    for t in tqdm(range(Nt)):
        particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, rho, phi = loop(particles, (Ex, Ey, Ez), (Bx, By, Bz), (Jx, Jy, Jz), rho, phi)
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


if __name__ == "__main__":
    ###################### JAX SETTINGS ########################################################################
    #jax.config.update("jax_enable_x64", True)
    # set Jax to use 64 bit precision
    #jax.config.update("jax_debug_nans", True)
    # debugging for nans
    #jax.config.update('jax_platform_name', 'cpu')
    # set Jax to use CPUs
    #jax.config.update("jax_disable_jit", True)
    ############################################################################################################

    toml_file = load_config_file()
    # load the configuration file

    run_PyPIC3D(toml_file)
    # run the PyPIC3D simulation