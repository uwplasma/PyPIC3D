# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities

########################################## IMPORT LIBRARIES #############################################
import time
import jax
from jax import block_until_ready
import jax.numpy as jnp
from tqdm import tqdm
#from memory_profiler import profile
# Importing relevant libraries

from PyPIC3D.plotting import (
    plotter
)

from PyPIC3D.utils import (
    dump_parameters_to_toml, load_config_file, compute_energy
)

from PyPIC3D.initialization import (
    initialize_simulation
)


# Importing functions from the PyPIC3D package
############################################################################################################


def run_PyPIC3D(config_file):
    ##################################### INITIALIZE SIMULATION ################################################

    loop, particles, E, B, J, \
        phi, rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, \
            plasma_parameters, solver, bc, electrostatic, verbose, GPUs, Nt, curl_func = initialize_simulation(config_file)
    # initialize the simulation

    jit_loop = jax.jit(loop, static_argnames=('curl_func', 'solver', 'bc'))

    ############################################################################################################

    ###################################################### SIMULATION LOOP #####################################

    for t in tqdm(range(Nt)):
        plotter(t, particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, simulation_parameters)
        # plot the data

        particles, E, B, J, phi, rho = jit_loop(particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, curl_func, solver, bc)
        # time loop to update the particles and fields

    ############################################################################################################

    return Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, E, B, J, world

def main():
    ###################### JAX SETTINGS ########################################################################
    jax.config.update("jax_enable_x64", True)
    # set Jax to use 64 bit precision
    jax.config.update("jax_debug_nans", True)
    # debugging for nans
    jax.config.update('jax_platform_name', 'cpu')
    # set Jax to use CPUs
    #jax.config.update("jax_disable_jit", True)
    ############################################################################################################

    toml_file = load_config_file()
    # load the configuration file

    start = time.time()
    # start the timer

    Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, E, B, J, world =  block_until_ready(run_PyPIC3D(toml_file))
    # run the PyPIC3D simulation

    end = time.time()
    # end the timer

    e_energy, b_energy, kinetic_energy = compute_energy(particles, E, B, world, constants)
    # compute the energy of the system
    print(f"Final Electric Field Energy: {e_energy}")
    print(f"Final Magnetic Field Energy: {b_energy}")
    print(f"Final Kinetic Energy: {kinetic_energy}")
    print(f"Total Final Energy: {e_energy + b_energy + kinetic_energy}\n")


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


if __name__ == "__main__":
    main()
    # run the main function