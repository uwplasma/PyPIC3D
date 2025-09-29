# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities

########################################## IMPORT LIBRARIES #############################################
from sys import path
import time
import jax
from jax import block_until_ready
import jax.numpy as jnp
from tqdm import tqdm
#from memory_profiler import profile
# Importing relevant libraries

from PyPIC3D.plotting import (
    plotter, write_particles_phase_space, write_data, plot_vtk_particles, plot_field_slice_vtk,
    plot_vectorfield_slice_vtk
)

from PyPIC3D.utils import (
    dump_parameters_to_toml, load_config_file, compute_energy
)

from PyPIC3D.initialization import (
    initialize_simulation
)

from PyPIC3D.rho import compute_rho, compute_mass_density


# Importing functions from the PyPIC3D package
############################################################################################################


def run_PyPIC3D(config_file):
    ##################################### INITIALIZE SIMULATION ################################################

    loop, particles, fields, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, \
            plasma_parameters, solver, bcs, electrostatic, verbose, GPUs, Nt, curl_func, J_func, relativistic = initialize_simulation(config_file)
    # initialize the simulation

    jit_loop = jax.jit(loop, static_argnames=('curl_func', 'J_func','solver', 'x_bc', 'y_bc', 'z_bc', 'relativistic'))

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']
    x_bc, y_bc, z_bc = bcs
    # unpack relevant parameters

    scalar_field_names = ["rho", "mass_density"]
    vector_field_names = ["E", "B", "J"]

    ############################################################################################################

    ###################################################### SIMULATION LOOP #####################################

    for t in tqdm(range(Nt)):

        # plot the data
        if t % plotting_parameters['plotting_interval'] == 0:
            E, B, J, rho, *rest = fields
            # unpack the fields
            e_energy, b_energy, kinetic_energy = compute_energy(particles, E, B, world, constants)
            # Compute the energy of the system
            write_data(f"{output_dir}/data/total_energy.txt", t * dt, e_energy + b_energy + kinetic_energy)
            write_data(f"{output_dir}/data/electric_field_energy.txt", t * dt, e_energy)
            write_data(f"{output_dir}/data/magnetic_field_energy.txt", t * dt, b_energy)
            write_data(f"{output_dir}/data/kinetic_energy.txt", t * dt, kinetic_energy)
            # Write the total energy to a file
            total_momentum = sum(particle_species.momentum() for particle_species in particles)
            # Total momentum of the particles
            write_data(f"{output_dir}/data/total_momentum.txt", t * dt, total_momentum)
            # Write the total momentum to a file

            for species in particles:
                write_data(f"{output_dir}/data/{species.name}_kinetic_energy.txt", t * dt, species.kinetic_energy())


            if plotting_parameters['plot_phasespace']:
                write_particles_phase_space(particles, t, output_dir)

            rho = compute_rho(particles, rho, world, constants)
            # calculate the charge density based on the particle positions

            mass_density = compute_mass_density(particles, rho, world)
            # calculate the mass density based on the particle positions

            fields_mag = [rho[:,:,world['Nz']//2], mass_density[:,:,world['Nz']//2]]
            plot_field_slice_vtk(fields_mag, scalar_field_names, 2, E_grid, t, "scalar_field", output_dir, world)
            # Plot the scalar fields in VTK format


            vector_field_slices = [ [E[0][:,world['Ny']//2,:], E[1][:,world['Ny']//2,:], E[2][:,world['Ny']//2,:]],
                                    [B[0][:,world['Ny']//2,:], B[1][:,world['Ny']//2,:], B[2][:,world['Ny']//2,:]],
                                    [J[0][:,world['Ny']//2,:], J[1][:,world['Ny']//2,:], J[2][:,world['Ny']//2,:]]]
            plot_vectorfield_slice_vtk(vector_field_slices, vector_field_names, 1, E_grid, t, 'vector_field', output_dir, world)
            # Plot the vector fields in VTK format

            if plotting_parameters['plot_vtk_particles']:
                plot_vtk_particles(particles, t, output_dir)
            # Plot the particles in VTK format

        particles, fields = jit_loop(particles, fields, E_grid, B_grid, world, constants, curl_func, J_func, solver, x_bc, y_bc, z_bc, relativistic=relativistic)
        # time loop to update the particles and fields

    return Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, fields, world

def main():
    ###################### JAX SETTINGS ########################################################################
    jax.config.update("jax_enable_x64", True)
    # set Jax to use 64 bit precision
    # jax.config.update("jax_debug_nans", True)
    # debugging for nans
    jax.config.update('jax_platform_name', 'cpu')
    # set Jax to use CPUs
    #jax.config.update("jax_disable_jit", True)
    ############################################################################################################

    toml_file = load_config_file()
    # load the configuration file

    start = time.time()
    # start the timer

    Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, fields, world =  block_until_ready(run_PyPIC3D(toml_file))
    # run the PyPIC3D simulation

    end = time.time()
    # end the timer

    E, B, J, *rest = fields
    # unpack the fields

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