# Christopher Woolford July 2024
# I am writing a 3D PIC code in Python using the Jax library
# to test the feasibility of using Jax for plasma simulations to take advantage of
# Jax's auto-differentiation capabilities

########################################## IMPORT LIBRARIES #############################################
from sys import path
import os
import time
import jax
from jax import block_until_ready
import jax.numpy as jnp
from tqdm import tqdm

#from memory_profiler import profile
# Importing relevant libraries

from PyPIC3D.diagnostics.plotting import (
    write_particles_phase_space, write_data
)

from PyPIC3D.diagnostics.openPMD import (
    write_openpmd_particles, write_openpmd_fields
)

from PyPIC3D.diagnostics.vtk import (
    plot_field_slice_vtk, plot_vectorfield_slice_vtk, plot_vtk_particles
)

from PyPIC3D.utils import (
    dump_parameters_to_toml, load_config_file, compute_energy,
    setup_pmd_files
)

from PyPIC3D.initialization import (
    initialize_simulation
)

from PyPIC3D.diagnostics.fluid_quantities import (
    compute_mass_density, compute_velocity_field
)

from PyPIC3D.deposition.rho import compute_rho


# Importing functions from the PyPIC3D package
############################################################################################################


def run_PyPIC3D(config_file):
    ##################################### INITIALIZE SIMULATION ################################################

    loop, particles, fields, world, simulation_parameters, constants, plotting_parameters, plasma_parameters, solver, electrostatic, verbose, GPUs, Nt, curl_func, J_func, relativistic = initialize_simulation(config_file)
    # initialize the simulation

    jit_loop = jax.jit(loop, static_argnames=('curl_func', 'J_func', 'solver', 'relativistic'))

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']
    vertex_grid = tuple(g[1:-1] for g in world['grids']['vertex'])
    # unpack the physical interior of the vertex grid (strip ghost cell positions)

    scalar_field_names = ["rho", "mass_density"]
    vector_field_names = ["E", "B", "J"]

    E, B, J, rho, *rest = fields
    # unpack the fields
    e_energy, b_energy, kinetic_energy = compute_energy(particles, E, B, world, constants)
    # Compute the energy of the system
    initial_energy = e_energy + b_energy + kinetic_energy

    if plotting_parameters['plot_openpmd_fields']: setup_pmd_files( os.path.join(output_dir, "data"), "fields", ".h5")
    if plotting_parameters['plot_openpmd_particles']: setup_pmd_files( os.path.join(output_dir, "data"), "particles", ".h5")
    # setup the openPMD files if needed

    ############################################################################################################

    ###################################################### SIMULATION LOOP #####################################

    for t in tqdm(range(Nt)):

        # plot the data
        if t % plotting_parameters['plotting_interval'] == 0:

            plot_num = t // plotting_parameters['plotting_interval']
            # determine the plot number

            E, B, J, rho, *rest = fields
            # unpack the fields

            e_energy, b_energy, kinetic_energy = compute_energy(particles, E, B, world, constants)
            # Compute the energy of the system
            write_data(f"{output_dir}/data/total_energy.txt", t * dt, e_energy + b_energy + kinetic_energy)
            write_data(f"{output_dir}/data/energy_error.txt", t * dt, abs( initial_energy - (e_energy + b_energy + kinetic_energy)) / max(initial_energy, 1e-10))
            write_data(f"{output_dir}/data/electric_field_energy.txt", t * dt, e_energy)
            write_data(f"{output_dir}/data/magnetic_field_energy.txt", t * dt, b_energy)
            write_data(f"{output_dir}/data/kinetic_energy.txt", t * dt, kinetic_energy)
            # Write the total energy to a file
            total_momentum = sum(particle_species.momentum() for particle_species in particles)
            # Total momentum of the particles
            write_data(f"{output_dir}/data/total_momentum.txt", t * dt, total_momentum)
            # Write the total momentum to a file

            # for species in particles:
            #     write_data(f"{output_dir}/data/{species.name}_kinetic_energy.txt", t * dt, species.kinetic_energy())


            if plotting_parameters['plot_phasespace']:
                write_particles_phase_space(particles, t, output_dir)



            if plotting_parameters['plot_vtk_scalars']:
                rho = compute_rho(particles, rho, world, constants)
                # calculate the charge density based on the particle positions
                mass_density = compute_mass_density(particles, rho, world)
                # calculate the mass density based on the particle positions

                y_mid = world['Ny']//2 + 1
                # midplane index shifted by 1 for ghost cells
                fields_mag = [rho[1:-1, y_mid, 1:-1], mass_density[1:-1, y_mid, 1:-1]]
                plot_field_slice_vtk(fields_mag, scalar_field_names, 1, vertex_grid, t, "scalar_field", output_dir, world)
                # Plot the scalar fields in VTK format


            if plotting_parameters['plot_vtk_vectors']:
                y_mid = world['Ny']//2 + 1
                # midplane index shifted by 1 for ghost cells
                vector_field_slices = [ [E[0][1:-1, y_mid, 1:-1], E[1][1:-1, y_mid, 1:-1], E[2][1:-1, y_mid, 1:-1]],
                                        [B[0][1:-1, y_mid, 1:-1], B[1][1:-1, y_mid, 1:-1], B[2][1:-1, y_mid, 1:-1]],
                                        [J[0][1:-1, y_mid, 1:-1], J[1][1:-1, y_mid, 1:-1], J[2][1:-1, y_mid, 1:-1]]]
                plot_vectorfield_slice_vtk(vector_field_slices, vector_field_names, 1, vertex_grid, t, 'vector_field', output_dir, world)
                # Plot the vector fields in VTK format

            if plotting_parameters['plot_vtk_particles']:
                plot_vtk_particles(particles, plot_num, output_dir)
            # Plot the particles in VTK format

            if plotting_parameters['plot_openpmd_particles']:
                write_openpmd_particles(particles, world, constants, os.path.join(output_dir, "data"), plot_num, t, "particles", ".h5")
            # Write the particles in openPMD format

            if plotting_parameters['plot_openpmd_fields']:
                interior = (slice(1, -1), slice(1, -1), slice(1, -1))

                Ux = compute_velocity_field(particles, rho, 0, world)
                Uy = compute_velocity_field(particles, rho, 1, world)
                Uz = compute_velocity_field(particles, rho, 2, world)
                # compute the velocity field for each component
                U = (Ux, Uy, Uz)

                
                field_map = {
                    "E": tuple(comp[interior] for comp in E),
                    "B": tuple(comp[interior] for comp in B),
                    "J": tuple(comp[interior] for comp in J),
                    "U": tuple(comp[interior] for comp in U),
                    "rho": rho[interior],
                }
                # create a field map for the interior of the domain (strip ghost cells)


                write_openpmd_fields(field_map, world, os.path.join(output_dir, "data"), plot_num, t,  "fields", ".h5")
            # Write the fields in openPMD format

            fields = (E, B, J, rho, *rest)
            # repack the fields

        particles, fields = jit_loop(
            particles,
            fields,
            world,
            constants,
            curl_func,
            J_func,
            solver,
            relativistic=relativistic,
        )
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
