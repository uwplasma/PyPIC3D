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
    plotter, write_particles_phase_space, write_data, plot_vtk_particles, plot_field_slice_vtk
)

from PyPIC3D.utils import (
    dump_parameters_to_toml, load_config_file, compute_energy
)

from PyPIC3D.initialization import (
    initialize_simulation
)

from PyPIC3D.vector_potential import (
    E_from_A, B_from_A, initialize_vector_potential, update_vector_potential
)


from PyPIC3D.boris import (
    particle_push
)


from PyPIC3D.fields import (
    calculateE, update_B, update_E
)

from PyPIC3D.J import (
    VB_correction, J_from_rhov
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

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']

    field_names = ["E_magnitude", "B_magnitude", "Ax", "Ay", "Az"]

    A2, A1, A0 = initialize_vector_potential(J, world, constants)
    # initialize the vector potential A based on the current density J

    ############################################################################################################

    ###################################################### SIMULATION LOOP #####################################

    for t in tqdm(range(Nt)):
        # plotter(t, particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, plotting_parameters, simulation_parameters)
        # plot the data
        if t % plotting_parameters['plotting_interval'] == 0:
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

            write_particles_phase_space(particles, t, output_dir)

            E_magnitude = jnp.sqrt(E[0]**2 + E[1]**2 + E[2]**2)[:,:,world['Nz']//2]
            B_magnitude = jnp.sqrt(B[0]**2 + B[1]**2 + B[2]**2)[:,:,world['Nz']//2]
            Ax, Ay, Az = A0
            Ax_ = Ax[:,:,world['Nz']//2]
            Ay_ = Ay[:,:,world['Nz']//2]
            Az_ = Az[:,:,world['Nz']//2]
            fields = [E_magnitude, B_magnitude, Ax_, Ay_, Az_]
            plot_field_slice_vtk(fields, field_names, 2, E_grid, t, "fields", output_dir, world)
            # Plot the fields in VTK format

            if plotting_parameters['plot_vtk_particles']:
                plot_vtk_particles(particles, t, output_dir)
            # Plot the particles in VTK format

        # particles, E, B, J, phi, rho = jit_loop(particles, E, B, J, rho, phi, E_grid, B_grid, world, constants, curl_func, solver, bc)
        # time loop to update the particles and fields


        ################ PARTICLE PUSH ########################################################################################
        for i in range(len(particles)):

            particles[i] = particle_push(particles[i], E, B, E_grid, B_grid, world['dt'], constants)
            # use boris push for particle velocities

            particles[i].update_position()
            # update the particle positions

        ################ FIELD UPDATE ################################################################################################
        J, rho = J_from_rhov(particles, J, rho, constants, world)
        # calculate the current density from the particle positions and velocities

        # if t == 0:
        #     A1 = initialize_vector_potential(J, world, constants)
        #     # initialize the vector potential A based on the current density J

        #     E, phi, rho = calculateE(world, particles, constants, rho, phi, solver, bc)
        #     # calculate the electric field using the Poisson equation

        #     Ax0, Ay0, Az0 = A0
        #     # unpack the vector potential components
        #     Ex, Ey, Ez = E
        #     # unpack the electric field components

        #     Ax2 = Ax0 + 2*dt*Ex
        #     Ay2 = Ay0 + 2*dt*Ey
        #     Az2 = Az0 + 2*dt*Ez
        #     # update the vector potential based on the electric field

        #     A2 = Ax2, Ay2, Az2
        #     # new vector potential components

        A0 = A1
        A1 = A2
        # update the vector potential for the next iteration
        A2 = update_vector_potential(J, world, constants, A1, A0)
        # update the vector potential based on the current density J

        E = E_from_A(A2, A0, world)
        # calculate the electric field from the vector potential using centered finite difference
        B = B_from_A(A2, world)
        # calculate the magnetic field from the vector potential using centered finite difference

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