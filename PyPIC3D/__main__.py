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

from PyPIC3D.rho import compute_rho, compute_mass_density


# Importing functions from the PyPIC3D package
############################################################################################################


def run_PyPIC3D(config_file):
    ##################################### INITIALIZE SIMULATION ################################################

    loop, particles, fields, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, \
            plasma_parameters, solver, bc, electrostatic, verbose, GPUs, Nt, curl_func, J_func, relativistic = initialize_simulation(config_file)
    # initialize the simulation

    jit_loop = jax.jit(loop, static_argnames=('curl_func', 'J_func','solver', 'bc', 'relativistic'))

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']

    field_names = ["E_magnitude", "B_magnitude", "Jz", "rho", "Ex", "Ey", "Bz", "mass_density"]

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

            write_particles_phase_space(particles, t, output_dir)

            rho = compute_rho(particles, rho, world, constants)
            # calculate the charge density based on the particle positions

            mass_density = compute_mass_density(particles, rho, world)
            # calculate the mass density based on the particle positions

            E_magnitude = jnp.sqrt(E[0]**2 + E[1]**2 + E[2]**2)[:,:,world['Nz']//2]
            B_magnitude = jnp.sqrt(B[0]**2 + B[1]**2 + B[2]**2)[:,:,world['Nz']//2]
            Ex_slice = E[0][:,:,world['Nz']//2]
            Ey_slice = E[1][:,:,world['Nz']//2]
            Bz_slice = B[2][:,:,world['Nz']//2]
            # Calculate the magnitudes of the electric and magnetic fields, and the Bz component
            fields_mag = [E_magnitude, B_magnitude, J[2][:,:,world['Nz']//2], rho[:,:,world['Nz']//2], Ex_slice, Ey_slice, Bz_slice, mass_density[:,:,world['Nz']//2]]
            plot_field_slice_vtk(fields_mag, field_names, 2, E_grid, t, "fields", output_dir, world)
            # Plot the fields in VTK format

            if plotting_parameters['plot_vtk_particles']:
                plot_vtk_particles(particles, t, output_dir)
            # Plot the particles in VTK format

        particles, fields = jit_loop(particles, fields, E_grid, B_grid, world, constants, curl_func, J_func, solver, bc, relativistic=relativistic)
        # time loop to update the particles and fields

    return Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, fields, world, Exs, Eys, Ezs, Bxs, Bys, Bzs, Jxs, Jys, Jzs

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

    Nt, plotting_parameters, simulation_parameters, plasma_parameters, constants, particles, fields, world, Exs, Eys, Ezs, Bxs, Bys, Bzs, Jxs, Jys, Jzs =  block_until_ready(run_PyPIC3D(toml_file))
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


    import matplotlib.pyplot as plt

    def plot_time_series(datas, fig, filename):
        plt.figure()
        im = plt.imshow(jnp.array(datas).T, aspect='auto', cmap='viridis')
        plt.xlabel('Time (t)')
        plt.ylabel('Position (x)')
        plt.title(f'{fig} Over Time')
        plt.colorbar(im, label=fig)
        plt.savefig(filename)
        plt.close()

    # plot_time_series(Jzs, 'Jz', f"{simulation_parameters['output_dir']}/data/Jz_over_time.png")
    # plot_time_series(Jxs, 'Jx', f"{simulation_parameters['output_dir']}/data/Jx_over_time.png")
    # plot_time_series(Jys, 'Jy', f"{simulation_parameters['output_dir']}/data/Jy_over_time.png")
    # plot_time_series(Exs, 'Ex', f"{simulation_parameters['output_dir']}/data/Ex_over_time.png")
    # plot_time_series(Eys, 'Ey', f"{simulation_parameters['output_dir']}/data/Ey_over_time.png")
    # plot_time_series(Ezs, 'Ez', f"{simulation_parameters['output_dir']}/data/Ez_over_time.png")
    # plot_time_series(Bys, 'By', f"{simulation_parameters['output_dir']}/data/By_over_time.png")
    # plot_time_series(Bxs, 'Bx', f"{simulation_parameters['output_dir']}/data/Bx_over_time.png")
    # plot_time_series(Bzs, 'Bz', f"{simulation_parameters['output_dir']}/data/Bz_over_time.png")

    jnp.save(f"{simulation_parameters['output_dir']}/data/Jx.npy", jnp.array(Jxs) )
    jnp.save(f"{simulation_parameters['output_dir']}/data/Jy.npy", jnp.array(Jys) )
    jnp.save(f"{simulation_parameters['output_dir']}/data/Jz.npy", jnp.array(Jzs) )
    jnp.save(f"{simulation_parameters['output_dir']}/data/Ex.npy", jnp.array(Exs) )
    jnp.save(f"{simulation_parameters['output_dir']}/data/Ey.npy", jnp.array(Eys) )
    jnp.save(f"{simulation_parameters['output_dir']}/data/Ez.npy", jnp.array(Ezs) )
    jnp.save(f"{simulation_parameters['output_dir']}/data/By.npy", jnp.array(Bys) )
    jnp.save(f"{simulation_parameters['output_dir']}/data/Bx.npy", jnp.array(Bxs) )
    jnp.save(f"{simulation_parameters['output_dir']}/data/Bz.npy", jnp.array(Bzs) )


if __name__ == "__main__":
    main()
    # run the main function