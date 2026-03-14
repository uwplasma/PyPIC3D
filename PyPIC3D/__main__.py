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
from jax import lax
import jax.numpy as jnp
from tqdm import tqdm

#from memory_profiler import profile
# Importing relevant libraries

from PyPIC3D.utils import (
    dump_parameters_to_toml, load_config_file, compute_energy,
    setup_pmd_files
)

from PyPIC3D.initialization import (
    initialize_simulation
)

# Importing functions from the PyPIC3D package
############################################################################################################


def run_PyPIC3D(config_file):
    ##################################### INITIALIZE SIMULATION ################################################

    loop, particles, fields, world, simulation_parameters, constants, plotting_parameters, plasma_parameters, solver, electrostatic, verbose, GPUs, Nt, curl_func, J_func, relativistic = initialize_simulation(config_file)
    # initialize the simulation

    # `loop` is already jitted in `PyPIC3D.evolve`; avoid double-jitting.
    jit_loop = loop
    step_impl = getattr(loop, "__wrapped__", None)

    dt = world['dt']
    output_dir = simulation_parameters['output_dir']
    vertex_grid = world['grids']['vertex']
    # unpack relevant parameters

    scalar_field_names = ["rho", "mass_density"]
    vector_field_names = ["E", "B", "J"]

    E, B, J, rho, *rest = fields
    # unpack the fields
    e_energy, b_energy, kinetic_energy = compute_energy(particles, E, B, world, constants)
    # Compute the energy of the system
    initial_energy = e_energy + b_energy + kinetic_energy

    if plotting_parameters.get('plot_openpmd_fields', False) or plotting_parameters.get('plot_openpmd_particles', False):
        setup_pmd_files(os.path.join(output_dir, "data"), "fields", ".h5")
        setup_pmd_files(os.path.join(output_dir, "data"), "particles", ".h5")
    # setup the openPMD files if needed

    ############################################################################################################

    ###################################################### SIMULATION LOOP #####################################
    do_plotting = bool(plotting_parameters.get("plotting", True))
    plotting_interval = int(plotting_parameters.get("plotting_interval", 0) or 0)

    def do_diagnostics(t, particles, fields):
        if not (do_plotting and plotting_interval > 0 and (t % plotting_interval == 0)):
            return fields

        from PyPIC3D.diagnostics.plotting import write_data, write_particles_phase_space

        plot_num = t // plotting_interval
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

        if plotting_parameters['plot_phasespace']:
            write_particles_phase_space(particles, t, output_dir)

        if plotting_parameters['plot_vtk_scalars']:
            from PyPIC3D.diagnostics.vtk import plot_field_slice_vtk
            from PyPIC3D.diagnostics.fluid_quantities import compute_mass_density
            from PyPIC3D.rho import compute_rho

            rho = compute_rho(particles, rho, world, constants)
            # calculate the charge density based on the particle positions
            mass_density = compute_mass_density(particles, rho, world)
            # calculate the mass density based on the particle positions

            fields_mag = [rho[:,world['Ny']//2,:], mass_density[:,world['Ny']//2,:]]
            plot_field_slice_vtk(fields_mag, scalar_field_names, 1, vertex_grid, t, "scalar_field", output_dir, world)
            # Plot the scalar fields in VTK format

        if plotting_parameters['plot_vtk_vectors']:
            from PyPIC3D.diagnostics.vtk import plot_vectorfield_slice_vtk
            vector_field_slices = [ [E[0][:,world['Ny']//2,:], E[1][:,world['Ny']//2,:], E[2][:,world['Ny']//2,:]],
                                    [B[0][:,world['Ny']//2,:], B[1][:,world['Ny']//2,:], B[2][:,world['Ny']//2,:]],
                                    [J[0][:,world['Ny']//2,:], J[1][:,world['Ny']//2,:], J[2][:,world['Ny']//2,:]]]
            plot_vectorfield_slice_vtk(vector_field_slices, vector_field_names, 1, vertex_grid, t, 'vector_field', output_dir, world)
            # Plot the vector fields in VTK format

        if plotting_parameters['plot_vtk_particles']:
            from PyPIC3D.diagnostics.vtk import plot_vtk_particles
            plot_vtk_particles(particles, plot_num, output_dir)
        # Plot the particles in VTK format

        if plotting_parameters['plot_openpmd_particles']:
            from PyPIC3D.diagnostics.openPMD import write_openpmd_particles
            write_openpmd_particles(particles, world, constants, os.path.join(output_dir, "data"), plot_num, t, "particles", ".h5")
        # Write the particles in openPMD format

        if plotting_parameters['plot_openpmd_fields']:
            from PyPIC3D.diagnostics.openPMD import write_openpmd_fields
            write_openpmd_fields(fields, world, os.path.join(output_dir, "data"), plot_num, t,  "fields", ".h5")
        # Write the fields in openPMD format

        return (E, B, J, rho, *rest)

    use_scan = bool(simulation_parameters.get("use_scan", False))
    scan_chunk = int(simulation_parameters.get("scan_chunk", 256) or 256)

    if use_scan:
        def _scan_chunk(particles, fields, *, n_steps: int):
            def body(carry, _):
                p, f = carry
                p, f = jit_loop(
                    p,
                    f,
                    world,
                    constants,
                    curl_func,
                    J_func,
                    solver,
                    relativistic=relativistic,
                )
                return (p, f), None

            (p, f), _ = lax.scan(body, (particles, fields), xs=None, length=n_steps)
            return p, f

        scan_chunk_jit = jax.jit(_scan_chunk, donate_argnums=(0, 1), static_argnames=("n_steps",))

        chunk = scan_chunk
        if do_plotting and plotting_interval > 0 and (plotting_interval % scan_chunk):
            chunk = plotting_interval

        pbar = tqdm(total=Nt)
        t = 0
        while t < Nt:
            fields = do_diagnostics(t, particles, fields)
            remaining = Nt - t
            n_steps = chunk if remaining >= chunk else remaining
            if do_plotting and plotting_interval > 0:
                n_steps = min(n_steps, plotting_interval - (t % plotting_interval))

            particles, fields = scan_chunk_jit(particles, fields, n_steps=n_steps)
            t += n_steps
            pbar.update(n_steps)

        pbar.close()

    else:
        for t in tqdm(range(Nt)):

            fields = do_diagnostics(t, particles, fields)

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
    toml_file = load_config_file()
    # load the configuration file

    enable_x64 = bool(toml_file.get("simulation_parameters", {}).get("enable_x64", True))
    jax.config.update("jax_enable_x64", enable_x64)
    # set Jax to use 64 bit precision (configurable via `simulation_parameters.enable_x64`)
    # jax.config.update("jax_debug_nans", True)
    # debugging for nans
    jax.config.update('jax_platform_name', 'cpu')
    # set Jax to use CPUs
    #jax.config.update("jax_disable_jit", True)
    ############################################################################################################

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
