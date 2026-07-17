import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import os
from functools import partial
import toml
import matplotlib.pyplot as plt
import jax.numpy as jnp
#from memory_profiler import profile


from PyPIC3D.particles.particle_initialization import (
    load_particles_from_toml
)

from PyPIC3D.utils import (
    courant_condition,
    update_parameters_from_toml,
    convert_to_jax_compatible, load_external_fields_from_toml,
    print_stats, particle_sanity_check, build_plasma_parameters_dict,
    make_dir, compute_energy, add_external_fields
)
from PyPIC3D.utilities.grids import (
    build_collocated_grid,
    build_tiled_yee_grids,
    build_yee_grid,
)

from PyPIC3D.diagnostics.plotting import (
    plot_initial_histograms
)
from PyPIC3D.diagnostics.output_adapters import particles_for_output

from PyPIC3D.diagnostics.openPMD import (
    write_openpmd_initial_particles, write_openpmd_initial_fields
)

from PyPIC3D.boundary_conditions.ghost_cells import (
    make_field_mesh,
    update_tiled_vector_ghost_cells,
)


from PyPIC3D.evolve import (
    time_loop_electrodynamic,
    time_loop_electrostatic,
)


from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC
from PyPIC3D.boundary_conditions.PML import (
    initialize_tiled_pml_state,
    load_pml_from_toml,
)
from PyPIC3D.parameters import build_dynamic_parameters, build_static_parameters


def _encode_field_bc(bc_name):
    """
    Encode field boundary condition labels into integer codes for JAX-safe storage.
    """
    bc_codes = {
        "periodic": BC_PERIODIC,
        "conducting": BC_CONDUCTING,
    }
    if bc_name not in bc_codes:
        raise ValueError(f"Unsupported field boundary condition: {bc_name}")
    return bc_codes[bc_name]


def _encode_particle_bc(bc_name):
    """
    Encode global particle boundary condition labels into integer codes.
    """
    bc_codes = {
        "periodic": 0,
        "reflecting": 1,
        "absorbing": 2,
    }
    if bc_name not in bc_codes:
        raise ValueError(f"Unsupported particle boundary condition: {bc_name}")
    return bc_codes[bc_name]


def validate_field_solver(solver):
    """
    Keep the active field-solver names explicit so stale configs do not silently
    fall through to a different numerical update.
    """
    supported_solvers = ("electrodynamic_yee", "electrostatic")
    if solver not in supported_solvers:
        raise ValueError(
            f"Unsupported solver: {solver}. Use 'electrodynamic_yee' or 'electrostatic'."
        )


def _tile_shape_from_parameters(simulation_parameters):
    return (
        int(simulation_parameters["particle_tile_nx"]),
        int(simulation_parameters["particle_tile_ny"]),
        int(simulation_parameters["particle_tile_nz"]),
    )


def _encode_current_calculation(current_calculation):
    if current_calculation not in ("j_from_rhov", "esirkepov"):
        raise ValueError("Unsupported current_calculation. Use 'j_from_rhov' or 'esirkepov'.")
    if current_calculation == "esirkepov":
        return "esirkepov"
    return "direct"


def _validate_current_filter_contract(simulation_parameters):
    if (
        simulation_parameters["current_calculation"] == "esirkepov"
        and simulation_parameters["filter_j"] != "none"
    ):
        raise ValueError("Esirkepov current filtering is not supported; use filter_j='none'.")


def _validate_tiled_yee_configuration(simulation_parameters, electrostatic, pml_active):
    """
    Keep the first tile-native PIC path tied to the kernels that exist today.
    """

    if simulation_parameters["current_calculation"] not in ("j_from_rhov", "esirkepov"):
        raise ValueError("Yee runtime currently supports current_calculation='j_from_rhov' or 'esirkepov'")
    if simulation_parameters["particle_pusher"] not in ("boris", "higuera_cary"):
        raise ValueError("Yee runtime currently supports only particle_pusher='boris' or 'higuera_cary'")
    if simulation_parameters["filter_j"] not in ("none", "digital", "bilinear"):
        raise ValueError("Yee runtime currently supports only filter_j='none', filter_j='digital', or 'bilinear'")
    tile_shape = _tile_shape_from_parameters(simulation_parameters)
    grid_shape = (
        int(simulation_parameters["Nx"]),
        int(simulation_parameters["Ny"]),
        int(simulation_parameters["Nz"]),
    )
    for cells, tile_width in zip(grid_shape, tile_shape):
        if cells % tile_width != 0:
            raise ValueError("Yee runtime requires the shared tile shape to divide Nx/Ny/Nz exactly")


def _apply_pml_field_boundaries(world):
    """
    PML-active axes use nonwrapping field halos from initialization onward.
    """

    _, pml_x, pml_y, pml_z, _ = world["pml"]
    for axis, pml_axis_active in zip(("x", "y", "z"), (pml_x, pml_y, pml_z)):
        if pml_axis_active and world["boundary_conditions"][axis] == BC_PERIODIC:
            world["boundary_conditions"][axis] = BC_CONDUCTING


def default_parameters():
    """
    Returns a dictionary of default parameters for the simulation.

    Returns:
    dict: A dictionary of default parameters.
    """
    plotting_parameters = {
    "plotting" : True,
    "save_data": False,
    "plotfields": False,
    "plotpositions": False,
    "plotvelocities": False,
    "plotenergy": True,
    "plotcurrent": False,
    "plasmaFreq": False,
    "plot_phasespace": False,
    "plot_errors": False,
    "plot_dispersion": False,
    'plot_chargeconservation': False,
    "plot_openpmd_particles": False,
    "plot_openpmd_fields": False,
    "plotting_interval": 10,
    "openpmd_field_queue_size": 2,
    "openpmd_particle_queue_size": 2,
    "dump_particles": False,
    "dump_fields": False,
    }
    # dictionary for plotting/saving data

    simulation_parameters = {
        "name": "Default Simulation",
        "output_dir": os.getcwd(),
        "solver": "electrodynamic_yee",  # solver: electrodynamic_yee or electrostatic
        "particle_x_bc": "periodic",  # particle x boundary conditions: periodic, absorbing, reflecting
        "particle_y_bc": "periodic",  # particle y boundary conditions: periodic, absorbing, reflecting
        "particle_z_bc": "periodic",  # particle z boundary conditions: periodic, absorbing, reflecting
        # "bc": "periodic",  # boundary conditions: periodic, dirichlet, neumann
        "x_bc": "periodic",  # x boundary conditions: periodic, conducting
        "y_bc": "periodic",  # y boundary conditions: periodic, conducting
        "z_bc": "periodic",  # z boundary conditions: periodic, conducting
        "Nx": 30,  # number of array spacings in x
        "Ny": 30,  # number of array spacings in y
        "Nz": 30,  # number of array spacings in z
        "x_wind": 1e-2,  # size of the spatial window in x in meters
        "y_wind": 1e-2,  # size of the spatial window in y in meters
        "z_wind": 1e-2,  # size of the spatial window in z in meters
        "t_wind": 1e-12,  # size of the temporal window in seconds
        "dt": None,  # time step in seconds
        "Nt": None,  # number of time steps
        "relativistic": True,  # boolean for relativistic simulation
        "particle_pusher": "boris",  # particle pusher: boris, higuera_cary
        "benchmark": False, # boolean for using the profiler
        "verbose": False, # boolean for printing verbose output
        "GPUs": False, # boolean for using GPUs
        "cfl"  : 1.0, # CFL condition number
        "ds_per_debye" : None, # number of grid spacings per debye length
        "shape_factor" : 1, # shape factor for the simulation (1 for 1st order, 2 for 2nd order)
        "guard_cells": 2, # tile guard-cell depth shared by tiled fields and tiled currents
        "particle_tile_nx": None, # number of x cells per shared field/particle tile
        "particle_tile_ny": None, # number of y cells per shared field/particle tile
        "particle_tile_nz": None, # number of z cells per shared field/particle tile
        "particle_tile_capacity_factor": 1.0, # inactive particle slot headroom per tile
        "current_calculation": "j_from_rhov",  # current calculation method: esirkepov, villasenor_buneman, j_from_rhov
        "filter_j": "bilinear",  # filter for the current density: bilinear, digital, none
    }
    # dictionary for simulation parameters

    constants = {
        "eps": 8.85418782e-12,  # permitivity of freespace
        "mu" : 1.25663706e-6, # permeability of free space
        "C": 2.99792458e8,  # Speed of light in m/s
        "kb": 1.380649e-23,  # Boltzmann's constant in J/K
        'alpha': 1.0,  # digital filter alpha value
    }

    return plotting_parameters, simulation_parameters, constants
    # return the dictionaries


def setup_write_dir(simulation_parameters, plotting_parameters):
        output_dir = simulation_parameters['output_dir']
        # get the output directory from the simulation parameters
        make_dir(f'{output_dir}/data')
        # make the directory for the data


#@profile
def initialize_simulation(toml_file):
    """
    Initializes the simulation environment based on the provided TOML configuration file.

    Args:
        toml_file (str): Path to the TOML configuration file. If None, default parameters are used.

    Returns:
        tuple: A tuple containing the following elements:
            evolve_loop (function): The function to evolve the simulation loop.
            particles (list): List of particle objects initialized from the configuration file.
            Ex, Ey, Ez (jax.numpy.ndarray): Electric field components.
            Bx, By, Bz (jax.numpy.ndarray): Magnetic field components.
            Jx, Jy, Jz (jax.numpy.ndarray): Current density components.
            phi (jax.numpy.ndarray): Electric potential.
            rho (jax.numpy.ndarray): Charge density.
            world (dict): Dictionary containing world parameters such as spatial resolution, domain size,
                field boundary conditions, and all simulation grids.
            simulation_parameters (dict): Dictionary containing simulation parameters.
            constants (dict): Dictionary containing physical constants.
            plotting_parameters (dict): Dictionary containing parameters for plotting.
            plasma_parameters (dict): Dictionary containing plasma parameters.
            M (jax.numpy.ndarray or None): Preconditioner matrix, if neural network preconditioning is used.
            solver (str): Solver type, either "electrodynamic_yee" or "electrostatic".
            electrostatic (bool): Derived flag indicating if the simulation is electrostatic.
            verbose (bool): Flag indicating if verbose output is enabled.
            GPUs (int): Number of GPUs to use.
            start (float): Start time of the simulation initialization.
            Nt (int): Number of time steps.
            pecs (list): List of perfectly electrical conductor boundaries.
            lasers (list): List of laser objects initialized from the configuration file.
            surfaces (list): List of material surfaces initialized from the configuration file.
    """

    plotting_parameters, simulation_parameters, constants = default_parameters()
    # load the default parameters

    if toml_file is not None:
        simulation_parameters, plotting_parameters, constants = update_parameters_from_toml(toml_file, simulation_parameters, plotting_parameters, constants)

    print(f"Initializing Simulation: { simulation_parameters['name'] }\n")
    print(f"Using boundary conditions: x: {simulation_parameters['x_bc']}, y: {simulation_parameters['y_bc']}, z: {simulation_parameters['z_bc']}\n")

    x_wind, y_wind, z_wind = simulation_parameters['x_wind'], simulation_parameters['y_wind'], simulation_parameters['z_wind']
    Nx, Ny, Nz = simulation_parameters['Nx'], simulation_parameters['Ny'], simulation_parameters['Nz']
    t_wind = simulation_parameters['t_wind']
    solver = simulation_parameters['solver']
    if "electrostatic" in simulation_parameters:
        raise ValueError("Use solver='electrostatic' instead of the deprecated electrostatic flag.")
    validate_field_solver(solver)
    electrostatic = solver == "electrostatic"
    relativistic = simulation_parameters['relativistic']
    particle_pusher = simulation_parameters['particle_pusher']

    if simulation_parameters['particle_tile_nx'] is None:
        simulation_parameters["particle_tile_nx"] = int(Nx)
    if simulation_parameters['particle_tile_ny'] is None:
        simulation_parameters["particle_tile_ny"] = int(Ny)
    if simulation_parameters['particle_tile_nz'] is None:
        simulation_parameters["particle_tile_nz"] = int(Nz)
    
    if electrostatic:
        simulation_parameters["particle_tile_nx"] = int(Nx)
        simulation_parameters["particle_tile_ny"] = int(Ny)
        simulation_parameters["particle_tile_nz"] = int(Nz)
        # Electrostatic uses one tile covering the whole domain.  The leading
        # tile axes remain present, but their extent is one in each direction.
    guard_cells = int(simulation_parameters["guard_cells"])
    guard_cells = max(guard_cells, 2)
    _validate_current_filter_contract(simulation_parameters)
    verbose = simulation_parameters['verbose']
    GPUs = simulation_parameters['GPUs']
    # set the simulation parameters


    setup_write_dir(simulation_parameters, plotting_parameters)
    # setup the write directory

    dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
    # compute the spatial resolution

    if simulation_parameters['dt'] is not None:
        print(f"Using user defined dt: {simulation_parameters['dt']}")
        dt = simulation_parameters['dt']
    else:
        courant_number = simulation_parameters['cfl']
        dt = courant_condition(courant_number, dx, dy, dz, simulation_parameters, constants)
    # compute the time step
    if simulation_parameters['Nt'] is not None:
        Nt = simulation_parameters['Nt']
    else:
        Nt     = int( t_wind / dt )
    # Nt for resolution

    if simulation_parameters['dt'] is not None and simulation_parameters['Nt'] is not None:
        t_wind = dt * Nt
        print(f"Adjusting t_wind to {t_wind} based on provided dt and Nt")
        simulation_parameters['t_wind'] = t_wind
    # adjust t_wind if both dt and Nt are provided


    world = {
        'dt': dt,
        'Nt': Nt,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'Nx': Nx,
        'Ny': Ny,
        'Nz': Nz,
        'x_wind': x_wind,
        'y_wind': y_wind,
        'z_wind': z_wind,
        'shape_factor': simulation_parameters['shape_factor'],
        'guard_cells': guard_cells,
        'current_deposition': _encode_current_calculation(simulation_parameters['current_calculation']),
        'current_filter': simulation_parameters['filter_j'],
        'boundary_conditions': {
            'x': _encode_field_bc(simulation_parameters['x_bc']),
            'y': _encode_field_bc(simulation_parameters['y_bc']),
            'z': _encode_field_bc(simulation_parameters['z_bc']),
        },
        'particle_boundary_conditions': {
            'x': _encode_particle_bc(simulation_parameters['particle_x_bc']),
            'y': _encode_particle_bc(simulation_parameters['particle_y_bc']),
            'z': _encode_particle_bc(simulation_parameters['particle_z_bc']),
        },
    }
    # set the simulation world parameters

    raw_pml = []
    if toml_file is not None:
        raw_pml = toml_file.get("pml", [])
    pml_active = bool(raw_pml)
    if pml_active and electrostatic:
        raise ValueError("PML is only supported for the electrodynamic_yee solver")
    _validate_tiled_yee_configuration(simulation_parameters, electrostatic, pml_active)
    world["pml"] = load_pml_from_toml(raw_pml, world, constants)
    _apply_pml_field_boundaries(world)

    world = convert_to_jax_compatible(world)
    constants = convert_to_jax_compatible(constants)
    simulation_parameters = convert_to_jax_compatible(simulation_parameters)
    plotting_parameters = convert_to_jax_compatible(plotting_parameters)
    # convert the world parameters to jax compatible format
    world["guard_cells"] = guard_cells
    simulation_parameters["guard_cells"] = guard_cells
    # Guard-cell depth controls static tiled array shapes, so keep it as a
    # Python integer for jitted tile kernels.

    if electrostatic:
        B_grid, E_grid = build_collocated_grid(world)
        # electrostatic E comes from a symmetric gradient and is colocated at cell centers
    else:
        B_grid, E_grid = build_yee_grid(world)
        # build the Yee grid for the electrodynamic fields

    world['grids'] = {
        'vertex': E_grid,
        'center': B_grid,
    }
    # set the grids in the world parameters
    tile_shape = _tile_shape_from_parameters(simulation_parameters)
    world["tile_shape"] = tile_shape
    tile_grid_shape = (
        int(Nx) // int(tile_shape[0]),
        int(Ny) // int(tile_shape[1]),
        int(Nz) // int(tile_shape[2]),
    )
    world["field_mesh"] = make_field_mesh(tile_grid_shape)
    tiled_vertex_grid, tiled_center_grid = build_tiled_yee_grids(world, tile_shape, guard_cells)
    world["grids"]["tiled_vertex_grid"] = tiled_vertex_grid
    world["grids"]["tiled_center_grid"] = tiled_center_grid
    simulation_parameters["tile_shape"] = tile_shape
    # set the shared tiled field/particle geometry before allocating fields

    if not os.path.exists(f"{simulation_parameters['output_dir']}/data"):
        os.makedirs(f"{simulation_parameters['output_dir']}/data")
        # create the data directory if it doesn't exist
    ################################### INITIALIZE PARTICLES AND FIELDS ########################################################

    particles, species_config, particle_species_names, particle_metadata = load_particles_from_toml(
        toml_file,
        simulation_parameters,
        world,
        constants,
    )
    # load particles directly into tiled runtime storage
    simulation_parameters["particle_species_names"] = particle_species_names
    simulation_parameters["particle_species_metadata"] = particle_metadata
    # keep species names and scalar metadata available for diagnostics/output

    initial_particle_records = particles_for_output(
        particles,
        species_config=species_config,
        species_names=particle_species_names,
        world=world,
    )
    for particle_record in initial_particle_records:
        name = particle_record.name.replace(" ", "_")
        # replace spaces with underscores in the name
        plot_initial_histograms(particle_record, world, path=f"{simulation_parameters['output_dir']}/data", name=name)
        # plot the initial histograms of the particles

    print_stats(world)
    # print the statistics of the simulation

    if particle_metadata:
        plasma_parameters = build_plasma_parameters_dict(world, constants, particle_metadata[0], dt)
        # build the plasma parameters dictionary
    else:
        plasma_parameters = {}
        # if no particles are loaded, set plasma parameters to empty dictionary

    particle_sanity_check(particles)
    # ensure the arrays for the particles are of the correct shape

    if plotting_parameters['dump_particles']:
        write_openpmd_initial_particles(
            particles,
            world,
            constants,
            simulation_parameters['output_dir'],
            species_config=species_config,
            species_names=particle_species_names,
        )
    # write the initial particles to an openPMD file

    E, B, J, phi, rho = initialize_fields(world)
    # initialize the electric and magnetic fields
    external_fields = (
        tuple(jax.numpy.zeros_like(comp) for comp in E),
        tuple(jax.numpy.zeros_like(comp) for comp in B),
    )
    # external_fields stores prescribed E/B that particles see but Maxwell does not evolve

    # load any external fields
    fields = [component for field in [E, B, J] for component in field]
    # convert the E, B, and J tuples into one big list
    fields, external_fields = load_external_fields_from_toml(fields, external_fields, toml_file, world)
    # route configured fields into either evolved fields or external-only fields
    E, B, J = fields[:3], fields[3:6], fields[6:9]
    # convert the fields list back into tuples

    E = update_tiled_vector_ghost_cells(E, world, num_guard_cells=guard_cells)
    B = update_tiled_vector_ghost_cells(B, world, num_guard_cells=guard_cells)
    # fill ghost cells for the initial E and B fields
    external_E, external_B = external_fields
    external_E = update_tiled_vector_ghost_cells(external_E, world, num_guard_cells=guard_cells)
    external_B = update_tiled_vector_ghost_cells(external_B, world, num_guard_cells=guard_cells)
    external_fields = (external_E, external_B)
    # fill ghost cells for external fields before they are interpolated to particles

    ######################### COMPUTE INITIAL ENERGY ########################################################
    total_E, total_B = add_external_fields(E, B, external_fields)
    # energy diagnostics use the same total fields that the particle pusher sees
    e_energy, b_energy, kinetic_energy = compute_energy(
        particles,
        total_E,
        total_B,
        world,
        constants,
        species_config=species_config,
    )
    # compute the initial energy of the system
    print(f"Initial Electric Field Energy: {e_energy:.2e} J")
    print(f"Initial Magnetic Field Energy: {b_energy:.2e} J")
    print(f"Initial Kinetic Energy: {kinetic_energy:.2e} J")
    print(f"Total Initial Energy: {e_energy + b_energy + kinetic_energy:.2e} J\n")
    # print the initial energy of the system

    if relativistic:
        print("Relativistic simulation")
    else:
        print("Non-relativistic simulation")
    print(f"Using {particle_pusher} particle pusher")
    # print the selected particle pusher

    if electrostatic:
        print("Using electrostatic solver")
        evolve_loop = time_loop_electrostatic
    else:
        print("Using electrodynamic Yee solver")
        evolve_loop = time_loop_electrodynamic
    # set the evolve loop function based on the electrostatic flag

    if simulation_parameters['current_calculation'] == "esirkepov":
        print("Using Esirkepov current calculation method")
    elif simulation_parameters['current_calculation'] == "j_from_rhov":
        print(f"Using J from rhov current calculation method with filter: {simulation_parameters['filter_j']}")


    external_E, external_B = external_fields
    external_fields = (external_E, external_B)
    print(f"Using tiled Yee storage with tile shape: {tile_shape}")

    overflow = jnp.asarray(False)
    if electrostatic:
        fields = (E, B, J, rho, phi, external_fields, None, overflow)
        # define the fields tuple for the electrostatic solver
    else:
        pml_state = None
        # electrodynamic Yee always carries the PML state slot; None means
        # ordinary, unstretched Yee derivatives.
        if pml_active:
            pml_state = initialize_tiled_pml_state(world, tile_shape)
        fields = (E, B, J, rho, phi, external_fields, pml_state, overflow)
        # define the fields tuple for the electrodynamic Yee solver

    if plotting_parameters['dump_fields']:
        write_openpmd_initial_fields(fields, world, simulation_parameters['output_dir'], filename="initial_fields.h5")
    # write the initial fields to an openPMD file


    if GPUs:
        print(f"GPUs Detected! Using GPUs for simulation\n")
        particles = jax.device_put(particles, jax.devices("gpu")[0])
        if species_config is not None:
            species_config = jax.device_put(species_config, jax.devices("gpu")[0])
    # put the particles on the GPU if GPUs are enabled

    static_parameters = build_static_parameters(
        world,
        solver=solver,
        electrostatic=electrostatic,
        relativistic=relativistic,
        particle_pusher=particle_pusher,
    )
    dynamic_parameters = build_dynamic_parameters(world, constants)
    simulation_parameters["static_parameters"] = static_parameters
    simulation_parameters["dynamic_parameters"] = dynamic_parameters
    # The jitted kernels use these split parameter groups.  The legacy world
    # and constants dictionaries remain available to diagnostics and output.

    return evolve_loop, particles, fields, world, simulation_parameters, constants, plotting_parameters, plasma_parameters, \
        solver, electrostatic, verbose, GPUs, Nt, relativistic, particle_pusher, species_config


def build_tiled_array(world, dtype=jnp.float64):
    """
    Build one zero-filled tiled field component from the geometry in world.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in world["tile_shape"]]
    Nx = int(world["Nx"])
    Ny = int(world["Ny"])
    Nz = int(world["Nz"])
    g = int(world["guard_cells"])
    ntx = Nx // tile_nx
    nty = Ny // tile_ny
    ntz = Nz // tile_nz
    tiled_shape = (ntx, nty, ntz, tile_nx + 2 * g, tile_ny + 2 * g, tile_nz + 2 * g)
    return jnp.zeros(shape=tiled_shape, dtype=dtype)


def initialize_fields(world):
    """
    Initializes the electric and magnetic field arrays, as well as the electric potential and charge density arrays.

    The field state is tiled. The tile shape and guard-cell depth are stored in
    world and are shared by E, B, J, phi, and rho.

    Returns:
        E, B, J, phi, rho initialized to zero in tiled storage.
    """

    Ex = build_tiled_array(world)
    Ey = build_tiled_array(world)
    Ez = build_tiled_array(world)
    # initialize the electric field arrays as 0
    Bx = build_tiled_array(world)
    By = build_tiled_array(world)
    Bz = build_tiled_array(world)
    # initialize the magnetic field arrays as 0
    Jx = build_tiled_array(world)
    Jy = build_tiled_array(world)
    Jz = build_tiled_array(world)
    # initialize the current density arrays as 0
    phi = build_tiled_array(world)
    rho = build_tiled_array(world)
    # initialize the electric potential and charge density arrays as 0

    return (Ex, Ey, Ez), (Bx, By, Bz), (Jx, Jy, Jz), phi, rho
