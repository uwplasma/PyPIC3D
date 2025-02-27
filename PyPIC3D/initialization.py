import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import os
import functools
from functools import partial
import toml

from PyPIC3D.particle import (
    load_particles_from_toml
)

from PyPIC3D.utils import (
    courant_condition,
    update_parameters_from_toml,
    build_yee_grid, convert_to_jax_compatible, load_external_fields_from_toml,
    check_stability, print_stats, particle_sanity_check, build_plasma_parameters_dict,
)

from PyPIC3D.fields import (
    calculateE, initialize_fields
)

from PyPIC3D.pstd import (
    spectral_curl
)

from PyPIC3D.fdtd import (
    centered_finite_difference_curl
)

from PyPIC3D.pec import (
    read_pec_boundaries_from_toml
)

from PyPIC3D.plotting import (
    plot_initial_KE
)

from PyPIC3D.laser import (
    load_lasers_from_toml
)

from PyPIC3D.boundaryconditions import (
    load_material_surfaces_from_toml
)

from PyPIC3D.evolve import (
    time_loop_electrodynamic, time_loop_electrostatic
)

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
    "phaseSpace": False,
    "plot_errors": False,
    "plot_dispersion": False,
    "plotting_interval": 10
    }
    # dictionary for plotting/saving data

    simulation_parameters = {
        "name": "Default Simulation",
        "output_dir": ".",
        "solver": "spectral",  # solver: spectral, fdtd, autodiff
        "bc": "spectral",  # boundary conditions: periodic, dirichlet, neumann
        "Nx": 30,  # number of array spacings in x
        "Ny": 30,  # number of array spacings in y
        "Nz": 30,  # number of array spacings in z
        "x_wind": 1e-2,  # size of the spatial window in x in meters
        "y_wind": 1e-2,  # size of the spatial window in y in meters
        "z_wind": 1e-2,  # size of the spatial window in z in meters
        "t_wind": 1e-12,  # size of the temporal window in seconds
        "electrostatic": False,  # boolean for electrostatic simulation
        "benchmark": False, # boolean for using the profiler
        "verbose": False, # boolean for printing verbose output
        "GPUs": False, # boolean for using GPUs
        "ncores": 4, # number of cores to use
        "ncpus": 1, # number of CPUs to use
        "cfl"  : 1, # CFL condition number
        "NN" : False, # boolean for using neural networks
        "model_name": None, # neural network model name
    }
    # dictionary for simulation parameters

    constants = {
        "eps": 8.854e-12,  # permitivity of freespace
        "mu" : 1.2566370613e-6, # permeability of free space
        "C": 3e8,  # Speed of light in m/s
        "kb": 1.380649e-23,  # Boltzmann's constant in J/K
    }

    return plotting_parameters, simulation_parameters, constants
    # return the dictionaries


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
            E_grid, B_grid (dict): Grids for electric and magnetic fields.
            world (dict): Dictionary containing world parameters such as spatial resolution and domain size.
            simulation_parameters (dict): Dictionary containing simulation parameters.
            constants (dict): Dictionary containing physical constants.
            plotting_parameters (dict): Dictionary containing parameters for plotting.
            plasma_parameters (dict): Dictionary containing plasma parameters.
            M (jax.numpy.ndarray or None): Preconditioner matrix, if neural network preconditioning is used.
            solver (str): Solver type, either "spectral" or "fdtd".
            bc (str): Boundary conditions.
            electrostatic (bool): Flag indicating if the simulation is electrostatic.
            verbose (bool): Flag indicating if verbose output is enabled.
            GPUs (int): Number of GPUs to use.
            start (float): Start time of the simulation initialization.
            Nt (int): Number of time steps.
            curl_func (function): Function to compute the curl of the fields.
            pecs (list): List of perfectly electrical conductor boundaries.
            lasers (list): List of laser objects initialized from the configuration file.
            surfaces (list): List of material surfaces initialized from the configuration file.
    """

    plotting_parameters, simulation_parameters, constants = default_parameters()
    # load the default parameters

    if toml_file is not None:
        simulation_parameters, plotting_parameters, constants = update_parameters_from_toml(toml_file, simulation_parameters, plotting_parameters, constants)

    print(f"Initializing Simulation: { simulation_parameters['name'] }\n")

    x_wind, y_wind, z_wind = simulation_parameters['x_wind'], simulation_parameters['y_wind'], simulation_parameters['z_wind']
    Nx, Ny, Nz = simulation_parameters['Nx'], simulation_parameters['Ny'], simulation_parameters['Nz']
    t_wind = simulation_parameters['t_wind']
    electrostatic = simulation_parameters['electrostatic']
    solver = simulation_parameters['solver']
    bc = simulation_parameters['bc']
    verbose = simulation_parameters['verbose']
    GPUs = simulation_parameters['GPUs']
    ncores = simulation_parameters['ncores']
    ncpus = simulation_parameters['ncpus']
    # set the simulation parameters

    os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={ncores}'
    # set the number of cores to use

    dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
    # compute the spatial resolution
    courant_number = simulation_parameters['cfl']
    dt = courant_condition(courant_number, dx, dy, dz, simulation_parameters, constants)
    Nt     = int( t_wind / dt )
    # Nt for resolution
    world = {'dt': dt, 'Nt': Nt, 'dx': dx, 'dy': dy, 'dz': dz, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'x_wind': x_wind, 'y_wind': y_wind, 'z_wind': z_wind}
    # set the simulation world parameters

    world = convert_to_jax_compatible(world)
    constants = convert_to_jax_compatible(constants)
    # convert the world parameters to jax compatible format

    E_grid, B_grid = build_yee_grid(world)
    # build the grid for the fields


    if not os.path.exists(f"{simulation_parameters['output_dir']}/data"):
        os.makedirs(f"{simulation_parameters['output_dir']}/data")
        # create the data directory if it doesn't exist
    ################################### INITIALIZE PARTICLES AND FIELDS ########################################################

    particles = load_particles_from_toml(toml_file, simulation_parameters, world, constants)
    # load the particles from the configuration file

    print_stats(world)
    # print the statistics of the simulation

    plot_initial_KE(particles, path=simulation_parameters['output_dir'])
    # plot the initial kinetic energy of the particles

    plasma_parameters = build_plasma_parameters_dict(world, constants, particles[0], dt)
    # build the plasma parameters dictionary

    particle_sanity_check(particles)
    # ensure the arrays for the particles are of the correct shape

    check_stability(plasma_parameters, dt)
    # check the stability of the simulation

    devices = mesh_utils.create_device_mesh((ncpus,))
    mesh = Mesh(devices, ('data',))
    sharding = NamedSharding(mesh, PartitionSpec('data',))
    init_fields = jax.jit(initialize_fields, out_shardings=sharding, static_argnums=(0,1,2))
    # create the mesh for the fields

    Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, phi, rho = init_fields(Nx, Ny, Nz)
    # initialize the electric and magnetic fields

    Ex_ext, Ey_ext, Ez_ext, Bx_ext, By_ext, Bz_ext = load_external_fields_from_toml([Ex, Ey, Ez, Bx, By, Bz], toml_file)
    # add any external fields to the simulation

    pecs = read_pec_boundaries_from_toml(toml_file, world)
    # read in perfectly electrical conductor boundaries

    M = None
    # specify the preconditioner matrix

    lasers = load_lasers_from_toml(toml_file, constants, world, E_grid, B_grid)
    # load the lasers from the configuration file

    surfaces = load_material_surfaces_from_toml(toml_file)
    # load the material surfaces from the configuration file

    if solver == "spectral":
        curl_func = functools.partial(spectral_curl, world=world)
    elif solver == "fdtd":
        curl_func = functools.partial(centered_finite_difference_curl, dx=dx, dy=dy, dz=dz, bc=bc)

    # if not electrostatic:
    #     Bx, By, Bz = initialize_magnetic_field(particles, E_grid, B_grid, world, constants, GPUs)
    # # initialize the magnetic field

    Ex, Ey, Ez, phi, rho = calculateE(Ex, Ey, Ez, world, particles, constants, rho, phi, M, solver, bc, verbose)
    # calculate the electric field using the Poisson equation

    Ex = Ex + Ex_ext
    Ey = Ey + Ey_ext
    Ez = Ez + Ez_ext
    # add the external fields to the electric field
    Bx = Bx + Bx_ext
    By = By + By_ext
    Bz = Bz + Bz_ext
    # add the external fields to the magnetic field

    if electrostatic:
        evolve_loop = partial(time_loop_electrostatic, E_grid=E_grid, B_grid=B_grid, world=world, constants=constants, pecs=pecs, lasers=lasers, surfaces=surfaces, \
            curl_func=curl_func, M=M, solver=solver, bc=bc, verbose=verbose, GPUs=GPUs)

    else:
        evolve_loop = partial(time_loop_electrodynamic, E_grid=E_grid, B_grid=B_grid, world=world, constants=constants, pecs=pecs, lasers=lasers, surfaces=surfaces, \
            curl_func=curl_func, M=M, solver=solver, bc=bc, verbose=verbose, GPUs=GPUs)

    return evolve_loop, particles, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, phi, \
        rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, plasma_parameters, M, \
            solver, bc, electrostatic, verbose, GPUs, Nt, curl_func, pecs, lasers, surfaces
