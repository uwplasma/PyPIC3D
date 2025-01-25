import jax
import jax.numpy as jnp
from jax import random
import os
import time
import functools
import equinox as eqx
import toml

from PyPIC3D.model import (
    PoissonPrecondition
)

from PyPIC3D.particle import (
    load_particles_from_toml
)

from PyPIC3D.utils import (
    courant_condition,
    update_parameters_from_toml,
    precondition, build_coallocated_grid,
    build_yee_grid, convert_to_jax_compatible, load_external_fields_from_toml,
    check_stability, print_stats, particle_sanity_check, build_plasma_parameters_dict
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

def default_parameters():
    """
    Returns a dictionary of default parameters for the simulation.

    Returns:
    dict: A dictionary of default parameters.
    """
    plotting_parameters = {
    "save_data": False,
    "plotfields": False,
    "plotpositions": False,
    "plotvelocities": False,
    "plotKE": False,
    "plotEnergy": False,
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


def initialize_simulation(config_file):
    """
    Initializes the simulation with the given configuration file.

    Parameters:
    config_file (str): Path to the configuration file.

    Returns:
    tuple: A tuple containing the following elements:
        - particles: Initialized particles.
        - Ex, Ey, Ez: Electric field components.
        - Bx, By, Bz: Magnetic field components.
        - Jx, Jy, Jz: Current density components.
        - phi: Electric potential.
        - rho: Charge density.
        - E_grid, B_grid: Yee grid for the fields.
        - world: Dictionary containing world parameters.
        - simulation_parameters: Dictionary containing simulation parameters.
        - constants: Dictionary containing physical constants.
        - plotting_parameters: Dictionary containing plotting parameters.
        - M: Preconditioner matrix.
        - solver: Solver type.
        - bc: Boundary conditions.
        - electrostatic: Boolean indicating if the simulation is electrostatic.
        - verbose: Boolean indicating if verbose output is enabled.
        - GPUs: Number of GPUs to use.
        - start: Start time of the simulation.
        - Nt: Number of time steps.
        - debye: Debye length.
        - theoretical_freq: Theoretical frequency.
        - thermal_velocity: Thermal velocity.
        - curl_func: Function for calculating the curl of the fields.
    """
    plotting_parameters, simulation_parameters, constants = default_parameters()
    # load the default parameters

    if os.path.exists(config_file):
        simulation_parameters, plotting_parameters, constants = update_parameters_from_toml(config_file, simulation_parameters, plotting_parameters, constants)

    print(f"Initializing Simulation: { simulation_parameters['name'] }")
    start = time.time()
    # start the timer

    x_wind, y_wind, z_wind = simulation_parameters['x_wind'], simulation_parameters['y_wind'], simulation_parameters['z_wind']
    Nx, Ny, Nz = simulation_parameters['Nx'], simulation_parameters['Ny'], simulation_parameters['Nz']
    t_wind = simulation_parameters['t_wind']
    electrostatic = simulation_parameters['electrostatic']
    solver = simulation_parameters['solver']
    bc = simulation_parameters['bc']
    verbose = simulation_parameters['verbose']
    GPUs = simulation_parameters['GPUs']
    # set the simulation parameters

    dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
    # compute the spatial resolution
    courant_number = 1
    dt = courant_condition(courant_number, dx, dy, dz, simulation_parameters, constants)
    # calculate temporal resolution using courant condition
    Nt     = int( t_wind / dt )
    # Nt for resolution

    world = {'dt': dt, 'Nt': Nt, 'dx': dx, 'dy': dy, 'dz': dz, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'x_wind': x_wind, 'y_wind': y_wind, 'z_wind': z_wind}
    # set the simulation world parameters

    world = convert_to_jax_compatible(world)
    constants = convert_to_jax_compatible(constants)
    # convert the world parameters to jax compatible format

    print_stats(world)
    ################################### INITIALIZE PARTICLES AND FIELDS ########################################################

    particles = load_particles_from_toml(config_file, simulation_parameters, world, constants)
    # load the particles from the configuration file

    plot_initial_KE(particles, path=simulation_parameters['output_dir'])
    # plot the initial kinetic energy of the particles

    plasma_parameters = build_plasma_parameters_dict(world, constants, particles[0], dt)
    # build the plasma parameters dictionary

    particle_sanity_check(particles)
    # ensure the arrays for the particles are of the correct shape

    check_stability(plasma_parameters, dt)
    # check the stability of the simulation

    Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, phi, rho = initialize_fields(world)
    # initialize the electric and magnetic fields

    Ex_ext, Ey_ext, Ez_ext, Bx_ext, By_ext, Bz_ext = load_external_fields_from_toml([Ex, Ey, Ez, Bx, By, Bz], config_file)
    # add any external fields to the simulation

    # import matplotlib.pyplot as plt
    # plt.plot(Ex[:, 15, 15])
    # plt.show()
    # exit()

    pecs = read_pec_boundaries_from_toml(config_file, world)
    # read in perfectly electrical conductor boundaries

    ##################################### Neural Network Preconditioner ################################################

    key = jax.random.PRNGKey(0)
    # define the random key
    M = None
    if simulation_parameters['NN']:
        model = PoissonPrecondition( Nx=Nx, Ny=Ny, Nz=Nz, hidden_dim=3000, key=key)
        # define the model
        model = eqx.tree_deserialise_leaves(simulation_parameters['model_name'], model)
        # load the model from file
    else: model = None

    M = precondition( simulation_parameters['NN'], phi, rho, model)
    # solve for the preconditioner using the neural network

    # if not electrostatic:
    #     Ex, Ey, Ez, phi, rho = calculateE(Ex, Ey, Ez, world, particles, constants, rho, phi, M, 0, solver, bc, verbose, GPUs, electrostatic)

    E_grid, B_grid = build_yee_grid(world)
    # build the grid for the fields

    if solver == "spectral":
        curl_func = functools.partial(spectral_curl, world=world)
    elif solver == "fdtd":
        curl_func = functools.partial(centered_finite_difference_curl, dx=dx, dy=dy, dz=dz, bc=bc)

    return particles, Ex, Ey, Ez, Ex_ext, Ey_ext, Ez_ext, Bx, By, Bz, Bx_ext, By_ext, Bz_ext, Jx, Jy, Jz, phi, \
        rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, plasma_parameters, M, \
            solver, bc, electrostatic, verbose, GPUs, start, Nt, curl_func, pecs