import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import os
import functools
from functools import partial
import toml
import matplotlib.pyplot as plt
import jax.numpy as jnp
#from memory_profiler import profile

from PyPIC3D.particle import (
    load_particles_from_toml
)

from PyPIC3D.utils import (
    courant_condition,
    update_parameters_from_toml,
    build_yee_grid, convert_to_jax_compatible, load_external_fields_from_toml,
    print_stats, particle_sanity_check, build_plasma_parameters_dict,
    make_dir
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


from PyPIC3D.plotting import (
    plot_initial_histograms
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
    'plot_chargeconservation': False,
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
        "dt": None,  # time step in seconds
        "electrostatic": False,  # boolean for electrostatic simulation
        "benchmark": False, # boolean for using the profiler
        "verbose": False, # boolean for printing verbose output
        "GPUs": False, # boolean for using GPUs
        "ncores": 4, # number of cores to use
        "ncpus": 1, # number of CPUs to use
        "cfl"  : 1, # CFL condition number
    }
    # dictionary for simulation parameters

    constants = {
        "eps": 8.85418782e-12,  # permitivity of freespace
        "mu" : 1.25663706e-7, # permeability of free space
        "C": 2.99792458e8,  # Speed of light in m/s
        "kb": 1.380649e-23,  # Boltzmann's constant in J/K
    }

    return plotting_parameters, simulation_parameters, constants
    # return the dictionaries


def setup_write_dir(simulation_parameters, plotting_parameters):
        output_dir = simulation_parameters['output_dir']
        # get the output directory from the simulation parameters
        make_dir(f'{output_dir}/data')
        # make the directory for the data
        if plotting_parameters['plotfields']:
            make_dir( f"{output_dir}/data/E_slice" )
            # make the directory for the electric field slices
            make_dir( f"{output_dir}/data/B_slice" )
            # make the directory for the magnetic field slices
            make_dir( f"{output_dir}/data/Exy_slice" )
            # make the directory for the electric field xy slices
            make_dir( f"{output_dir}/data/Exz_slice" )
            # make the directory for the electric field xz slices
            make_dir( f"{output_dir}/data/Eyz_slice" )
            # make the directory for the electric field yz slices
            make_dir( f"{output_dir}/data/Bxy_slice" )
            # make the directory for the magnetic field xy slices
            make_dir( f"{output_dir}/data/Bxz_slice" )
            # make the directory for the magnetic field xz slices
            make_dir( f"{output_dir}/data/Byz_slice" )
            # make the directory for the magnetic field yz slices


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
    #ncores = simulation_parameters['ncores']
    ncpus = simulation_parameters['ncpus']
    # set the simulation parameters

    if 'ncores' in simulation_parameters:
        os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={simulation_parameters['ncores']}'
    # set the number of cores to use

    setup_write_dir(simulation_parameters, plotting_parameters)
    # setup the write directory

    dx, dy, dz = x_wind/Nx, y_wind/Ny, z_wind/Nz
    # compute the spatial resolution
    #print(simulation_parameters)
    if simulation_parameters['dt'] is not None:
        print(f"Using user defined dt: {simulation_parameters['dt']}")
        dt = simulation_parameters['dt']
    else:
        courant_number = simulation_parameters['cfl']
        dt = courant_condition(courant_number, dx, dy, dz, simulation_parameters, constants)
    # compute the time step
    Nt     = int( t_wind / dt )
    # Nt for resolution
    world = {'dt': dt, 'Nt': Nt, 'dx': dx, 'dy': dy, 'dz': dz, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 'x_wind': x_wind, 'y_wind': y_wind, 'z_wind': z_wind}
    # set the simulation world parameters

    world = convert_to_jax_compatible(world)
    constants = convert_to_jax_compatible(constants)
    simulation_parameters = convert_to_jax_compatible(simulation_parameters)
    plotting_parameters = convert_to_jax_compatible(plotting_parameters)
    # convert the world parameters to jax compatible format

    E_grid, B_grid = build_yee_grid(world)
    # build the grid for the fields

    if not os.path.exists(f"{simulation_parameters['output_dir']}/data"):
        os.makedirs(f"{simulation_parameters['output_dir']}/data")
        # create the data directory if it doesn't exist
    ################################### INITIALIZE PARTICLES AND FIELDS ########################################################

    particles = load_particles_from_toml(toml_file, simulation_parameters, world, constants)
    # load the particles from the configuration file

    for species in particles:
        name = species.get_name()
        name = name.replace(" ", "_")
        # replace spaces with underscores in the name
        plot_initial_histograms(species, world, path=f"{simulation_parameters['output_dir']}/data", name=name)
        # plot the initial histograms of the particles

    print_stats(world)
    # print the statistics of the simulation

    plasma_parameters = build_plasma_parameters_dict(world, constants, particles[0], dt)
    # build the plasma parameters dictionary

    particle_sanity_check(particles)
    # ensure the arrays for the particles are of the correct shape

    E, B, J, phi, rho = initialize_fields(Nx, Ny, Nz)
    # initialize the electric and magnetic fields

    # load any external fields
    fields = [component for field in [E, B, J] for component in field]
    # convert the E, B, and J tuples into one big list
    fields = load_external_fields_from_toml(fields, toml_file)
    # add any external fields to the simulation
    E, B, J = fields[:3], fields[3:6], fields[6:9]
    # convert the fields list back into tuples

    if solver == "spectral":
        curl_func = functools.partial(spectral_curl, world=world)
    elif solver == "fdtd":
        curl_func = functools.partial(centered_finite_difference_curl, dx=dx, dy=dy, dz=dz, bc=bc)


    E, phi, rho = calculateE(world, particles, constants, rho, phi, solver, bc)
    # calculate the electric field using the Poisson equation

    ######################### COMPUTE INITIAL ENERGY ########################################################
    Ex, Ey, Ez = E
    Bx, By, Bz = B
    E2_integral = jnp.trapezoid(  jnp.trapezoid(  jnp.trapezoid(Ex**2 + Ey**2 + Ez**2, dx=dx, axis=0), dx=dy, axis=0), dx=dz, axis=0)
    B2_integral = jnp.trapezoid(  jnp.trapezoid(  jnp.trapezoid(Bx**2 + By**2 + Bz**2, dx=dx, axis=0), dx=dy, axis=0), dx=dz, axis=0)
    # Integral of E^2 and B^2 over the entire grid
    e_energy = 0.5 * constants['eps'] * E2_integral
    b_energy = 0.5 / constants['mu'] * B2_integral
    # Electric and magnetic field energy
    print(f"Initial Electric Field Energy: {e_energy:.2e} J")
    print(f"Initial Magnetic Field Energy: {b_energy:.2e} J")
    # print the initial electric and magnetic field energy
    kinetic_energy = sum([species.kinetic_energy() for species in particles])
    # compute the kinetic energy of the particles
    print(f"Initial Kinetic Energy: {kinetic_energy:.2e} J")
    # print the initial kinetic energy
    print(f"Total Initial Energy: {e_energy + b_energy + kinetic_energy:.2e} J\n")
    
    



    if electrostatic:
        evolve_loop = time_loop_electrostatic

    else:
        evolve_loop = time_loop_electrodynamic

    return evolve_loop, particles, E, B, J, phi, \
        rho, E_grid, B_grid, world, simulation_parameters, constants, plotting_parameters, plasma_parameters, \
            solver, bc, electrostatic, verbose, GPUs, Nt, curl_func
