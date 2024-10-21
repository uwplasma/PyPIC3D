import time
import numpy as np
import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
from jax import lax
from jax._src.scipy.sparse.linalg import _vdot_real_tree, _add, _sub, _mul
from jax.tree_util import tree_leaves
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools
from functools import partial
# import external libraries

def default_parameters():
    """
    Returns a dictionary of default parameters for the simulation.

    Returns:
    dict: A dictionary of default parameters.
    """
    plotting_parameters = {
    "save_data": False,
    "plotfields": True,
    "plotpositions": False,
    "plotvelocities": False,
    "plotKE": False,
    "plotEnergy": False,
    "plasmaFreq": False,
    "phaseSpace": True,
    "plotting_interval": 10
    }
    # dictionary for plotting/saving data

    simulation_parameters = {
        "name": "Default Simulation",
        "output_dir": ".",
        "bc": "spectral",  # boundary conditions: periodic, dirichlet, neumann, spectral
        "eps": 8.854e-12,  # permitivity of freespace
        "mu" : 1.2566370613e-6, # permeability of free space
        "C": 3e8,  # Speed of light in m/s
        "kb": 1.380649e-23,  # Boltzmann's constant in J/K
        "me": 9.1093837e-31,  # mass of the electron in Kg
        "mi": 1.67e-27,  # mass of the ion in Kg
        "q_e": -1.602e-19,  # charge of electron
        "q_i": 1.602e-19,  # charge of ion
        "Te": 233000,  # electron temperature in K
        "Ti": 233000,  # ion temperature in K
        "N_electrons": 5000,  # number of electrons
        "N_ions": 5000,  # number of ions
        "Nx": 30,  # number of array spacings in x
        "Ny": 30,  # number of array spacings in y
        "Nz": 30,  # number of array spacings in z
        "x_wind": 1e-2,  # size of the spatial window in x in meters
        "y_wind": 1e-2,  # size of the spatial window in y in meters
        "z_wind": 1e-2,  # size of the spatial window in z in meters
        "t_wind": 1e-12,  # size of the temporal window in seconds
        "benchmark": False, # boolean for using the profiler
        "verbose": False # boolean for printing verbose output
    }
    # dictionary for simulation parameters

    return plotting_parameters, simulation_parameters