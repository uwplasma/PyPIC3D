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
    return {
        'N_electrons': 1e6,
        'N_ions': 1e6,
        'x_wind': 1.0,
        'y_wind': 1.0,
        'z_wind': 1.0,
        'eps': 8.854e-12,
        'me': 9.10938356e-31,
        'mi': 1.6726219e-27,
        'q_e': 1.60217662e-19,
        'q_i': 1.60217662e-19,
        'dx': 0.01,
        'dy': 0.01,
        'dz': 0.01,
        'dt': 1e-9,
        'C': 0.1,
        'timesteps': 1000,
        'output_dir': 'output'
    }