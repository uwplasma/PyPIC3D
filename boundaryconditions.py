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

# Christopher Woolford, Oct 22 2024
# This file contains functions that apply boundary conditions to a field.

def apply_supergaussian_boundary_condition(field, boundary_thickness, order, strength):
    """
    Apply Super-Gaussian absorbing boundary conditions to the given field.

    Parameters:
    field (ndarray): The field to which the Super-Gaussian boundary condition is applied.
    boundary_thickness (int): The thickness of the absorbing boundary layer.
    order (int): The order of the Super-Gaussian function.
    strength (float): The strength of the absorption.

    Returns:
    ndarray: The field with Super-Gaussian boundary conditions applied.
    """
    def supergaussian_factor(x, thickness, order, strength):
        return np.exp(-strength * (x / thickness)**order)

    nx, ny, nz = field.shape
    for i in range(boundary_thickness):
        factor = supergaussian_factor(i, boundary_thickness, order, strength)
        field[i, :, :] *= factor
        field[nx - 1 - i, :, :] *= factor
        field[:, i, :] *= factor
        field[:, ny - 1 - i, :] *= factor
        field[:, :, i] *= factor
        field[:, :, nz - 1 - i] *= factor

    return field