from jax import jit
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp

# Christopher Woolford, Oct 22 2024
# This file contains functions that apply boundary conditions to a field.

def apply_supergaussian_boundary_condition(field, boundary_thickness, order, strength):
    """
    Apply Super-Gaussian absorbing boundary conditions to the given field.

    Args:
        field (ndarray): The field to which the Super-Gaussian boundary condition is applied.
        boundary_thickness (int): The thickness of the absorbing boundary layer.
        order (int): The order of the Super-Gaussian function.
        strength (float): The strength of the absorption.

    Returns:
        ndarray: The field with Super-Gaussian boundary conditions applied.
    """
    def supergaussian_factor(x, thickness, order, strength):
        return jnp.exp(-strength * (x / thickness)**order)

    nx, ny, nz = field.shape
    for i in range(boundary_thickness):
        factor = supergaussian_factor(i, boundary_thickness, order, strength)
        field = field.at[i, :, :].mul(factor)
        field = field.at[nx - 1 - i, :, :].mul(factor)
        field = field.at[:, i, :].mul(factor)
        field = field.at[:, ny - 1 - i, :].mul(factor)
        field = field.at[:, :, i].mul(factor)
        field = field.at[:, :, nz - 1 - i].mul(factor)

    return field

@jit
def apply_zero_boundary_condition(field):
    """
    Apply zero boundary conditions to the given field.

    Args:
        field (ndarray): The field to which the zero boundary condition is applied.

    Returns:
        ndarray: The field with zero boundary conditions applied.
    """
    field = field.at[0, :, :].set(0)
    field = field.at[-1, :, :].set(0)
    field = field.at[:, 0, :].set(0)
    field = field.at[:, -1, :].set(0)
    field = field.at[:, :, 0].set(0)
    field = field.at[:, :, -1].set(0)

    return field