import matplotlib.pyplot as plt
import jax
from jax import random
from jax import jit
from jax import lax
from jax._src.scipy.sparse.linalg import _vdot_real_tree, _add, _sub, _mul
from jax import tree_util
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
import math
from pyevtk.hl import gridToVTK
import functools
from functools import partial
import toml

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

    Parameters:
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

def get_material_surface_keys(config):
    """
    Extracts and returns a list of keys from the given configuration dictionary
    that start with the prefix 'surface'.
    Args:
        config (dict): A dictionary containing configuration keys and values.
    Returns:
        list: A list of keys from the config dictionary that start with 'surface'.
    """
    surface_keys = []
    for key in config.keys():
        if key[:7] == 'surface':
            surface_keys.append(key)
    return surface_keys

def load_material_surfaces_from_toml(config):
    """
    Load material surfaces from a TOML file.

    Parameters:
    - config (dict): Dictionary containing the configuration.

    Returns:
    - list: List of MaterialSurface objects.
    """
    surface_keys = get_material_surface_keys(config)
    surfaces = []

    for toml_key in surface_keys:
        try:
            surface_config = config[toml_key]
            name = surface_config['name']
            material = surface_config['material']
            work_function_x_path = surface_config['work_function_x']
            work_function_y_path = surface_config['work_function_y']
            work_function_z_path = surface_config['work_function_z']
            # load the work function paths
            work_function_x = jnp.load(work_function_x_path)
            work_function_y = jnp.load(work_function_y_path)
            work_function_z = jnp.load(work_function_z_path)
            # load the work functions
            surface = MaterialSurface(name, material, work_function_x, work_function_y, work_function_z)
            # define the material surface
            surfaces.append(surface)
        except Exception as e:
            print(f"Error loading material surface: {toml_key}")
            print(e)

    return surfaces

@register_pytree_node_class
class MaterialSurface:
    """
    Class representing a material surface.
    """

    def __init__(self, name, material, work_function_x, work_function_y, work_function_z):
        """
        Initialize a MaterialSurface object.

        Parameters:
        - name (str): Name of the material surface.
        - material (str): Material of the surface.
        - work_function_x (float): Work function in the x-direction.
        - work_function_y (float): Work function in the y-direction.
        - work_function_z (float): Work function in the z-direction.
        """

        print(f"Initializing material surface: {name}")
        self.name = name
        self.material = material
        self.work_function = jnp.array([work_function_x, work_function_y, work_function_z])

    def get_material(self):
        """
        Get the material of the surface.

        Returns:
        - str: Material of the surface.
        """
        return self.material

    def get_work_function(self):
        """
        Get the work function of the material surface.

        Returns:
        - ndarray: Work function of the material surface.
        """
        return self.work_function[0], self.work_function[1], self.work_function[2]
    
    def get_work_function_x(self):
        """
        Get the work function in the x-direction.

        Returns:
        - float: Work function in the x-direction.
        """
        return self.work_function[0]
    
    def get_work_function_y(self):
        """
        Get the work function in the y-direction.

        Returns:
        - float: Work function in the y-direction.
        """
        return self.work_function[1]
    
    def get_work_function_z(self):
        """
        Get the work function in the z-direction.

        Returns:
        - float: Work function in the z-direction.
        """
        return self.work_function[2]
    

    def tree_flatten(self):
        children = None
        aux_data = (self.name, self.material, self.work_function)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        name, material, work_function = aux_data

        return cls(
            name=name,
            material=material,
            work_function_x=work_function[0],
            work_function_y=work_function[1],
            work_function_z=work_function[2]
        )