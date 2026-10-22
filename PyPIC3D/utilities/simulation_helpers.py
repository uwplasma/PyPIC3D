import os

import jax.numpy as jnp
from jax.tree_util import tree_map

from PyPIC3D.utilities.grids import grid_axis_width, grid_domain_bounds


def setup_pmd_files(file_path, name, extension=".bp"):
    """
    Set up the openPMD file structure for storing simulation data.
    """

    file = os.path.join(file_path, name + ".pmd")
    with open(file, 'w') as f:
        f.write(f"{name}{extension}\n")


def make_dir(path):
    """
    Create a directory if it does not exist.
    """

    if not os.path.exists(path):
        os.makedirs(path)


def convert_to_jax_compatible(data):
    """
    Convert a dictionary to a JAX-compatible PyTree.
    """
    return tree_map(lambda x: jnp.array(x) if isinstance(x, (int, float, list, tuple)) else x, data)


def particle_sanity_check(particles):
    """
    Perform a basic shape check for tiled particle storage.
    """

    assert particles.x.shape == particles.u.shape
    assert particles.x.shape[-1] == 3
    assert particles.active.shape == particles.x.shape[:-1]


def print_stats(static_parameters, dynamic_parameters):
    """
    Print the spatial and temporal simulation statistics.
    """

    Nt = static_parameters.Nt
    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    dt = dynamic_parameters.dt
    x_bounds, y_bounds, z_bounds = grid_domain_bounds(dynamic_parameters)
    x_wind = grid_axis_width(dynamic_parameters.grids.center[0])
    y_wind = grid_axis_width(dynamic_parameters.grids.center[1])
    z_wind = grid_axis_width(dynamic_parameters.grids.center[2])
    t_wind = Nt*dt
    time_unit = 'code units' if static_parameters.solver == 'dynamic_metric' else 's'
    length_unit = 'code units' if static_parameters.solver == 'dynamic_metric' else 'm'
    print(f'\ntime window: {t_wind} {time_unit} with {Nt} time steps of {dt} {time_unit}')
    print(f'x window: {x_wind} {length_unit} [{x_bounds[0]}, {x_bounds[1]}] with dx: {dx} {length_unit}')
    print(f'y window: {y_wind} {length_unit} [{y_bounds[0]}, {y_bounds[1]}] with dy: {dy} {length_unit}')
    print(f'z window: {z_wind} {length_unit} [{z_bounds[0]}, {z_bounds[1]}] with dz: {dz} {length_unit}\n')


def courant_condition(courant_number, dx, dy, dz, dynamic_parameters):
    """
    Calculate the Courant-limited timestep for the active dimensions.
    """

    C = dynamic_parameters.C

    Nx = dynamic_parameters.Nx
    Ny = dynamic_parameters.Ny
    Nz = dynamic_parameters.Nz
    Ns = [Nx, Ny, Nz]
    dxs = [dx, dy, dz]

    dx_inv = []
    for d, N in zip(dxs, Ns):
        if N > 1:
            dx_inv.append(1/d)

    dx_inv = sum(dx_inv)

    dt = courant_number / (C * dx_inv)

    return dt
