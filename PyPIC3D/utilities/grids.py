import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    build_collocated_axis,
    build_staggered_axis,
)

def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)

def build_collocated_grid(dynamic_parameters):
    """
    Build the collocated vertex/center grid including one ghost cell per side.
    """

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    # get the spatial resolutions
    x_wind = dynamic_parameters.x_wind
    y_wind = dynamic_parameters.y_wind
    z_wind = dynamic_parameters.z_wind
    # get the physical domain sizes
    Nx = dynamic_parameters.Nx
    Ny = dynamic_parameters.Ny
    Nz = dynamic_parameters.Nz
    # get the number of grid points

    grid = (
        build_collocated_axis(-x_wind / 2, dx, Nx),
        build_collocated_axis(-y_wind / 2, dy, Ny),
        build_collocated_axis(-z_wind / 2, dz, Nz),
    )
    # construct a collocated grid with ghost cells

    return grid, grid


def build_yee_grid(dynamic_parameters):
    """
    Build the legacy PyPIC3D Yee center and vertex grids.

    In the runtime contract, ``center`` is the collocated/base index grid and
    ``vertex`` is the staggered half-cell grid.
    """

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    # get the spatial resolutions
    x_wind = dynamic_parameters.x_wind
    y_wind = dynamic_parameters.y_wind
    z_wind = dynamic_parameters.z_wind
    # get the physical domain sizes
    Nx = dynamic_parameters.Nx
    Ny = dynamic_parameters.Ny
    Nz = dynamic_parameters.Nz
    # get the number of grid points

    center_grid = (
        build_collocated_axis(-x_wind / 2, dx, Nx),
        build_collocated_axis(-y_wind / 2, dy, Ny),
        build_collocated_axis(-z_wind / 2, dz, Nz),
    )
    # construct the collocated/base grid with ghost cells

    vertex_grid = (
        build_staggered_axis(-x_wind / 2, dx, Nx),
        build_staggered_axis(-y_wind / 2, dy, Ny),
        build_staggered_axis(-z_wind / 2, dz, Nz),
    )
    # construct the staggered half-cell grid with ghost cells

    return center_grid, vertex_grid


def _tile_grid_axis(global_axis_grid, dynamic_parameters, tile_shape, tile_counts, axis_index, num_guard_cells):
    tile_width = tile_shape[axis_index]
    tile_count = tile_counts[axis_index]
    g = num_guard_cells

    d = jax.lax.cond(
        axis_index == 0,
        lambda _: dynamic_parameters.dx,
        lambda _: jax.lax.cond(
            axis_index == 1,
            lambda _: dynamic_parameters.dy,
            lambda _: dynamic_parameters.dz,
            None,
        ),
        None,
    )
    # determine the grid spacing for the given axis index

    offsets = jnp.arange(tile_width + 2 * g, dtype=global_axis_grid.dtype)
    # determine the offsets for the tile including guard cells
    tile_indices = jnp.arange(tile_count, dtype=global_axis_grid.dtype)
    # get the tile indices for the given axis
    axis_lines = global_axis_grid[0] + (
        offsets[jnp.newaxis, :] + tile_indices[:, jnp.newaxis] * tile_width - (g - 1)
    ) * d
    # build the local tile axes

    axis_shape = [1, 1, 1, tile_width + 2 * g]
    axis_shape[axis_index] = tile_count
    tiled_shape = tuple(tile_counts) + (tile_width + 2 * g,)

    return jnp.broadcast_to(axis_lines.reshape(axis_shape), tiled_shape)


def _tile_grid_axes(grid, dynamic_parameters, tile_shape, num_guard_cells=2):
    """
    Build tile-local coordinate lines for a center or vertex grid.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    Nx = int(grid[0].shape[0]) - 2
    Ny = int(grid[1].shape[0]) - 2
    Nz = int(grid[2].shape[0]) - 2
    tile_counts = (
        _tile_axis_count(Nx, tile_nx),
        _tile_axis_count(Ny, tile_ny),
        _tile_axis_count(Nz, tile_nz),
    )

    return tuple(
        _tile_grid_axis(grid[axis], dynamic_parameters, tile_shape, tile_counts, axis, num_guard_cells)
        for axis in range(3)
    )


def build_tiled_yee_grids(static_parameters, dynamic_parameters):
    """
    Build the tiled center and vertex grids from the untiled grids.
    """

    tile_shape = static_parameters.tile_shape
    # get the tile shape from the static parameters
    num_guard_cells = static_parameters.guard_cells
    # get the number of guard cells from the static parameters
    grids = dynamic_parameters.grids
    # get the grids from the dynamic parameters

    center_grid = grids.center
    vertex_grid = grids.vertex
    # get the center and vertex grids from the grids object

    tiled_center_grid = _tile_grid_axes(
        center_grid,
        dynamic_parameters,
        tile_shape,
        num_guard_cells=num_guard_cells,
    )
    tiled_vertex_grid = _tile_grid_axes(
        vertex_grid,
        dynamic_parameters,
        tile_shape,
        num_guard_cells=num_guard_cells,
    )
    # build the tiled center and vertex grids

    return tiled_center_grid, tiled_vertex_grid
