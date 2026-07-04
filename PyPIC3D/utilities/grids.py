import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    build_collocated_axis,
    build_staggered_axis,
)


def _tile_axis_count(n_cells, cells_per_tile):
    if int(n_cells) % int(cells_per_tile) != 0:
        raise ValueError("Shared tile sizes must divide the physical grid dimensions exactly.")
    return int(n_cells) // int(cells_per_tile)


def build_collocated_grid(world):
    """
    Build the collocated vertex/center grid including one ghost cell per side.
    """

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    x_wind = world["x_wind"]
    y_wind = world["y_wind"]
    z_wind = world["z_wind"]
    Nx = world["Nx"]
    Ny = world["Ny"]
    Nz = world["Nz"]

    grid = (
        build_collocated_axis(-x_wind / 2, dx, Nx),
        build_collocated_axis(-y_wind / 2, dy, Ny),
        build_collocated_axis(-z_wind / 2, dz, Nz),
    )

    return grid, grid


def build_yee_grid(world):
    """
    Build the Yee vertex and center grids including one ghost cell per side.
    """

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    x_wind = world["x_wind"]
    y_wind = world["y_wind"]
    z_wind = world["z_wind"]
    Nx = world["Nx"]
    Ny = world["Ny"]
    Nz = world["Nz"]

    vertex_grid = (
        build_collocated_axis(-x_wind / 2, dx, Nx),
        build_collocated_axis(-y_wind / 2, dy, Ny),
        build_collocated_axis(-z_wind / 2, dz, Nz),
    )
    center_grid = (
        build_staggered_axis(-x_wind / 2, dx, Nx),
        build_staggered_axis(-y_wind / 2, dy, Ny),
        build_staggered_axis(-z_wind / 2, dz, Nz),
    )

    return vertex_grid, center_grid


def _tile_grid_axis(global_axis_grid, world, tile_shape, tile_counts, axis_index, num_guard_cells):
    tile_width = int(tile_shape[axis_index])
    tile_count = int(tile_counts[axis_index])
    g = int(num_guard_cells)

    if axis_index == 0:
        d = world["dx"]
    elif axis_index == 1:
        d = world["dy"]
    else:
        d = world["dz"]

    offsets = jnp.arange(tile_width + 2 * g, dtype=global_axis_grid.dtype)
    tile_indices = jnp.arange(tile_count, dtype=global_axis_grid.dtype)
    axis_lines = global_axis_grid[0] + (
        offsets[jnp.newaxis, :] + tile_indices[:, jnp.newaxis] * tile_width - (g - 1)
    ) * d

    axis_shape = [1, 1, 1, tile_width + 2 * g]
    axis_shape[axis_index] = tile_count
    tiled_shape = tuple(tile_counts) + (tile_width + 2 * g,)

    return jnp.broadcast_to(axis_lines.reshape(axis_shape), tiled_shape)


def _tile_grid_axes(grid, world, tile_shape, num_guard_cells=2):
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
        _tile_grid_axis(grid[axis], world, tile_shape, tile_counts, axis, num_guard_cells)
        for axis in range(3)
    )


def build_tiled_yee_grids(world, tile_shape=None, num_guard_cells=None):
    """
    Build the tiled vertex and center grids from the untiled grids in world.
    """

    if tile_shape is None:
        tile_shape = tuple(int(width) for width in world["tile_shape"])
    if num_guard_cells is None:
        num_guard_cells = int(world["guard_cells"])

    tiled_vertex_grid = _tile_grid_axes(
        world["grids"]["vertex"],
        world,
        tile_shape,
        num_guard_cells=num_guard_cells,
    )
    tiled_center_grid = _tile_grid_axes(
        world["grids"]["center"],
        world,
        tile_shape,
        num_guard_cells=num_guard_cells,
    )

    return tiled_vertex_grid, tiled_center_grid
