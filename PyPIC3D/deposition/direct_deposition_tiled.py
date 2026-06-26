import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.solvers.yee_tiled import (
    fold_tiled_vector_ghost_cells,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.utils import bilinear_filter, digital_filter


def digital_filter_tiled_current_density(J_tiles, alpha, world):
    """
    Apply the global digital current filter to compact current tiles.

    Tile halos are refreshed before the stencil so every tile sees the same
    neighbor values that the assembled ghost-celled current would see.  The
    halos are refreshed again after filtering so downstream Yee updates read
    filtered guard-cell values.
    """

    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world)

    def filter_component(component_tiles):
        filter_tiles = jax.vmap(jax.vmap(jax.vmap(lambda tile: digital_filter(tile, alpha))))
        return filter_tiles(component_tiles)

    J_tiles = tuple(filter_component(component) for component in J_tiles)
    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world)

    return J_tiles


def _tile_axis_grid(global_axis_grid, tile_index, tile_n, local_n, d):
    """
    Build the ghost-celled coordinate line for one compact tile.

    The first local grid point is the global ghost-cell origin shifted by the
    tile's active-cell offset.  With this convention, local active indices are
    1..tile_n and local ghost indices are 0 and tile_n + 1.
    """

    offsets = jnp.arange(local_n, dtype=global_axis_grid.dtype)
    return global_axis_grid[0] + (offsets + tile_index * tile_n) * d


@partial(jit, static_argnames=("filter",))
def direct_J_from_tiled_particles(
    tiled_particles,
    J_tiles,
    constants,
    world,
    grid=None,
    filter="bilinear",
):
    """
    Compute direct current deposition using tile-local stencils only.

    ``tiled_particles.x`` is assumed to already be centered at the current
    deposition time and refreshed into the tiles that own those centered
    positions.  In other words, callers that store particles at the forward
    position should pass a temporary, refreshed deposition view with
    ``x_dep = x_forward - 0.5 * u * dt``.
    """

    if grid is None:
        grid = world["grids"]["center"]

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]

    Jx_tiles, Jy_tiles, Jz_tiles = J_tiles
    ntx, nty, ntz = Jx_tiles.shape[:3]
    tile_nx = Jx_tiles.shape[3] - 2
    tile_ny = Jx_tiles.shape[4] - 2
    tile_nz = Jx_tiles.shape[5] - 2
    local_Nx = tile_nx + 2
    local_Ny = tile_ny + 2
    local_Nz = tile_nz + 2
    shape_factor = world["shape_factor"]

    Jx_template = jnp.zeros_like(Jx_tiles[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy_tiles[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz_tiles[0, 0, 0])

    # Tile boundaries are not physical boundaries.  Deposits that cross a tile
    # edge should land in tile ghost cells and be exchanged by the tiled fold.
    local_bc = 2

    def deposit_one_tile(x_tile, u_tile, charge_tile, weight_tile, active_tile, tx, ty, tz):
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        vx = u_tile[..., 0].reshape(-1)
        vy = u_tile[..., 1].reshape(-1)
        vz = u_tile[..., 2].reshape(-1)
        active = active_tile.reshape(-1).astype(x.dtype)
        dq = (charge_tile * weight_tile).reshape(-1) / (dx * dy * dz)

        x_grid = _tile_axis_grid(grid[0], tx, tile_nx, local_Nx, dx)
        y_grid = _tile_axis_grid(grid[1], ty, tile_ny, local_Ny, dy)
        z_grid = _tile_axis_grid(grid[2], tz, tile_nz, local_Nz, dz)

        x, x0, deltax_node, xpts = prepare_particle_axis_stencil(
            x,
            x_grid,
            local_Nx,
            shape_factor,
            local_bc,
            wind=tile_nx * dx,
            ghost_cells=True,
        )
        y, y0, deltay_node, ypts = prepare_particle_axis_stencil(
            y,
            y_grid,
            local_Ny,
            shape_factor,
            local_bc,
            wind=tile_ny * dy,
            ghost_cells=True,
        )
        z, z0, deltaz_node, zpts = prepare_particle_axis_stencil(
            z,
            z_grid,
            local_Nz,
            shape_factor,
            local_bc,
            wind=tile_nz * dz,
            ghost_cells=True,
        )

        deltax_face = (x - x_grid[0]) - (x0 + 0.5) * dx
        deltay_face = (y - y_grid[0]) - (y0 + 0.5) * dy
        deltaz_face = (z - z_grid[0]) - (z0 + 0.5) * dz

        x_weights_node, y_weights_node, z_weights_node = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            operand=None,
        )
        x_weights_face, y_weights_face, z_weights_face = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz),
            operand=None,
        )

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        x_weights_node = jnp.asarray(x_weights_node)
        y_weights_node = jnp.asarray(y_weights_node)
        z_weights_node = jnp.asarray(z_weights_node)
        x_weights_face = jnp.asarray(x_weights_face)
        y_weights_face = jnp.asarray(y_weights_face)
        z_weights_face = jnp.asarray(z_weights_face)

        xpts, x_weights_node = collapse_axis_stencil(xpts, x_weights_node, local_Nx, ghost_cells=True)
        _, x_weights_face = collapse_axis_stencil(xpts, x_weights_face, local_Nx, ghost_cells=True)
        ypts, y_weights_node = collapse_axis_stencil(ypts, y_weights_node, local_Ny, ghost_cells=True)
        _, y_weights_face = collapse_axis_stencil(ypts, y_weights_face, local_Ny, ghost_cells=True)
        zpts, z_weights_node = collapse_axis_stencil(zpts, z_weights_node, local_Nz, ghost_cells=True)
        _, z_weights_face = collapse_axis_stencil(zpts, z_weights_face, local_Nz, ghost_cells=True)

        tile_Jx = Jx_template
        tile_Jy = Jy_template
        tile_Jz = Jz_template

        for i in range(xpts.shape[0]):
            for j in range(ypts.shape[0]):
                for k in range(zpts.shape[0]):
                    ix = xpts[i]
                    iy = ypts[j]
                    iz = zpts[k]
                    tile_Jx = tile_Jx.at[ix, iy, iz].add(
                        active * dq * vx * x_weights_face[i] * y_weights_node[j] * z_weights_node[k],
                        mode="drop",
                    )
                    tile_Jy = tile_Jy.at[ix, iy, iz].add(
                        active * dq * vy * x_weights_node[i] * y_weights_face[j] * z_weights_node[k],
                        mode="drop",
                    )
                    tile_Jz = tile_Jz.at[ix, iy, iz].add(
                        active * dq * vz * x_weights_node[i] * y_weights_node[j] * z_weights_face[k],
                        mode="drop",
                    )

        return tile_Jx, tile_Jy, tile_Jz

    tx, ty, tz = jnp.meshgrid(
        jnp.arange(ntx),
        jnp.arange(nty),
        jnp.arange(ntz),
        indexing="ij",
    )

    deposit_tiles = deposit_one_tile
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)

    Jx_tiles, Jy_tiles, Jz_tiles = deposit_tiles(
        tiled_particles.x,
        tiled_particles.u,
        tiled_particles.charge,
        tiled_particles.weight,
        tiled_particles.active,
        tx,
        ty,
        tz,
    )

    J_tiles = fold_tiled_vector_ghost_cells((Jx_tiles, Jy_tiles, Jz_tiles), world)
    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world)

    if filter == "bilinear":
        apply_filter = jax.vmap(jax.vmap(jax.vmap(bilinear_filter)))
        J_tiles = tuple(apply_filter(component) for component in J_tiles)
        J_tiles = update_tiled_vector_ghost_cells(J_tiles, world)
    elif filter == "digital":
        J_tiles = digital_filter_tiled_current_density(J_tiles, constants["alpha"], world)

    return J_tiles