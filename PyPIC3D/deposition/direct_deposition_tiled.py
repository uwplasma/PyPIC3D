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
    digital_filter_tiled_vector,
    fold_tiled_vector_ghost_cells,
    tiled_grid_axes_from_world,
    update_tiled_vector_ghost_cells,
)


def bilinear_filter_tiled_current_density(J_tiles, world, num_guard_cells=2, tile_shape=None):
    """
    Apply the global bilinear current filter to compact current tiles.

    The filter is the same separable [1, 2, 1]/4 stencil used by
    ``bilinear_filter`` on assembled fields.  Only the physical tile cells are
    overwritten; guard cells are refreshed before and after the stencil.
    """

    g = int(num_guard_cells)
    active = slice(g, -g)
    backward = slice(g - 1, -g - 1)
    forward = slice(g + 1, None if g == 1 else -g + 1)

    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world, g, tile_shape)

    def filter_component(component):
        center = component[:, :, :, active, active, active]
        filtered = (
            8.0 * center
            + 4.0 * (
                component[:, :, :, backward, active, active]
                + component[:, :, :, forward, active, active]
                + component[:, :, :, active, backward, active]
                + component[:, :, :, active, forward, active]
                + component[:, :, :, active, active, backward]
                + component[:, :, :, active, active, forward]
            )
            + 2.0 * (
                component[:, :, :, backward, backward, active]
                + component[:, :, :, backward, forward, active]
                + component[:, :, :, forward, backward, active]
                + component[:, :, :, forward, forward, active]
                + component[:, :, :, backward, active, backward]
                + component[:, :, :, backward, active, forward]
                + component[:, :, :, forward, active, backward]
                + component[:, :, :, forward, active, forward]
                + component[:, :, :, active, backward, backward]
                + component[:, :, :, active, backward, forward]
                + component[:, :, :, active, forward, backward]
                + component[:, :, :, active, forward, forward]
            )
            + (
                component[:, :, :, backward, backward, backward]
                + component[:, :, :, backward, backward, forward]
                + component[:, :, :, backward, forward, backward]
                + component[:, :, :, backward, forward, forward]
                + component[:, :, :, forward, backward, backward]
                + component[:, :, :, forward, backward, forward]
                + component[:, :, :, forward, forward, backward]
                + component[:, :, :, forward, forward, forward]
            )
        ) / 64.0
        return component.at[:, :, :, active, active, active].set(filtered)

    J_tiles = tuple(filter_component(component) for component in J_tiles)
    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world, g, tile_shape)

    return J_tiles


def digital_filter_tiled_current_density(J_tiles, alpha, world, num_guard_cells=2, tile_shape=None):
    """
    Apply the global digital current filter to compact current tiles.

    Tile halos are refreshed before the stencil so every tile sees the same
    neighbor values that the assembled ghost-celled current would see.  The
    halos are refreshed again after filtering so downstream Yee updates read
    filtered guard-cell values.
    """

    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world, num_guard_cells, tile_shape)

    J_tiles = digital_filter_tiled_vector(J_tiles, alpha, num_guard_cells)
    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world, num_guard_cells, tile_shape)

    return J_tiles


def _collapse_tiled_axis_stencil(points, weights, local_n, reduced_axis, g):
    if reduced_axis:
        collapsed_points = jnp.full((1, points.shape[1]), int(g), dtype=points.dtype)
        collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
        return collapsed_points, collapsed_weights
    return collapse_axis_stencil(points, weights, local_n, ghost_cells=True)


@partial(jit, static_argnames=("filter", "tile_shape", "g"))
def direct_J_from_tiled_particles(
    tiled_particles,
    species_config,
    J_tiles,
    constants,
    world,
    grid=None,
    filter="bilinear",
    tile_shape=None,
    g=2,
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
    tiled_grid = tiled_grid_axes_from_world(
        world,
        grid,
        "tiled_center_grid",
        tile_shape,
        g,
    )

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]

    Jx_tiles, Jy_tiles, Jz_tiles = J_tiles
    ntx, nty, ntz = Jx_tiles.shape[:3]
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(g)
    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g
    shape_factor = world["shape_factor"]
    reduced_x = int(ntx) == 1 and tile_nx == 1
    reduced_y = int(nty) == 1 and tile_ny == 1
    reduced_z = int(ntz) == 1 and tile_nz == 1

    Jx_template = jnp.zeros_like(Jx_tiles[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy_tiles[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz_tiles[0, 0, 0])

    # Tile boundaries are not physical boundaries.  Deposits that cross a tile
    # edge should land in tile ghost cells and be exchanged by the tiled fold.
    local_bc = 2

    species_weighted_charge = species_config.charge * species_config.weight

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        vx = u_tile[..., 0].reshape(-1)
        vy = u_tile[..., 1].reshape(-1)
        vz = u_tile[..., 2].reshape(-1)
        active = active_tile.reshape(-1).astype(x.dtype)
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        dq = q / (dx * dy * dz)

        x_grid = tiled_grid[0][tx, ty, tz]
        y_grid = tiled_grid[1][tx, ty, tz]
        z_grid = tiled_grid[2][tx, ty, tz]

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

        xpts, x_weights_node = _collapse_tiled_axis_stencil(xpts, x_weights_node, local_Nx, reduced_x, g)
        _, x_weights_face = _collapse_tiled_axis_stencil(xpts, x_weights_face, local_Nx, reduced_x, g)
        ypts, y_weights_node = _collapse_tiled_axis_stencil(ypts, y_weights_node, local_Ny, reduced_y, g)
        _, y_weights_face = _collapse_tiled_axis_stencil(ypts, y_weights_face, local_Ny, reduced_y, g)
        zpts, z_weights_node = _collapse_tiled_axis_stencil(zpts, z_weights_node, local_Nz, reduced_z, g)
        _, z_weights_face = _collapse_tiled_axis_stencil(zpts, z_weights_face, local_Nz, reduced_z, g)

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
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)

    Jx_tiles, Jy_tiles, Jz_tiles = deposit_tiles(
        tiled_particles.x,
        tiled_particles.u,
        tiled_particles.active,
        tx,
        ty,
        tz,
    )

    J_tiles = fold_tiled_vector_ghost_cells((Jx_tiles, Jy_tiles, Jz_tiles), world, g, tile_shape)
    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world, g, tile_shape)

    if filter == "bilinear":
        J_tiles = bilinear_filter_tiled_current_density(J_tiles, world, num_guard_cells=g, tile_shape=tile_shape)
    elif filter == "digital":
        J_tiles = digital_filter_tiled_current_density(J_tiles, constants["alpha"], world, num_guard_cells=g, tile_shape=tile_shape)

    return J_tiles
