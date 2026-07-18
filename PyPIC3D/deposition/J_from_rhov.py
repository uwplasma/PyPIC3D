from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)

from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.boundary_conditions.ghost_cells import (
    fold_tiled_vector_ghost_cells,
    update_tiled_vector_ghost_cells,
)

from PyPIC3D.utilities.filters import (
    bilinear_filter_vector,
    digital_filter_vector,
)

import jax
import jax.numpy as jnp
from functools import partial

def _collapse_tiled_axis_stencil(points, weights, local_n, reduced_axis, g):
    if reduced_axis:
        collapsed_points = jnp.full((1, points.shape[1]), int(g), dtype=points.dtype)
        collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
        return collapsed_points, collapsed_weights
    return collapse_axis_stencil(points, weights, local_n, ghost_cells=True)


@partial(jax.jit, static_argnames="static_parameters")
def J_from_rhov(
    particles,
    species_config,
    J,
    static_parameters,
    dynamic_parameters,
):
    """Compute tile-local direct current from centered tiled particles."""


    current_filter = static_parameters.current_filter
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    g = int(static_parameters.guard_cells)
    g = int(g)
    # determine the number of guard cells and the shape of each of the tiles

    tiled_grid = dynamic_parameters.grids.tiled_center_grid
    # get the grid for the tiles

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    # get the grid spacing

    Jx_tiles, Jy_tiles, Jz_tiles = J
    # unpack the current density tiles

    ntx, nty, ntz = Jx_tiles.shape[:3]
    # get the number of tiles in each dimension
    tile_nx, tile_ny, tile_nz = tile_shape
    # unpack the tile shape
    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g
    # piece together the total local tile shape

    shape_factor = static_parameters.shape_factor
    # get the shape factor

    reduced_x = int(tile_nx) == 1 and int(ntx) == 1
    reduced_y = int(tile_ny) == 1 and int(nty) == 1
    reduced_z = int(tile_nz) == 1 and int(ntz) == 1
    # determine if any of the axes are dummy axes

    Jx_template = jnp.zeros_like(Jx_tiles[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy_tiles[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz_tiles[0, 0, 0])
    # build template tiles

    # Tile boundaries are not physical boundaries.  Deposits that cross a tile
    # edge should land in tile ghost cells and be exchanged by the tiled fold.
    local_bc = 2

    species_weighted_charge = species_config.charge * species_config.weight
    # compute the weighted charge for each species

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        # deposit the current density for a single tile, given the particle positions, velocities, and active mask
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        # reshape the particle positions into 1D arrays for processing
        vx = u_tile[..., 0].reshape(-1)
        vy = u_tile[..., 1].reshape(-1)
        vz = u_tile[..., 2].reshape(-1)
        # reshape the particle velocities into 1D arrays for processing
        active = active_tile.reshape(-1).astype(x.dtype)
        # reshape the active particle mask into a 1D array for processing
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        # reshape the particle charges into a 1D array for processing
        dq = q / (dx * dy * dz)
        # compute the charge density contribution of each particle

        x_grid = tiled_grid[0][tx, ty, tz]
        y_grid = tiled_grid[1][tx, ty, tz]
        z_grid = tiled_grid[2][tx, ty, tz]
        # get the grid points for the current tile in each dimension

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
        # prepare the particle positions and compute the stencil points and deltas for each axis

        deltax_face = (x - x_grid[0]) - (x0 + 0.5) * dx
        deltay_face = (y - y_grid[0]) - (y0 + 0.5) * dy
        deltaz_face = (z - z_grid[0]) - (z0 + 0.5) * dz
        # compute the deltas for the face-centered weights based on the particle positions and grid points

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
        # compute the weights for the node-centered and face-centered contributions based on the shape factor and deltas

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        x_weights_node = jnp.asarray(x_weights_node)
        y_weights_node = jnp.asarray(y_weights_node)
        z_weights_node = jnp.asarray(z_weights_node)
        x_weights_face = jnp.asarray(x_weights_face)
        y_weights_face = jnp.asarray(y_weights_face)
        z_weights_face = jnp.asarray(z_weights_face)
        # convert the stencil points and weights to JAX arrays for further processing

        xpts, x_weights_node = _collapse_tiled_axis_stencil(xpts, x_weights_node, local_Nx, reduced_x, g)
        _, x_weights_face = _collapse_tiled_axis_stencil(xpts, x_weights_face, local_Nx, reduced_x, g)
        ypts, y_weights_node = _collapse_tiled_axis_stencil(ypts, y_weights_node, local_Ny, reduced_y, g)
        _, y_weights_face = _collapse_tiled_axis_stencil(ypts, y_weights_face, local_Ny, reduced_y, g)
        zpts, z_weights_node = _collapse_tiled_axis_stencil(zpts, z_weights_node, local_Nz, reduced_z, g)
        _, z_weights_face = _collapse_tiled_axis_stencil(zpts, z_weights_face, local_Nz, reduced_z, g)
        # collapse the stencil points and weights for each axis, taking into account any reduced axes and guard cells

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
    # build the tile index arrays for each dimension

    deposit_tiles = deposit_one_tile
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    # vectorize the deposit_one_tile function over the tile indices using jax.vmap

    Jx, Jy, Jz = deposit_tiles(
        particles.x,
        particles.u,
        particles.active,
        tx,
        ty,
        tz,
    )
    # compute the current density contributions for all tiles by applying the vectorized deposit function to the particle data and tile indices

    J = fold_tiled_vector_ghost_cells((Jx, Jy, Jz), static_parameters, g, bc_type=1)
    # fold the ghost cells of the current density tiles to ensure continuity across tile boundaries
    J = update_tiled_vector_ghost_cells(J, static_parameters, g, bc_type=1)
    # update the ghost cells of the current density tiles to reflect the contributions from neighboring tiles



    ################# CURRENT FILTERING #################
    def bilinear_filtered_current(J):
        J = bilinear_filter_vector(J, num_guard_cells=g)
        J = update_tiled_vector_ghost_cells(J, static_parameters, num_guard_cells=g, bc_type=1)
        return J
    
    def digital_filtered_current(J):
        J = digital_filter_vector(J, dynamic_parameters.alpha, num_guard_cells=g)
        J = update_tiled_vector_ghost_cells(J, static_parameters, num_guard_cells=g, bc_type=1)
        return J

    J = jax.lax.cond(
        current_filter == "bilinear",
        lambda J: bilinear_filtered_current(J),
        lambda J: digital_filtered_current(J),
        J,
    )
    # apply the specified filter to the current density tiles, either bilinear or digital, based on the current_filter argument
    ########################################################


    
    return J
