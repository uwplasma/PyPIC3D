from functools import partial

import jax
from jax import jit
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.boundary_conditions.ghost_cells import fold_tiled_ghost_cells, update_tiled_ghost_cells
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.particles.particle_class import TiledParticles
from PyPIC3D.utilities.filters import digital_filter
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)


def _collapse_tiled_axis_stencil(points, weights, local_n, reduced_axis, g):
    if reduced_axis:
        collapsed_points = jnp.full((1, points.shape[1]), int(g), dtype=points.dtype)
        collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
        return collapsed_points, collapsed_weights
    return collapse_axis_stencil(points, weights, local_n, ghost_cells=True)

@partial(jax.jit, static_argnames="static_parameters")
def compute_rho(
        particles,
        species_config,
        rho,
        static_parameters,
        dynamic_parameters,
):

    
    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    shape_factor = static_parameters.shape_factor
    # unpack grid and tile parameters

    tile_nx, tile_ny, tile_nz = [ int(width) for width in static_parameters.tile_shape ]
    # get the tile shape
    g = static_parameters.guard_cells

    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g
    # piece together the total local tile shape

    tiled_grid = dynamic_parameters.grids.tiled_center_grid
    # get the grid for the tiles

    local_bc = 2

    species_weighted_charge = species_config.charge * species_config.weight / (dx * dy * dz)
    # compute the weighted charge for each species divided by the cell volume

    rho_template = jnp.zeros_like(rho[0, 0, 0])

    ntx, nty, ntz = rho.shape[:3]
    # get the number of tiles in each dimension

    reduced_x = int(tile_nx) == 1 and int(ntx) == 1
    reduced_y = int(tile_ny) == 1 and int(nty) == 1
    reduced_z = int(tile_nz) == 1 and int(ntz) == 1
    # determine if any of the axes are dummy axes

    def deposit_one_tile(x_tile, active_tile, tx, ty, tz):
        # deposit the current density for a single tile, given the particle positions, velocities, and active mask
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        # reshape the particle positions into 1D arrays for processing
        active = active_tile.reshape(-1).astype(x.dtype)
        # reshape the active particle mask into a 1D array for processing
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        # reshape the species-weighted charges into the same flat slot layout

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

        x_weights_node, y_weights_node, z_weights_node = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            operand=None,
        )

        # compute the weights for the node-centered stencil based on the deltas and shape factor

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        x_weights_node = jnp.asarray(x_weights_node)
        y_weights_node = jnp.asarray(y_weights_node)
        z_weights_node = jnp.asarray(z_weights_node)
        # convert the stencil points and weights to JAX arrays for further processing

        xpts, x_weights_node = _collapse_tiled_axis_stencil(xpts, x_weights_node, local_Nx, reduced_x, g)
        ypts, y_weights_node = _collapse_tiled_axis_stencil(ypts, y_weights_node, local_Ny, reduced_y, g)
        zpts, z_weights_node = _collapse_tiled_axis_stencil(zpts, z_weights_node, local_Nz, reduced_z, g)
        # collapse the stencil points and weights for each axis, taking into account any reduced axes and guard cells

        rho_tile = rho_template

        for i in range(xpts.shape[0]):
            for j in range(ypts.shape[0]):
                for k in range(zpts.shape[0]):
                    ix = xpts[i]
                    iy = ypts[j]
                    iz = zpts[k]
                    rho_tile = rho_tile.at[ix, iy, iz].add(
                        active * q * x_weights_node[i] * y_weights_node[j] * z_weights_node[k],
                        mode="drop",
                    )
        # deposit the charge density for each stencil point in the tile, accumulating contributions from all particles

        return rho_tile



    tx, ty, tz = jnp.meshgrid(
        jnp.arange(ntx),
        jnp.arange(nty),
        jnp.arange(ntz),
        indexing="ij",
    )
    # build the tile index arrays for each dimension

    deposit_charge = deposit_one_tile
    deposit_charge = jax.vmap(deposit_charge, in_axes=(0, 0, 0, 0, 0), out_axes=0)
    deposit_charge = jax.vmap(deposit_charge, in_axes=(0, 0, 0, 0, 0), out_axes=0)
    deposit_charge = jax.vmap(deposit_charge, in_axes=(0, 0, 0, 0, 0), out_axes=0)
    # vectorize the deposit_one_tile function over the tile indices using jax.vmap

    rho = deposit_charge(particles.x, particles.active, tx, ty, tz)
    # deposit the charge density for all tiles by applying the vectorized deposit_charge function to the particle positions, active mask, and tile indices

    rho = fold_tiled_ghost_cells(rho, static_parameters, g)
    # fold charge deposited into tile ghost cells back to the owner interiors

    rho = update_tiled_ghost_cells(rho, static_parameters, g)
    # update the ghost cells of the charge density array to ensure proper boundary conditions
    
    return rho
