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
from PyPIC3D.particles.particle_batching import (
    number_of_particle_batches,
    particle_batch_indices,
    prepare_particle_batches,
)
from PyPIC3D.utilities.filters import tiled_digital_filter
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
        particle_capacity, batch_size, active_indices, n_active = prepare_particle_batches(
            active_tile,
            static_parameters.particle_batch_size,
        )
        if particle_capacity == 0:
            return rho_template

        x_flat = x_tile.reshape(-1, 3)
        slots_per_species = active_tile.shape[-1]
        n_batches = number_of_particle_batches(n_active, batch_size)

        x_grid = tiled_grid[0][tx, ty, tz]
        y_grid = tiled_grid[1][tx, ty, tz]
        z_grid = tiled_grid[2][tx, ty, tz]
        # get the grid points for the current tile in each dimension

        def deposit_batch(batch_state):
            batch_index, rho_tile = batch_state
            particle_indices, valid = particle_batch_indices(
                active_indices,
                n_active,
                batch_index,
                batch_size,
            )

            x_batch = x_flat[particle_indices]
            x = x_batch[:, 0]
            y = x_batch[:, 1]
            z = x_batch[:, 2]

            species_indices = particle_indices // slots_per_species
            q_batch = species_weighted_charge[species_indices]
            dq = jnp.where(valid, q_batch, jnp.zeros_like(q_batch))

            _, _, deltax_node, xpts = prepare_particle_axis_stencil(
                x,
                x_grid,
                local_Nx,
                shape_factor,
                local_bc,
                wind=tile_nx * dx,
                ghost_cells=True,
            )
            _, _, deltay_node, ypts = prepare_particle_axis_stencil(
                y,
                y_grid,
                local_Ny,
                shape_factor,
                local_bc,
                wind=tile_ny * dy,
                ghost_cells=True,
            )
            _, _, deltaz_node, zpts = prepare_particle_axis_stencil(
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

            for i in range(xpts.shape[0]):
                for j in range(ypts.shape[0]):
                    for k in range(zpts.shape[0]):
                        rho_tile = rho_tile.at[
                            xpts[i],
                            ypts[j],
                            zpts[k],
                        ].add(
                            dq * x_weights_node[i] * y_weights_node[j] * z_weights_node[k],
                            mode="drop",
                        )
            # deposit the charge density for each stencil point in the tile

            return batch_index + 1, rho_tile

        def batches_remaining(batch_state):
            batch_index, _ = batch_state
            return batch_index < n_batches

        _, rho_tile = jax.lax.while_loop(
            batches_remaining,
            deposit_batch,
            (jnp.asarray(0), rho_template),
        )

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

    rho = fold_tiled_ghost_cells(rho, static_parameters, g, bc_type=1)
    # fold charge deposited into tile ghost cells back to the owner interiors

    def filter(rho):
        return tiled_digital_filter(
            rho,
            dynamic_parameters.alpha,
            static_parameters,
            bc_type=1,
        )

    rho = jax.lax.cond(
        static_parameters.current_filter == "digital",
        filter,
        lambda rho: update_tiled_ghost_cells(rho, static_parameters, g, bc_type=1),
        rho,
    )
    # apply an additional digital filter to the charge density if specified in the static parameters


    
    return rho
