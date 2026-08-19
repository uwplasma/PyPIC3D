from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
from PyPIC3D.particles.particle_batching import (
    number_of_particle_batches,
    particle_batch_indices,
    prepare_particle_batches,
)

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
    tiled_bilinear_filter_vector,
    tiled_digital_filter_vector,
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

    tiled_center_grid = dynamic_parameters.grids.tiled_center_grid
    tiled_vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
    # direct scatter must use the same component-grid anchors as the Yee gather

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
        particle_capacity, batch_size, active_indices, n_active = prepare_particle_batches(
            active_tile,
            static_parameters.particle_batch_size,
        )
        if particle_capacity == 0:
            return Jx_template, Jy_template, Jz_template

        slots_per_species = active_tile.shape[-1]
        x_flat = x_tile.reshape(-1, 3)
        u_flat = u_tile.reshape(-1, 3)
        n_batches = number_of_particle_batches(n_active, batch_size)

        center_x = tiled_center_grid[0][tx, ty, tz]
        center_y = tiled_center_grid[1][tx, ty, tz]
        center_z = tiled_center_grid[2][tx, ty, tz]
        vertex_x = tiled_vertex_grid[0][tx, ty, tz]
        vertex_y = tiled_vertex_grid[1][tx, ty, tz]
        vertex_z = tiled_vertex_grid[2][tx, ty, tz]
        # get the collocated and staggered grid points for the current tile

        def deposit_batch(batch_state):
            batch_index, tile_Jx, tile_Jy, tile_Jz = batch_state
            particle_indices, valid = particle_batch_indices(
                active_indices,
                n_active,
                batch_index,
                batch_size,
            )

            x_batch = x_flat[particle_indices]
            u_batch = u_flat[particle_indices]
            x, y, z = x_batch[:, 0], x_batch[:, 1], x_batch[:, 2]
            vx, vy, vz = u_batch[:, 0], u_batch[:, 1], u_batch[:, 2]

            species_indices = particle_indices // slots_per_species
            q_batch = species_weighted_charge[species_indices]
            update_x_batch = species_config.update_x[species_indices]
            physical_mask = valid.astype(x.dtype)
            dq = physical_mask * q_batch / (dx * dy * dz)

            _, _, deltax_center, xpts_center = prepare_particle_axis_stencil(
                x,
                center_x,
                local_Nx,
                shape_factor,
                local_bc,
                wind=tile_nx * dx,
                ghost_cells=True,
            )
            _, _, deltay_center, ypts_center = prepare_particle_axis_stencil(
                y,
                center_y,
                local_Ny,
                shape_factor,
                local_bc,
                wind=tile_ny * dy,
                ghost_cells=True,
            )
            _, _, deltaz_center, zpts_center = prepare_particle_axis_stencil(
                z,
                center_z,
                local_Nz,
                shape_factor,
                local_bc,
                wind=tile_nz * dz,
                ghost_cells=True,
            )
            _, _, deltax_vertex, xpts_vertex = prepare_particle_axis_stencil(
                x,
                vertex_x,
                local_Nx,
                shape_factor,
                local_bc,
                wind=tile_nx * dx,
                ghost_cells=True,
            )
            _, _, deltay_vertex, ypts_vertex = prepare_particle_axis_stencil(
                y,
                vertex_y,
                local_Ny,
                shape_factor,
                local_bc,
                wind=tile_ny * dy,
                ghost_cells=True,
            )
            _, _, deltaz_vertex, zpts_vertex = prepare_particle_axis_stencil(
                z,
                vertex_z,
                local_Nz,
                shape_factor,
                local_bc,
                wind=tile_nz * dz,
                ghost_cells=True,
            )
            # compute the independent stencils on each actual Yee grid axis

            x_weights_center, y_weights_center, z_weights_center = jax.lax.cond(
                shape_factor == 1,
                lambda _: get_first_order_weights(deltax_center, deltay_center, deltaz_center, dx, dy, dz),
                lambda _: get_second_order_weights(deltax_center, deltay_center, deltaz_center, dx, dy, dz),
                operand=None,
            )
            x_weights_vertex, y_weights_vertex, z_weights_vertex = jax.lax.cond(
                shape_factor == 1,
                lambda _: get_first_order_weights(deltax_vertex, deltay_vertex, deltaz_vertex, dx, dy, dz),
                lambda _: get_second_order_weights(deltax_vertex, deltay_vertex, deltaz_vertex, dx, dy, dz),
                operand=None,
            )
            # compute the center- and vertex-grid weights for the selected shape

            xpts_center, x_weights_center = _collapse_tiled_axis_stencil(
                jnp.asarray(xpts_center), jnp.asarray(x_weights_center), local_Nx, reduced_x, g
            )
            xpts_vertex, x_weights_vertex = _collapse_tiled_axis_stencil(
                jnp.asarray(xpts_vertex), jnp.asarray(x_weights_vertex), local_Nx, reduced_x, g
            )
            ypts_center, y_weights_center = _collapse_tiled_axis_stencil(
                jnp.asarray(ypts_center), jnp.asarray(y_weights_center), local_Ny, reduced_y, g
            )
            ypts_vertex, y_weights_vertex = _collapse_tiled_axis_stencil(
                jnp.asarray(ypts_vertex), jnp.asarray(y_weights_vertex), local_Ny, reduced_y, g
            )
            zpts_center, z_weights_center = _collapse_tiled_axis_stencil(
                jnp.asarray(zpts_center), jnp.asarray(z_weights_center), local_Nz, reduced_z, g
            )
            zpts_vertex, z_weights_vertex = _collapse_tiled_axis_stencil(
                jnp.asarray(zpts_vertex), jnp.asarray(z_weights_vertex), local_Nz, reduced_z, g
            )
            # collapse each center and vertex stencil independently on reduced axes

            update_x1 = update_x_batch[:, 0]
            update_x2 = update_x_batch[:, 1]
            update_x3 = update_x_batch[:, 2]

            for i in range(xpts_center.shape[0]):
                for j in range(ypts_center.shape[0]):
                    for k in range(zpts_center.shape[0]):
                        tile_Jx = tile_Jx.at[
                            xpts_vertex[i], ypts_center[j], zpts_center[k]
                        ].add(
                            update_x1 * dq * vx
                            * x_weights_vertex[i] * y_weights_center[j] * z_weights_center[k],
                            mode="drop",
                        )
                        tile_Jy = tile_Jy.at[
                            xpts_center[i], ypts_vertex[j], zpts_center[k]
                        ].add(
                            update_x2 * dq * vy
                            * x_weights_center[i] * y_weights_vertex[j] * z_weights_center[k],
                            mode="drop",
                        )
                        tile_Jz = tile_Jz.at[
                            xpts_center[i], ypts_center[j], zpts_vertex[k]
                        ].add(
                            update_x3 * dq * vz
                            * x_weights_center[i] * y_weights_center[j] * z_weights_vertex[k],
                            mode="drop",
                        )

            return batch_index + 1, tile_Jx, tile_Jy, tile_Jz

        def batches_remaining(batch_state):
            batch_index, _, _, _ = batch_state
            return batch_index < n_batches

        _, tile_Jx, tile_Jy, tile_Jz = jax.lax.while_loop(
            batches_remaining,
            deposit_batch,
            (jnp.asarray(0), Jx_template, Jy_template, Jz_template),
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

    ################# CURRENT FILTERING #################
    def bilinear_filtered_current(J):
        return tiled_bilinear_filter_vector(J, static_parameters, bc_type=1)

    def digital_filtered_current(J):
        return tiled_digital_filter_vector(
            J,
            dynamic_parameters.alpha,
            static_parameters,
            bc_type=1,
        )

    J = jax.lax.cond(
        current_filter == "bilinear",
        bilinear_filtered_current,
        lambda J: jax.lax.cond(
            current_filter == "digital",
            digital_filtered_current,
            lambda J: update_tiled_vector_ghost_cells(J, static_parameters, g, bc_type=1),
            J,
        ),
        J,
    )
    # apply current filtering

    return J
