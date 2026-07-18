import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.particles.particle_class import TiledParticles


def _collapse_tiled_axis_stencil(points, weights, local_n, reduced_axis, g):
    if reduced_axis:
        collapsed_points = jnp.full((1, points.shape[1]), int(g), dtype=points.dtype)
        collapsed_weights = jnp.sum(weights, axis=0, keepdims=True)
        return collapsed_points, collapsed_weights
    return collapse_axis_stencil(points, weights, local_n, ghost_cells=True)


def fluid_velocity(
        particles,
        species_config,
        field,
        direction,
        static_parameters,
        dynamic_parameters,
):

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    shape_factor = static_parameters.shape_factor
    # unpack grid and tile parameters

    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    # get the tile shape
    g = static_parameters.guard_cells

    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g
    # piece together the total local tile shape

    tiled_grid = dynamic_parameters.grids.tiled_center_grid
    # get the grid for the tiles

    local_bc = 2

    species_weight = species_config.weight
    # compute the particle weights for each species

    field_template = jnp.zeros_like(field[0, 0, 0])

    ntx, nty, ntz = field.shape[:3]
    # get the number of tiles in each dimension

    reduced_x = int(tile_nx) == 1 and int(ntx) == 1
    reduced_y = int(tile_ny) == 1 and int(nty) == 1
    reduced_z = int(tile_nz) == 1 and int(ntz) == 1
    # determine if any of the axes are dummy axes

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        # deposit the weighted velocity numerator and shape-weight denominator for one tile
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        # reshape the particle positions into 1D arrays for processing

        u = u_tile[..., direction].reshape(-1)
        active = active_tile.reshape(-1).astype(x.dtype)
        weight = jnp.broadcast_to(species_weight[:, jnp.newaxis], active_tile.shape).reshape(-1)
        # reshape velocity, active mask, and species weights into particle-slot arrays

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

        velocity_numerator_tile = field_template
        velocity_weight_tile = field_template

        for i in range(xpts.shape[0]):
            for j in range(ypts.shape[0]):
                for k in range(zpts.shape[0]):
                    ix = xpts[i]
                    iy = ypts[j]
                    iz = zpts[k]
                    shape_weight = x_weights_node[i] * y_weights_node[j] * z_weights_node[k]
                    particle_weight = active * weight * shape_weight
                    velocity_numerator_tile = velocity_numerator_tile.at[ix, iy, iz].add(
                        particle_weight * u,
                        mode="drop",
                    )
                    velocity_weight_tile = velocity_weight_tile.at[ix, iy, iz].add(
                        particle_weight,
                        mode="drop",
                    )
        # deposit the velocity moment and its matching shape weight on the same stencil

        return velocity_numerator_tile, velocity_weight_tile

    tx, ty, tz = jnp.meshgrid(
        jnp.arange(ntx),
        jnp.arange(nty),
        jnp.arange(ntz),
        indexing="ij",
    )
    # build the tile index arrays for each dimension

    deposit_velocity = deposit_one_tile
    deposit_velocity = jax.vmap(deposit_velocity, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_velocity = jax.vmap(deposit_velocity, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_velocity = jax.vmap(deposit_velocity, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    # vectorize the deposit_one_tile function over the tile indices using jax.vmap

    velocity_numerator, velocity_weight = deposit_velocity(particles.x, particles.u, particles.active, tx, ty, tz)
    # deposit the velocity numerator and denominator for all tiles

    velocity_numerator = update_tiled_ghost_cells(velocity_numerator, static_parameters, g)
    velocity_weight = update_tiled_ghost_cells(velocity_weight, static_parameters, g)
    # update ghost cells before forming the local average so halos use the same values as their source interiors

    occupied = velocity_weight > 0.0
    field = jnp.where(occupied, velocity_numerator / jnp.where(occupied, velocity_weight, 1.0), 0.0)
    # compute the local fluid velocity and leave cells with no deposited particle weight at zero

    return field


def compute_velocity_field(particles, field, direction, static_parameters, dynamic_parameters, species_config=None):

    return fluid_velocity(
        particles,
        species_config,
        field,
        int(direction),
        static_parameters,
        dynamic_parameters,
    )