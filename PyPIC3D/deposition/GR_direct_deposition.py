from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import (
    BC_TYPE_PARTICLE,
    fold_tiled_vector_ghost_cells,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_tiled_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.relativity.core import contravariant_three_velocity
from PyPIC3D.relativity.interpolate_metric import interpolate_metric, safe_inactive_positions
from PyPIC3D.utilities.filters import tiled_bilinear_filter_vector, tiled_digital_filter_vector


@partial(jax.jit, static_argnames="static_parameters")
def GR_direct_deposition(
    particles,
    species_config,
    J,
    metric,
    static_parameters,
    dynamic_parameters,
):
    """
    Direct current deposition for a fixed 3+1 metric.

    ``particles.u`` stores covariant spatial components ``u_i``.  Deposition,
    ghost folding, and filtering operate on, and return, the conformal current
    of the densitized Maxwell update

        sqrt(gamma) J^i = (q / d^3x) * S(x) * (alpha v^i - beta^i).
    """

    current_filter = static_parameters.current_filter
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    g = int(static_parameters.guard_cells)
    tiled_grid = dynamic_parameters.grids.tiled_center_grid
    grid = dynamic_parameters.grids.center
    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz

    Jx_tiles, Jy_tiles, Jz_tiles = J
    ntx, nty, ntz = Jx_tiles.shape[:3]
    tile_nx, tile_ny, tile_nz = tile_shape
    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g
    shape_factor = static_parameters.shape_factor

    reduced_x = int(tile_nx) == 1 and int(ntx) == 1
    reduced_y = int(tile_ny) == 1 and int(nty) == 1
    reduced_z = int(tile_nz) == 1 and int(ntz) == 1

    Jx_template = jnp.zeros_like(Jx_tiles[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy_tiles[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz_tiles[0, 0, 0])
    local_bc = 2
    species_weighted_charge = species_config.charge * species_config.weight

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        x = x_tile[..., 0].reshape(-1)
        y = x_tile[..., 1].reshape(-1)
        z = x_tile[..., 2].reshape(-1)
        u_cov = u_tile.reshape(-1, 3)
        active = active_tile.reshape(-1).astype(x.dtype)
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x1 = jnp.broadcast_to(species_config.update_x[:, 0, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x2 = jnp.broadcast_to(species_config.update_x[:, 1, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x3 = jnp.broadcast_to(species_config.update_x[:, 2, jnp.newaxis], active_tile.shape).reshape(-1)
        dq = q / (dx * dy * dz)

        tiled_x_grid = tiled_grid[0][tx, ty, tz]
        tiled_y_grid = tiled_grid[1][tx, ty, tz]
        tiled_z_grid = tiled_grid[2][tx, ty, tz]
        center_grid = (tiled_x_grid, tiled_y_grid, tiled_z_grid)

        active_axes = (not reduced_x, not reduced_y, not reduced_z)
        evaluation_position = safe_inactive_positions(
            jnp.stack((x, y, z), axis=-1), active.astype(bool), center_grid, active_axes, g)
        u_cov = jnp.where(active[:, None].astype(bool), u_cov, 0.)
        metric_at_particles = interpolate_metric(
            jax.tree.map(lambda array: array[tx, ty, tz], metric.center),
            evaluation_position,
            center_grid,
            static_parameters.metric,
            active_axes,
            (g, g, g),
            derivatives=False,
        )
        v_con = contravariant_three_velocity(u_cov, metric_at_particles.gamma_inv)
        source_velocity = metric_at_particles.lapse[:, jnp.newaxis] * v_con - metric_at_particles.shift
        vx = source_velocity[:, 0]
        vy = source_velocity[:, 1]
        vz = source_velocity[:, 2]

        x_grid, y_grid, z_grid = grid

        x, _, deltax_node, xpts_node = prepare_particle_axis_stencil(
            x,
            x_grid,
            x_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_nx * dx,
            ghost_cells=True,
        )
        _, _, deltax_face, xpts_face = prepare_particle_axis_stencil(
            x,
            x_grid + 0.5 * dx,
            x_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_nx * dx,
            ghost_cells=True,
        )
        y, _, deltay_node, ypts_node = prepare_particle_axis_stencil(
            y,
            y_grid,
            y_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_ny * dy,
            ghost_cells=True,
        )
        _, _, deltay_face, ypts_face = prepare_particle_axis_stencil(
            y,
            y_grid + 0.5 * dy,
            y_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_ny * dy,
            ghost_cells=True,
        )
        z, _, deltaz_node, zpts_node = prepare_particle_axis_stencil(
            z,
            z_grid,
            z_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_nz * dz,
            ghost_cells=True,
        )
        _, _, deltaz_face, zpts_face = prepare_particle_axis_stencil(
            z,
            z_grid + 0.5 * dz,
            z_grid.shape[0],
            shape_factor,
            local_bc,
            wind=tile_nz * dz,
            ghost_cells=True,
        )
        # Center- and vertex-located quantities need independent anchors.

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

        xpts_node = jnp.asarray(xpts_node)
        ypts_node = jnp.asarray(ypts_node)
        zpts_node = jnp.asarray(zpts_node)
        xpts_face = jnp.asarray(xpts_face)
        ypts_face = jnp.asarray(ypts_face)
        zpts_face = jnp.asarray(zpts_face)
        x_local_offset = tx * tile_nx - (g - 1)
        y_local_offset = ty * tile_ny - (g - 1)
        z_local_offset = tz * tile_nz - (g - 1)
        xpts_node = xpts_node - x_local_offset
        xpts_face = xpts_face - x_local_offset
        ypts_node = ypts_node - y_local_offset
        ypts_face = ypts_face - y_local_offset
        zpts_node = zpts_node - z_local_offset
        zpts_face = zpts_face - z_local_offset
        x_weights_node = jnp.asarray(x_weights_node)
        y_weights_node = jnp.asarray(y_weights_node)
        z_weights_node = jnp.asarray(z_weights_node)
        x_weights_face = jnp.asarray(x_weights_face)
        y_weights_face = jnp.asarray(y_weights_face)
        z_weights_face = jnp.asarray(z_weights_face)

        xpts_node, x_weights_node = collapse_tiled_axis_stencil(
            xpts_node, x_weights_node, local_Nx, reduced_x, g
        )
        xpts_face, x_weights_face = collapse_tiled_axis_stencil(
            xpts_face, x_weights_face, local_Nx, reduced_x, g
        )
        ypts_node, y_weights_node = collapse_tiled_axis_stencil(
            ypts_node, y_weights_node, local_Ny, reduced_y, g
        )
        ypts_face, y_weights_face = collapse_tiled_axis_stencil(
            ypts_face, y_weights_face, local_Ny, reduced_y, g
        )
        zpts_node, z_weights_node = collapse_tiled_axis_stencil(
            zpts_node, z_weights_node, local_Nz, reduced_z, g
        )
        zpts_face, z_weights_face = collapse_tiled_axis_stencil(
            zpts_face, z_weights_face, local_Nz, reduced_z, g
        )

        tile_Jx = Jx_template
        tile_Jy = Jy_template
        tile_Jz = Jz_template

        for i in range(xpts_node.shape[0]):
            for j in range(ypts_node.shape[0]):
                for k in range(zpts_node.shape[0]):
                    tile_Jx = tile_Jx.at[xpts_face[i], ypts_node[j], zpts_node[k]].add(
                        active * update_x1 * dq * vx * x_weights_face[i] * y_weights_node[j] * z_weights_node[k],
                        mode="drop",
                    )
                    tile_Jy = tile_Jy.at[xpts_node[i], ypts_face[j], zpts_node[k]].add(
                        active * update_x2 * dq * vy * x_weights_node[i] * y_weights_face[j] * z_weights_node[k],
                        mode="drop",
                    )
                    tile_Jz = tile_Jz.at[xpts_node[i], ypts_node[j], zpts_face[k]].add(
                        active * update_x3 * dq * vz * x_weights_node[i] * y_weights_node[j] * z_weights_face[k],
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

    Jx, Jy, Jz = deposit_tiles(
        particles.x,
        particles.u,
        particles.active,
        tx,
        ty,
        tz,
    )

    conformal_J = fold_tiled_vector_ghost_cells(
        (Jx, Jy, Jz),
        static_parameters,
        g,
        bc_type=BC_TYPE_PARTICLE,
    )

    def bilinear_filtered_current(conformal_J):
        return tiled_bilinear_filter_vector(
            conformal_J,
            static_parameters,
            bc_type=BC_TYPE_PARTICLE,
        )

    def digital_filtered_current(conformal_J):
        return tiled_digital_filter_vector(
            conformal_J,
            dynamic_parameters.alpha,
            static_parameters,
            bc_type=BC_TYPE_PARTICLE,
        )

    return jax.lax.cond(
        current_filter == "bilinear",
        bilinear_filtered_current,
        lambda conformal_J: jax.lax.cond(
            current_filter == "digital",
            digital_filtered_current,
            lambda conformal_J: update_tiled_vector_ghost_cells(
                conformal_J,
                static_parameters,
                g,
                bc_type=BC_TYPE_PARTICLE,
            ),
            conformal_J,
        ),
        conformal_J,
    )
