import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    compute_particle_anchor,
    particle_axis_offset,
)
from PyPIC3D.deposition.Esirkepov import (
    collapse_redundant_axis,
    get_1D_esirkepov_weights,
    get_2D_esirkepov_weights,
    get_3D_esirkepov_weights,
    shift_old_stencil,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.solvers.yee_tiled import (
    tiled_grid_axes_from_world,
    update_tiled_vector_ghost_cells,
)


def _active_stencil_indices(axis_active):
    if axis_active:
        return (0, 1, 2, 3, 4)
    return (2,)


def fold_tiled_esirkepov_ghost_cells(field_tiles, world, component_axis, num_guard_cells=2, tile_shape=None):
    """
    Fold Esirkepov current deposits using the configured guard depth.

    Internal tile faces are pure ownership transfers.  At global particle
    boundaries, periodic deposits wrap, reflecting deposits mirror into the
    same-side physical layer with the Esirkepov component sign, and absorbing
    deposits are discarded.
    """

    g = int(num_guard_cells)
    if tile_shape is None:
        tile_shape = tuple(int(width) - 2 * g for width in field_tiles.shape[3:])
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    ntx, nty, ntz = field_tiles.shape[:3]
    reduced_axes = (
        int(ntx) == 1 and tile_nx == 1,
        int(nty) == 1 and tile_ny == 1,
        int(ntz) == 1 and tile_nz == 1,
    )
    particle_bc = world.get("particle_boundary_conditions", {"x": 0, "y": 0, "z": 0})

    def fold_axis(field_tiles, axis, bc, reduced_axis):
        tiles = jnp.moveaxis(field_tiles, (axis, 3 + axis), (0, 3))
        lower_ghost = tiles[:, :, :, :g, :, :]
        upper_ghost = tiles[:, :, :, -g:, :, :]
        sign = -1.0 if axis == component_axis else 1.0

        if reduced_axis:
            ghost_sum = (
                jnp.sum(lower_ghost, axis=3, keepdims=True)
                + jnp.sum(upper_ghost, axis=3, keepdims=True)
            )

            def periodic_boundary(tiles):
                return tiles.at[:, :, :, g:g + 1, :, :].add(ghost_sum)

            def reflecting_boundary(tiles):
                return tiles.at[:, :, :, g:g + 1, :, :].add(sign * ghost_sum)

            def absorbing_boundary(tiles):
                return tiles

            tiles = jax.lax.switch(bc, (periodic_boundary, reflecting_boundary, absorbing_boundary), tiles)
            tiles = tiles.at[:, :, :, :g, :, :].set(0.0)
            tiles = tiles.at[:, :, :, -g:, :, :].set(0.0)
            return jnp.moveaxis(tiles, (0, 3), (axis, 3 + axis))

        tiles = tiles.at[:-1, :, :, -2 * g:-g, :, :].add(lower_ghost[1:, :, :, :, :, :])
        tiles = tiles.at[1:, :, :, g:2 * g, :, :].add(upper_ghost[:-1, :, :, :, :, :])

        def periodic_boundary(tiles):
            tiles = tiles.at[-1, :, :, -2 * g:-g, :, :].add(lower_ghost[0, :, :, :, :, :])
            tiles = tiles.at[0, :, :, g:2 * g, :, :].add(upper_ghost[-1, :, :, :, :, :])
            return tiles

        def reflecting_boundary(tiles):
            tiles = tiles.at[0, :, :, g:2 * g, :, :].add(sign * lower_ghost[0, :, :, :, :, :])
            tiles = tiles.at[-1, :, :, -2 * g:-g, :, :].add(sign * upper_ghost[-1, :, :, :, :, :])
            return tiles

        def absorbing_boundary(tiles):
            return tiles

        tiles = jax.lax.switch(bc, (periodic_boundary, reflecting_boundary, absorbing_boundary), tiles)
        tiles = tiles.at[:, :, :, :g, :, :].set(0.0)
        tiles = tiles.at[:, :, :, -g:, :, :].set(0.0)
        return jnp.moveaxis(tiles, (0, 3), (axis, 3 + axis))

    field_tiles = fold_axis(field_tiles, 0, particle_bc["x"], reduced_axes[0])
    field_tiles = fold_axis(field_tiles, 1, particle_bc["y"], reduced_axes[1])
    field_tiles = fold_axis(field_tiles, 2, particle_bc["z"], reduced_axes[2])

    return field_tiles


def fold_tiled_esirkepov_vector_ghost_cells(field_tiles, world, num_guard_cells=2, tile_shape=None):
    return tuple(
        fold_tiled_esirkepov_ghost_cells(component, world, component_axis, num_guard_cells, tile_shape)
        for component_axis, component in enumerate(field_tiles)
    )


@partial(jit, static_argnames=("tile_shape", "g"))
def tiled_esirkepov_current(tiled_particles, species_config, J_tiles, constants, world, grid=None, tile_shape=None, g=2):
    """
    Deposit Esirkepov current into tile-local current buffers.

    ``tiled_particles.x`` is the old particle position after the velocity push.
    The future position used in Esirkepov's charge-conserving difference is
    predicted locally as ``x + u*dt``.  Particle boundary conditions and retile
    ownership are applied by the caller after deposition.
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
    dt = world["dt"]
    shape_factor = world["shape_factor"]

    Jx_tiles, Jy_tiles, Jz_tiles = J_tiles
    ntx, nty, ntz = Jx_tiles.shape[:3]
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    g = int(g)
    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g

    x_active = ntx * tile_nx > 1
    y_active = nty * tile_ny > 1
    z_active = ntz * tile_nz > 1

    Jx_template = jnp.zeros_like(Jx_tiles[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy_tiles[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz_tiles[0, 0, 0])

    species_weighted_charge = species_config.charge * species_config.weight

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        old_x = x_tile[..., 0].reshape(-1)
        old_y = x_tile[..., 1].reshape(-1)
        old_z = x_tile[..., 2].reshape(-1)
        vx = u_tile[..., 0].reshape(-1)
        vy = u_tile[..., 1].reshape(-1)
        vz = u_tile[..., 2].reshape(-1)
        active = active_tile.reshape(-1).astype(old_x.dtype)
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        N_particles = old_x.shape[0]
        update_x1 = jnp.broadcast_to(species_config.update_x[:, 0, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x2 = jnp.broadcast_to(species_config.update_x[:, 1, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x3 = jnp.broadcast_to(species_config.update_x[:, 2, jnp.newaxis], active_tile.shape).reshape(-1)

        x = old_x + jnp.where(update_x1, vx * dt, 0.0)
        y = old_y + jnp.where(update_x2, vy * dt, 0.0)
        z = old_z + jnp.where(update_x3, vz * dt, 0.0)

        x_grid = tiled_grid[0][tx, ty, tz]
        y_grid = tiled_grid[1][tx, ty, tz]
        z_grid = tiled_grid[2][tx, ty, tz]

        x0 = compute_particle_anchor(x, x_grid, shape_factor)
        y0 = compute_particle_anchor(y, y_grid, shape_factor)
        z0 = compute_particle_anchor(z, z_grid, shape_factor)
        old_x0 = compute_particle_anchor(old_x, x_grid, shape_factor)
        old_y0 = compute_particle_anchor(old_y, y_grid, shape_factor)
        old_z0 = compute_particle_anchor(old_z, z_grid, shape_factor)

        deltax = particle_axis_offset(x, x0, x_grid)
        deltay = particle_axis_offset(y, y0, y_grid)
        deltaz = particle_axis_offset(z, z0, z_grid)
        old_deltax = particle_axis_offset(old_x, old_x0, x_grid)
        old_deltay = particle_axis_offset(old_y, old_y0, y_grid)
        old_deltaz = particle_axis_offset(old_z, old_z0, z_grid)

        shift_x = x0 - old_x0
        shift_y = y0 - old_y0
        shift_z = z0 - old_z0

        offsets = jnp.asarray([-2, -1, 0, 1, 2], dtype=x0.dtype)
        xpts = x0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
        ypts = y0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
        zpts = z0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]

        xw, yw, zw = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None,
        )
        oxw, oyw, ozw = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            operand=None,
        )

        tmp = jnp.zeros_like(xw[0])
        xw = [tmp, xw[0], xw[1], xw[2], tmp]
        yw = [tmp, yw[0], yw[1], yw[2], tmp]
        zw = [tmp, zw[0], zw[1], zw[2], tmp]
        oxw = [tmp, oxw[0], oxw[1], oxw[2], tmp]
        oyw = [tmp, oyw[0], oyw[1], oyw[2], tmp]
        ozw = [tmp, ozw[0], ozw[1], ozw[2], tmp]

        oxw = shift_old_stencil(oxw, shift_x)
        oyw = shift_old_stencil(oyw, shift_y)
        ozw = shift_old_stencil(ozw, shift_z)

        xpts, xw, oxw = collapse_redundant_axis(xpts, xw, oxw, x_active, local_Nx)
        ypts, yw, oyw = collapse_redundant_axis(ypts, yw, oyw, y_active, local_Ny)
        zpts, zw, ozw = collapse_redundant_axis(zpts, zw, ozw, z_active, local_Nz)

        if x_active and y_active and z_active:
            Wx_, Wy_, Wz_ = get_3D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles)
        elif (x_active and y_active and (not z_active)) or (x_active and z_active and (not y_active)) or (
            y_active and z_active and (not x_active)
        ):
            if not x_active:
                null_dim = 0
            elif not y_active:
                null_dim = 1
            else:
                null_dim = 2
            Wx_, Wy_, Wz_ = get_2D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, null_dim=null_dim)
        elif x_active and (not y_active) and (not z_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=0)
        elif y_active and (not x_active) and (not z_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=1)
        else:
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=2)

        dJx = jax.lax.cond(
            x_active,
            lambda _: active * (-(q / (dy * dz)) / dt),
            lambda _: active * q * vx / (dx * dy * dz),
            operand=None,
        )
        dJy = jax.lax.cond(
            y_active,
            lambda _: active * (-(q / (dx * dz)) / dt),
            lambda _: active * q * vy / (dx * dy * dz),
            operand=None,
        )
        dJz = jax.lax.cond(
            z_active,
            lambda _: active * (-(q / (dx * dy)) / dt),
            lambda _: active * q * vz / (dx * dy * dz),
            operand=None,
        )

        Fx = dJx * Wx_
        Fy = dJy * Wy_
        Fz = dJz * Wz_
        Jx_loc = jnp.cumsum(Fx, axis=0)
        Jy_loc = jnp.cumsum(Fy, axis=1)
        Jz_loc = jnp.cumsum(Fz, axis=2)

        tile_Jx = Jx_template
        tile_Jy = Jy_template
        tile_Jz = Jz_template

        x_stencil_indices = _active_stencil_indices(x_active)
        y_stencil_indices = _active_stencil_indices(y_active)
        z_stencil_indices = _active_stencil_indices(z_active)

        for i in x_stencil_indices:
            for j in y_stencil_indices:
                for k in z_stencil_indices:
                    ix = xpts[i, :]
                    iy = ypts[j, :]
                    iz = zpts[k, :]
                    tile_Jx = tile_Jx.at[ix, iy, iz].add(
                        jax.lax.cond(x_active, lambda _: Jx_loc[i, j, k, :], lambda _: Fx[i, j, k, :], operand=None),
                        mode="drop",
                    )
                    tile_Jy = tile_Jy.at[ix, iy, iz].add(
                        jax.lax.cond(y_active, lambda _: Jy_loc[i, j, k, :], lambda _: Fy[i, j, k, :], operand=None),
                        mode="drop",
                    )
                    tile_Jz = tile_Jz.at[ix, iy, iz].add(
                        jax.lax.cond(z_active, lambda _: Jz_loc[i, j, k, :], lambda _: Fz[i, j, k, :], operand=None),
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

    J_tiles = fold_tiled_esirkepov_vector_ghost_cells((Jx_tiles, Jy_tiles, Jz_tiles), world, num_guard_cells=g, tile_shape=tile_shape)
    J_tiles = update_tiled_vector_ghost_cells(J_tiles, world, num_guard_cells=g, tile_shape=tile_shape)

    return J_tiles
