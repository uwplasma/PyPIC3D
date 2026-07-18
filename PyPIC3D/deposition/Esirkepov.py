import jax
import jax.numpy as jnp
from functools import partial

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    compute_particle_anchor,
    particle_axis_offset,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.boundary_conditions.ghost_cells import (
    fold_tiled_vector_ghost_cells,
    update_tiled_vector_ghost_cells,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles


def shift_old_stencil(old_w_list, shift):
    """Roll old weights into the new-cell frame for Esirkepov deposition."""
    old_w = jnp.stack(old_w_list, axis=0)

    def roll_one_particle(w5, s):
        return jnp.roll(w5, -s, axis=0)

    rolled = jax.vmap(roll_one_particle, in_axes=(1, 0), out_axes=1)(old_w, shift)
    return [rolled[i, :] for i in range(5)]


def collapse_redundant_axis(points, current_weights, old_weights, axis_active, extended_axis_size):
    """Collapse a singleton axis onto the physical cell of the two-ghost buffer."""
    if axis_active:
        return points, current_weights, old_weights

    collapsed_index = extended_axis_size // 2
    # the physical cell is the middle point of the extended axis, which has two ghost cells on either side.
    zero = jnp.zeros_like(current_weights[0])
    # initialize a zero array for padding the weights after collapsing the axis
    current_total = sum(current_weights)
    # sum the current weights along the collapsed axis to preserve the total current contribution of the particle
    old_total = sum(old_weights)
    # sum the old weights along the collapsed axis to preserve the total old weight for differencing in Esirkepov's formula
    collapsed_points = jnp.full(points.shape, collapsed_index, dtype=points.dtype)
    collapsed_current = [zero, zero, current_total, zero, zero]
    collapsed_old = [zero, zero, old_total, zero, zero]
    # build the 5 point stencil with the total weights at the physical cell and zeros at the redundant ghost points.
    return collapsed_points, collapsed_current, collapsed_old


@partial(jax.jit, static_argnames=("static_parameters", "bc_type"))
def Esirkepov_current(
    particles: TiledParticles,
    species_config: SpeciesConfig,
    J,
    static_parameters,
    dynamic_parameters,
    bc_type=1,
):
    """
    Deposit Esirkepov current into tile-local current buffers.

    ``particles.x`` is the old particle position after the velocity push. The
    future position used in Esirkepov's charge-conserving difference is
    predicted locally as ``x + u*dt``. Particle retile ownership is applied by
    the caller after deposition.
    """

    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    # get the tile shape
    g = int(static_parameters.guard_cells)
    # get the number of guard cells on the tiles
    tiled_grid = dynamic_parameters.grids.tiled_center_grid
    # get the tile grid for the current deposition

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    # get spatial resolution
    dt = dynamic_parameters.dt
    # get temporal resolution
    shape_factor = static_parameters.shape_factor
    # get shape factor

    Jx, Jy, Jz = J
    # unpack current density
    ntx, nty, ntz = Jx.shape[:3]
    # get the number of tiles
    tile_nx, tile_ny, tile_nz = [int(width) for width in tile_shape]
    # unpack the tile shapes
    local_Nx = tile_nx + 2 * g
    local_Ny = tile_ny + 2 * g
    local_Nz = tile_nz + 2 * g
    # compute the local shape including guard cells

    x_active = ntx * tile_nx > 1
    y_active = nty * tile_ny > 1
    z_active = ntz * tile_nz > 1
    # determine which axes are actually active and which ones are redundant

    Jx_template = jnp.zeros_like(Jx[0, 0, 0])
    Jy_template = jnp.zeros_like(Jy[0, 0, 0])
    Jz_template = jnp.zeros_like(Jz[0, 0, 0])
    # build a template array for the local J tiles

    species_weighted_charge = species_config.charge * species_config.weight
    # compute the species weighted charge

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        old_x = x_tile[..., 0].reshape(-1)
        old_y = x_tile[..., 1].reshape(-1)
        old_z = x_tile[..., 2].reshape(-1)
        # get the old positions and reshape them as 1D arrays
        vx = u_tile[..., 0].reshape(-1)
        vy = u_tile[..., 1].reshape(-1)
        vz = u_tile[..., 2].reshape(-1)
        # get the velocities and reshape them as 1D arrays
        active = active_tile.reshape(-1).astype(old_x.dtype)
        # get the active mask and reshape it as a 1D array
        q = jnp.broadcast_to(species_weighted_charge[:, jnp.newaxis], active_tile.shape).reshape(-1)
        # broadcast the species weighted charge to the shape of the active tile and reshape it as a 1D array
        N_particles = old_x.shape[0]
        # get the number of particles in the tile
        update_x1 = jnp.broadcast_to(species_config.update_x[:, 0, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x2 = jnp.broadcast_to(species_config.update_x[:, 1, jnp.newaxis], active_tile.shape).reshape(-1)
        update_x3 = jnp.broadcast_to(species_config.update_x[:, 2, jnp.newaxis], active_tile.shape).reshape(-1)
        # determine which axes are updated for each particle and reshape them as 1D arrays

        x = old_x + jnp.where(update_x1, vx * dt, 0.0)
        y = old_y + jnp.where(update_x2, vy * dt, 0.0)
        z = old_z + jnp.where(update_x3, vz * dt, 0.0)
        # step the particle positions forward in time using the velocity and dt, but only for the axes that are updated

        x_grid = tiled_grid[0][tx, ty, tz]
        y_grid = tiled_grid[1][tx, ty, tz]
        z_grid = tiled_grid[2][tx, ty, tz]
        # get the local grid for the tile in each axis

        x0 = compute_particle_anchor(x, x_grid, shape_factor)
        y0 = compute_particle_anchor(y, y_grid, shape_factor)
        z0 = compute_particle_anchor(z, z_grid, shape_factor)
        # get the particle anchor points for the new positions in each axis
        old_x0 = compute_particle_anchor(old_x, x_grid, shape_factor)
        old_y0 = compute_particle_anchor(old_y, y_grid, shape_factor)
        old_z0 = compute_particle_anchor(old_z, z_grid, shape_factor)
        # get the particle anchor points for the old positions in each axis

        deltax = particle_axis_offset(x, x0, x_grid)
        deltay = particle_axis_offset(y, y0, y_grid)
        deltaz = particle_axis_offset(z, z0, z_grid)
        # compute the particle offsets from the anchor points for the new positions in each axis
        old_deltax = particle_axis_offset(old_x, old_x0, x_grid)
        old_deltay = particle_axis_offset(old_y, old_y0, y_grid)
        old_deltaz = particle_axis_offset(old_z, old_z0, z_grid)
        # compute the particle offsets from the anchor points for the old positions in each axis

        shift_x = x0 - old_x0
        shift_y = y0 - old_y0
        shift_z = z0 - old_z0
        # get the difference between the new and old anchor points to determine how much the old weights need to be shifted to align with the new anchor points

        offsets = jnp.asarray([-2, -1, 0, 1, 2], dtype=x0.dtype)
        xpts = x0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
        ypts = y0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
        zpts = z0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
        # compute the 5-point stencil indices for the new anchor points in each axis

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
        # get the current weights for the new and old positions based on the shape factor

        tmp = jnp.zeros_like(xw[0])
        xw = [tmp, xw[0], xw[1], xw[2], tmp]
        yw = [tmp, yw[0], yw[1], yw[2], tmp]
        zw = [tmp, zw[0], zw[1], zw[2], tmp]
        oxw = [tmp, oxw[0], oxw[1], oxw[2], tmp]
        oyw = [tmp, oyw[0], oyw[1], oyw[2], tmp]
        ozw = [tmp, ozw[0], ozw[1], ozw[2], tmp]
        # build the 5 point stencil weights for the new and old positions, padding with zeros at the ghost points

        oxw = shift_old_stencil(oxw, shift_x)
        oyw = shift_old_stencil(oyw, shift_y)
        ozw = shift_old_stencil(ozw, shift_z)
        # shift the old weights to align with the new anchor points based on the computed shifts

        xpts, xw, oxw = collapse_redundant_axis(xpts, xw, oxw, x_active, local_Nx)
        ypts, yw, oyw = collapse_redundant_axis(ypts, yw, oyw, y_active, local_Ny)
        zpts, zw, ozw = collapse_redundant_axis(zpts, zw, ozw, z_active, local_Nz)
        # collapse any redundant axes (if the axis is inactive) to ensure that the weights and points are correctly aligned for deposition

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
        # compute the local current contributions for each axis based on whether the axis is active or not, using the Esirkepov formula for charge-conserving current deposition.

        tile_Jx = Jx_template
        tile_Jy = Jy_template
        tile_Jz = Jz_template
        # initialize the local tile current arrays to zero

        if x_active and y_active and z_active:
            # if all three axes are active, compute the 3D Esirkepov weights and deposit the currents accordingly
            Wx_, Wy_, Wz_ = _3D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles)
            Fx = dJx * Wx_
            Fy = dJy * Wy_
            Fz = dJz * Wz_
            Jx_loc = jnp.cumsum(Fx, axis=0)
            Jy_loc = jnp.cumsum(Fy, axis=1)
            Jz_loc = jnp.cumsum(Fz, axis=2)

            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        ix = xpts[i, :]
                        iy = ypts[j, :]
                        iz = zpts[k, :]
                        tile_Jx = tile_Jx.at[ix, iy, iz].add(Jx_loc[i, j, k, :], mode="drop")
                        tile_Jy = tile_Jy.at[ix, iy, iz].add(Jy_loc[i, j, k, :], mode="drop")
                        tile_Jz = tile_Jz.at[ix, iy, iz].add(Jz_loc[i, j, k, :], mode="drop")
        elif (x_active and y_active and (not z_active)) or (x_active and z_active and (not y_active)) or (
            y_active and z_active and (not x_active)
        ): # if two axes are active and one is inactive, compute the 2D Esirkepov weights and deposit the currents accordingly
            if not x_active:
                null_dim = 0
            elif not y_active:
                null_dim = 1
            else:
                null_dim = 2
            Wx_, Wy_, Wz_ = _2d_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, null_dim=null_dim)
            Fx = dJx * Wx_
            Fy = dJy * Wy_
            Fz = dJz * Wz_

            if null_dim == 0:
                Jy_loc = jnp.cumsum(Fy, axis=0)
                Jz_loc = jnp.cumsum(Fz, axis=1)
                for j in range(5):
                    for k in range(5):
                        ix = xpts[2, :]
                        iy = ypts[j, :]
                        iz = zpts[k, :]
                        tile_Jx = tile_Jx.at[ix, iy, iz].add(Fx[j, k, :], mode="drop")
                        tile_Jy = tile_Jy.at[ix, iy, iz].add(Jy_loc[j, k, :], mode="drop")
                        tile_Jz = tile_Jz.at[ix, iy, iz].add(Jz_loc[j, k, :], mode="drop")
            elif null_dim == 1:
                Jx_loc = jnp.cumsum(Fx, axis=0)
                Jz_loc = jnp.cumsum(Fz, axis=1)
                for i in range(5):
                    for k in range(5):
                        ix = xpts[i, :]
                        iy = ypts[2, :]
                        iz = zpts[k, :]
                        tile_Jx = tile_Jx.at[ix, iy, iz].add(Jx_loc[i, k, :], mode="drop")
                        tile_Jy = tile_Jy.at[ix, iy, iz].add(Fy[i, k, :], mode="drop")
                        tile_Jz = tile_Jz.at[ix, iy, iz].add(Jz_loc[i, k, :], mode="drop")
            else:
                Jx_loc = jnp.cumsum(Fx, axis=0)
                Jy_loc = jnp.cumsum(Fy, axis=1)
                for i in range(5):
                    for j in range(5):
                        ix = xpts[i, :]
                        iy = ypts[j, :]
                        iz = zpts[2, :]
                        tile_Jx = tile_Jx.at[ix, iy, iz].add(Jx_loc[i, j, :], mode="drop")
                        tile_Jy = tile_Jy.at[ix, iy, iz].add(Jy_loc[i, j, :], mode="drop")
                        tile_Jz = tile_Jz.at[ix, iy, iz].add(Fz[i, j, :], mode="drop")
        elif x_active and (not y_active) and (not z_active):
            # if only the x-axis is active, compute the 1D Esirkepov weights and deposit the currents accordingly
            Wx_, Wy_, Wz_ = _1d_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, dim=0)
            Fx = dJx * Wx_
            Fy = dJy * Wy_
            Fz = dJz * Wz_
            Jx_loc = jnp.cumsum(Fx, axis=0)
            for i in range(5):
                ix = xpts[i, :]
                iy = ypts[2, :]
                iz = zpts[2, :]
                tile_Jx = tile_Jx.at[ix, iy, iz].add(Jx_loc[i, :], mode="drop")
                tile_Jy = tile_Jy.at[ix, iy, iz].add(Fy[i, :], mode="drop")
                tile_Jz = tile_Jz.at[ix, iy, iz].add(Fz[i, :], mode="drop")
        elif y_active and (not x_active) and (not z_active):
            # if only the y-axis is active, compute the 1D Esirkepov weights and deposit the currents accordingly
            Wx_, Wy_, Wz_ = _1d_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, dim=1)
            Fx = dJx * Wx_
            Fy = dJy * Wy_
            Fz = dJz * Wz_
            Jy_loc = jnp.cumsum(Fy, axis=0)
            for j in range(5):
                ix = xpts[2, :]
                iy = ypts[j, :]
                iz = zpts[2, :]
                tile_Jx = tile_Jx.at[ix, iy, iz].add(Fx[j, :], mode="drop")
                tile_Jy = tile_Jy.at[ix, iy, iz].add(Jy_loc[j, :], mode="drop")
                tile_Jz = tile_Jz.at[ix, iy, iz].add(Fz[j, :], mode="drop")
        else:
            # if only the z-axis is active, compute the 1D Esirkepov weights and deposit the currents accordingly
            Wx_, Wy_, Wz_ = _1d_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, dim=2)
            Fx = dJx * Wx_
            Fy = dJy * Wy_
            Fz = dJz * Wz_
            Jz_loc = jnp.cumsum(Fz, axis=0)
            for k in range(5):
                ix = xpts[2, :]
                iy = ypts[2, :]
                iz = zpts[k, :]
                tile_Jx = tile_Jx.at[ix, iy, iz].add(Fx[k, :], mode="drop")
                tile_Jy = tile_Jy.at[ix, iy, iz].add(Fy[k, :], mode="drop")
                tile_Jz = tile_Jz.at[ix, iy, iz].add(Jz_loc[k, :], mode="drop")

        return tile_Jx, tile_Jy, tile_Jz

    tx, ty, tz = jnp.meshgrid(
        jnp.arange(ntx),
        jnp.arange(nty),
        jnp.arange(ntz),
        indexing="ij",
    )
    # build a meshgrid of tile indices for the x, y, and z axes to pass to the deposit function

    deposit_tiles = deposit_one_tile
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0), out_axes=0)
    # nested vmap to vectorize the deposit function over the tile indices for x, y, and z axes

    Jx, Jy, Jz = deposit_tiles(
        particles.x,
        particles.u,
        particles.active,
        tx,
        ty,
        tz,
    )
    # deposit the currents for all tiles in parallel using the vectorized deposit function

    J = fold_tiled_vector_ghost_cells((Jx, Jy, Jz), static_parameters, num_guard_cells=g, bc_type=bc_type)
    # fold the deposited currents across tile boundaries, applying the appropriate boundary conditions for ghost cells
    J = update_tiled_vector_ghost_cells(J, static_parameters, num_guard_cells=g, bc_type=bc_type)
    # update the ghost cells of the folded currents to ensure consistency across tile boundaries

    return J


def _3D_esirkepov_weights(
        x_weights, 
        y_weights, 
        z_weights, 
        old_x_weights, 
        old_y_weights, 
        old_z_weights, 
        N_particles
    ):
    


    Wx_ = jnp.zeros((len(x_weights), len(y_weights), len(z_weights), N_particles))
    Wy_ = jnp.zeros_like(Wx_)
    Wz_ = jnp.zeros_like(Wx_)

    for i in range(len(x_weights)):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[i, j, k, :].set(
                    (x_weights[i] - old_x_weights[i])
                    * (
                        1 / 3 * (y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k])
                        + 1 / 6 * (y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k])
                    )
                )

                Wy_ = Wy_.at[i, j, k, :].set(
                    (y_weights[j] - old_y_weights[j])
                    * (
                        1 / 3 * (x_weights[i] * z_weights[k] + old_x_weights[i] * old_z_weights[k])
                        + 1 / 6 * (x_weights[i] * old_z_weights[k] + old_x_weights[i] * z_weights[k])
                    )
                )

                Wz_ = Wz_.at[i, j, k, :].set(
                    (z_weights[k] - old_z_weights[k])
                    * (
                        1 / 3 * (x_weights[i] * y_weights[j] + old_x_weights[i] * old_y_weights[j])
                        + 1 / 6 * (x_weights[i] * old_y_weights[j] + old_x_weights[i] * y_weights[j])
                    )
                )

    return Wx_, Wy_, Wz_



def _1d_esirkepov_weights(
        xw,
        yw,
        zw, 
        oxw, 
        oyw, 
        ozw, 
        dim=0
    ):

    if dim == 0:
        Wx = jnp.stack([xw[i] - oxw[i] for i in range(5)], axis=0)
        Wy = jnp.stack([(xw[i] + oxw[i]) / 2 for i in range(5)], axis=0)
        Wz = jnp.stack([(xw[i] + oxw[i]) / 2 for i in range(5)], axis=0)
    elif dim == 1:
        Wy = jnp.stack([yw[j] - oyw[j] for j in range(5)], axis=0)
        Wx = jnp.stack([(yw[j] + oyw[j]) / 2 for j in range(5)], axis=0)
        Wz = jnp.stack([(yw[j] + oyw[j]) / 2 for j in range(5)], axis=0)
    else:
        Wz = jnp.stack([zw[k] - ozw[k] for k in range(5)], axis=0)
        Wx = jnp.stack([(zw[k] + ozw[k]) / 2 for k in range(5)], axis=0)
        Wy = jnp.stack([(zw[k] + ozw[k]) / 2 for k in range(5)], axis=0)

    return Wx, Wy, Wz


def _2d_esirkepov_weights(
        xw,
        yw, 
        zw, 
        oxw, 
        oyw, 
        ozw, 
        null_dim=2
    ):


    def stack_plane(rows):
        return jnp.stack([jnp.stack(row, axis=0) for row in rows], axis=0)

    if null_dim == 0:
        Wx = stack_plane(
            [
                [
                    1 / 3 * (yw[j] * zw[k] + oyw[j] * ozw[k])
                    + 1 / 6 * (yw[j] * ozw[k] + oyw[j] * zw[k])
                    for k in range(5)
                ]
                for j in range(5)
            ]
        )
        Wy = stack_plane(
            [[1 / 2 * (yw[j] - oyw[j]) * (zw[k] + ozw[k]) for k in range(5)] for j in range(5)]
        )
        Wz = stack_plane(
            [[1 / 2 * (zw[k] - ozw[k]) * (yw[j] + oyw[j]) for k in range(5)] for j in range(5)]
        )
    elif null_dim == 1:
        Wx = stack_plane(
            [[1 / 2 * (xw[i] - oxw[i]) * (zw[k] + ozw[k]) for k in range(5)] for i in range(5)]
        )
        Wy = stack_plane(
            [
                [
                    1 / 3 * (xw[i] * zw[k] + oxw[i] * ozw[k])
                    + 1 / 6 * (xw[i] * ozw[k] + oxw[i] * zw[k])
                    for k in range(5)
                ]
                for i in range(5)
            ]
        )
        Wz = stack_plane(
            [[1 / 2 * (zw[k] - ozw[k]) * (xw[i] + oxw[i]) for k in range(5)] for i in range(5)]
        )
    else:
        Wx = stack_plane(
            [[1 / 2 * (xw[i] - oxw[i]) * (yw[j] + oyw[j]) for j in range(5)] for i in range(5)]
        )
        Wy = stack_plane(
            [[1 / 2 * (yw[j] - oyw[j]) * (xw[i] + oxw[i]) for j in range(5)] for i in range(5)]
        )
        Wz = stack_plane(
            [
                [
                    1 / 3 * (xw[i] * yw[j] + oxw[i] * oyw[j])
                    + 1 / 6 * (xw[i] * oyw[j] + oxw[i] * yw[j])
                    for j in range(5)
                ]
                for i in range(5)
            ]
        )

    return Wx, Wy, Wz