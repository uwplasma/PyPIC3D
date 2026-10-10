import jax
import jax.numpy as jnp
from functools import partial

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    compute_particle_anchor,
    particle_axis_offset,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.boundary_conditions.ghost_cells import (
    BC_TYPE_PARTICLE,
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


@partial(jax.jit, static_argnames="static_parameters")
def Esirkepov_current(
    particles_old: TiledParticles,
    particles_new: TiledParticles,
    species_config: SpeciesConfig,
    J,
    static_parameters,
    dynamic_parameters,
    *,
    coordinate_velocity=None,
):
    """Deposit charge-conserving coordinate current from explicit endpoints.

    Both states must have matching slots, activity, and tile ownership, before
    wrapping, particle boundary handling, or retiling. The new state's active
    mask is used. Species position masks suppress displacement and current in
    frozen directions.

    The result is Cartesian current for ordinary Yee and densitized current
    ``sqrt(gamma) J^i`` for static-metric Maxwell; no metric conversion is needed.
    Particle momentum is never read. On unresolved grid axes, current uses the
    optional coordinate velocity (an array shaped like ``particles_new.x``), or
    the masked endpoint displacement divided by dt when omitted. Flat pushers
    supply their coordinate velocity to avoid subtraction error; GR pushers
    use their actual displacement, since their stored momentum is covariant.

    No current-only filter is applied: it would break discrete continuity with
    the raw deposited charge.
    """
    update_axes = species_config.update_x.reshape(
        (1, 1, 1, species_config.update_x.shape[0], 1, 3)
    )
    endpoints = jnp.where(update_axes, particles_new.x, particles_old.x)
    if coordinate_velocity is None:
        coordinate_velocity = (endpoints - particles_old.x) / dynamic_parameters.dt
    coordinate_velocity = jnp.where(update_axes, coordinate_velocity, 0.0)
    return _deposit_esirkepov_tiles(
        particles_old.x, endpoints, coordinate_velocity, particles_new.active,
        species_config, J, static_parameters, dynamic_parameters,
    )


def _deposit_esirkepov_tiles(
    old_positions, endpoints, coordinate_velocity, active, species,
    current_template, static, dynamic,
):
    """Map the metric-free endpoint kernel over owned tiles and fold halos."""
    guard = int(static.guard_cells)
    tile_shape = tuple(int(width) for width in static.tile_shape)
    tile_counts = current_template[0].shape[:3]
    local_shape = tuple(width + 2 * guard for width in tile_shape)
    active_axes = tuple(count * width > 1 for count, width in zip(tile_counts, tile_shape))
    templates = tuple(jnp.zeros_like(component[0, 0, 0]) for component in current_template)
    grids = dynamic.grids.tiled_center_grid
    weighted_charge = species.charge * species.weight

    def deposit_one_tile(old_tile, endpoint_tile, velocity_tile, active_tile, tx, ty, tz):
        old_position = tuple(old_tile[..., axis].reshape(-1) for axis in range(3))
        new_position = tuple(endpoint_tile[..., axis].reshape(-1) for axis in range(3))
        velocity = tuple(velocity_tile[..., axis].reshape(-1) for axis in range(3))
        charge = jnp.broadcast_to(weighted_charge[:, None], active_tile.shape).reshape(-1)
        update_axes = tuple(
            jnp.broadcast_to(species.update_x[:, axis, None], active_tile.shape).reshape(-1)
            for axis in range(3)
        )
        return esirkepov_tile_currents(
            new_position, old_position, velocity, charge,
            active_tile.reshape(-1).astype(old_position[0].dtype), update_axes,
            tuple(grid[tx, ty, tz] for grid in grids), active_axes, local_shape,
            templates, static.shape_factor, dynamic.dx, dynamic.dy, dynamic.dz, dynamic.dt,
        )

    tile_indices = jnp.meshgrid(*(jnp.arange(count) for count in tile_counts), indexing="ij")
    deposit_tiles = deposit_one_tile
    for _ in range(3):
        deposit_tiles = jax.vmap(deposit_tiles, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=0)
    current = deposit_tiles(old_positions, endpoints, coordinate_velocity, active, *tile_indices)
    current = fold_tiled_vector_ghost_cells(
        current, static, num_guard_cells=guard, bc_type=BC_TYPE_PARTICLE,
    )
    return update_tiled_vector_ghost_cells(
        current, static, num_guard_cells=guard, bc_type=BC_TYPE_PARTICLE,
    )


def esirkepov_tile_currents(
    new_position,
    old_position,
    out_of_plane_velocity,
    charge,
    active,
    update_axes,
    tile_axis_grids,
    axis_active,
    local_shape,
    templates,
    shape_factor,
    dx,
    dy,
    dz,
    dt,
):
    """
    Esirkepov charge-conserving current for the particles owned by one tile.

    This is the metric-free core of the scheme.  It consumes the old and new
    particle positions directly and never looks at a metric, which is what lets
    the same decomposition serve both flat space and a fixed 3+1 metric: the
    conformal charge density ``sqrt(gamma) rho = q S / (dx dy dz)`` carries no
    metric, so the conformal continuity equation

        d_t( sqrt(gamma) rho ) + d_i( sqrt(gamma) J^i ) = 0

    is the flat Esirkepov identity verbatim, with the conformal current
    ``sqrt(gamma) J^i`` in the role of the Cartesian current. Densitized Maxwell
    consumes this result directly, without division by ``sqrt(gamma)``.

    ``out_of_plane_velocity`` is only read on axes that are inactive (a
    quasi-dimensional run), where the component takes no part in the continuity
    equation and is deposited directly.
    """

    x, y, z = new_position
    old_x, old_y, old_z = old_position
    vx, vy, vz = out_of_plane_velocity
    q = charge
    update_x1, update_x2, update_x3 = update_axes
    x_grid, y_grid, z_grid = tile_axis_grids
    x_active, y_active, z_active = axis_active
    local_Nx, local_Ny, local_Nz = local_shape
    Jx_template, Jy_template, Jz_template = templates
    N_particles = old_x.shape[0]
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

    dJx = jnp.where(update_x1, dJx, 0.0)
    dJy = jnp.where(update_x2, dJy, 0.0)
    dJz = jnp.where(update_x3, dJz, 0.0)
    # suppress current in every direction where that species is fixed

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
