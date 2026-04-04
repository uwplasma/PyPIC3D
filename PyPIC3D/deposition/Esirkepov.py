import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax

from PyPIC3D.utils import digital_filter, bilinear_filter
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.boundaryconditions import fold_ghost_cells


def _roll_old_weights_to_new_frame(old_w_list, shift):
    """Roll old weights into the new-cell frame for Esirkepov deposition."""
    old_w = jnp.stack(old_w_list, axis=0)

    def roll_one_particle(w5, s):
        return jnp.roll(w5, -s, axis=0)

    rolled = jax.vmap(roll_one_particle, in_axes=(1, 0), out_axes=1)(old_w, shift)
    return [rolled[i, :] for i in range(5)]


def Esirkepov_current(particles, J, constants, world, grid=None, filter=None):
    """Esirkepov current deposition supporting 1D/2D/3D via inactive dims."""
    if grid is None:
        grid = world["grids"]["center"]

    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    dx, dy, dz, dt = world["dx"], world["dy"], world["dz"], world["dt"]
    xmin, ymin, zmin = grid[0][0], grid[1][0], grid[2][0]

    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)

    # use physical grid sizes to determine active dimensions
    # (array shape includes ghost cells, so shape != 1 is always True)
    x_active = world["Nx"] != 1
    y_active = world["Ny"] != 1
    z_active = world["Nz"] != 1

    for species in particles:
        q = species.get_charge()
        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
        shape_factor = species.get_shape()
        N_particles = species.get_number_of_particles()

        old_x = x - vx * dt
        old_y = y - vy * dt
        old_z = z - vz * dt

        x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((x - xmin) / dx).astype(int),
            lambda _: jnp.round((x - xmin) / dx).astype(int),
            operand=None,
        )
        y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((y - ymin) / dy).astype(int),
            lambda _: jnp.round((y - ymin) / dy).astype(int),
            operand=None,
        )
        z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((z - zmin) / dz).astype(int),
            lambda _: jnp.round((z - zmin) / dz).astype(int),
            operand=None,
        )

        old_x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((old_x - xmin) / dx).astype(int),
            lambda _: jnp.round((old_x - xmin) / dx).astype(int),
            operand=None,
        )
        old_y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((old_y - ymin) / dy).astype(int),
            lambda _: jnp.round((old_y - ymin) / dy).astype(int),
            operand=None,
        )
        old_z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((old_z - zmin) / dz).astype(int),
            lambda _: jnp.round((old_z - zmin) / dz).astype(int),
            operand=None,
        )

        deltax = (x - xmin) - x0 * dx
        deltay = (y - ymin) - y0 * dy
        deltaz = (z - zmin) - z0 * dz
        old_deltax = (old_x - xmin) - old_x0 * dx
        old_deltay = (old_y - ymin) - old_y0 * dy
        old_deltaz = (old_z - zmin) - old_z0 * dz

        shift_x = x0 - old_x0
        shift_y = y0 - old_y0
        shift_z = z0 - old_z0

        # Esirkepov uses a 5-point stencil (x0-2 to x0+2) which can exceed
        # the single ghost cell layer. Use mod wrapping to keep indices valid.
        # Nx here is the full array size (physical Nx + 2 ghost cells).
        x1 = jnp.mod(x0 + 1, Nx)
        y1 = jnp.mod(y0 + 1, Ny)
        z1 = jnp.mod(z0 + 1, Nz)
        x2 = jnp.mod(x0 + 2, Nx)
        y2 = jnp.mod(y0 + 2, Ny)
        z2 = jnp.mod(z0 + 2, Nz)
        x_minus1 = jnp.mod(x0 - 1, Nx)
        y_minus1 = jnp.mod(y0 - 1, Ny)
        z_minus1 = jnp.mod(z0 - 1, Nz)
        x_minus2 = jnp.mod(x0 - 2, Nx)
        y_minus2 = jnp.mod(y0 - 2, Ny)
        z_minus2 = jnp.mod(z0 - 2, Nz)

        xpts = [x_minus2, x_minus1, x0, x1, x2]
        ypts = [y_minus2, y_minus1, y0, y1, y2]
        zpts = [z_minus2, z_minus1, z0, z1, z2]

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

        oxw = _roll_old_weights_to_new_frame(oxw, shift_x)
        oyw = _roll_old_weights_to_new_frame(oyw, shift_y)
        ozw = _roll_old_weights_to_new_frame(ozw, shift_z)

        if x_active and y_active and z_active:
            Wx_, Wy_, Wz_ = get_3D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles)
        elif (x_active and y_active and (not z_active)) or (x_active and z_active and (not y_active)) or (
            y_active and z_active and (not x_active)
        ):
            null_dim = lax.cond(
                not x_active,
                lambda _: 0,
                lambda _: lax.cond(
                    not y_active,
                    lambda _: 1,
                    lambda _: 2,
                    operand=None,
                ),
                operand=None,
            )

            Wx_, Wy_, Wz_ = get_2D_esirkepov_weights(
                xw, yw, zw, oxw, oyw, ozw, N_particles, null_dim=null_dim
            )
        elif x_active and (not y_active) and (not z_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=0)
        elif y_active and (not x_active) and (not z_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=1)
        elif z_active and (not x_active) and (not y_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=2)

        dJx = jax.lax.cond(
            x_active,
            lambda _: -(q / (dy * dz)) / dt * jnp.ones(N_particles),
            lambda _: q * vx / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        dJy = jax.lax.cond(
            y_active,
            lambda _: -(q / (dx * dz)) / dt * jnp.ones(N_particles),
            lambda _: q * vy / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        dJz = jax.lax.cond(
            z_active,
            lambda _: -(q / (dx * dy)) / dt * jnp.ones(N_particles),
            lambda _: q * vz / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        Fx = dJx * Wx_
        Fy = dJy * Wy_
        Fz = dJz * Wz_

        Jx_loc = jnp.zeros_like(Fx)
        Jy_loc = jnp.zeros_like(Fy)
        Jz_loc = jnp.zeros_like(Fz)

        Jx_loc = jnp.cumsum(Fx, axis=0)
        Jy_loc = jnp.cumsum(Fy, axis=1)
        Jz_loc = jnp.cumsum(Fz, axis=2)

        if x_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jx = Jx.at[xpts[i], ypts[j], zpts[k]].add(Jx_loc[i, j, k, :], mode="drop")
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jx = Jx.at[xpts[i], ypts[j], zpts[k]].add(Fx[i, j, k, :], mode="drop")

        if y_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jy = Jy.at[xpts[i], ypts[j], zpts[k]].add(Jy_loc[i, j, k, :], mode="drop")
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jy = Jy.at[xpts[i], ypts[j], zpts[k]].add(Fy[i, j, k, :], mode="drop")

        if z_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jz = Jz.at[xpts[i], ypts[j], zpts[k]].add(Jz_loc[i, j, k, :], mode="drop")
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jz = Jz.at[xpts[i], ypts[j], zpts[k]].add(Fz[i, j, k, :], mode="drop")

    bc_x = world['boundary_conditions']['x']
    bc_y = world['boundary_conditions']['y']
    bc_z = world['boundary_conditions']['z']
    Jx = fold_ghost_cells(Jx, bc_x, bc_y, bc_z)
    Jy = fold_ghost_cells(Jy, bc_x, bc_y, bc_z)
    Jz = fold_ghost_cells(Jz, bc_x, bc_y, bc_z)
    # fold ghost cell deposits back into interior

    return (Jx, Jy, Jz)


def get_3D_esirkepov_weights(
    x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=None
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


def get_2D_esirkepov_weights(
    x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=2
):
    d_Sx = [x_weights[i] - old_x_weights[i] for i in range(len(x_weights))]
    d_Sy = [y_weights[i] - old_y_weights[i] for i in range(len(y_weights))]
    d_Sz = [z_weights[i] - old_z_weights[i] for i in range(len(z_weights))]

    Wx_ = jnp.zeros((len(x_weights), len(y_weights), len(z_weights), N_particles))
    Wy_ = jnp.zeros_like(Wx_)
    Wz_ = jnp.zeros_like(Wx_)

    def xy_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            for j in range(len(y_weights)):
                Wx_ = Wx_.at[i, j, 2, :].set(1 / 2 * d_Sx[i] * (y_weights[j] + old_y_weights[j]))
                Wy_ = Wy_.at[i, j, 2, :].set(1 / 2 * d_Sy[j] * (x_weights[i] + old_x_weights[i]))
                Wz_ = Wz_.at[i, j, 2, :].set(
                    1 / 3 * (x_weights[i] * y_weights[j] + old_x_weights[i] * old_y_weights[j])
                    + 1 / 6 * (x_weights[i] * old_y_weights[j] + old_x_weights[i] * y_weights[j])
                )
        return Wx_, Wy_, Wz_

    def xz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[i, 2, k, :].set(1 / 2 * d_Sx[i] * (z_weights[k] + old_z_weights[k]))
                Wy_ = Wy_.at[i, 2, k, :].set(
                    1 / 3 * (x_weights[i] * z_weights[k] + old_x_weights[i] * old_z_weights[k])
                    + 1 / 6 * (x_weights[i] * old_z_weights[k] + old_x_weights[i] * z_weights[k])
                )
                Wz_ = Wz_.at[i, 2, k, :].set(1 / 2 * d_Sz[k] * (x_weights[i] + old_x_weights[i]))
        return Wx_, Wy_, Wz_

    def yz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[2, j, k, :].set(
                    1 / 3 * (y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k])
                    + 1 / 6 * (y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k])
                )
                Wy_ = Wy_.at[2, j, k, :].set(1 / 2 * d_Sy[j] * (z_weights[k] + old_z_weights[k]))
                Wz_ = Wz_.at[2, j, k, :].set(1 / 2 * d_Sz[k] * (y_weights[j] + old_y_weights[j]))
        return Wx_, Wy_, Wz_

    Wx_, Wy_, Wz_ = lax.cond(
        null_dim == 0,
        lambda _: yz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
        lambda _: lax.cond(
            null_dim == 1,
            lambda _: xz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            lambda _: xy_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            operand=None,
        ),
        operand=None,
    )

    return Wx_, Wy_, Wz_


def get_1D_esirkepov_weights(
    x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, dim=0
):
    Wx_ = jnp.zeros((len(x_weights), len(y_weights), len(z_weights), N_particles))
    Wy_ = jnp.zeros_like(Wx_)
    Wz_ = jnp.zeros_like(Wx_)

    def x_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            Wx_ = Wx_.at[i, 2, 2, :].set((x_weights[i] - old_x_weights[i]))
            Wy_ = Wy_.at[i, 2, 2, :].set((x_weights[i] + old_x_weights[i]) / 2)
            Wz_ = Wz_.at[i, 2, 2, :].set((x_weights[i] + old_x_weights[i]) / 2)
        return Wx_, Wy_, Wz_

    def y_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for j in range(len(y_weights)):
            Wy_ = Wy_.at[2, j, 2, :].set((y_weights[j] - old_y_weights[j]))
            Wx_ = Wx_.at[2, j, 2, :].set((y_weights[j] + old_y_weights[j]) / 2)
            Wz_ = Wz_.at[2, j, 2, :].set((y_weights[j] + old_y_weights[j]) / 2)
        return Wx_, Wy_, Wz_

    def z_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for k in range(len(z_weights)):
            Wz_ = Wz_.at[2, 2, k, :].set((z_weights[k] - old_z_weights[k]))
            Wx_ = Wx_.at[2, 2, k, :].set((z_weights[k] + old_z_weights[k]) / 2)
            Wy_ = Wy_.at[2, 2, k, :].set((z_weights[k] + old_z_weights[k]) / 2)
        return Wx_, Wy_, Wz_

    Wx_, Wy_, Wz_ = lax.cond(
        dim == 0,
        lambda _: x_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
        lambda _: lax.cond(
            dim == 1,
            lambda _: y_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            lambda _: z_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            operand=None,
        ),
        operand=None,
    )

    return Wx_, Wy_, Wz_
