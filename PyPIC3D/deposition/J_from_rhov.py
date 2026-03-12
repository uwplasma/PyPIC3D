import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax

from PyPIC3D.utils import digital_filter, wrap_around, bilinear_filter
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights


@partial(jit, static_argnames=("filter",))
def J_from_rhov(particles, J, constants, world, grid=None, filter="bilinear"):
    """Compute current density (Jx,Jy,Jz) by depositing particle velocities."""

    if grid is None:
        grid = world["grids"]["center"]

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    Nx = world["Nx"]
    Ny = world["Ny"]
    Nz = world["Nz"]

    Jx, Jy, Jz = J
    x_active = Jx.shape[0] != 1
    y_active = Jx.shape[1] != 1
    z_active = Jx.shape[2] != 1

    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)

    for species in particles:
        shape_factor = species.get_shape()
        charge = species.get_charge()
        dq = charge / (dx * dy * dz)

        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()

        x = x - vx * world["dt"] / 2
        y = y - vy * world["dt"] / 2
        z = z - vz * world["dt"] / 2

        x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((x - grid[0][0]) / dx).astype(int),
            lambda _: jnp.round((x - grid[0][0]) / dx).astype(int),
            operand=None,
        )
        y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((y - grid[1][0]) / dy).astype(int),
            lambda _: jnp.round((y - grid[1][0]) / dy).astype(int),
            operand=None,
        )
        z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor((z - grid[2][0]) / dz).astype(int),
            lambda _: jnp.round((z - grid[2][0]) / dz).astype(int),
            operand=None,
        )

        deltax_node = (x - grid[0][0]) - (x0 * dx)
        deltay_node = (y - grid[1][0]) - (y0 * dy)
        deltaz_node = (z - grid[2][0]) - (z0 * dz)

        deltax_face = (x - grid[0][0]) - (x0 + 0.5) * dx
        deltay_face = (y - grid[1][0]) - (y0 + 0.5) * dy
        deltaz_face = (z - grid[2][0]) - (z0 + 0.5) * dz

        x0 = wrap_around(x0, Nx)
        y0 = wrap_around(y0, Ny)
        z0 = wrap_around(z0, Nz)
        x1 = wrap_around(x0 + 1, Nx)
        y1 = wrap_around(y0 + 1, Ny)
        z1 = wrap_around(z0 + 1, Nz)
        x_minus1 = x0 - 1
        y_minus1 = y0 - 1
        z_minus1 = z0 - 1

        xpts = [x_minus1, x0, x1]
        ypts = [y_minus1, y0, y1]
        zpts = [z_minus1, z0, z1]

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

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)

        x_weights_face = jnp.asarray(x_weights_face)
        y_weights_face = jnp.asarray(y_weights_face)
        z_weights_face = jnp.asarray(z_weights_face)

        x_weights_node = jnp.asarray(x_weights_node)
        y_weights_node = jnp.asarray(y_weights_node)
        z_weights_node = jnp.asarray(z_weights_node)

        if x_active:
            xpts_eff = xpts
            x_weights_node_eff = x_weights_node
            x_weights_face_eff = x_weights_face
        else:
            xpts_eff = jnp.zeros((1, xpts.shape[1]), dtype=xpts.dtype)
            x_weights_node_eff = jnp.sum(x_weights_node, axis=0, keepdims=True)
            x_weights_face_eff = jnp.sum(x_weights_face, axis=0, keepdims=True)

        if y_active:
            ypts_eff = ypts
            y_weights_node_eff = y_weights_node
            y_weights_face_eff = y_weights_face
        else:
            ypts_eff = jnp.zeros((1, ypts.shape[1]), dtype=ypts.dtype)
            y_weights_node_eff = jnp.sum(y_weights_node, axis=0, keepdims=True)
            y_weights_face_eff = jnp.sum(y_weights_face, axis=0, keepdims=True)

        if z_active:
            zpts_eff = zpts
            z_weights_node_eff = z_weights_node
            z_weights_face_eff = z_weights_face
        else:
            zpts_eff = jnp.zeros((1, zpts.shape[1]), dtype=zpts.dtype)
            z_weights_node_eff = jnp.sum(z_weights_node, axis=0, keepdims=True)
            z_weights_face_eff = jnp.sum(z_weights_face, axis=0, keepdims=True)

        ii, jj, kk = jnp.meshgrid(
            jnp.arange(xpts_eff.shape[0]),
            jnp.arange(ypts_eff.shape[0]),
            jnp.arange(zpts_eff.shape[0]),
            indexing="ij",
        )
        combos = jnp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)

        def idx_and_dJ_values(idx):
            i, j, k = idx
            ix = xpts_eff[i, ...]
            iy = ypts_eff[j, ...]
            iz = zpts_eff[k, ...]
            valx = (
                (dq * vx)
                * x_weights_face_eff[i, ...]
                * y_weights_node_eff[j, ...]
                * z_weights_node_eff[k, ...]
            )
            valy = (
                (dq * vy)
                * x_weights_node_eff[i, ...]
                * y_weights_face_eff[j, ...]
                * z_weights_node_eff[k, ...]
            )
            valz = (
                (dq * vz)
                * x_weights_node_eff[i, ...]
                * y_weights_node_eff[j, ...]
                * z_weights_face_eff[k, ...]
            )
            return ix, iy, iz, valx, valy, valz

        ix, iy, iz, valx, valy, valz = jax.vmap(idx_and_dJ_values)(combos)

        Jx = Jx.at[(ix, iy, iz)].add(valx, mode="drop")
        Jy = Jy.at[(ix, iy, iz)].add(valy, mode="drop")
        Jz = Jz.at[(ix, iy, iz)].add(valz, mode="drop")

    def filter_func(J_, filter):
        J_ = jax.lax.cond(
            filter == "bilinear",
            lambda J_: bilinear_filter(J_),
            lambda J_: J_,
            operand=J_,
        )
        return J_

    Jx = filter_func(Jx, filter)
    Jy = filter_func(Jy, filter)
    Jz = filter_func(Jz, filter)

    return (Jx, Jy, Jz)