import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax
# import external libraries

from PyPIC3D.utils import digital_filter, wrap_around, bilinear_filter
from PyPIC3D.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.indexed_particles import _advance_index_and_frac


def _weights_order1(r):
    w0 = 1.0 - r
    w1 = r
    return jnp.stack((w0, w1), axis=0)


def _weights_order2(r):
    w0 = 0.5 * (0.5 - r) ** 2
    w1 = 0.75 - r**2
    w2 = 0.5 * (0.5 + r) ** 2
    return jnp.stack((w0, w1, w2), axis=0)


def _weights_face(r, shape_factor):
    if shape_factor == 1:
        return _weights_order1(r - 0.5)
    return _weights_order2(r - 0.5)


def _deposit_1d(J_stack, dq, vx, vy, vz, x, grid_x0, dx, dt, shape_factor, Nx):
    if shape_factor == 1:
        x0 = jnp.floor((x - grid_x0) / dx).astype(jnp.int32)
        deltax_node = (x - grid_x0) - x0 * dx
        deltax_face = (x - grid_x0) - (x0 + 0.5) * dx

        r_node = deltax_node / dx
        r_face = deltax_face / dx

        w_node = _weights_order1(r_node)  # (2,Np)
        w_face = _weights_order1(r_face)  # (2,Np)

        ix = jnp.stack((x0, x0 + 1), axis=0)
        ix = wrap_around(ix, Nx)
    else:
        x0 = jnp.round((x - grid_x0) / dx).astype(jnp.int32)
        deltax_node = (x - grid_x0) - x0 * dx
        deltax_face = (x - grid_x0) - (x0 + 0.5) * dx

        r_node = deltax_node / dx
        r_face = deltax_face / dx

        w_node = _weights_order2(r_node)  # (3,Np)
        w_face = _weights_order2(r_face)  # (3,Np)

        ix = jnp.stack((x0 - 1, x0, x0 + 1), axis=0)
        ix = wrap_around(ix, Nx)

    # Jx uses face weights; Jy/Jz use node weights.
    val = jnp.stack(
        (
            (dq * vx)[None, :] * w_face,
            (dq * vy)[None, :] * w_node,
            (dq * vz)[None, :] * w_node,
        ),
        axis=-1,
    )  # (S,Np,3)

    comp = jnp.arange(3, dtype=ix.dtype)[None, None, :]  # (1,1,3)
    idx = ix[:, :, None] + comp * jnp.asarray(Nx, dtype=ix.dtype)  # (S,Np,3)

    out = jnp.bincount(
        idx.reshape(-1),
        weights=val.reshape(-1),
        length=Nx * 3,
    ).reshape(3, Nx)

    Jx, Jy, Jz = out[0], out[1], out[2]
    return jnp.stack((Jx, Jy, Jz), axis=-1).reshape((Nx, 1, 1, 3))


def _deposit_2d(J_stack, dq, vx, vy, vz, x, y, xmin, ymin, dx, dy, dt, shape_factor, Nx, Ny):
    if shape_factor == 1:
        x0 = jnp.floor((x - xmin) / dx).astype(jnp.int32)
        y0 = jnp.floor((y - ymin) / dy).astype(jnp.int32)
        deltax_node = (x - xmin) - x0 * dx
        deltay_node = (y - ymin) - y0 * dy
        deltax_face = (x - xmin) - (x0 + 0.5) * dx
        deltay_face = (y - ymin) - (y0 + 0.5) * dy

        wx_node = _weights_order1(deltax_node / dx)  # (2,Np)
        wy_node = _weights_order1(deltay_node / dy)  # (2,Np)
        wx_face = _weights_order1(deltax_face / dx)  # (2,Np)
        wy_face = _weights_order1(deltay_face / dy)  # (2,Np)

        ix = jnp.stack((x0, x0 + 1), axis=0)
        iy = jnp.stack((y0, y0 + 1), axis=0)
        ix = wrap_around(ix, Nx)
        iy = wrap_around(iy, Ny)
    else:
        x0 = jnp.round((x - xmin) / dx).astype(jnp.int32)
        y0 = jnp.round((y - ymin) / dy).astype(jnp.int32)
        deltax_node = (x - xmin) - x0 * dx
        deltay_node = (y - ymin) - y0 * dy
        deltax_face = (x - xmin) - (x0 + 0.5) * dx
        deltay_face = (y - ymin) - (y0 + 0.5) * dy

        wx_node = _weights_order2(deltax_node / dx)  # (3,Np)
        wy_node = _weights_order2(deltay_node / dy)  # (3,Np)
        wx_face = _weights_order2(deltax_face / dx)  # (3,Np)
        wy_face = _weights_order2(deltay_face / dy)  # (3,Np)

        ix = jnp.stack((x0 - 1, x0, x0 + 1), axis=0)
        iy = jnp.stack((y0 - 1, y0, y0 + 1), axis=0)
        ix = wrap_around(ix, Nx)
        iy = wrap_around(iy, Ny)

    idx = ix[:, None, :] + Nx * iy[None, :, :]  # (Sx,Sy,Np)
    idx_flat = idx.reshape(-1)

    # weights for each component
    wjx = wx_face[:, None, :] * wy_node[None, :, :]
    wjy = wx_node[:, None, :] * wy_face[None, :, :]
    wjz = wx_node[:, None, :] * wy_node[None, :, :]

    valx = (dq * vx)[None, None, :] * wjx
    valy = (dq * vy)[None, None, :] * wjy
    valz = (dq * vz)[None, None, :] * wjz

    vals = jnp.stack((valx, valy, valz), axis=-1).reshape(-1, 3)
    J_flat = jax.ops.segment_sum(vals, idx_flat, num_segments=Nx * Ny)  # (Nx*Ny,3)
    J2 = J_flat.reshape((Nx, Ny, 3))[:, :, None, :]
    return J2


def _deposit_1d_indexed(dq, vx, vy, vz, i0, r, shape_factor, Nx):
    if shape_factor == 1:
        ix = jnp.stack((i0, wrap_around(i0 + 1, Nx)), axis=0)
    else:
        ix = jnp.stack((wrap_around(i0 - 1, Nx), i0, wrap_around(i0 + 1, Nx)), axis=0)

    w_node = _weights_order1(r) if shape_factor == 1 else _weights_order2(r)
    w_face = _weights_face(r, shape_factor)

    val = jnp.stack(
        (
            (dq * vx)[None, :] * w_face,
            (dq * vy)[None, :] * w_node,
            (dq * vz)[None, :] * w_node,
        ),
        axis=-1,
    )

    comp = jnp.arange(3, dtype=ix.dtype)[None, None, :]
    idx = ix[:, :, None] + comp * jnp.asarray(Nx, dtype=ix.dtype)

    out = jnp.bincount(idx.reshape(-1), weights=val.reshape(-1), length=Nx * 3).reshape(3, Nx)
    Jx, Jy, Jz = out[0], out[1], out[2]
    return jnp.stack((Jx, Jy, Jz), axis=-1).reshape((Nx, 1, 1, 3))


def _deposit_2d_indexed(dq, vx, vy, vz, i0, j0, rx, ry, shape_factor, Nx, Ny):
    if shape_factor == 1:
        ix = jnp.stack((i0, wrap_around(i0 + 1, Nx)), axis=0)
        iy = jnp.stack((j0, wrap_around(j0 + 1, Ny)), axis=0)
    else:
        ix = jnp.stack((wrap_around(i0 - 1, Nx), i0, wrap_around(i0 + 1, Nx)), axis=0)
        iy = jnp.stack((wrap_around(j0 - 1, Ny), j0, wrap_around(j0 + 1, Ny)), axis=0)

    wx_node = _weights_order1(rx) if shape_factor == 1 else _weights_order2(rx)
    wy_node = _weights_order1(ry) if shape_factor == 1 else _weights_order2(ry)
    wx_face = _weights_face(rx, shape_factor)
    wy_face = _weights_face(ry, shape_factor)

    idx = ix[:, None, :] + Nx * iy[None, :, :]
    idx_flat = idx.reshape(-1)

    wjx = wx_face[:, None, :] * wy_node[None, :, :]
    wjy = wx_node[:, None, :] * wy_face[None, :, :]
    wjz = wx_node[:, None, :] * wy_node[None, :, :]

    valx = (dq * vx)[None, None, :] * wjx
    valy = (dq * vy)[None, None, :] * wjy
    valz = (dq * vz)[None, None, :] * wjz

    vals = jnp.stack((valx, valy, valz), axis=-1).reshape(-1, 3)
    J_flat = jax.ops.segment_sum(vals, idx_flat, num_segments=Nx * Ny)
    return J_flat.reshape((Nx, Ny, 3))[:, :, None, :]


def _deposit_3d_indexed(dq, vx, vy, vz, i0, j0, k0, rx, ry, rz, shape_factor, Nx, Ny, Nz):
    if shape_factor == 1:
        xpts = jnp.stack((i0, wrap_around(i0 + 1, Nx)), axis=0)
        ypts = jnp.stack((j0, wrap_around(j0 + 1, Ny)), axis=0)
        zpts = jnp.stack((k0, wrap_around(k0 + 1, Nz)), axis=0)
        x_weights_node = _weights_order1(rx)
        y_weights_node = _weights_order1(ry)
        z_weights_node = _weights_order1(rz)
        x_weights_face = _weights_face(rx, shape_factor)
        y_weights_face = _weights_face(ry, shape_factor)
        z_weights_face = _weights_face(rz, shape_factor)
    else:
        xpts = jnp.stack((wrap_around(i0 - 1, Nx), i0, wrap_around(i0 + 1, Nx)), axis=0)
        ypts = jnp.stack((wrap_around(j0 - 1, Ny), j0, wrap_around(j0 + 1, Ny)), axis=0)
        zpts = jnp.stack((wrap_around(k0 - 1, Nz), k0, wrap_around(k0 + 1, Nz)), axis=0)
        x_weights_node = _weights_order2(rx)
        y_weights_node = _weights_order2(ry)
        z_weights_node = _weights_order2(rz)
        x_weights_face = _weights_face(rx, shape_factor)
        y_weights_face = _weights_face(ry, shape_factor)
        z_weights_face = _weights_face(rz, shape_factor)

    if shape_factor == 1:
        xpts = xpts
        ypts = ypts
        zpts = zpts

    ii, jj, kk = jnp.meshgrid(
        jnp.arange(xpts.shape[0]),
        jnp.arange(ypts.shape[0]),
        jnp.arange(zpts.shape[0]),
        indexing="ij",
    )
    combos = jnp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)

    def idx_and_dJ_values(idx):
        i, j, k = idx
        ix = xpts[i, ...]
        iy = ypts[j, ...]
        iz = zpts[k, ...]
        valx = (dq * vx) * x_weights_face[i, ...] * y_weights_node[j, ...] * z_weights_node[k, ...]
        valy = (dq * vy) * x_weights_node[i, ...] * y_weights_face[j, ...] * z_weights_node[k, ...]
        valz = (dq * vz) * x_weights_node[i, ...] * y_weights_node[j, ...] * z_weights_face[k, ...]
        return ix, iy, iz, jnp.stack((valx, valy, valz), axis=-1)

    ix, iy, iz, dJ = jax.vmap(idx_and_dJ_values)(combos)

    ix_flat = ix.reshape(-1)
    iy_flat = iy.reshape(-1)
    iz_flat = iz.reshape(-1)
    dJ_flat = dJ.reshape(-1, 3)

    idx_flat = ix_flat + Nx * (iy_flat + Ny * iz_flat)
    J_flat = jax.ops.segment_sum(dJ_flat, idx_flat, num_segments=Nx * Ny * Nz)
    return J_flat.reshape((Nx, Ny, Nz, 3))


@partial(jit, static_argnames=("filter", "shape_factor"))
def J_from_rhov(particles, J, constants, world, grid=None, filter='bilinear', shape_factor=2):
    """
    Compute the current density from the charge density and particle velocities.

    Args:
        particles (list): List of particle species, each with methods to get charge, subcell position, resolution, and index.
        rho (ndarray): Charge density array.
        J (tuple): Current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
        constants (dict): Dictionary containing physical constants.

    Returns:
        tuple: Updated current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
    """

    if grid is None:
        grid = world['grids']['center']

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    # get the world parameters
    x_active = Jx.shape[0] != 1
    y_active = Jx.shape[1] != 1
    z_active = Jx.shape[2] != 1
    # infer effective dimensionality from the current-grid shape

    J_stack = jnp.zeros((Nx, Ny, Nz, 3), dtype=Jx.dtype)
    # keep J together so deposition and filtering can be fused across components

    for species in particles:
        charge = species.get_charge()
        dq = charge / (dx * dy * dz)
        # calculate the charge density contribution per particle
        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
        # get the particles positions and velocities

        dt = world["dt"]
        x = x - vx * dt / 2
        # step back to half time step positions for proper time staggering

        if Ny == 1 and Nz == 1:
            J_stack = J_stack + _deposit_1d(J_stack, dq, vx, vy, vz, x, grid[0][0], dx, world["dt"], shape_factor, Nx)
            continue
        if Nz == 1:
            y = y - vy * dt / 2
            J_stack = J_stack + _deposit_2d(
                J_stack,
                dq,
                vx,
                vy,
                vz,
                x,
                y,
                grid[0][0],
                grid[1][0],
                dx,
                dy,
                world["dt"],
                shape_factor,
                Nx,
                Ny,
            )
            continue

        y = y - vy * dt / 2
        z = z - vz * dt / 2

        if shape_factor == 1:
            x0 = jnp.floor((x - grid[0][0]) / dx).astype(jnp.int32)
            y0 = jnp.floor((y - grid[1][0]) / dy).astype(jnp.int32)
            z0 = jnp.floor((z - grid[2][0]) / dz).astype(jnp.int32)
        else:
            x0 = jnp.round((x - grid[0][0]) / dx).astype(jnp.int32)
            y0 = jnp.round((y - grid[1][0]) / dy).astype(jnp.int32)
            z0 = jnp.round((z - grid[2][0]) / dz).astype(jnp.int32)
        # calculate the nearest grid point based on shape factor

        deltax_node = (x - grid[0][0]) - (x0 * dx)
        deltay_node = (y - grid[1][0]) - (y0 * dy)
        deltaz_node = (z - grid[2][0]) - (z0 * dz)
        # Calculate the difference between the particle position and the nearest grid point

        deltax_face = (x - grid[0][0]) - (x0 + 0.5) * dx
        deltay_face = (y - grid[1][0]) - (y0 + 0.5) * dy
        deltaz_face = (z - grid[2][0]) - (z0 + 0.5) * dz
        # Calculate the difference between the particle position and the nearest staggered cell face

        x0 = wrap_around(x0, Nx)
        y0 = wrap_around(y0, Ny)
        z0 = wrap_around(z0, Nz)
        # wrap around the grid points for periodic boundary conditions
        x1 = wrap_around(x0 + 1, Nx)
        y1 = wrap_around(y0 + 1, Ny)
        z1 = wrap_around(z0 + 1, Nz)
        # calculate the right grid point
        x_minus1 = x0 - 1
        y_minus1 = y0 - 1
        z_minus1 = z0 - 1
        # calculate the left grid point
        xpts = [x_minus1, x0, x1]
        ypts = [y_minus1, y0, y1]
        zpts = [z_minus1, z0, z1]
        # place all the points in a list

        if shape_factor == 1:
            x_weights_node, y_weights_node, z_weights_node = get_first_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz)
            x_weights_face, y_weights_face, z_weights_face = get_first_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz)
        else:
            x_weights_node, y_weights_node, z_weights_node = get_second_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz)
            x_weights_face, y_weights_face, z_weights_face = get_second_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz)
        # get the weights for node and face positions

        xpts = jnp.asarray(xpts)  # (Sx, Np)
        ypts = jnp.asarray(ypts)  # (Sy, Np)
        zpts = jnp.asarray(zpts)  # (Sz, Np)

        x_weights_face = jnp.asarray(x_weights_face)  # (Sx, Np)
        y_weights_face = jnp.asarray(y_weights_face)  # (Sy, Np)
        z_weights_face = jnp.asarray(z_weights_face)  # (Sz, Np)

        x_weights_node = jnp.asarray(x_weights_node)  # (Sx, Np)
        y_weights_node = jnp.asarray(y_weights_node)  # (Sy, Np)
        z_weights_node = jnp.asarray(z_weights_node)  # (Sz, Np)

        if shape_factor == 1:
            # drop the redundant (-1) stencil point for first-order (its weights are identically 0)
            xpts = xpts[1:, ...]
            ypts = ypts[1:, ...]
            zpts = zpts[1:, ...]
            x_weights_face = x_weights_face[1:, ...]
            y_weights_face = y_weights_face[1:, ...]
            z_weights_face = z_weights_face[1:, ...]
            x_weights_node = x_weights_node[1:, ...]
            y_weights_node = y_weights_node[1:, ...]
            z_weights_node = z_weights_node[1:, ...]

        # Keep full shape-factor computation but collapse inactive axes to an
        # effective stencil of size 1 to avoid redundant deposition work.
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
        combos = jnp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)  # (Sx*Sy*Sz, 3)

        def idx_and_dJ_values(idx):
            i, j, k = idx
            # unpack the stencil indices
            ix = xpts_eff[i, ...]
            iy = ypts_eff[j, ...]
            iz = zpts_eff[k, ...]
            # get the grid indices for this stencil point
            valx = (dq * vx) * x_weights_face_eff[i, ...] * y_weights_node_eff[j, ...] * z_weights_node_eff[k, ...]
            valy = (dq * vy) * x_weights_node_eff[i, ...] * y_weights_face_eff[j, ...] * z_weights_node_eff[k, ...]
            valz = (dq * vz) * x_weights_node_eff[i, ...] * y_weights_node_eff[j, ...] * z_weights_face_eff[k, ...]
            # calculate the current contributions for this stencil point
            return ix, iy, iz, jnp.stack((valx, valy, valz), axis=-1)
        
        ix, iy, iz, dJ = jax.vmap(idx_and_dJ_values)(combos)  # (M,Np), (M,Np), (M,Np), (M,Np,3)
        # vectorized computation of indices and current contributions

        ix_flat = ix.reshape(-1)
        iy_flat = iy.reshape(-1)
        iz_flat = iz.reshape(-1)
        dJ_flat = dJ.reshape(-1, 3)

        in_bounds = (
            (ix_flat >= 0)
            & (ix_flat < Nx)
            & (iy_flat >= 0)
            & (iy_flat < Ny)
            & (iz_flat >= 0)
            & (iz_flat < Nz)
        )

        ix_flat = jnp.clip(ix_flat, 0, Nx - 1)
        iy_flat = jnp.clip(iy_flat, 0, Ny - 1)
        iz_flat = jnp.clip(iz_flat, 0, Nz - 1)

        idx_flat = ix_flat + Nx * (iy_flat + Ny * iz_flat)
        dJ_flat = jnp.where(in_bounds[:, None], dJ_flat, 0)

        J_flat = jax.ops.segment_sum(dJ_flat, idx_flat, num_segments=Nx * Ny * Nz)
        J_stack = J_stack + J_flat.reshape((Nx, Ny, Nz, 3))
        # segment_sum avoids large scatter updates on CPU

    if filter == "bilinear":
        J_stack = bilinear_filter(J_stack)
    # (optional) digital filter disabled by default

    return (J_stack[..., 0], J_stack[..., 1], J_stack[..., 2])


@partial(jit, static_argnames=("filter",))
def J_from_rhov_indexed(particles, J, constants, world, grid=None, filter="bilinear"):
    if grid is None:
        grid = world["grids"]["center"]

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape

    J_stack = jnp.zeros((Nx, Ny, Nz, 3), dtype=Jx.dtype)

    dt = world["dt"]
    for species in particles:
        charge = species.get_charge()
        dq = charge / (dx * dy * dz)
        vx, vy, vz = species.get_velocity()
        i0, j0, k0, rx, ry, rz = species.get_indexed_position()
        shape_factor = species.get_shape()

        i0b, rxb = _advance_index_and_frac(i0, rx, -vx * dt / (2 * dx), Nx, shape_factor)

        if Ny == 1 and Nz == 1:
            J_stack = J_stack + _deposit_1d_indexed(dq, vx, vy, vz, i0b, rxb, shape_factor, Nx)
            continue

        j0b, ryb = _advance_index_and_frac(j0, ry, -vy * dt / (2 * dy), Ny, shape_factor)

        if Nz == 1:
            J_stack = J_stack + _deposit_2d_indexed(dq, vx, vy, vz, i0b, j0b, rxb, ryb, shape_factor, Nx, Ny)
            continue

        k0b, rzb = _advance_index_and_frac(k0, rz, -vz * dt / (2 * dz), Nz, shape_factor)
        J_stack = J_stack + _deposit_3d_indexed(dq, vx, vy, vz, i0b, j0b, k0b, rxb, ryb, rzb, shape_factor, Nx, Ny, Nz)

    if filter == "bilinear":
        J_stack = bilinear_filter(J_stack)

    return (J_stack[..., 0], J_stack[..., 1], J_stack[..., 2])

def _roll_old_weights_to_new_frame(old_w_list, shift):
    """
    old_w_list: list of 5 arrays, each (Np,)
    shift: (Np,) integer = old_i0 - new_i0 (expected in {-1,0,1} for Esirkepov)
    Returns a list of 5 arrays rolled per particle so old weights align with new-cell frame.
    """
    old_w = jnp.stack(old_w_list, axis=0)  # (5, Np)

    def roll_one_particle(w5, s):
        return jnp.roll(w5, -s, axis=0)

    rolled = jax.vmap(roll_one_particle, in_axes=(1, 0), out_axes=1)(old_w, shift)  # (5,Np)
    return [rolled[i, :] for i in range(5)]


def Esirkepov_current(particles, J, constants, world, grid=None, filter=None):
    """
    Local per-particle Esirkepov deposition that works for 1D/2D/3D by setting inactive dims to size 1.
    J is a tuple (Jx,Jy,Jz) arrays shaped (Nx,Ny,Nz).
    """
    if grid is None:
        grid = world['grids']['center']

    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    dx, dy, dz, dt = world["dx"], world["dy"], world["dz"], world["dt"]
    xmin, ymin, zmin = grid[0][0], grid[1][0], grid[2][0]
 
    # zero current arrays
    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)

    x_active = (Nx != 1)
    y_active = (Ny != 1)
    z_active = (Nz != 1)
    # determine which axis are null

    for species in particles:
        q = species.get_charge()
        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
        shape_factor = species.get_shape()
        N_particles = species.get_number_of_particles()
        # get the particle properties

        old_x = x - vx * dt
        old_y = y - vy * dt
        old_z = z - vz * dt
        # calculate old positions from new positions and velocities
        
        x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (x - xmin) / dx).astype(int),
            lambda _: jnp.round( (x - xmin) / dx).astype(int),
            operand=None
        )
        y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (y - ymin) / dy).astype(int),
            lambda _: jnp.round( (y - ymin) / dy).astype(int),
            operand=None
        )
        z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (z - zmin) / dz).astype(int),
            lambda _: jnp.round( (z - zmin) / dz).astype(int),
            operand=None
        ) # calculate the nearest grid point based on shape factor for new positions

        old_x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_x - xmin) / dx).astype(int),
            lambda _: jnp.round( (old_x - xmin) / dx).astype(int),
            operand=None
        )
        old_y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_y - ymin) / dy).astype(int),
            lambda _: jnp.round( (old_y - ymin) / dy).astype(int),
            operand=None
        )
        old_z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_z - zmin) / dz).astype(int),
            lambda _: jnp.round( (old_z - zmin) / dz).astype(int),
            operand=None
        ) # calculate the nearest grid point based on shape factor for old positions

        deltax = (x - xmin) - x0 * dx
        deltay = (y - ymin) - y0 * dy
        deltaz = (z - zmin) - z0 * dz
        # get the difference between the particle position and the nearest grid point
        old_deltax = (old_x - xmin) - old_x0 * dx
        old_deltay = (old_y - ymin) - old_y0 * dy
        old_deltaz = (old_z - zmin) - old_z0 * dz
        # get the difference between the particle position and the nearest grid point

        shift_x = x0 - old_x0
        shift_y = y0 - old_y0
        shift_z = z0 - old_z0
        # calculate the shift between old and new grid points

        x0 = wrap_around(x0, Nx)
        y0 = wrap_around(y0, Ny)
        z0 = wrap_around(z0, Nz)
        # wrap around the grid points for periodic boundary conditions
        x1 = wrap_around(x0+1, Nx)
        y1 = wrap_around(y0+1, Ny)
        z1 = wrap_around(z0+1, Nz)
        # calculate the right grid point
        x2 = wrap_around(x0+2, Nx)
        y2 = wrap_around(y0+2, Ny)
        z2 = wrap_around(z0+2, Nz)
        # calculate the second right grid point
        x_minus1 = wrap_around(x0 - 1, Nx)
        y_minus1 = wrap_around(y0 - 1, Ny)
        z_minus1 = wrap_around(z0 - 1, Nz)
        # calculate the left grid point
        x_minus2 = wrap_around(x0 - 2, Nx)
        y_minus2 = wrap_around(y0 - 2, Ny)
        z_minus2 = wrap_around(z0 - 2, Nz)
        # calculate the second left grid point

        xpts = [x_minus2, x_minus1, x0, x1, x2]
        ypts = [y_minus2, y_minus1, y0, y1, y2]
        zpts = [z_minus2, z_minus1, z0, z1, z2]
        # place all the points in a list

        xw, yw, zw = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None,
        )
        # get the weights for the new positions
        oxw, oyw, ozw = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            operand=None,
        ) # get the weights for the old positions

        tmp = jnp.zeros_like(xw[0])
        # build the temporary zero array for padding

        xw = [tmp, xw[0], xw[1], xw[2], tmp]
        yw = [tmp, yw[0], yw[1], yw[2], tmp]
        zw = [tmp, zw[0], zw[1], zw[2], tmp]
        # pad the weights to 5 points for consistency

        oxw = [tmp, oxw[0], oxw[1], oxw[2], tmp]
        oyw = [tmp, oyw[0], oyw[1], oyw[2], tmp]
        ozw = [tmp, ozw[0], ozw[1], ozw[2], tmp]
        # pad the old weights to 5 points for consistency

        oxw = _roll_old_weights_to_new_frame(oxw, shift_x)
        oyw = _roll_old_weights_to_new_frame(oyw, shift_y)
        ozw = _roll_old_weights_to_new_frame(ozw, shift_z)

        # --- build Esirkepov W on compact stencil ---
        if x_active and y_active and z_active:
            Wx_, Wy_, Wz_ = get_3D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles)
        elif (x_active and y_active and (not z_active)) or (x_active and z_active and (not y_active)) or (y_active and z_active and (not x_active)):
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
            # determine which dimension is inactive

            Wx_, Wy_, Wz_ = get_2D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, null_dim=null_dim)
        elif x_active and (not y_active) and (not z_active):
            # 1D in x: Esirkepov reduces to 1D continuity;
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
        # calculate prefactors for current deposition

        # local “difference RHS”
        Fx = dJx * Wx_   # (Sx,Sy,Sz,Np)
        Fy = dJy * Wy_
        Fz = dJz * Wz_

        Jx_loc = jnp.zeros_like(Fx)
        Jy_loc = jnp.zeros_like(Fy)
        Jz_loc = jnp.zeros_like(Fz)

        # Using Backward Finite Difference approach for prefix sum #################################
        # Jx currents
        Jx_loc = jnp.cumsum(Fx, axis=0)
        # Jy currents
        Jy_loc = jnp.cumsum(Fy, axis=1)
        # Jz currents
        Jz_loc = jnp.cumsum(Fz, axis=2)
        # This assumes 5 cells in each dimension for the stencil, but 6 faces (so 5 differences).
        # This should give periodic wrap around J(1) = J(6) = 0 as required.
        ################################################################################################
        if x_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jx = Jx.at[xpts[i], ypts[j], zpts[k]].add(Jx_loc[i, j, k, :], mode="drop")
                        # deposit Jx using Esirkepov weights
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jx = Jx.at[xpts[i], ypts[j], zpts[k]].add(Fx[i, j, k, :], mode="drop")
                        # deposit Jx using midpoint weights for inactive dimension

        if y_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jy = Jy.at[xpts[i], ypts[j], zpts[k]].add(Jy_loc[i, j, k, :], mode="drop")
                        # deposit Jy using Esirkepov weights
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jy = Jy.at[xpts[i], ypts[j], zpts[k]].add(Fy[i, j, k, :], mode="drop")
                        # deposit Jy using midpoint weights for inactive dimension

        if z_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jz = Jz.at[xpts[i], ypts[j], zpts[k]].add(Jz_loc[i, j, k, :], mode="drop")
                        # deposit Jz using Esirkepov weights
        
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jz = Jz.at[xpts[i], ypts[j], zpts[k]].add(Fz[i, j, k, :], mode="drop")
                        # deposit Jz using midpoint weights for inactive dimension


    return (Jx, Jy, Jz)
        
        
def get_3D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=None):

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros_like( Wx_)
    Wz_ = jnp.zeros_like( Wx_)


    for i in range(len(x_weights)):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[i,j,k,:].set( (x_weights[i] - old_x_weights[i]) * ( 1/3 * (y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k])     \
                                                    +  1/6 * (y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k]) ) )

                Wy_ = Wy_.at[i,j,k,:].set( (y_weights[j] - old_y_weights[j]) * ( 1/3 * (x_weights[i] * z_weights[k] + old_x_weights[i] * old_z_weights[k])     \
                                                    +  1/6 * (x_weights[i] * old_z_weights[k] + old_x_weights[i] * z_weights[k]) ) )

                Wz_ = Wz_.at[i,j,k,:].set( (z_weights[k] - old_z_weights[k]) * ( 1/3 * (x_weights[i] * y_weights[j] + old_x_weights[i] * old_y_weights[j])     \
                                                    +  1/6 * (x_weights[i] * old_y_weights[j] + old_x_weights[i] * y_weights[j]) ) )

    return Wx_, Wy_, Wz_

def get_2D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=2):
    d_Sx = []
    d_Sy = []
    d_Sz = []

    d_Sx = [ x_weights[i] - old_x_weights[i] for i in range(len(x_weights)) ]
    d_Sy = [ y_weights[i] - old_y_weights[i] for i in range(len(y_weights)) ]
    d_Sz = [ z_weights[i] - old_z_weights[i] for i in range(len(z_weights)) ]

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros_like( Wx_)
    Wz_ = jnp.zeros_like( Wx_)
    # initialize the weight arrays

    # XY Plane
    def xy_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            for j in range(len(y_weights)):
                Wx_ = Wx_.at[i,j,2,:].set( 1/2 * d_Sx[i] * ( y_weights[j] + old_y_weights[j] ) )
                Wy_ = Wy_.at[i,j,2,:].set( 1/2 * d_Sy[j] * ( x_weights[i] + old_x_weights[i] ) )
                Wz_ = Wz_.at[i,j,2,:].set( 1/3 * ( x_weights[i] * y_weights[j] + old_x_weights[i] * old_y_weights[j] )     \
                                        +  1/6 * ( x_weights[i] * old_y_weights[j] + old_x_weights[i] * y_weights[j] ) )
        # Weights if the 2D plane is in the XY plane

        return Wx_, Wy_, Wz_
    

    # XZ Plane
    def xz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[i,2,k,:].set( 1/2 * d_Sx[i] * ( z_weights[k] + old_z_weights[k] ) )
                Wy_ = Wy_.at[i,2,k,:].set( 1/3 * ( x_weights[i] * z_weights[k] + old_x_weights[i] * old_z_weights[k] )     \
                                        +  1/6 * ( x_weights[i] * old_z_weights[k] + old_x_weights[i] * z_weights[k] ) )
                Wz_ = Wz_.at[i,2,k,:].set( 1/2 * d_Sz[k] * ( x_weights[i] + old_x_weights[i] ) )
        # Weights if the 2D plane is in the XZ plane
        return Wx_, Wy_, Wz_
    

    # YZ Plane
    def yz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[2,j,k,:].set( 1/3 * ( y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k] )     \
                                        +  1/6 * ( y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k] ) )
                Wy_ = Wy_.at[2,j,k,:].set( 1/2 * d_Sy[j] * ( z_weights[k] + old_z_weights[k] ) )
                Wz_ = Wz_.at[2,j,k,:].set( 1/2 * d_Sz[k] * ( y_weights[j] + old_y_weights[j] ) )
        # Weights if the 2D plane is in the YZ plane
        return Wx_, Wy_, Wz_
    

    Wx_, Wy_, Wz_ = lax.cond(
        null_dim == 0,
        lambda _: yz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
        lambda _: lax.cond(
            null_dim == 1,
            lambda _: xz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            lambda _: xy_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            operand=None
        ),
        operand=None
    )

    return Wx_, Wy_, Wz_


def get_1D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, dim=0):

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros_like( Wx_)
    Wz_ = jnp.zeros_like( Wx_)
    # initialize the weight arrays

    def x_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            Wx_ = Wx_.at[i, 2, 2, :].set( (x_weights[i] - old_x_weights[i]) )
            # get the weights for x direction
            Wy_ = Wy_.at[i, 2, 2, :].set( (x_weights[i] + old_x_weights[i]) / 2 )
            Wz_ = Wz_.at[i, 2, 2, :].set( (x_weights[i] + old_x_weights[i]) / 2 )
            # use a midpoint average for inactive directions
        # weights if x direction is active
        return Wx_, Wy_, Wz_

    def y_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for j in range(len(y_weights)):
            Wy_ = Wy_.at[2, j, 2, :].set( (y_weights[j] - old_y_weights[j]) )
            # weights for y direction
            Wx_ = Wx_.at[2, j, 2, :].set( (y_weights[j] + old_y_weights[j]) / 2 )
            Wz_ = Wz_.at[2, j, 2, :].set( (y_weights[j] + old_y_weights[j]) / 2 )
            # use a midpoint average for inactive directions
        # weights if y direction is active
        return Wx_, Wy_, Wz_
    
    def z_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for k in range(len(z_weights)):
            Wz_ = Wz_.at[2, 2, k, :].set( (z_weights[k] - old_z_weights[k]) )
            # weights for z direction
            Wx_ = Wx_.at[2, 2, k, :].set( (z_weights[k] + old_z_weights[k]) / 2 )
            Wy_ = Wy_.at[2, 2, k, :].set( (z_weights[k] + old_z_weights[k]) / 2 )
            # use a midpoint average for inactive directions
        # weights if z direction is active
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
    # determine which dimension is active and calculate weights accordingly


    return Wx_, Wy_, Wz_
