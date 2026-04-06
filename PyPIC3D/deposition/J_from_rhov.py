import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    axis_has_active_cells,
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.boundary_conditions.boundaryconditions import fold_ghost_cells, update_ghost_cells
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.utils import bilinear_filter


def _remap_staggered_periodic_ghosts(points, position, axis_size, wind):
    """Route out-of-domain staggered deposits into the ghost that folds across the seam."""
    points = jnp.where(
        position[jnp.newaxis, ...] > 0.5 * wind,
        jnp.where(points == axis_size - 1, 0, points),
        points,
    )
    points = jnp.where(
        position[jnp.newaxis, ...] < -0.5 * wind,
        jnp.where(points == 0, axis_size - 1, points),
        points,
    )
    return points


@partial(jit, static_argnames=("filter",))
def J_from_rhov(particles, J, constants, world, grid=None, filter="bilinear"):
    """Compute current density (Jx,Jy,Jz) by depositing particle velocities."""

    if grid is None:
        grid = world["grids"]["center"]

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]

    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    x_active = axis_has_active_cells(Nx, ghost_cells=True)
    y_active = axis_has_active_cells(Ny, ghost_cells=True)
    z_active = axis_has_active_cells(Nz, ghost_cells=True)
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]

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

        x, x0, deltax_node, xpts = prepare_particle_axis_stencil(
            x,
            grid[0],
            Nx,
            shape_factor,
            bc_x,
            wind=world["x_wind"],
            ghost_cells=True,
        )
        y, y0, deltay_node, ypts = prepare_particle_axis_stencil(
            y,
            grid[1],
            Ny,
            shape_factor,
            bc_y,
            wind=world["y_wind"],
            ghost_cells=True,
        )
        z, z0, deltaz_node, zpts = prepare_particle_axis_stencil(
            z,
            grid[2],
            Nz,
            shape_factor,
            bc_z,
            wind=world["z_wind"],
            ghost_cells=True,
        )

        deltax_face = (x - grid[0][0]) - (x0 + 0.5) * dx
        deltay_face = (y - grid[1][0]) - (y0 + 0.5) * dy
        deltaz_face = (z - grid[2][0]) - (z0 + 0.5) * dz

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
        xpts = jax.lax.cond(
            bc_x == 0,
            lambda pts: _remap_staggered_periodic_ghosts(pts, x, Nx, world["x_wind"]),
            lambda pts: pts,
            operand=xpts,
        )
        ypts = jax.lax.cond(
            bc_y == 0,
            lambda pts: _remap_staggered_periodic_ghosts(pts, y, Ny, world["y_wind"]),
            lambda pts: pts,
            operand=ypts,
        )
        zpts = jax.lax.cond(
            bc_z == 0,
            lambda pts: _remap_staggered_periodic_ghosts(pts, z, Nz, world["z_wind"]),
            lambda pts: pts,
            operand=zpts,
        )

        x_weights_face = jnp.asarray(x_weights_face)
        y_weights_face = jnp.asarray(y_weights_face)
        z_weights_face = jnp.asarray(z_weights_face)

        x_weights_node = jnp.asarray(x_weights_node)
        y_weights_node = jnp.asarray(y_weights_node)
        z_weights_node = jnp.asarray(z_weights_node)
        xpts, x_weights_node = collapse_axis_stencil(xpts, x_weights_node, Nx, ghost_cells=True)
        _, x_weights_face = collapse_axis_stencil(xpts, x_weights_face, Nx, ghost_cells=True)
        ypts, y_weights_node = collapse_axis_stencil(ypts, y_weights_node, Ny, ghost_cells=True)
        _, y_weights_face = collapse_axis_stencil(ypts, y_weights_face, Ny, ghost_cells=True)
        zpts, z_weights_node = collapse_axis_stencil(zpts, z_weights_node, Nz, ghost_cells=True)
        _, z_weights_face = collapse_axis_stencil(zpts, z_weights_face, Nz, ghost_cells=True)

        xpts_eff = xpts
        ypts_eff = ypts
        zpts_eff = zpts
        x_weights_node_eff = x_weights_node
        y_weights_node_eff = y_weights_node
        z_weights_node_eff = z_weights_node
        x_weights_face_eff = x_weights_face
        y_weights_face_eff = y_weights_face
        z_weights_face_eff = z_weights_face

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

    Jx = fold_ghost_cells(Jx, bc_x, bc_y, bc_z)
    Jy = fold_ghost_cells(Jy, bc_x, bc_y, bc_z)
    Jz = fold_ghost_cells(Jz, bc_x, bc_y, bc_z)
    # fold ghost cell deposits back into interior
    Jx = update_ghost_cells(Jx, bc_x, bc_y, bc_z)
    Jy = update_ghost_cells(Jy, bc_x, bc_y, bc_z)
    Jz = update_ghost_cells(Jz, bc_x, bc_y, bc_z)
    # refresh ghost cells before any stencil-based post-processing

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
    Jx = update_ghost_cells(Jx, bc_x, bc_y, bc_z)
    Jy = update_ghost_cells(Jy, bc_x, bc_y, bc_z)
    Jz = update_ghost_cells(Jz, bc_x, bc_y, bc_z)

    return (Jx, Jy, Jz)
