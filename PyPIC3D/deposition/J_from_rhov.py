import jax
from jax import jit
import jax.numpy as jnp
from functools import partial

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    axis_has_active_cells,
    collapse_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.boundary_conditions.boundaryconditions import fold_ghost_cells, update_ghost_cells
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.utils import bilinear_filter


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
    # get the grid dimensions for the current density arrays
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    # determine the boundary conditions for each axis to handle ghost cells correctly

    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)
    # initialize current density arrays to zero before deposition

    for species in particles:
        shape_factor = species.get_shape()
        charge = species.get_charge()
        dq = charge / (dx * dy * dz)
        # get the particle information needed for deposition, including charge and shape factor

        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
        active = species.get_active_mask().astype(x.dtype)
        # get the particle velocities and positions at the forward time step (t + dt) for deposition

        x = x - vx * world["dt"] / 2
        y = y - vy * world["dt"] / 2
        z = z - vz * world["dt"] / 2
        # half step back the particle positions to align with the time-centered current deposition scheme

        x, x0, deltax_node, xpts = prepare_particle_axis_stencil(
            x,
            grid[0],
            Nx,
            shape_factor,
            bc_x,
            wind=world["x_wind"],
            ghost_cells=True,
        )
        # prepare the particle axis stencil for the x-axis, which includes determining the grid points that 
        # contribute to the deposition and the corresponding weights based on the particle's position and 
        # shape factor. This is done for all three axes (x, y, z) to handle 3D deposition correctly.

        y, y0, deltay_node, ypts = prepare_particle_axis_stencil(
            y,
            grid[1],
            Ny,
            shape_factor,
            bc_y,
            wind=world["y_wind"],
            ghost_cells=True,
        ) 
        # prepare the particle axis stencil for the y-axis, which includes determining the grid points that 
        # contribute to the deposition and the corresponding weights based on the particle's position and 
        # shape factor. This is done for all three axes (x, y, z) to handle 3D deposition correctly.

        z, z0, deltaz_node, zpts = prepare_particle_axis_stencil(
            z,
            grid[2],
            Nz,
            shape_factor,
            bc_z,
            wind=world["z_wind"],
            ghost_cells=True,
        ) 
        # prepare the particle axis stencil for the z-axis, which includes determining the grid points that 
        # contribute to the deposition and the corresponding weights based on the particle's position and 
        # shape factor. This is done for all three axes (x, y, z) to handle 3D deposition correctly.

        deltax_face = (x - grid[0][0]) - (x0 + 0.5) * dx
        deltay_face = (y - grid[1][0]) - (y0 + 0.5) * dy
        deltaz_face = (z - grid[2][0]) - (z0 + 0.5) * dz
        # compute the distances from the particles to the face-centered grid points.

        x_weights_node, y_weights_node, z_weights_node = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            operand=None,
        )
        # compute the node-centered weights for deposition

        x_weights_face, y_weights_face, z_weights_face = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz),
            operand=None,
        )
        # compute the face-centered weights for deposition

        xpts = jnp.asarray(xpts)
        ypts = jnp.asarray(ypts)
        zpts = jnp.asarray(zpts)
        # Boundary ownership stays in fold_ghost_cells: deposition may write
        # into ghost cells, and folding maps those deposits back to the domain.

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
        # Collapse stencils for inactive (single-cell) axes so deposition uses the
        # reduced effective stencil; ghost-cell folding is handled later.

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
        # get the list of indices to map over

        def idx_and_dJ_values(idx):
            i, j, k = idx
            ix = xpts_eff[i, ...]
            iy = ypts_eff[j, ...]
            iz = zpts_eff[k, ...]
            valx = (
                (active * dq * vx)
                * x_weights_face_eff[i, ...]
                * y_weights_node_eff[j, ...]
                * z_weights_node_eff[k, ...]
            )
            valy = (
                (active * dq * vy)
                * x_weights_node_eff[i, ...]
                * y_weights_face_eff[j, ...]
                * z_weights_node_eff[k, ...]
            )
            valz = (
                (active * dq * vz)
                * x_weights_node_eff[i, ...]
                * y_weights_node_eff[j, ...]
                * z_weights_face_eff[k, ...]
            )
            return ix, iy, iz, valx, valy, valz

        ix, iy, iz, valx, valy, valz = jax.vmap(idx_and_dJ_values)(combos)
        # use vectorized mapping to compute the current depositions.

        Jx = Jx.at[(ix, iy, iz)].add(valx, mode="drop")
        Jy = Jy.at[(ix, iy, iz)].add(valy, mode="drop")
        Jz = Jz.at[(ix, iy, iz)].add(valz, mode="drop")
        # deposit the currents onto the faces.

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
    # apply any filters
    Jx = update_ghost_cells(Jx, bc_x, bc_y, bc_z)
    Jy = update_ghost_cells(Jy, bc_x, bc_y, bc_z)
    Jz = update_ghost_cells(Jz, bc_x, bc_y, bc_z)
    # update the ghost cells

    return (Jx, Jy, Jz)
