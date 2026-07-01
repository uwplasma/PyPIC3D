import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    axis_has_active_cells,
    compute_particle_anchor,
    particle_axis_offset,
)
from PyPIC3D.boundary_conditions.boundaryconditions import update_ghost_cells
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.particles.tiled_particles import TiledParticles


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


def clear_ghost_cells(field, axis):
    """Clear the two ghost layers along one axis."""
    field_axis = jnp.moveaxis(field, axis, 0)
    # move the axis to be cleared to the front for easier indexing
    field_axis = field_axis.at[0, :, :].set(0.0)
    field_axis = field_axis.at[1, :, :].set(0.0)
    field_axis = field_axis.at[-2, :, :].set(0.0)
    field_axis = field_axis.at[-1, :, :].set(0.0)
    # clear the two ghost layers by setting them to zero
    field = jnp.moveaxis(field_axis, 0, axis)
    # move the axis back to its original position
    return field


def enforce_bc_along_axis(field, axis, bc, component_axis):
    """
    Enforce boundary conditions along a specified axis of a field array.
    
    This function applies boundary conditions to ghost cells of a field by moving
    the specified axis to the front for easier manipulation, applying the appropriate
    boundary condition logic, and then moving the axis back to its original position.
    
    Parameters
    ----------
    field : jax.Array
        The field array to which boundary conditions will be applied.
        Expected to have two ghost cells at indices 0, 1, -2, -1 along the specified axis.
    axis : int
        The axis along which to enforce boundary conditions (0, 1, or 2).
    bc : str
        The type of boundary condition to apply. Options are:
        - "reflecting": Ghost cells are reflected across the boundary with optional sign flip
        - "absorbing": Ghost cells are left unchanged (will be cleared afterward)
        - "periodic": Ghost cells are folded by adding them to opposite physical layers (default)
    component_axis : int
        The axis corresponding to the field component direction. Used to determine
        the sign for reflecting boundary conditions (sign = -1 if axis == component_axis,
        else 1).
    
    Returns
    -------
    jax.Array
        The field array with boundary conditions enforced and ghost cells cleared.
    
    Notes
    -----
    - Ghost cells are assumed to be at indices [0, 1, -2, -1] along the axis.
    - The extended physical region is at indices [2:-2] along the axis.
    - After applying boundary conditions, all ghost cells are cleared to zero.
    """


    field_axis = jnp.moveaxis(field, axis, 0)
    # move the axis to be folded to the front for easier indexing.

    if bc == "reflecting":
        sign = -1.0 if axis == component_axis else 1.0
        field_axis = field_axis.at[2, :, :].add(sign * field_axis[0, :, :])
        field_axis = field_axis.at[3, :, :].add(sign * field_axis[1, :, :])
        field_axis = field_axis.at[-4, :, :].add(sign * field_axis[-2, :, :])
        field_axis = field_axis.at[-3, :, :].add(sign * field_axis[-1, :, :])
        # if the boundary is reflecting, then reflect the ghost layers across the boundary
    elif bc == "absorbing":
        field_axis = field_axis
        # if the boundary is absorbing, then neglect the ghost layers by leaving them as they are, which will be cleared to zero in the next step.
    else:
        # PERIODIC BC IS THE DEFAULT
        field_axis = field_axis.at[-4, :, :].add(field_axis[0, :, :])
        field_axis = field_axis.at[-3, :, :].add(field_axis[1, :, :])
        field_axis = field_axis.at[2, :, :].add(field_axis[-2, :, :])
        field_axis = field_axis.at[3, :, :].add(field_axis[-1, :, :])
        # if the boundary is periodic, fold the ghost layers by adding them to the opposite physical layer.


    field = jnp.moveaxis(field_axis, 0, axis)
    # move the axis back to its original position after folding the ghost layers according to the boundary condition.
    field = clear_ghost_cells(field, axis)
    # clear the ghost cells

    return field


def enforce_particle_bc_code_along_axis(field, axis, bc, component_axis):
    """
    Fold two Esirkepov ghost layers using global particle boundary-condition codes.
    """

    field_axis = jnp.moveaxis(field, axis, 0)
    sign = -1.0 if axis == component_axis else 1.0

    def periodic_bc(field_axis):
        field_axis = field_axis.at[-4, :, :].add(field_axis[0, :, :])
        field_axis = field_axis.at[-3, :, :].add(field_axis[1, :, :])
        field_axis = field_axis.at[2, :, :].add(field_axis[-2, :, :])
        field_axis = field_axis.at[3, :, :].add(field_axis[-1, :, :])
        return field_axis

    def reflecting_bc(field_axis):
        field_axis = field_axis.at[2, :, :].add(sign * field_axis[0, :, :])
        field_axis = field_axis.at[3, :, :].add(sign * field_axis[1, :, :])
        field_axis = field_axis.at[-4, :, :].add(sign * field_axis[-2, :, :])
        field_axis = field_axis.at[-3, :, :].add(sign * field_axis[-1, :, :])
        return field_axis

    def absorbing_bc(field_axis):
        return field_axis

    field_axis = lax.switch(bc, (periodic_bc, reflecting_bc, absorbing_bc), field_axis)
    field = jnp.moveaxis(field_axis, 0, axis)
    field = clear_ghost_cells(field, axis)

    return field


def ghost_cell_particle_bc_codes_esirkepov(field, bc_x, bc_y, bc_z, component_axis):
    field = enforce_particle_bc_code_along_axis(field, axis=0, bc=bc_x, component_axis=component_axis)
    field = enforce_particle_bc_code_along_axis(field, axis=1, bc=bc_y, component_axis=component_axis)
    field = enforce_particle_bc_code_along_axis(field, axis=2, bc=bc_z, component_axis=component_axis)
    return field


def ghost_cell_bc_esirkepov(field, bc_x, bc_y, bc_z, component_axis):
    """
    Apply boundary conditions to ghost cells using Esirkepov method.

    Enforces boundary conditions along all three axes (x, y, z) for a field
    by applying the appropriate boundary condition type along each spatial dimension.

    Parameters
    ----------
    field : ndarray
        The field array to which boundary conditions will be applied.
    bc_x : str or callable
        Boundary condition to enforce along the x-axis (axis=0).
    bc_y : str or callable
        Boundary condition to enforce along the y-axis (axis=1).
    bc_z : str or callable
        Boundary condition to enforce along the z-axis (axis=2).
    component_axis : int
        The axis index of the field component for which boundary conditions
        are being applied.

    Returns
    -------
    ndarray
        The field array with boundary conditions applied along all three axes.
    """
    field = enforce_bc_along_axis(field, axis=0, bc=bc_x, component_axis=component_axis)
    # enforce the x boundary conditions along the x axis (axis=0)
    field = enforce_bc_along_axis(field, axis=1, bc=bc_y, component_axis=component_axis)
    # enforce the y boundary conditions along the y axis (axis=1)
    field = enforce_bc_along_axis(field, axis=2, bc=bc_z, component_axis=component_axis)
    # enforce the z boundary conditions along the z axis (axis=2)
    return field


def eliminate_esirkepov_ghost_cells(field):
    slices = [slice(1, -1), slice(1, -1), slice(1, -1)]
    # remove one layer on each side, returning the solver's ordinary one-ghost current array.
    return field[tuple(slices)]

def Esirkepov_current(
    particles,
    J,
    constants,
    world,
    grid=None,
    filter=None,
    species_config=None,
    tile_shape=None,
    g=None,
):
    """Esirkepov current deposition supporting 1D/2D/3D via inactive dims."""
    if isinstance(particles, TiledParticles):
        if filter not in (None, "none"):
            raise ValueError("Esirkepov current filtering is not supported; use filter='none'.")

        if species_config is None:
            species_config = J
            J = constants
            constants = world
            world = grid
            grid = None
        # Tiled Esirkepov stores old particle positions.  The tiled kernel
        # predicts the new positions locally and leaves particle retile staging
        # to the caller, matching the Task 0 old/new position contract.

        if tile_shape is None:
            tile_shape = tuple(int(width) for width in world["tile_shape"])
        if g is None:
            g = int(world["guard_cells"])

        from PyPIC3D.deposition.esirkepov_tiled import _tiled_esirkepov_current

        return _tiled_esirkepov_current(
            particles,
            species_config,
            J,
            constants,
            world,
            grid=grid,
            tile_shape=tile_shape,
            g=int(g),
        )

    if grid is None:
        grid = world["grids"]["center"]

    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    dx, dy, dz, dt = world["dx"], world["dy"], world["dz"], world["dt"]
    bc_x = world["boundary_conditions"]["x"]
    bc_y = world["boundary_conditions"]["y"]
    bc_z = world["boundary_conditions"]["z"]
    shape_factor = world["shape_factor"]

    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)

    x_active = axis_has_active_cells(Nx, ghost_cells=True)
    y_active = axis_has_active_cells(Ny, ghost_cells=True)
    z_active = axis_has_active_cells(Nz, ghost_cells=True)
    # determine which directions are not quasi-1D.

    Nx_ext, Ny_ext, Nz_ext = Nx + 2, Ny + 2, Nz + 2
    # calculate the size of the extended grid with two ghost layers on each side for Esirkepov deposition.

    x_extended = jnp.concatenate([ grid[0][:1] - dx, grid[0], grid[0][-1:] + dx,])
    # add +- dx ghost points to the x grid axis for an extra ghost cell on each side
    y_extended = jnp.concatenate([ grid[1][:1] - dy, grid[1], grid[1][-1:] + dy,])
    # add +- dy ghost points to the y grid axis for an extra ghost cell on each side
    z_extended = jnp.concatenate([ grid[2][:1] - dz, grid[2], grid[2][-1:] + dz,])
    # add +- dz ghost points to the z grid axis for an extra ghost cell on each side

    extended_grid = (x_extended, y_extended, z_extended)
    # build the extended grid tuple for passing to deposition functions

    for species in particles:
        Jx_ext = jnp.zeros((Nx_ext, Ny_ext, Nz_ext), dtype=Jx.dtype)
        Jy_ext = jnp.zeros_like(Jx_ext)
        Jz_ext = jnp.zeros_like(Jx_ext)
        # Deposit one species at a time so particle boundary labels can fold
        # the extended ghost layers correctly.

        q = species.get_charge()
        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
        active = species.get_active_mask().astype(x.dtype)
        N_particles = species.get_number_of_particles()
        # get particle information

        old_x = x - vx * dt
        old_y = y - vy * dt
        old_z = z - vz * dt
        # compute old positions by backstepping along the velocity

        x0 = compute_particle_anchor(x, extended_grid[0], shape_factor)
        y0 = compute_particle_anchor(y, extended_grid[1], shape_factor)
        z0 = compute_particle_anchor(z, extended_grid[2], shape_factor)
        # compute the nearest grid point to the particle's current position

        old_x0 = compute_particle_anchor(old_x, extended_grid[0], shape_factor)
        old_y0 = compute_particle_anchor(old_y, extended_grid[1], shape_factor)
        old_z0 = compute_particle_anchor(old_z, extended_grid[2], shape_factor)
        # compute the nearest grid point to the particle's old position

        deltax = particle_axis_offset(x, x0, extended_grid[0])
        deltay = particle_axis_offset(y, y0, extended_grid[1])
        deltaz = particle_axis_offset(z, z0, extended_grid[2])
        old_deltax = particle_axis_offset(old_x, old_x0, extended_grid[0])
        old_deltay = particle_axis_offset(old_y, old_y0, extended_grid[1])
        old_deltaz = particle_axis_offset(old_z, old_z0, extended_grid[2])
        # compute the particle's offset from the nearest grid point in each direction at both time steps

        shift_x = x0 - old_x0
        shift_y = y0 - old_y0
        shift_z = z0 - old_z0
        # compute how many grid points the particles have shifted in each direction.

        offsets = jnp.asarray([-2, -1, 0, 1, 2], dtype=x0.dtype)
        # define the offsets for the 5-point Esirkepov stencil in each direction

        xpts = x0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
        ypts = y0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
        zpts = z0[jnp.newaxis, ...] + offsets[:, jnp.newaxis]
        # Build the 5-point stencil in the two-ghost deposition buffer.

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
        # compute the particle shape weights at the current and old time steps

        tmp = jnp.zeros_like(xw[0])

        xw = [tmp, xw[0], xw[1], xw[2], tmp]
        yw = [tmp, yw[0], yw[1], yw[2], tmp]
        zw = [tmp, zw[0], zw[1], zw[2], tmp]

        oxw = [tmp, oxw[0], oxw[1], oxw[2], tmp]
        oyw = [tmp, oyw[0], oyw[1], oyw[2], tmp]
        ozw = [tmp, ozw[0], ozw[1], ozw[2], tmp]
        # pad the weights with zeros so they can be rolled.

        oxw = shift_old_stencil(oxw, shift_x)
        oyw = shift_old_stencil(oyw, shift_y)
        ozw = shift_old_stencil(ozw, shift_z)
        # shift the old weights into the new-cell frame so they can be differenced with the current weights.
        xpts, xw, oxw = collapse_redundant_axis(xpts, xw, oxw, x_active, Nx_ext)
        ypts, yw, oyw = collapse_redundant_axis(ypts, yw, oyw, y_active, Ny_ext)
        zpts, zw, ozw = collapse_redundant_axis(zpts, zw, ozw, z_active, Nz_ext)
        # remove dummy stencil points for non 3D simulations

        if x_active and y_active and z_active:
            # if full 3D, calculate full Esirkepov weights with the 3D formula
            Wx_, Wy_, Wz_ = get_3D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles)
        elif (x_active and y_active and (not z_active)) or (x_active and z_active and (not y_active)) or (
            y_active and z_active and (not x_active)
            # if 2D, calculate Esirkepov weights with the 2D formula for the active plane
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
            # if 1D in x, calculate Esirkepov weights with the 1D formula for x and pad with zeros for y and z
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=0)
        elif y_active and (not x_active) and (not z_active):
            # if 1D in y, calculate Esirkepov weights with the 1D formula for y and pad with zeros for x and z
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=1)
        elif z_active and (not x_active) and (not y_active):
            # if 1D in z, calculate Esirkepov weights with the 1D formula for z and pad with zeros for x and y
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=2)
        # calculate the Esirkepov weights for the active dimensions

        dJx = jax.lax.cond(
            x_active,
            lambda _: active * (-(q / (dy * dz)) / dt) * jnp.ones(N_particles),
            lambda _: active * q * vx / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        dJy = jax.lax.cond(
            y_active,
            lambda _: active * (-(q / (dx * dz)) / dt) * jnp.ones(N_particles),
            lambda _: active * q * vy / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        dJz = jax.lax.cond(
            z_active,
            lambda _: active * (-(q / (dx * dy)) / dt) * jnp.ones(N_particles),
            lambda _: active * q * vz / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )
        # compute the current contribution prefactors for each active dimension according to Esirkepov's formula

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
                        Jx_ext = Jx_ext.at[xpts[i, :], ypts[j, :], zpts[k, :]].add(Jx_loc[i, j, k, :], mode="drop")
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jx_ext = Jx_ext.at[xpts[i, :], ypts[j, :], zpts[k, :]].add(Fx[i, j, k, :], mode="drop")

        if y_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jy_ext = Jy_ext.at[xpts[i, :], ypts[j, :], zpts[k, :]].add(Jy_loc[i, j, k, :], mode="drop")
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jy_ext = Jy_ext.at[xpts[i, :], ypts[j, :], zpts[k, :]].add(Fy[i, j, k, :], mode="drop")

        if z_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jz_ext = Jz_ext.at[xpts[i, :], ypts[j, :], zpts[k, :]].add(Jz_loc[i, j, k, :], mode="drop")
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jz_ext = Jz_ext.at[xpts[i, :], ypts[j, :], zpts[k, :]].add(Fz[i, j, k, :], mode="drop")
        # deposit the current contributions onto the grid.

        particle_bc = world.get("particle_boundary_conditions", {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC})
        # determine the particle boundary condition type from the global world state

        Jx_ext = ghost_cell_particle_bc_codes_esirkepov(
            Jx_ext, particle_bc["x"], particle_bc["y"], particle_bc["z"], component_axis=0
        )
        Jy_ext = ghost_cell_particle_bc_codes_esirkepov(
            Jy_ext, particle_bc["x"], particle_bc["y"], particle_bc["z"], component_axis=1
        )
        Jz_ext = ghost_cell_particle_bc_codes_esirkepov(
            Jz_ext, particle_bc["x"], particle_bc["y"], particle_bc["z"], component_axis=2
        )
        # Fold the two Esirkepov ghost layers before returning to the solver's
        # ordinary one-ghost current arrays.

        Jx = Jx + eliminate_esirkepov_ghost_cells(Jx_ext)
        Jy = Jy + eliminate_esirkepov_ghost_cells(Jy_ext)
        Jz = Jz + eliminate_esirkepov_ghost_cells(Jz_ext)

    Jx = update_ghost_cells(Jx, bc_x, bc_y, bc_z)
    Jy = update_ghost_cells(Jy, bc_x, bc_y, bc_z)
    Jz = update_ghost_cells(Jz, bc_x, bc_y, bc_z)
    # refresh the ordinary one-ghost current halos before the field update

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
