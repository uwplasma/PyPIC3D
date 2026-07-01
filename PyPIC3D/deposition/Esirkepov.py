import jax
import jax.numpy as jnp
from jax import lax

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

    raise ValueError("Public Esirkepov_current requires TiledParticles.")


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
