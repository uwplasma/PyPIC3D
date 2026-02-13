import jax
import jax.numpy as jnp

from PyPIC3D.solvers.fdtd import (
    centered_finite_difference_curl,
    centered_finite_difference_laplacian,
    centered_finite_difference_divergence,
    centered_finite_difference_gradient
)
from PyPIC3D.boris import interpolate_field_to_particles
from PyPIC3D.utils import digital_filter


def initialize_vector_potential(J, world, constants):
    """
    Initialize the vector potential A based on the current density J.

    Args:
        J (tuple): Current density components (Jx, Jy, Jz).
        world (dict): Simulation world parameters including 'dx', 'dy', 'dz'.
        constants (dict): Physical constants including 'mu'.

    Returns:
        tuple: Initial vector potential components (Ax, Ay, Az).
    """
    Jx, Jy, Jz = J

    Ax = jnp.zeros_like(Jx)
    Ay = jnp.zeros_like(Jy)
    Az = jnp.zeros_like(Jz)
    # initialize A as zero

    A = Ax, Ay, Az

    return A, A, A

@jax.jit
def update_vector_potential(J, world, constants, A1, A0):
    """
    Update the electromagnetic vector potential using an explicit second-order
    finite-difference time-domain (FDTD) scheme with periodic boundary conditions.

    This routine advances the vector potential components (Ax, Ay, Az) from the
    previous two time levels (A0, A1) using the wave equation form with a
    current-density source and a gauge-like correction term:
        A^{n+1} = 2 A^{n} - A^{n-1} + C^2 dt^2 * ( mu * J + ∇²A - ∇(∇·A) )
    A simple digital filter is applied to each updated component to reduce numerical
    noise.

    Parameters
    ----------
    J : tuple[array_like, array_like, array_like]
        Current density components (Jx, Jy, Jz) defined on the grid.
    world : dict
        Simulation/grid parameters. Must contain:
            - 'dx', 'dy', 'dz' : float
                Grid spacings in each direction.
            - 'dt' : float
                Time step.
    constants : dict
        Physical/numerical constants. Must contain:
            - 'mu' : float
                Magnetic permeability used in the source term.
            - 'C' : float
                Wave propagation speed (e.g., speed of light).
            - 'alpha' : float
                Digital filter parameter passed to `digital_filter`.
    A1 : tuple[array_like, array_like, array_like]
        Vector potential at the current time level n: (Ax, Ay, Az).
    A0 : tuple[array_like, array_like, array_like]
        Vector potential at the previous time level n-1: (Ax0, Ay0, Az0).

    Returns
    -------
    tuple[array_like, array_like, array_like]
        Updated vector potential at the next time level n+1: (Ax_new, Ay_new, Az_new).

    Notes
    -----
    - Spatial operators are computed using centered finite differences with
      'periodic' boundary conditions via:
        `centered_finite_difference_laplacian`,
        `centered_finite_difference_divergence`,
        `centered_finite_difference_gradient`.
    - The term ∇(∇·A) helps control the divergence of A depending on the chosen gauge.
    - The update is explicit; stability depends on dt relative to grid spacing and C
      (CFL-like constraint).
    """


    Ax0, Ay0, Az0 = A0
    Ax, Ay, Az = A1
    Jx, Jy, Jz = J

    dx, dy, dz = world['dx'], world['dy'], world['dz']

    mu = constants['mu']
    C  = constants['C']
    dt = world['dt']

    laplacian_Ax = centered_finite_difference_laplacian(Ax, dx, dy, dz, 'periodic')
    laplacian_Ay = centered_finite_difference_laplacian(Ay, dx, dy, dz, 'periodic')
    laplacian_Az = centered_finite_difference_laplacian(Az, dx, dy, dz, 'periodic')
    # calculate the Laplacian of the vector potential using centered finite difference

    divA         = centered_finite_difference_divergence(Ax, Ay, Az, dx, dy, dz, 'periodic')
    # calculate the divergence of the vector potential using centered finite difference

    gradx, grady, gradz = centered_finite_difference_gradient(divA, dx, dy, dz, 'periodic')
    # calculate the gradient of the divergence of the vector potential using centered finite difference

    Ax_new = 2 * Ax - Ax0 + C**2 * dt**2 * ( mu * Jx  + laplacian_Ax - gradx )
    Ay_new = 2 * Ay - Ay0 + C**2 * dt**2 * ( mu * Jy  + laplacian_Ay - grady )
    Az_new = 2 * Az - Az0 + C**2 * dt**2 * ( mu * Jz  + laplacian_Az - gradz )
    # update the vector potential using centered finite difference

    alpha = constants['alpha']
    Ax_new = digital_filter(Ax_new, alpha)
    Ay_new = digital_filter(Ay_new, alpha)
    Az_new = digital_filter(Az_new, alpha)
    # apply a digital filter to the vector potential components

    return Ax_new, Ay_new, Az_new

@jax.jit
def E_from_A(A2, A1, A0, world, grid, staggered_grid, interpolation_order=2):
    """
    Compute the electric field components from the vector potential via a leapfrog-consistent
    finite difference in time and interpolate them onto Yee-staggered grids.

    This routine uses:
        E(t + 1/2) = - (A(t + 1) - A(t)) / dt
    where `A1` and `A0` represent the vector potential at successive time levels. The raw
    finite-difference fields are computed on the base `grid` and then interpolated to the
    appropriate staggered (Yee) locations for each component.

    Parameters
    ----------
    A2 : tuple[array-like, array-like, array-like]
        Vector potential components (Ax, Ay, Az) at a time level not used by this function
        (kept for interface/time-stepping consistency).
    A1 : tuple[array-like, array-like, array-like]
        Vector potential components (Ax, Ay, Az) at the newer time level.
    A0 : tuple[array-like, array-like, array-like]
        Vector potential components (Ax, Ay, Az) at the older time level.
    world : dict
        Simulation state containing at least:
            - 'dt' (float): time step size.
    grid : tuple
        Base grid specification for the vector potential and intermediate fields. The exact
        structure is passed through to `interpolate_field`.
    staggered_grid : tuple
        Staggered grid specification (typically Yee offsets). Each component of E is placed on:
            - Ex: (staggered_grid[0], grid[1], grid[2])
            - Ey: (grid[0], staggered_grid[1], grid[2])
            - Ez: (grid[0], grid[1], staggered_grid[2])
    interpolation_order : int, optional
        Interpolation order forwarded to `interpolate_field` (default is 2).

    Returns
    -------
    Ex_ : array-like
        x-component of the electric field interpolated to its Yee-staggered grid.
    Ey_ : array-like
        y-component of the electric field interpolated to its Yee-staggered grid.
    Ez_ : array-like
        z-component of the electric field interpolated to its Yee-staggered grid.

    Notes
    -----
    - The function assumes `interpolate_field(field, src_grid, dst_grid, interpolation_order=...)`
      is available in scope.
    - `A2` is accepted but not used in the present implementation.
    """


    Ax2, Ay2, Az2 = A2
    Ax1, Ay1, Az1 = A1
    Ax0, Ay0, Az0 = A0
    dt = world['dt']

    Ex = -1 * (Ax1 - Ax0) / dt
    Ey = -1 * (Ay1 - Ay0) / dt
    Ez = -1 * (Az1 - Az0) / dt
    # calculate the electric field from the vector potential using a simple finite difference to get
    # E_t+1/2 = - (A_t+1 - A_t) / dt, which is more consistent with the leapfrog time-stepping scheme

    ############## INTERPOLATION GRIDS ###################################
    Ex_grid = staggered_grid[0], grid[1], grid[2]
    Ey_grid = grid[0], staggered_grid[1], grid[2]
    Ez_grid = grid[0], grid[1], staggered_grid[2]
    # create the grids for the electric field components
    ##########################################################################

    Ex_ = interpolate_field(Ex, grid, Ex_grid, interpolation_order=interpolation_order)
    Ey_ = interpolate_field(Ey, grid, Ey_grid, interpolation_order=interpolation_order)
    Ez_ = interpolate_field(Ez, grid, Ez_grid, interpolation_order=interpolation_order)
    # interpolate the electric field components to the Yee grid

    return Ex_, Ey_, Ez_

@jax.jit
def B_from_A(A, world, grid, staggered_grid, interpolation_order=2):
    """
    Compute the magnetic field **B** from a vector potential **A** via a centered
    finite-difference curl, then interpolate the result onto a Yee (staggered) grid.

    This routine:
    1) Computes B = ∇×A on the provided *cell-centered* grid using
        `centered_finite_difference_curl(..., bc='periodic')`.
    2) Defines the appropriate staggered-grid locations for each B component.
    3) Interpolates each component from the original grid to its Yee-grid location
        with `interpolate_field`.

    Parameters
    ----------
    A : tuple(array_like, array_like, array_like)
         Vector potential components `(Ax, Ay, Az)` defined on `grid`.
    world : dict
         Simulation metadata containing grid spacings with keys `'dx'`, `'dy'`, `'dz'`.
    grid : tuple(array_like, array_like, array_like)
         Base grid coordinate arrays `(x, y, z)` (typically cell-centered) on which
         `A` is defined.
    staggered_grid : tuple(array_like, array_like, array_like)
         Staggered coordinate arrays `(x_s, y_s, z_s)` defining half-cell offsets
         used for Yee-grid placement.
    interpolation_order : int, optional
         Order passed to `interpolate_field` when mapping each B component from
         `grid` to its staggered target grid. Default is 2.

    Returns
    -------
    Bx, By, Bz : array_like
         Magnetic field components interpolated to the Yee grid:
         - `Bx` on `(grid[0], staggered_grid[1], staggered_grid[2])`
         - `By` on `(staggered_grid[0], grid[1], staggered_grid[2])`
         - `Bz` on `(staggered_grid[0], staggered_grid[1], grid[2])`

    Notes
    -----
    - Boundary conditions for the curl are currently hard-coded to `'periodic'`.
    - This function assumes `centered_finite_difference_curl` and `interpolate_field`
      are available in scope and accept the argument patterns used here.
    """

    Ax, Ay, Az = A
    dx, dy, dz = world['dx'], world['dy'], world['dz']
    # unpack the vector potential components and grid spacing from the input parameters


    Bx, By, Bz = centered_finite_difference_curl(Ax, Ay, Az, dx, dy, dz, 'periodic')
    # calculate the magnetic field from the vector potential using centered finite difference curl

    ################## INTERPOLATION GRIDS ###################################
    Bx_grid = grid[0], staggered_grid[1], staggered_grid[2]
    By_grid = staggered_grid[0], grid[1], staggered_grid[2]
    Bz_grid = staggered_grid[0], staggered_grid[1], grid[2]
    # create the staggered grids for the magnetic field components
    ##########################################################################

    Bx_ = interpolate_field(Bx, grid, Bx_grid, interpolation_order)
    By_ = interpolate_field(By, grid, By_grid, interpolation_order)
    Bz_ = interpolate_field(Bz, grid, Bz_grid, interpolation_order)
    # interpolate the magnetic field components to the Yee grid

    return Bx_, By_, Bz_

def interpolate_field(field, grid, target_grid, interpolation_order=1):
    """
    Interpolate a field defined on a source grid onto a target grid.

    This function creates a 3D mesh from the 1D coordinate arrays provided in
    ``target_grid``, flattens the mesh coordinates, and calls
    :func:`interpolate_field_to_particles` to evaluate/interpolate the field at those
    points. The interpolated values are then reshaped back to the target mesh shape.

    Parameters
    ----------
    field : array-like
        Field values defined on ``grid``. Typically a JAX array with shape matching
        the source grid (or compatible with ``interpolate_field_to_particles``).
    grid : tuple
        Source grid specification passed through to
        :func:`interpolate_field_to_particles` (e.g., coordinate arrays and/or
        metadata describing the field's native grid).
    target_grid : tuple of array-like
        Target coordinate arrays ``(x, y, z)``. Each element is a 1D array of
        coordinates at which to construct the target mesh.
    interpolation_order : int, optional
        Interpolation order forwarded to :func:`interpolate_field_to_particles`.
        Default is 1 (typically linear).

    Returns
    -------
    jax.numpy.ndarray
        Interpolated field on the target grid with shape
        ``(len(x), len(y), len(z))``.

    Notes
    -----
    This routine uses ``jax.numpy.meshgrid(..., indexing="ij")`` so the output axis
    ordering follows ``(x, y, z)``.
    """


    x, y, z = target_grid
    # Unpack target grid coordinates

    X_target, Y_target, Z_target = jnp.meshgrid(x, y, z, indexing='ij')
    # Create a meshgrid for target coordinates

    x_flat = X_target.flatten()
    y_flat = Y_target.flatten()
    z_flat = Z_target.flatten()
    # Flatten target coordinates for interpolation

    interp_flat = interpolate_field_to_particles(field, x_flat, y_flat, z_flat, grid, interpolation_order)
    # Interpolate the field values at the flattened target coordinates using the specified interpolation order

    interp_field = interp_flat.reshape(X_target.shape)
    # Reshape back to the target grid shape

    return interp_field