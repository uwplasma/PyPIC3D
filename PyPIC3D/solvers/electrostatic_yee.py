import jax.numpy as jnp
from jax import lax

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.utilities.filters import digital_filter
from PyPIC3D.boundary_conditions import ghost_cells


def _active_slice(g):
    return slice(g, -g)


def _forward_slice(g):
    return slice(g + 1, None if g == 1 else -g + 1)


def _backward_slice(g):
    return slice(g - 1, -g - 1)


def _as_single_tile(field):
    return field[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]


def _refresh_single_tile_scalar(field, world, g, apply_conducting=False):
    field_tiles = _as_single_tile(field)
    if apply_conducting:
        field_tiles = ghost_cells.apply_tiled_scalar_conducting_bc(field_tiles, world, num_guard_cells=g)
    field_tiles = ghost_cells.update_tiled_ghost_cells(field_tiles, world, g)
    return field_tiles[0, 0, 0]


def _centered_finite_difference_gradient(field, dx, dy, dz):
    """
    Compute the centered finite-difference gradient on a periodic scalar field.

    This is the electrostatic single-tile post-processing stencil.  The tiled
    electrostatic path below uses the same centered difference directly on
    compact tile arrays with refreshed halos.
    """

    grad_x = (jnp.roll(field, shift=-1, axis=0) - jnp.roll(field, shift=1, axis=0)) / (2.0 * dx)
    grad_y = (jnp.roll(field, shift=-1, axis=1) - jnp.roll(field, shift=1, axis=1)) / (2.0 * dy)
    grad_z = (jnp.roll(field, shift=-1, axis=2) - jnp.roll(field, shift=1, axis=2)) / (2.0 * dz)

    return grad_x, grad_y, grad_z


def solve_poisson_with_conjugate_gradient(rho, phi, constants, world, tol=1e-12, max_iter=5000):
    """
    Solve Poisson's equation using matrix-free conjugate gradient.

    Uses ghost-cell slicing for the Laplacian stencil instead of jnp.roll.

    Args:
        rho (ndarray): Charge density field with shape (Nx+2*g, Ny+2*g, Nz+2*g).
        phi (ndarray): Initial guess for potential with shape (Nx+2*g, Ny+2*g, Nz+2*g).
        constants (dict): Physical constants with key ``eps``.
        world (dict): Simulation metadata with ``dx``, ``dy``, and ``dz``.
        tol (float): Residual tolerance.
        max_iter (int): Maximum number of CG iterations.

    Returns:
        ndarray: Electrostatic potential field with shape (Nx+2*g, Ny+2*g, Nz+2*g).
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    eps = constants['eps']

    g = int(world["guard_cells"])
    active = _active_slice(g)
    forward = _forward_slice(g)
    backward = _backward_slice(g)
    # get the single-tile layout from the world contract

    def lapl(field):
        # Laplacian using tile-local ghost-cell neighbors instead of jnp.roll.
        dfdx2 = (field[forward, active, active] + field[backward, active, active] - 2.0 * field[active, active, active]) / (dx * dx)
        dfdy2 = (field[active, forward, active] + field[active, backward, active] - 2.0 * field[active, active, active]) / (dy * dy)
        dfdz2 = (field[active, active, forward] + field[active, active, backward] - 2.0 * field[active, active, active]) / (dz * dz)
        return dfdx2 + dfdy2 + dfdz2
    # compute the laplacian on the interior using ghost cell neighbors

    def apply_bc(field):
        return _refresh_single_tile_scalar(field, world, g, apply_conducting=True)
    # apply scalar conducting boundaries and refresh ghost cells through the tiled halo path

    def body_fun(state):
        phi, r, p, k = state

        lapl_p = -lapl(p)
        alpha = jnp.sum(r * r) / jnp.sum(p[active, active, active] * lapl_p)
        # compute the optimal step size in the direction of the residual
        phi_next = phi.at[active, active, active].add(alpha * p[active, active, active])
        # update the potential guess
        phi_next = apply_bc(phi_next)
        # apply boundary conditions to the new potential guess

        r_next = r - alpha * lapl_p
        # compute the new residual after stepping in the direction of p
        beta = jnp.sum(r_next * r_next) / jnp.sum(r * r)
        # compute the optimal scaling for the new search direction
        p_next = p.at[active, active, active].set(r_next + beta * p[active, active, active])
        p_next = apply_bc(p_next)
        # compute the new search direction

        return phi_next, r_next, p_next, k + 1
    # perform one iteration of the conjugate gradient method

    def cond_fun(state):
        phi, r, p, k = state

        norm_r = jnp.sum(r * r)
        # compute the squared L2 norm of the residual for convergence checking
        return jnp.logical_and(k < max_iter, norm_r > tol**2)
    # if the residual norm is above the tolerance and we haven't exceeded max iterations, continue iterating

    phi = apply_bc(phi)
    residual = rho[active, active, active] / eps + lapl(phi)
    # compute the error in the Poisson equation with the current potential guess

    p0 = jnp.zeros_like(phi)
    p0 = p0.at[active, active, active].set(residual)
    p0 = apply_bc(p0)
    # initialize the search direction with ghost cells

    phi, residual, p, iteration = lax.while_loop(
        cond_fun,
        body_fun,
        (phi, residual, p0, 0),
    )
    # run the conjugate gradient iterations until convergence or max iterations reached

    return apply_bc(phi)


def calculate_electrostatic_fields(world, particles, constants, rho, phi, solver, bc):
    """
    Compute electrostatic fields from charge deposition and Poisson solve.

    All field arrays have shape (Nx+2*g, Ny+2*g, Nz+2*g) with ghost cells.

    Args:
        world (dict): Simulation world dictionary.
        particles (list): Particle species list.
        constants (dict): Physical constants.
        rho (ndarray): Charge density array with shape (Nx+2*g, Ny+2*g, Nz+2*g).
        phi (ndarray): Potential array with shape (Nx+2*g, Ny+2*g, Nz+2*g).
        solver (str): Electrostatic solver mode.
        bc (str): Boundary condition for finite-difference gradient.

    Returns:
        tuple: ``((Ex, Ey, Ez), phi, rho)``.
    """
    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    tile_shape = tuple(int(width) for width in world["tile_shape"])
    g = int(world["guard_cells"])
    active = _active_slice(g)

    phi = solve_poisson_with_conjugate_gradient(rho, phi, constants, world)

    phi = _refresh_single_tile_scalar(phi, world, g)
    # refresh ghost cells before any stencil-based post-processing

    alpha = constants['alpha']
    phi = digital_filter(phi, alpha, num_guard_cells=g)
    phi = _refresh_single_tile_scalar(phi, world, g, apply_conducting=True)
    # update ghost cells after filtering

    del solver, bc

    Ex, Ey, Ez = _centered_finite_difference_gradient(-1.0 * phi[active, active, active], dx, dy, dz)
    # compute gradient on the interior

    # Place the gradient results into ghost-celled arrays
    Ex_full = jnp.zeros_like(phi)
    Ey_full = jnp.zeros_like(phi)
    Ez_full = jnp.zeros_like(phi)
    Ex_full = Ex_full.at[active, active, active].set(Ex)
    Ey_full = Ey_full.at[active, active, active].set(Ey)
    Ez_full = Ez_full.at[active, active, active].set(Ez)
    Ex_full = _refresh_single_tile_scalar(Ex_full, world, g)
    Ey_full = _refresh_single_tile_scalar(Ey_full, world, g)
    Ez_full = _refresh_single_tile_scalar(Ez_full, world, g)

    return (Ex_full, Ey_full, Ez_full), phi, rho


def _centered_tiled_electrostatic_gradient(phi_tiles, world, tile_shape, g):
    """
    Compute ``E = -grad(phi)`` on compact scalar tiles.

    The potential halos must already contain neighboring tile/global boundary
    values.  This mirrors ``centered_finite_difference_gradient`` on the
    assembled physical interior, then refreshes vector halos for particle
    interpolation.
    """

    dx = world["dx"]
    dy = world["dy"]
    dz = world["dz"]
    g = int(g)
    active = slice(g, -g)
    forward = slice(g + 1, None if g == 1 else -g + 1)
    backward = slice(g - 1, -g - 1)

    phi_tiles = ghost_cells.update_tiled_ghost_cells(phi_tiles, world, g)

    Ex = jnp.zeros_like(phi_tiles)
    Ey = jnp.zeros_like(phi_tiles)
    Ez = jnp.zeros_like(phi_tiles)

    Ex = Ex.at[:, :, :, active, active, active].set(
        -1.0 * (phi_tiles[:, :, :, forward, active, active] - phi_tiles[:, :, :, backward, active, active]) / (2.0 * dx)
    )
    Ey = Ey.at[:, :, :, active, active, active].set(
        -1.0 * (phi_tiles[:, :, :, active, forward, active] - phi_tiles[:, :, :, active, backward, active]) / (2.0 * dy)
    )
    Ez = Ez.at[:, :, :, active, active, active].set(
        -1.0 * (phi_tiles[:, :, :, active, active, forward] - phi_tiles[:, :, :, active, active, backward]) / (2.0 * dz)
    )

    return ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), world, g)


def calculate_tiled_electrostatic_fields(world, particles, species_config, constants, rho_tiles, phi_tiles):
    """
    Compute electrostatic fields from single-tile rho deposition and a Poisson solve.

    Electrostatic runs use one tile covering the whole physical domain.  The
    leading tile axes remain singleton axes, so the Poisson solve acts directly
    on ``rho_tiles[0, 0, 0]`` and ``phi_tiles[0, 0, 0]`` without assembling a
    separate global representation.
    """

    tile_shape = tuple(int(width) for width in world["tile_shape"])
    g = int(world["guard_cells"])
    rho_tiles = compute_rho(particles, species_config, rho_tiles, constants, world)
    rho = rho_tiles[0, 0, 0]
    phi = phi_tiles[0, 0, 0]

    phi = solve_poisson_with_conjugate_gradient(rho, phi, constants, world)
    phi_tiles = phi_tiles.at[0, 0, 0].set(phi)
    phi_tiles = ghost_cells.update_tiled_ghost_cells(phi_tiles, world, g)
    # refresh ghost cells before filtering and tiled differentiation

    alpha = constants["alpha"]
    phi_tiles = digital_filter(phi_tiles, alpha, num_guard_cells=g)
    phi_tiles = ghost_cells.apply_tiled_scalar_conducting_bc(phi_tiles, world, num_guard_cells=g)
    phi_tiles = ghost_cells.update_tiled_ghost_cells(phi_tiles, world, g)
    # keep the same phi post-processing order as the previous electrostatic solver

    E_tiles = _centered_tiled_electrostatic_gradient(phi_tiles, world, tile_shape, g)

    return E_tiles, phi_tiles, rho_tiles
