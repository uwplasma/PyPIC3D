import jax.numpy as jnp
from jax import lax

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.utilities.filters import digital_filter
from PyPIC3D.boundary_conditions import ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONDUCTING, BC_PERIODIC


def _active_slice(g):
    return slice(g, -g)


def _forward_slice(g):
    return slice(g + 1, None if g == 1 else -g + 1)


def _backward_slice(g):
    return slice(g - 1, -g - 1)


def _apply_tiled_phi_zero_boundaries(field_tiles, static_parameters, g):
    bc_x, bc_y, bc_z = static_parameters.boundary_conditions
    boundary_conditions = (bc_x, bc_y, bc_z)

    # Ground each physical conducting face before refreshing the inter-tile and
    # exterior halos. Explicit constant boundaries retain their normal-copy
    # behavior in the generic ghost-cell updater.
    for axis, boundary_condition in enumerate(boundary_conditions):
        if int(boundary_condition) == BC_CONDUCTING:
            apply_boundary = ghost_cells.make_distributed_zero_boundary(
                static_parameters.field_mesh,
                static_parameters.tile_shape,
                axis,
                g,
            )
            field_tiles = apply_boundary(field_tiles)

    return ghost_cells.update_tiled_ghost_cells(
        field_tiles,
        static_parameters,
        g,
    )


def _free_potential_mask(residual, static_parameters):
    """Mask out fixed-potential planes on the global conducting walls."""

    free = jnp.ones_like(residual, dtype=bool)
    boundary_conditions = tuple(
        int(boundary_condition)
        for boundary_condition in static_parameters.boundary_conditions
    )

    for axis, boundary_condition in enumerate(boundary_conditions):
        if boundary_condition != BC_CONDUCTING:
            continue

        lower_wall = [slice(None)] * residual.ndim
        upper_wall = [slice(None)] * residual.ndim
        lower_wall[axis] = 0
        lower_wall[axis + 3] = 0
        upper_wall[axis] = -1
        upper_wall[axis + 3] = -1
        free = free.at[tuple(lower_wall)].set(False)
        free = free.at[tuple(upper_wall)].set(False)

    return free


def _free_poisson_residual(
    rho_tiles,
    phi_tiles,
    static_parameters,
    dynamic_parameters,
    g,
):
    residual = _poisson_residual(
        rho_tiles,
        phi_tiles,
        dynamic_parameters,
        g,
    )
    return jnp.where(
        _free_potential_mask(residual, static_parameters),
        residual,
        0.0,
    )


def _tiled_laplacian(field_tiles, dynamic_parameters, g):
    """Apply the seven-point Laplacian to every tile owned interior."""

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    active = _active_slice(g)
    forward = _forward_slice(g)
    backward = _backward_slice(g)

    dfdx2 = (
        field_tiles[..., forward, active, active]
        + field_tiles[..., backward, active, active]
        - 2.0 * field_tiles[..., active, active, active]
    ) / (dx * dx)
    dfdy2 = (
        field_tiles[..., active, forward, active]
        + field_tiles[..., active, backward, active]
        - 2.0 * field_tiles[..., active, active, active]
    ) / (dy * dy)
    dfdz2 = (
        field_tiles[..., active, active, forward]
        + field_tiles[..., active, active, backward]
        - 2.0 * field_tiles[..., active, active, active]
    ) / (dz * dz)

    return dfdx2 + dfdy2 + dfdz2


def _poisson_residual(rho_tiles, phi_tiles, dynamic_parameters, g):
    """Return ``rho / eps + laplacian(phi)`` on each tile's owned cells."""

    active = _active_slice(g)
    rho_owned = rho_tiles[..., active, active, active]
    return rho_owned / dynamic_parameters.eps + _tiled_laplacian(
        phi_tiles,
        dynamic_parameters,
        g,
    )


def _local_tile_cg_solve(
    rho_tiles,
    phi_tiles,
    static_parameters,
    dynamic_parameters,
    g,
    local_cg_tol,
    local_cg_max_iterations,
):
    """Solve each tile with fixed Dirichlet halos and conducting wall planes."""

    active = _active_slice(g)
    residual = _free_poisson_residual(
        rho_tiles,
        phi_tiles,
        static_parameters,
        dynamic_parameters,
        g,
    )
    free = _free_potential_mask(residual, static_parameters)
    search_direction = jnp.zeros_like(phi_tiles)
    search_direction = search_direction.at[..., active, active, active].set(residual)
    rr = jnp.sum(
        residual * residual,
        axis=(-3, -2, -1),
    )
    local_cg_tol_squared = jnp.asarray(local_cg_tol, dtype=rr.dtype) ** 2
    tile_active = rr > local_cg_tol_squared
    local_cg_iteration = jnp.asarray(0, dtype=jnp.int32)

    def cg_not_converged(state):
        _, _, _, _, tile_active, local_cg_iteration = state
        return jnp.any(tile_active) & (
            local_cg_iteration < local_cg_max_iterations
        )

    def cg_iteration(state):
        phi_tiles, residual, search_direction, rr, tile_active, local_cg_iteration = state

        search_owned = search_direction[..., active, active, active]
        Ap = -_tiled_laplacian(
            search_direction,
            dynamic_parameters,
            g,
        )
        Ap = jnp.where(free, Ap, 0.0)
        pAp = jnp.sum(
            search_owned * Ap,
            axis=(-3, -2, -1),
        )
        valid_step = tile_active & (pAp > 0.0)
        safe_pAp = jnp.where(valid_step, pAp, 1.0)
        alpha = jnp.where(valid_step, rr / safe_pAp, 0.0)
        alpha = alpha[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]

        phi_owned = phi_tiles[..., active, active, active]
        phi_owned_next = phi_owned + alpha * search_owned
        phi_tiles_next = phi_tiles.at[..., active, active, active].set(phi_owned_next)
        residual_next = residual - alpha * Ap
        rr_next = jnp.sum(
            residual_next * residual_next,
            axis=(-3, -2, -1),
        )

        tile_active_next = valid_step & (rr_next > local_cg_tol_squared)
        safe_rr = jnp.where(tile_active_next, rr, 1.0)
        beta = jnp.where(tile_active_next, rr_next / safe_rr, 0.0)
        beta = beta[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
        tile_active_owned = tile_active_next[
            ..., jnp.newaxis, jnp.newaxis, jnp.newaxis
        ]
        search_owned_next = jnp.where(
            tile_active_owned,
            residual_next + beta * search_owned,
            jnp.zeros_like(residual_next),
        )
        # Search directions remain zero in every guard cell. Neighbor values
        # are fixed Dirichlet data until the next outer Schwarz halo refresh.
        search_direction_next = jnp.zeros_like(search_direction)
        search_direction_next = search_direction_next.at[
            ..., active, active, active
        ].set(search_owned_next)

        return (
            phi_tiles_next,
            residual_next,
            search_direction_next,
            rr_next,
            tile_active_next,
            local_cg_iteration + 1,
        )

    phi_tiles, _, _, rr, _, _ = lax.while_loop(
        cg_not_converged,
        cg_iteration,
        (
            phi_tiles,
            residual,
            search_direction,
            rr,
            tile_active,
            local_cg_iteration,
        ),
    )

    local_cg_residual = jnp.sqrt(rr)
    return phi_tiles, local_cg_residual


def _schwarz_interface_mask(residual, static_parameters, g):
    interface_mask = jnp.zeros_like(residual, dtype=bool)
    boundary_conditions = tuple(
        int(boundary_condition)
        for boundary_condition in static_parameters.boundary_conditions
    )
    has_interface = False

    for axis, boundary_condition in enumerate(boundary_conditions):
        num_tiles = int(residual.shape[axis])
        num_owned_cells = int(residual.shape[axis + 3])

        tile_shape = [1] * residual.ndim
        tile_shape[axis] = num_tiles
        tile_index = jnp.arange(num_tiles).reshape(tile_shape)

        cell_shape = [1] * residual.ndim
        cell_shape[axis + 3] = num_owned_cells
        cell_index = jnp.arange(num_owned_cells).reshape(cell_shape)

        if boundary_condition == BC_PERIODIC:
            lower_has_neighbor = jnp.ones(tile_shape, dtype=bool)
            upper_has_neighbor = jnp.ones(tile_shape, dtype=bool)
            has_interface = True
        else:
            lower_has_neighbor = tile_index > 0
            upper_has_neighbor = tile_index < num_tiles - 1
            has_interface = has_interface or num_tiles > 1

        lower_interface = lower_has_neighbor & (cell_index < g)
        upper_interface = upper_has_neighbor & (cell_index >= num_owned_cells - g)
        interface_mask = interface_mask | lower_interface | upper_interface

    return interface_mask, has_interface


def _schwarz_interface_residual(residual, static_parameters, g):
    """Return the maximum residual in the ``g``-cell inter-tile face slabs."""

    interface_mask, has_interface = _schwarz_interface_mask(
        residual,
        static_parameters,
        g,
    )
    if not has_interface:
        # A single nonperiodic tile has no Schwarz interface. Its local domain
        # residual still has to drive the one required solve and convergence.
        return jnp.max(jnp.abs(residual))

    return jnp.max(jnp.where(interface_mask, jnp.abs(residual), 0.0))


def solve_poisson_with_tiled_local_schwarz(
    rho_tiles,
    phi_tiles,
    static_parameters,
    dynamic_parameters,
    *,
    schwarz_tol=1.0e-6,
    schwarz_max_iterations=500,
    local_cg_tol=1.0e-6,
    local_cg_max_iterations=500,
    return_diagnostics=False,
):
    """
    Solve tiled Poisson equations with residual-controlled local Schwarz steps.

    Each Schwarz iteration solves the owned cells of every tile with local CG
    while holding that tile's ``g = guard_cells`` halo fixed as Dirichlet data.
    Search directions are zero in the guard cells, and CG reductions cover only
    the three owned spatial axes, so no global Krylov solve is formed.
    Owned points on global conducting walls are grounded and excluded from the
    Krylov updates and convergence residuals.

    Each parallel Schwarz update is averaged with the previous iterate before
    the potential halos are refreshed. This fixed under-relaxation damps the
    interface two-cycle of an undamped parallel update without changing the
    converged Poisson solution. ``schwarz_residual`` is the maximum absolute
    residual in the ``g``-cell-wide owned slabs adjacent to tile interfaces.
    ``phi_tiles`` remains the previous-timestep warm start, and neither the
    potential nor a CG search direction is assembled into a global field.

    If requested, diagnostics are returned as ``(local_cg_residual,
    schwarz_residual, schwarz_iteration)``. The local residual contains one L2
    norm per tile for the undamped local solve; the Schwarz residual describes
    the accepted relaxed iterate, and it and the iteration count are scalars.
    """

    g = int(static_parameters.guard_cells)
    phi_tiles = _apply_tiled_phi_zero_boundaries(
        phi_tiles,
        static_parameters,
        g,
    )
    residual = _free_poisson_residual(
        rho_tiles,
        phi_tiles,
        static_parameters,
        dynamic_parameters,
        g,
    )
    local_cg_residual = jnp.sqrt(
        jnp.sum(residual * residual, axis=(-3, -2, -1))
    )
    interface_mask, _ = _schwarz_interface_mask(
        residual,
        static_parameters,
        g,
    )
    interior_residual = jnp.where(interface_mask, 0.0, residual)
    initial_interior_rr = jnp.sum(
        interior_residual * interior_residual,
        axis=(-3, -2, -1),
    )
    initial_local_solve_needed = jnp.any(
        initial_interior_rr > local_cg_tol**2
    )
    schwarz_residual = _schwarz_interface_residual(
        residual,
        static_parameters,
        g,
    )
    schwarz_iteration = jnp.asarray(0, dtype=jnp.int32)
    schwarz_relaxation = 0.5

    def schwarz_not_converged(state):
        _, _, schwarz_residual, schwarz_iteration = state
        initial_interior_not_converged = (
            (schwarz_iteration == 0) & initial_local_solve_needed
        )
        return (
            initial_interior_not_converged
            | (schwarz_residual > schwarz_tol)
        ) & (schwarz_iteration < schwarz_max_iterations)

    def schwarz_sweep(state):
        phi_tiles, _, _, schwarz_iteration = state
        local_phi_tiles, local_cg_residual = _local_tile_cg_solve(
            rho_tiles,
            phi_tiles,
            static_parameters,
            dynamic_parameters,
            g,
            local_cg_tol,
            local_cg_max_iterations,
        )
        # Parallel Dirichlet tile solves can alternate between two interface
        # states. Average old and new potentials to damp that mode while
        # preserving the fixed point of the discrete Poisson equation.
        phi_tiles = phi_tiles + schwarz_relaxation * (
            local_phi_tiles - phi_tiles
        )
        phi_tiles = _apply_tiled_phi_zero_boundaries(
            phi_tiles,
            static_parameters,
            g,
        )
        residual = _free_poisson_residual(
            rho_tiles,
            phi_tiles,
            static_parameters,
            dynamic_parameters,
            g,
        )
        schwarz_residual = _schwarz_interface_residual(
            residual,
            static_parameters,
            g,
        )

        return (
            phi_tiles,
            local_cg_residual,
            schwarz_residual,
            schwarz_iteration + 1,
        )

    phi_tiles, local_cg_residual, schwarz_residual, schwarz_iteration = lax.while_loop(
        schwarz_not_converged,
        schwarz_sweep,
        (
            phi_tiles,
            local_cg_residual,
            schwarz_residual,
            schwarz_iteration,
        ),
    )

    if not return_diagnostics:
        return phi_tiles

    diagnostics = (
        local_cg_residual,
        schwarz_residual,
        schwarz_iteration,
    )
    return phi_tiles, diagnostics


def _centered_tiled_electrostatic_gradient(phi_tiles, static_parameters, dynamic_parameters, g):
    """
    Compute ``E = -grad(phi)`` on compact scalar tiles.

    The potential halos must already contain neighboring tile and physical
    boundary values. The returned vector halos are refreshed for particle
    interpolation.
    """

    dx = dynamic_parameters.dx
    dy = dynamic_parameters.dy
    dz = dynamic_parameters.dz
    g = int(g)
    active = slice(g, -g)
    forward = slice(g + 1, None if g == 1 else -g + 1)
    backward = slice(g - 1, -g - 1)

    phi_tiles = _apply_tiled_phi_zero_boundaries(phi_tiles, static_parameters, g)

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

    return ghost_cells.update_tiled_vector_ghost_cells((Ex, Ey, Ez), static_parameters, g)


def calculate_electrostatic_fields(
    static_parameters,
    dynamic_parameters,
    particles,
    species_config,
    rho_tiles,
    phi_tiles,
):
    """
    Deposit charge and solve Poisson directly in compact tiled field storage.

    The supplied potential is the previous-timestep warm start. The local
    Schwarz solver exchanges only halos and never assembles a global field or
    forms a CG reduction across tiles.
    """

    g = static_parameters.guard_cells
    rho_tiles = compute_rho(particles, species_config, rho_tiles, static_parameters, dynamic_parameters)
    phi_tiles = solve_poisson_with_tiled_local_schwarz(
        rho_tiles,
        phi_tiles,
        static_parameters,
        dynamic_parameters,
        schwarz_tol=static_parameters.electrostatic_schwarz_tol,
        schwarz_max_iterations=static_parameters.electrostatic_schwarz_max_iterations,
        local_cg_tol=static_parameters.electrostatic_local_cg_tol,
        local_cg_max_iterations=static_parameters.electrostatic_local_cg_max_iterations,
    )
    # The final Schwarz sweep returns refreshed halos for post-solve filtering.

    alpha = dynamic_parameters.alpha
    phi_tiles = digital_filter(phi_tiles, alpha, num_guard_cells=g)
    phi_tiles = _apply_tiled_phi_zero_boundaries(phi_tiles, static_parameters, g)
    # preserve the established solve -> filter -> halo refresh ordering

    E_tiles = _centered_tiled_electrostatic_gradient(phi_tiles, static_parameters, dynamic_parameters, g)

    return E_tiles, phi_tiles, rho_tiles
