"""Cartesian, time-symmetric Hamiltonian solve for fixed charged dust.

Called by initial_data.py after loading particles and the spherical electric
field. JAX_BSSN supplies the discrete Hamiltonian and its spatial derivatives.
"""

import math

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from JAX_BSSN.bssn.constraints import compute_hamiltonian_constraint_with_matter
from JAX_BSSN.evolution.spatial_derivatives import diff1_physical


MAX_NEWTON_ITERATIONS = 20
BOUNDARY_TOLERANCE = 1.0e-10


def metric_residual(log_W, bssn, particle_energy, electromagnetic_energy):
    """Return Hamiltonian, asymptotic Robin residual, and outer-node mask.

    Fixed particles at rest have rho_dust = W^3 rho_dust_flat. Fixed
    densitized electric fields have rho_EM = W^4 rho_EM_flat. The asymptotic
    condition x.grad(psi) + psi - 1 = 0 permits an unknown ADM mass; at a
    finite outer boundary it neglects higher powers of 1/r, including charge.
    """
    W = jnp.exp(log_W)
    state = bssn.state._replace(conformal_factor=W)
    params = bssn.evolution_parameters
    rho = W**3 * particle_energy + W**4 * electromagnetic_energy
    hamiltonian = compute_hamiltonian_constraint_with_matter(state, params, rho)

    psi = W**-0.5
    robin = psi - 1.0
    boundary = jnp.zeros_like(W, dtype=bool)
    for axis, origin in enumerate((params.x_min, params.y_min, params.z_min)):
        shape = [1, 1, 1]
        shape[axis] = W.shape[axis]
        index = jnp.arange(W.shape[axis]).reshape(shape)
        coordinate = origin + index * params.dx
        robin = robin + coordinate * diff1_physical(psi, axis, params)
        boundary = boundary | (index == 0) | (index == W.shape[axis] - 1)
    return hamiltonian, robin, boundary


def solve_initial_metric(bssn, particle_energy, electromagnetic_energy,
                         *, constraint_tolerance=None,
                         max_newton_iterations=MAX_NEWTON_ITERATIONS):
    """Return solved BSSN state, H/Robin histories, and Hamiltonian tolerance.

    Newton corrections use autodiff Jacobian-vector products, keeping memory
    proportional to the grid rather than forming a dense 3D Jacobian.
    """
    dx = float(bssn.evolution_parameters.dx)
    length = dx * max(bssn.state.conformal_factor.shape)
    source_scale = max(float(16 * jnp.pi * jnp.max(
        jnp.abs(particle_energy) + jnp.abs(electromagnetic_energy))), 1.0 / length**2)
    tolerance = (max(1.0e-12 / length**2, 1.0e-8 * source_scale)
                 if constraint_tolerance is None else float(constraint_tolerance))

    def residual(unknown):
        hamiltonian, robin, boundary = metric_residual(
            unknown, bssn, particle_energy, electromagnetic_energy)
        return jnp.where(boundary, robin, hamiltonian / source_scale)

    evaluate = jax.jit(residual)
    unknown = jnp.log(bssn.state.conformal_factor)
    _, _, boundary = metric_residual(unknown, bssn, particle_energy, electromagnetic_energy)

    @jax.jit
    def newton_correction(unknown, value):
        _, jacobian_product = jax.linearize(residual, unknown)
        # Approximate diagonal of 4 W^2 Laplacian(log W) and the Robin rows.
        # This only scales GMRES; the residual and its derivative stay exact.
        W = jnp.exp(unknown)
        diagonal = jnp.where(boundary, -0.5 * W**-0.5 * (1 + length/dx),
                             -30 * W**2 / (dx**2 * source_scale))
        rhs = -value / diagonal
        rhs_norm = jnp.linalg.norm(rhs)
        # Normalize the linear problem even near Newton convergence; GMRES's
        # basis normalization otherwise loses very small right-hand sides.
        correction, _ = gmres(lambda vector: jacobian_product(vector) / diagonal,
                              rhs / rhs_norm, tol=1.0e-10, restart=100,
                              maxiter=100, solve_method="incremental")
        correction = correction * rhs_norm
        linear_error = jnp.linalg.norm(jacobian_product(correction) + value)
        return correction, linear_error

    residual_history, boundary_history = [], []
    for iteration in range(max_newton_iterations + 1):
        value = evaluate(unknown)
        h_norm = float(jnp.max(jnp.where(boundary, 0.0, jnp.abs(value)))) * source_scale
        boundary_error = float(jnp.max(jnp.where(boundary, jnp.abs(value), 0.0)))
        residual_history.append(h_norm)
        boundary_history.append(boundary_error)
        print(f"Newton {iteration:2d}: H_linf={h_norm:.3e}, Robin_linf={boundary_error:.3e}", flush=True)
        if h_norm <= tolerance and boundary_error <= BOUNDARY_TOLERANCE:
            return (bssn.state._replace(conformal_factor=jnp.exp(unknown)),
                    residual_history, boundary_history, tolerance)
        if iteration == max_newton_iterations:
            break

        correction, linear_error = newton_correction(unknown, value)
        if not math.isfinite(float(linear_error)) or float(linear_error) > 1.0e-6 * float(jnp.linalg.norm(value)):
            raise RuntimeError(f"Initial metric GMRES failed: residual={float(linear_error):.3e}")
        merit = max(h_norm / tolerance, boundary_error / BOUNDARY_TOLERANCE)
        for backtrack in range(20):
            trial = unknown + 0.5**backtrack * correction
            trial_value = evaluate(trial)
            trial_merit = max(
                float(jnp.max(jnp.where(boundary, 0.0, jnp.abs(trial_value)))) * source_scale / tolerance,
                float(jnp.max(jnp.where(boundary, jnp.abs(trial_value), 0.0))) / BOUNDARY_TOLERANCE)
            if bool(jnp.all(jnp.isfinite(trial_value))) and math.isfinite(trial_merit) and trial_merit < merit:
                unknown = trial
                break
        else:
            raise RuntimeError(f"Initial metric Newton step failed: H_linf={h_norm:.3e}, "
                               f"Robin_linf={boundary_error:.3e}")
    raise RuntimeError(f"Initial metric did not converge: H_linf={h_norm:.3e}, "
                       f"Robin_linf={boundary_error:.3e}")
