"""Periodic initial metric solve for the warm electron-ion two-stream demo.

Called by initial_data.py after loading particles and solving Gauss's law.
"""

import math

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.constraints import compute_hamiltonian_constraint_with_matter
from PyPIC3D.relativity.filtering import filter_gr_grid


# Newton solve controls and proper-volume normalization tolerance.
MAX_NEWTON_ITERATIONS = 20
VOLUME_TOLERANCE = 1.0e-12


def solve_initial_metric(bssn, particle_energy, electromagnetic_energy, omega_pe,
                         *, constraint_tolerance=None,
                         max_newton_iterations=MAX_NEWTON_ITERATIONS,
                         static=None, dynamic=None):
    """Solve discrete Hamiltonian plus mean(W^-3)=1 on the expanding branch.

    Local orthonormal particle momenta are fixed initial-data inputs. Their
    Lorentz factors therefore remain fixed. Filtering coordinate densities
    gives rho(W)=W^3 F[rho_particle(1)+W*rho_EM(1)] for fixed densitized D.
    Omitting static/dynamic retains the unfiltered solve used by older callers.
    The production-source check verifies the same formula after conversion
    using the metric actually gathered by particles.
    """
    flat = bssn.state
    gravity_G = float(bssn.gravity_G)
    source_scale = float(16.0 * jnp.pi * gravity_G * jnp.mean(particle_energy + electromagnetic_energy))
    tolerance = (max(1.0e-11 * omega_pe**2, 1.0e-8 * source_scale)
                 if constraint_tolerance is None else float(constraint_tolerance))
    if gravity_G == 0.0:
        return flat, [0.0], [0.0], tolerance

    nx = flat.conformal_factor.size

    def state_and_density(unknown):
        W = jnp.exp(unknown[:-1]).reshape(flat.conformal_factor.shape)
        # Scale K^2 by the matter source to condition the last Newton column.
        K = -jnp.sqrt(source_scale * unknown[-1])
        state = flat._replace(conformal_factor=W, trace_K=jnp.full_like(W, K))
        coordinate_energy = particle_energy + W * electromagnetic_energy
        if static is not None:
            coordinate_energy = filter_gr_grid(coordinate_energy, static, dynamic)
        rho = W**3 * coordinate_energy
        return state, rho

    def residual(unknown):
        state, rho = state_and_density(unknown)
        hamiltonian = compute_hamiltonian_constraint_with_matter(
            state, bssn.evolution_parameters, gravity_G * rho,
        )
        volume = jnp.mean(state.conformal_factor**-3) - 1.0
        return jnp.concatenate((hamiltonian.ravel() / source_scale, jnp.asarray([volume])))

    evaluate = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual))
    unknown = jnp.concatenate((jnp.zeros(nx, dtype=jnp.float64), jnp.asarray([1.5])))
    residual_history, volume_history = [], []
    for iteration in range(max_newton_iterations + 1):
        value = evaluate(unknown)
        h_norm = float(jnp.max(jnp.abs(value[:-1]))) * source_scale
        volume_error = float(jnp.abs(value[-1]))
        residual_history.append(h_norm)
        volume_history.append(volume_error)
        if h_norm <= tolerance and volume_error <= VOLUME_TOLERANCE:
            return state_and_density(unknown)[0], residual_history, volume_history, tolerance
        if iteration == max_newton_iterations:
            break

        correction = jnp.linalg.solve(jacobian(unknown), -value)
        merit = max(h_norm / tolerance, volume_error / VOLUME_TOLERANCE)
        for backtrack in range(20):
            trial = unknown + (0.5**backtrack) * correction
            if float(trial[-1]) <= 0.0 or not bool(jnp.all(jnp.isfinite(trial))):
                continue
            trial_value = evaluate(trial)
            trial_merit = max(
                float(jnp.max(jnp.abs(trial_value[:-1]))) * source_scale / tolerance,
                float(jnp.abs(trial_value[-1])) / VOLUME_TOLERANCE,
            )
            if math.isfinite(trial_merit) and trial_merit < merit:
                unknown = trial
                break
        else:
            raise RuntimeError(f"Initial metric Newton step failed: H_linf={h_norm:.3e}, "
                               f"volume_error={volume_error:.3e}")
    raise RuntimeError(f"Initial metric did not converge: H_linf={h_norm:.3e}, "
                       f"volume_error={volume_error:.3e}")
