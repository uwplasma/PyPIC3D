"""Tile-native Proca fields: E/phi at n and A at n - 1/2.

A and E use electric Yee edges; phi uses (C, C, C) nodes. The forward
scalar gradient and backward vector divergence form a compatible pair.
``dark_mu`` is an inverse length, not electromagnetic permeability.

The live dark state (E, A, phi) always carries refreshed halos, so the
operators below read their inputs directly. Gradient, divergence and B are
interior-only intermediates with zero halos; ``synchronized_dark_fields``
refreshes B because particle gathers and output read its guard cells.
"""

from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.solvers.dark_matter_yee.boundaries import dark_field_boundaries
from PyPIC3D.solvers.yee.first_order_yee import yee_curl_e_to_b, yee_curl_b_to_e


def _interior(static_parameters):
    g = static_parameters.guard_cells
    return (slice(None),) * 3 + (slice(g, -g),) * 3


def dark_gradient(phi, static_parameters, dynamic_parameters):
    """Return +grad(phi) on electric edges with zero halos; phi halos must be fresh."""

    interior = _interior(static_parameters)
    g = static_parameters.guard_cells
    result = []
    for axis, spacing in enumerate((dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)):
        forward = list(interior)
        forward[axis + 3] = slice(g + 1, None if g == 1 else -g + 1)
        derivative = (phi[tuple(forward)] - phi[interior]) / spacing
        result.append(jnp.zeros_like(phi).at[interior].set(derivative))

    return tuple(result)

def dark_divergence_derivatives(A, static_parameters, dynamic_parameters):
    """Return the three backward edge-to-node derivatives; A halos must be fresh."""

    interior = _interior(static_parameters)
    g = static_parameters.guard_cells
    derivatives = []
    for axis, spacing in enumerate((dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)):
        backward = list(interior)
        backward[axis + 3] = slice(g - 1, -g - 1)
        derivatives.append((A[axis][interior] - A[axis][tuple(backward)]) / spacing)

    return tuple(derivatives)


def dark_divergence(A, static_parameters, dynamic_parameters):
    """Return the backward edge-to-node divergence with zero halos."""
    derivatives = dark_divergence_derivatives(A, static_parameters, dynamic_parameters)
    return jnp.zeros_like(A[0]).at[_interior(static_parameters)].set(sum(derivatives))

def compute_dark_B(A_n, static_parameters, dynamic_parameters):
    """Compute curl(A) on magnetic Yee faces with zero halos."""
    curl = yee_curl_e_to_b(A_n, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    return tuple(jnp.zeros_like(a).at[interior].set(c) for a, c in zip(A_n, curl))

def update_dark_E(E_n, B_half, A_half, J_half, static_parameters, dynamic_parameters, dt):
    """Kick E from n to n+1 with A, B and deposited J at n+1/2."""
    curl = yee_curl_b_to_e(B_half, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    c2 = dynamic_parameters.C**2
    mass2 = static_parameters.dark_mu**2
    source = static_parameters.sin_chi / dynamic_parameters.eps
    E = tuple(e.at[interior].add(dt * (c2 * (b + mass2 * a[interior]) + source * j[interior]))
              for e, b, a, j in zip(E_n, curl, A_half, J_half))

    E = dark_field_boundaries(E, static_parameters)
    # update E to full ghost-celled tiles, including the electric Yee edges

    return E

def update_dark_A(E_n, A_half, phi_n, static_parameters, dynamic_parameters, dt):
    """Drift A with E/phi at the intervening integer time."""
    gradient = dark_gradient(phi_n, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    A = tuple(a.at[interior].add(-dt * (e[interior] + grad[interior]))
              for a, e, grad in zip(A_half, E_n, gradient))
    A = dark_field_boundaries(A, static_parameters)
    # update A to full ghost-celled tiles, including the electric Yee edges

    return A


def update_dark_phi(A_half, phi_n, static_parameters, dynamic_parameters, dt):
    """Kick phi from n to n+1 with A at n+1/2."""
    divergence = dark_divergence(A_half, static_parameters, dynamic_parameters)
    interior = _interior(static_parameters)
    phi = phi_n.at[interior].add(-dt * dynamic_parameters.C**2 * divergence[interior])
    phi = dark_field_boundaries(phi, static_parameters, "phi")
    # update phi to full ghost-celled tiles on the (C, C, C) nodes

    return phi


@partial(jax.jit, static_argnames="static_parameters")
def synchronized_dark_fields(dark_fields, static_parameters, dynamic_parameters, pml_state=None):
    """Return (E, A, phi, B) at integer time for gathering and diagnostics.

    Without PML the half drift equals the average of adjacent A levels.
    With PML it previews A/B and the ADE histories over half a stage. Neither
    reconstruction modifies the live state.
    ``pml_state`` is the dark-sector state (runtime ``fields[6][1]``).
    """
    E, A, phi = dark_fields
    if pml_state is not None:
        from PyPIC3D.solvers.dark_matter_yee.pml import drift_dark_pml
        A, preview = drift_dark_pml(
            dark_fields, pml_state, static_parameters, dynamic_parameters, dynamic_parameters.dt / 2,
        )
        return E, A, phi, preview[0]
    A = update_dark_A(E, A, phi, static_parameters, dynamic_parameters, dynamic_parameters.dt / 2)
    B = dark_field_boundaries(compute_dark_B(A, static_parameters, dynamic_parameters), static_parameters, "B")
    # refresh B halos for the particle gather and field output
    return E, A, phi, B


def initialize_dark_photon_fields(static_parameters, dynamic_parameters, E=None, A=None, phi=None):
    """Initialize physical tiled data at t=0 and seed A onto t=-dt/2.

    Missing components default to zero. Nonzero data must satisfy
    div(E) + dark_mu**2 * phi = -sin_chi * rho / eps for a constraint-compatible
    initial state; this initializer does not solve that constraint.
    """
    from PyPIC3D.initialization import build_tiled_array

    zero = build_tiled_array(static_parameters, dynamic_parameters)
    if E is None and A is None and phi is None:
        # Zero satisfies the periodic/PEC halos and the half-step seed.
        return (zero, zero, zero), (zero, zero, zero), zero

    def scalar(value):
        value = zero if value is None else jnp.asarray(value)
        if value.shape != zero.shape or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError(f"Dark fields require real floating tiled arrays with shape {zero.shape}")
        return value

    def vector(value):
        if value is None:
            value = (zero, zero, zero)
        if len(value) != 3:
            raise ValueError("Dark vectors require three components")
        return tuple(scalar(v) for v in value)

    E = dark_field_boundaries(vector(E), static_parameters)
    # refresh E to full ghost-celled tiles, including the electric Yee edges
    phi = dark_field_boundaries(scalar(phi), static_parameters, "phi")
    # refresh phi to full ghost-celled tiles on the (C, C, C) nodes
    A = dark_field_boundaries(vector(A), static_parameters)
    A = update_dark_A(E, A, phi, static_parameters, dynamic_parameters, -dynamic_parameters.dt / 2)
    # seed A to t=-dt/2 with a half drift, which also refreshes its halos
    return E, A, phi
