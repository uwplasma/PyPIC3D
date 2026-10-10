"""Coordinate-stretched Maxwell--Proca leapfrog, including longitudinal waves.

The auxiliary state is (B_half, grad_memory, faraday_memory, div_memory,
ampere_memory, profiles). Gradient/Faraday histories live at half times;
divergence/Ampere histories live at integer times. The 18 histories have no
halos: they attach to derivatives on each tile's physical interior.
"""

import jax.numpy as jnp

from PyPIC3D.boundary_conditions.PML import tile_pml_profiles
from PyPIC3D.solvers.yee.first_order_yee import (
    yee_derivatives_e_to_b, yee_derivatives_b_to_e, assemble_yee_curl,
)
from PyPIC3D.solvers.dark_matter_yee.boundaries import dark_field_boundaries
from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import (
    _interior, dark_gradient, dark_divergence_derivatives, synchronized_dark_fields,
)


def stretch_dark_derivatives(derivatives, memory, sigma, axes, dt):
    """Integrate each ADE with a derivative held at the stage midpoint.

    psi_dot = -sigma*(psi + derivative). Use the interval average of
    derivative + psi in the field update, not its end-of-stage value. This
    centers the damping in time and has the exact zero-conductivity limit.
    """
    stretched, updated = [], []
    for derivative, previous, axis in zip(derivatives, memory, axes):
        q = sigma[axis] * dt
        change = jnp.expm1(-q)
        average = jnp.where(q == 0, 1., -change / jnp.where(q == 0, 1., q))
        stretched.append(average * (derivative + previous))
        updated.append(jnp.exp(-q) * previous + change * derivative)
    return tuple(stretched), tuple(updated)


def initialize_dark_pml(dark_fields, static, dynamic, pml_profiles):
    """Reseed physical t=0 fields with zero ADE histories at t=0.

    The ordinary initializer has already seeded A backwards by half a step.
    Undo that seed, then seed both A/B and the half-time histories with the
    same PML drift used in evolution and in diagnostic reconstruction.
    """
    E, A, phi, B = synchronized_dark_fields(dark_fields, static, dynamic)
    interior = _interior(static)
    zero = jnp.zeros_like(phi[interior])

    # Conductivity is sampled on C nodes by the existing profile builder.
    # Extend it constantly to the exterior endpoint before averaging to V;
    # zero-filled profile ghosts must not halve sigma at the outer wall.
    extended = []
    for axis, profile in enumerate(pml_profiles):
        for endpoint, neighbor in ((0, 1), (-1, -2)):
            wall, adjacent = [slice(None)] * 3, [slice(None)] * 3
            wall[axis], adjacent[axis] = endpoint, neighbor
            profile = profile.at[tuple(wall)].set(profile[tuple(adjacent)])
        extended.append(profile)
    profiles = tile_pml_profiles(static, tuple(extended), static.tile_shape)
    sigma_c = tuple(profile[interior] for profile in profiles)
    sigma_v = []
    g = static.guard_cells
    for axis, profile in enumerate(profiles):
        forward = list(interior)
        forward[axis + 3] = slice(g + 1, None if g == 1 else -g + 1)
        sigma_v.append((profile[interior] + profile[tuple(forward)]) / 2)

    state = B, (zero,) * 3, (zero,) * 6, (zero,) * 3, (zero,) * 6, (sigma_c, tuple(sigma_v))
    A, state = drift_dark_pml((E, A, phi), state, static, dynamic, -dynamic.dt / 2)
    return (E, A, phi), state


def drift_dark_pml(dark_fields, pml_state, static, dynamic, dt):
    """Drift A and B between half times using intervening E/phi.

    A positive half drift on a copy reconstructs integer-time fields without
    advancing live histories. The negative half drift seeds physical data.
    """
    E, A, phi = dark_fields
    B, grad_memory, faraday_memory, div_memory, ampere_memory, profiles = pml_state
    sigma_c, sigma_v = profiles
    interior = _interior(static)

    gradient = tuple(value[interior] for value in dark_gradient(phi, static, dynamic))
    gradient, grad_memory = stretch_dark_derivatives(gradient, grad_memory, sigma_v, (0, 1, 2), dt)
    derivatives = yee_derivatives_e_to_b(E, static, dynamic)
    derivatives, faraday_memory = stretch_dark_derivatives(
        derivatives, faraday_memory, sigma_v, (1, 2, 2, 0, 0, 1), dt,
    )
    curl = assemble_yee_curl(derivatives)

    A = tuple(a.at[interior].add(-dt * (e[interior] + grad)) for a, e, grad in zip(A, E, gradient))
    B = tuple(b.at[interior].add(-dt * value) for b, value in zip(B, curl))
    A = dark_field_boundaries(A, static)
    B = dark_field_boundaries(B, static, "B")
    return A, (B, grad_memory, faraday_memory, div_memory, ampere_memory, profiles)


def advance_dark_pml(dark_fields, J, pml_state, static, dynamic):
    """Drift A/B, then kick E/phi with the centered deposited current."""
    E, A, phi = dark_fields
    dt = dynamic.dt
    A, pml_state = drift_dark_pml(dark_fields, pml_state, static, dynamic, dt)
    B, grad_memory, faraday_memory, div_memory, ampere_memory, profiles = pml_state
    sigma_c, sigma_v = profiles
    interior = _interior(static)

    derivatives = dark_divergence_derivatives(A, static, dynamic)
    derivatives, div_memory = stretch_dark_derivatives(derivatives, div_memory, sigma_c, (0, 1, 2), dt)
    phi = phi.at[interior].add(-dt * dynamic.C**2 * sum(derivatives))
    derivatives = yee_derivatives_b_to_e(B, static, dynamic)
    derivatives, ampere_memory = stretch_dark_derivatives(
        derivatives, ampere_memory, sigma_c, (1, 2, 2, 0, 0, 1), dt,
    )
    curl = assemble_yee_curl(derivatives)
    E = tuple(e.at[interior].add(dt * (dynamic.C**2 * (value + static.dark_mu**2 * a[interior])
                                      + static.sin_chi * j[interior] / dynamic.eps))
              for e, a, j, value in zip(E, A, J, curl))
    E = dark_field_boundaries(E, static)
    phi = dark_field_boundaries(phi, static, "phi")
    return (E, A, phi), (B, grad_memory, faraday_memory, div_memory, ampere_memory, profiles)
