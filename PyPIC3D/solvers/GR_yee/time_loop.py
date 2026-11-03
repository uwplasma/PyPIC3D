import jax
import jax.numpy as jnp

from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.diagnostics.static_metric import step_diagnostics
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.filtering import filter_gr_grid, validate_gr_filter
from PyPIC3D.utilities.field_helpers import add_external_fields

from .curved_maxwell import compute_covariant_E, compute_covariant_H, update_B, update_D


__all__ = ["time_loop_static_metric"]


def time_loop_static_metric(
    particles,
    species_config,
    fields,
    static_parameters,
    dynamic_parameters,
    return_diagnostics=False,
    current_transform=None,
    *,
    absorber=None,
    pair_injector=None,
    injection_key=None,
    step=0,
    injection_interval_steps=1,
):
    """
    Advance a tiled PIC system in a prescribed 3+1 metric.

    The D/B/J slots, external D/B and previous D/B are native densities
    ``sqrt(gamma) V^i``; the particle push divides by the grid-node volumes.
    The particle velocity slot stores covariant spatial components ``u_i``.
    An optional ``current_transform`` may apply a compatible filter to the
    densitized current. Its caller must use the same commuting transform for
    deposited charge when checking Gauss's law; arbitrary current-only
    smoothing is not valid.
    An optional pair_injector(particles, species, fields, static, dynamic,
    key, event) returns (particles, next_key, (requested, inserted)). Injection runs
    before the push at step zero and every injection_interval_steps thereafter.
    Pass step and injection_key as JAX inputs; close over the callable and interval.
    Enabled calls append (next_key, counts) to the usual outputs. The caller must
    reject any nonzero requested - inserted counts on the host.
    E/H use target-site lapse-metric coefficients and unlimited upstream
    reconstruction of the shift cross products, with velocity -beta. At least
    two guards are required. Native curls and time staggering are unchanged.
    Empty particle storage takes a vacuum step, skipping the particle push,
    deposition, current transform, and particle tile communication.
    """

    if static_parameters.guard_cells < 2:
        raise ValueError("Upstream shift reconstruction requires guard_cells >= 2")
    validate_gr_filter(static_parameters)
    if current_transform is not None and static_parameters.current_filter != "none":
        raise ValueError("Use either matched GR filter_j or current_transform, not both")

    if pair_injector is not None:
        if static_parameters.solver != "static_metric":
            raise ValueError("Pair injection requires solver='static_metric'")
        if injection_key is None or injection_interval_steps < 1:
            raise ValueError("Pair injection requires an RNG key and a positive step interval")

        def inject(_):
            return pair_injector(
                particles, species_config, fields, static_parameters,
                dynamic_parameters, injection_key, step // injection_interval_steps)

        def skip(_):
            zero = jnp.zeros(particles.active.shape[:3], dtype=jnp.int32)
            return particles, injection_key, (zero, zero)

        particles, injection_key, injection_counts = jax.lax.cond(
            step % injection_interval_steps == 0, inject, skip, operand=None)

    D_n, B_n_minushalf, J_n_minushalf, rho, phi, external_fields, metric, previous_fields, overflow_previous = fields
    # unpack the fixed-metric field state

    d_absorber, b_absorber = (None, None) if absorber is None else absorber

    D_n_minusone, B_n_minusthreehalves = previous_fields
    # unpack the previous fixed-metric field state.  Every D and B arrives with
    # refreshed halos and walls from update_D/B or initialization.

    D_n_minushalf = tuple( 0.5 * (D_n[i] + D_n_minusone[i]) for i in range(3) )
    B_n_minusone = tuple( 0.5 * (B_n_minushalf[i] + B_n_minusthreehalves[i]) for i in range(3) )
    # compute the centered fields for the current time step

    E_n_minusonehalf = compute_covariant_E(
        D_n_minushalf, B_n_minushalf, metric)
    # compute the covariant electric field from the centered displacement and magnetic fields

    B_n = update_B(E_n_minusonehalf, B_n_minusone, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt, absorber=b_absorber)
    # update the densitized magnetic field using the centered displacement field

    particles_n = particles
    # keep the time level n positions for the charge-conserving deposition.  The
    # push does not retile and does not wrap, so particles_n and the pushed
    # particles share one tile frame and differencing their positions gives the
    # true displacement.

    if particles.active.size:
        push_D, push_B = add_external_fields(D_n, B_n, external_fields)
        # Evolved and external fields have already been refreshed.
        particles, centered_particles = hybrid_boris_geodesic_push(
            particles, species_config, push_D, push_B, metric,
            static_parameters, dynamic_parameters,
        )

        if static_parameters.current_deposition == "esirkepov":
            # Endpoint deposition must precede wrapping and full-step migration.
            # Midpoint particles are only needed by direct deposition.
            centered_overflow = False
            J_n_plushalf = Esirkepov_current(
                particles_n, particles, species_config, J_n_minushalf,
                static_parameters, dynamic_parameters,
            )
            J_n_plushalf = tuple(filter_gr_grid(
                value, static_parameters, dynamic_parameters, tiled=True) for value in J_n_plushalf)
        else:
            centered_particles, centered_overflow = refresh_tiled_particle_tiles(
                centered_particles, static_parameters, dynamic_parameters,
            )
            J_n_plushalf = GR_direct_deposition(
                centered_particles, species_config, J_n_minushalf,
                metric, static_parameters, dynamic_parameters,
            )
        if current_transform is not None:
            J_n_plushalf = current_transform(J_n_plushalf)
    else:
        # Storage shape is static under JIT, so vacuum runs never trace the
        # particle kernels. Clear both time levels to exclude stale current.
        J_n_plushalf = tuple(jnp.zeros_like(component) for component in J_n_minushalf)
        J_n_minushalf = J_n_plushalf
        centered_overflow = False

    boundary_diagnostics = None
    if return_diagnostics:
        boundary_diagnostics = step_diagnostics(particles_n, particles, J_n_plushalf, species_config,
                                                static_parameters, dynamic_parameters)
    fullstep_overflow = False
    if particles.active.size:
        particles, fullstep_overflow = refresh_tiled_particle_tiles(
            particles,
            static_parameters,
            dynamic_parameters,
        )
    overflow = overflow_previous | centered_overflow | fullstep_overflow
    # apply boundaries and restore full-step tile ownership for the next push while preserving all overflow events


    E_n = compute_covariant_E(
        D_n, B_n, metric)
    # compute the covariant electric field from the updated displacement and magnetic fields
    H_n = compute_covariant_H(
        D_n, B_n, metric)
    # compute the covariant magnetic field from the updated displacement and magnetic fields

    B_n_plushalf = update_B(E_n, B_n_minushalf, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt, absorber=b_absorber)
    # update the densitized magnetic field using the updated displacement field

    J_n = tuple( 0.5 * (J_n_plushalf[i] + J_n_minushalf[i]) for i in range(3))
    # compute the centered current for the current time step

    D_n_plushalf = update_D(D_n_minushalf, H_n, J_n, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt, absorber=d_absorber)
    # update the densitized displacement field using the updated magnetic field and current

    H_n_plushalf = compute_covariant_H(
        D_n_plushalf, B_n_plushalf, metric)
    # compute the covariant magnetic field from the updated displacement and magnetic fields

    D_n_plusone = update_D(D_n, H_n_plushalf, J_n_plushalf, metric, static_parameters, dynamic_parameters, dynamic_parameters.dt, absorber=d_absorber)
    # update the densitized displacement field using the updated magnetic field and current


    previous_fields = (D_n, B_n_minushalf)
    # store the current fields for the next time step

    fields = (
        D_n_plusone,
        B_n_plushalf,
        J_n_plushalf,
        rho,
        phi,
        external_fields,
        metric,
        previous_fields,
        overflow,
    )
    # pack the fixed-metric field state

    result = (particles, fields, boundary_diagnostics) if return_diagnostics else (particles, fields)
    return result + (injection_key, injection_counts) if pair_injector is not None else result
