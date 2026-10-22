"""Split evolution of PIC, Maxwell fields, and the BSSN geometry."""

import jax
import jax.numpy as jnp

from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs_with_matter,
    enforce_algebraic_constraints,
)
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.particles.particle_tile_communication import refresh_tiled_particle_tiles
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.dynamical_relativity import (
    build_bssn_metric,
    rephase_pic_state,
    synchronize_particles,
    synchronized_magnetic_field,
)
from PyPIC3D.relativity.energy_momentum_tensor import (
    compute_energy_momentum_tensor,
    electromagnetic_moments,
)

from .curved_maxwell import compute_covariant_E, compute_covariant_H, update_B, update_D


__all__ = ["advance_pic_half_step", "time_loop_dynamical_relativity"]


def advance_pic_half_step(
    particles, species_config, fields, static_parameters, dynamic_parameters,
):
    """Advance one retained PIC spacing h=dt/2 with the metric held fixed.

    D is at integer h times, while B, covariant particle momentum, and J
    lag by h/2. The five-entry history also carries a separate synchronized B
    for the gravitational sources. D/B/J are native coordinate densities.
    """
    D_n, B_n_minushalf, J_n_minushalf, rho, phi, external, metric, history, overflow = fields
    D_previous, B_previous, _, half_dt, absorber = history
    dynamic = dynamic_parameters._replace(dt=half_dt)
    d_absorber, b_absorber = (None, None) if absorber is None else absorber

    # Predictor fields at the particle position time. Retain incoming history
    # until the synchronized endpoint B has been reconstructed below.
    D_n_minushalf = tuple(0.5 * (d + old) for d, old in zip(D_n, D_previous))
    B_n_minusone = tuple(0.5 * (b + old) for b, old in zip(B_n_minushalf, B_previous))
    E_n_minushalf = compute_covariant_E(D_n_minushalf, B_n_minushalf, metric)
    B_n = update_B(
        E_n_minushalf, B_n_minusone, metric, static_parameters, dynamic,
        half_dt, absorber=b_absorber,
    )

    particles_n = particles
    if particles.active.size:
        particles, centered_particles = hybrid_boris_geodesic_push(
            particles, species_config, D_n, B_n, metric, static_parameters, dynamic,
        )
        if static_parameters.current_deposition == "esirkepov":
            # Deposit true endpoint displacements before wrapping or retile.
            J_n_plushalf = Esirkepov_current(
                particles_n, particles, species_config, J_n_minushalf,
                static_parameters, dynamic,
            )
            centered_overflow = False
        else:
            centered_particles, centered_overflow = refresh_tiled_particle_tiles(
                centered_particles, static_parameters, dynamic,
            )
            J_n_plushalf = GR_direct_deposition(
                centered_particles, species_config, J_n_minushalf, metric,
                static_parameters, dynamic,
            )
        particles, endpoint_overflow = refresh_tiled_particle_tiles(
            particles, static_parameters, dynamic,
        )
        overflow = overflow | centered_overflow | endpoint_overflow
    else:
        # The storage shape is static under JIT. Vacuum calls skip all particle
        # kernels and erase stale currents at both retained time levels.
        J_n_plushalf = tuple(jnp.zeros_like(j) for j in J_n_minushalf)
        J_n_minushalf = J_n_plushalf

    E_n = compute_covariant_E(D_n, B_n, metric)
    H_n = compute_covariant_H(D_n, B_n, metric)
    B_n_plushalf = update_B(
        E_n, B_n_minushalf, metric, static_parameters, dynamic,
        half_dt, absorber=b_absorber,
    )
    J_n = tuple(0.5 * (new + old) for new, old in zip(J_n_plushalf, J_n_minushalf))
    D_n_plushalf = update_D(
        D_n_minushalf, H_n, J_n, metric, static_parameters, dynamic,
        half_dt, absorber=d_absorber,
    )
    H_n_plushalf = compute_covariant_H(D_n_plushalf, B_n_plushalf, metric)
    # Final Ampere uses the endpoint Esirkepov current, preserving the same
    # discrete continuity/Gauss identity as the prescribed-metric solver.
    D_n_plusone = update_D(
        D_n, H_n_plushalf, J_n_plushalf, metric, static_parameters, dynamic,
        half_dt, absorber=d_absorber,
    )

    updated_fields = (
        D_n_plusone, B_n_plushalf, J_n_plushalf, rho, phi, external, metric,
        (D_n, B_n_minushalf, history[2], half_dt, absorber), overflow,
    )
    B_synchronized = synchronized_magnetic_field(
        fields, updated_fields, static_parameters, dynamic,
    )
    updated_fields = updated_fields[:7] + (
        (D_n, B_n_minushalf, B_synchronized, half_dt, absorber), overflow,
    )
    return particles, updated_fields


def time_loop_dynamical_relativity(
    particles, species_config, fields, static_parameters, dynamic_parameters,
    *, prescribed_gauge=None, gravity_boundary=None,
):
    """Advance PIC(h), BSSN(dt), PIC(h), with h=dt/2 and retained histories.

    The caller JITs this loop with static parameters and optional pure
    callbacks closed over. ``prescribed_gauge(state,time)`` returns lapse,
    shift, dt_lapse, dt_shift; ``gravity_boundary(state,time)`` changes only
    BSSN boundary data. Keep dt fixed after initializing the staggered state.

    At each RK stage, gravity sees the same midpoint positions, covariant
    momenta, and densitized EM moments through that stage's spatial metric.
    BSSN is RK4; the coupled split scheme is second order in time.
    """
    bssn = dynamic_parameters.bssn_parameters
    dt, time = dynamic_parameters.dt, bssn.time
    evolution_parameters = bssn.evolution_parameters

    def prepare_geometry(state, stage_time):
        if gravity_boundary is not None:
            state = gravity_boundary(state, stage_time)
        if prescribed_gauge is not None:
            lapse, shift, _, _ = prescribed_gauge(state, stage_time)
            state = state._replace(lapse=lapse, shift=shift)
        return enforce_algebraic_constraints(state)

    state = prepare_geometry(bssn.state, time)
    # Usually this is the previously stored metric. Boundary/gauge callbacks
    # can change it, so align retained histories before the first PIC step.
    metric = build_bssn_metric(state, static_parameters, dynamic_parameters)
    particles, fields = rephase_pic_state(
        particles, species_config, fields, metric, static_parameters, dynamic_parameters,
    )
    particles, fields = advance_pic_half_step(
        particles, species_config, fields, static_parameters, dynamic_parameters,
    )

    midpoint_particles = synchronize_particles(
        particles, species_config, fields, static_parameters, dynamic_parameters,
    )
    moments = electromagnetic_moments(fields[0], fields[7][2], static_parameters)

    def gravity_rhs(stage_state, stage_time):
        rho, momentum, stress = compute_energy_momentum_tensor(
            midpoint_particles, species_config, moments, stage_state,
            static_parameters, dynamic_parameters,
        )
        rhs = compute_bssn_rhs_with_matter(
            stage_state, evolution_parameters,
            bssn.gravity_G * rho, bssn.gravity_G * momentum, bssn.gravity_G * stress,
        )
        if prescribed_gauge is not None:
            _, _, lapse_dot, shift_dot = prescribed_gauge(stage_state, stage_time)
            rhs = rhs._replace(lapse=lapse_dot, shift=shift_dot)
        return rhs

    # Each intermediate geometry is projected before evaluating its sources.
    k1 = gravity_rhs(state, time)
    stage2 = jax.tree_util.tree_map(lambda q, k: q + 0.5 * dt * k, state, k1)
    stage2 = prepare_geometry(stage2, time + 0.5 * dt)
    k2 = gravity_rhs(stage2, time + 0.5 * dt)

    stage3 = jax.tree_util.tree_map(lambda q, k: q + 0.5 * dt * k, state, k2)
    stage3 = prepare_geometry(stage3, time + 0.5 * dt)
    k3 = gravity_rhs(stage3, time + 0.5 * dt)

    stage4 = jax.tree_util.tree_map(lambda q, k: q + dt * k, state, k3)
    stage4 = prepare_geometry(stage4, time + dt)
    k4 = gravity_rhs(stage4, time + dt)

    final_state = jax.tree_util.tree_map(
        lambda q, a, b, c, d: q + dt * (a + 2 * b + 2 * c + d) / 6,
        state, k1, k2, k3, k4,
    )
    final_state = prepare_geometry(final_state, time + dt)

    metric = build_bssn_metric(final_state, static_parameters, dynamic_parameters)
    particles, fields = rephase_pic_state(
        particles, species_config, fields, metric, static_parameters, dynamic_parameters,
        synchronized_particles=midpoint_particles,
    )
    particles, fields = advance_pic_half_step(
        particles, species_config, fields, static_parameters, dynamic_parameters,
    )
    dynamic_parameters = dynamic_parameters._replace(
        bssn_parameters=bssn._replace(state=final_state, time=time + dt),
    )
    return particles, fields, dynamic_parameters
