"""Split evolution of PIC, Maxwell fields, and the BSSN geometry."""

import jax

from JAX_BSSN.evolution.time_evolve import (
    compute_bssn_rhs_with_matter,
    enforce_algebraic_constraints,
)
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

from .time_loop import time_loop_static_metric


__all__ = ["time_loop_dynamical_relativity"]


def time_loop_dynamical_relativity(
    particles, species_config, fields, static_parameters, dynamic_parameters,
    *, prescribed_gauge=None, gravity_boundary=None, source_filter="none",
):
    """Advance PIC(h), BSSN(dt), PIC(h), with h=dt/2 and retained histories.

    The caller JITs this loop with static parameters and optional pure
    callbacks closed over. ``prescribed_gauge(state,time)`` returns lapse,
    shift, dt_lapse, dt_shift; ``gravity_boundary(state,time)`` changes only
    BSSN boundary data. Keep dt fixed after initializing the staggered state.

    At each RK stage, gravity sees the same midpoint positions, covariant
    momenta, and densitized EM moments through that stage's spatial metric.
    BSSN is RK4; the coupled split scheme is second order in time.
    ``source_filter`` optionally smooths electrovacuum stress-energy at every
    RK stage, leaving native fields, geometry and staggering unchanged.
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

    if static_parameters.gr_work_energy_theorem:
        from PyPIC3D.diagnostics.gr_work_energy_theorem import (
            sample_energy, start_tracker, advance_tracker,
        )
        # The initial callbacks may change geometry before the first push.
        # Establish E0 and its source sample only after matching that geometry.
        # Later intervals retain their previous endpoint, including any error
        # from a discontinuous user callback rather than resetting the budget.
        def prepared_baseline(_):
            prepared = dynamic_parameters._replace(bssn_parameters=bssn._replace(state=state))
            sample = sample_energy(particles, species_config, fields, static_parameters, prepared)
            return start_tracker(sample, static_parameters.gr_work_energy_reference)

        tracker = jax.lax.cond(dynamic_parameters.work_energy_tracker.steps == 0,
                               prepared_baseline, lambda _: dynamic_parameters.work_energy_tracker, None)

    # First PIC half-step with the initial metric. Retain the incoming
    # history to reconstruct B at the new particle position time.
    before = fields
    previous_D, previous_B, _, half_dt, absorber = before[7]
    pic_parameters = dynamic_parameters._replace(dt=half_dt)
    pic_fields = before[:7] + ((previous_D, previous_B), before[8])
    particles, fields = time_loop_static_metric(
        particles, species_config, pic_fields, static_parameters, pic_parameters,
        absorber=absorber,
    )
    common_B = synchronized_magnetic_field(before, fields, static_parameters, pic_parameters)
    history = (*fields[7], common_B, half_dt, absorber)
    fields = fields[:7] + (history, fields[8])

    midpoint_particles = synchronize_particles(
        particles, species_config, fields, static_parameters, dynamic_parameters,
    )
    moments = electromagnetic_moments(fields[0], fields[7][2], static_parameters)

    def gravity_rhs(stage_state, stage_time):
        rho, momentum, stress = compute_energy_momentum_tensor(
            midpoint_particles, species_config, moments, stage_state,
            static_parameters, dynamic_parameters,
            source_filter=source_filter,
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

    # Second PIC half-step with the updated metric and corrected histories.
    before = fields
    previous_D, previous_B, _, half_dt, absorber = before[7]
    pic_parameters = dynamic_parameters._replace(dt=half_dt)
    pic_fields = before[:7] + ((previous_D, previous_B), before[8])
    particles, fields = time_loop_static_metric(
        particles, species_config, pic_fields, static_parameters, pic_parameters,
        absorber=absorber,
    )
    common_B = synchronized_magnetic_field(before, fields, static_parameters, pic_parameters)
    history = (*fields[7], common_B, half_dt, absorber)
    fields = fields[:7] + (history, fields[8])
    dynamic_parameters = dynamic_parameters._replace(
        bssn_parameters=bssn._replace(state=final_state, time=time + dt),
    )
    if static_parameters.gr_work_energy_theorem:
        sample = sample_energy(particles, species_config, fields, static_parameters, dynamic_parameters)
        dynamic_parameters = dynamic_parameters._replace(
            work_energy_tracker=advance_tracker(tracker, sample))
    return particles, fields, dynamic_parameters
