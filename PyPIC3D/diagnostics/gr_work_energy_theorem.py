"""Synchronized Eulerian matter/EM work-energy theorem on the numerical geometry.

The native coupled integrator is second-order Strang splitting, not a common
matter/geometry RK4 method. Sample its integer endpoints and integrate powers
with the trapezoidal rule. Measured energy is reconstructed independently.
"""

from functools import partial
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from PyPIC3D.relativity.energy_momentum_tensor import (
    electromagnetic_moments,
    compute_electromagnetic_energy_momentum,
    compute_particle_energy_momentum,
)
from PyPIC3D.relativity.interpolate_metric import particle_lorentz_factor


class EnergySample(NamedTuple):
    """One synchronized measurement: energies, and the powers driving them."""
    time: object
    electromagnetic: object
    particles: object
    curvature_power: object
    lapse_power: object
    boundary_power: object
    absorbing_boundary_power: object


class WorkEnergyTracker(NamedTuple):
    """A baseline, the latest sample, and the work integrated between them.

    Each ``*_work`` field is the trapezoidal integral of the matching
    ``*_power`` field of the samples, accumulated every macro step and
    independently of the output cadence.
    """
    baseline: object
    latest: object
    reference: object
    curvature_work: object
    lapse_work: object
    boundary_work: object
    absorbing_boundary_work: object
    step_residual: object
    steps: object


def _axis_weights(size, boundary, spacing, dtype):
    """Integrate C data over the declared N-cell coordinate interval.

    Open axes omit the upper C endpoint. Both face values are extrapolated
    from interior nodes, avoiding the first-order/zero-filled lower halo in
    native EM moments. These are the resulting trapezoidal weights, without
    allocating extended grids. Conducting axes already own both endpoints.
    """
    weight = jnp.ones(size, dtype=dtype)
    if boundary == 1:
        weight = weight.at[0].set(0.5).at[-1].set(0.5)
    elif boundary != 0:
        weight = weight.at[0].set(0.0)
        weight = weight.at[1].add(1.0).at[2].add(-0.5)
        weight = weight.at[-1].add(1.0).at[-2].add(-0.5)
    return spacing * weight


def volume_integral(integrand, static, dynamic):
    """Coordinate integral of a global C-node density, excluding all halos."""
    weights = tuple(_axis_weights(n, bc, dx, integrand.dtype)
                    for n, bc, dx in zip(integrand.shape, static.boundary_conditions,
                                         (dynamic.dx, dynamic.dy, dynamic.dz)))
    return jnp.einsum("ijk,i,j,k->", integrand, *weights)


def surface_integral(flux, static, dynamic):
    """Integral of densitized coordinate flux, upper minus lower faces.

    sqrt(gamma)*F^i d^2x equals F^i s_i dA even for non-diagonal metrics.
    Periodic pairs cancel; tile interfaces are absent from this global array.
    """
    spacing = (dynamic.dx, dynamic.dy, dynamic.dz)
    result = jnp.zeros((), dtype=flux.dtype)
    for axis, bc in enumerate(static.boundary_conditions):
        if bc == 0:
            continue
        q = jnp.moveaxis(flux[axis], axis, 0)
        difference = (q[-1] - q[0] if bc == 1 else
                      (2 * q[-1] - q[-2]) - (2 * q[1] - q[2]))
        transverse = tuple(a for a in range(3) if a != axis)
        weights = tuple(_axis_weights(flux.shape[a + 1], static.boundary_conditions[a],
                                      spacing[a], flux.dtype) for a in transverse)
        result = result + jnp.einsum("ij,i,j->", difference, *weights)
    return result


def _physical_geometry(state, metric, static):
    """Physical inverse metric, volume element and extrinsic curvature on C.

    Geometry exchange uses determinant-normalized conformal gamma and the
    same W floor as the matter helper. Reconstruct K in that physical basis.
    """
    from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
    from PyPIC3D.relativity.dynamical_relativity import tiles_to_bssn

    gamma = tiles_to_bssn(jnp.moveaxis(metric.center.gamma, (-2, -1), (0, 1)), static)
    inverse = tiles_to_bssn(jnp.moveaxis(metric.center.gamma_inv, (-2, -1), (0, 1)), static)
    volume = tiles_to_bssn(metric.center.sqrt_gamma, static)
    W = jnp.maximum(state.conformal_factor, W_FLOOR_VALUE)
    return inverse, volume, state.traceless_K / W**2 + gamma * state.trace_K / 3


def _absorbing_boundary_power(D, B, state, metric, volume, absorber, template, static, dynamic):
    """Signed model power of the absorbing layer, excluded from the budget.

    A reference sponge may supply energy. This is not exact discrete removal
    from the predictor/history/synchronization operations of the split step.
    """
    if absorber is None and not static.supergaussian_active:
        return jnp.zeros_like(template)

    from PyPIC3D.boundary_conditions.absorbing_sponge import sponge_advance
    from PyPIC3D.boundary_conditions.staggered import refresh_fields
    from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
    from PyPIC3D.relativity.core import D_FIELD_LOCATIONS, B_FIELD_LOCATIONS

    def damp(duration):
        result = []
        kinds = (None, None) if absorber is None else absorber
        for vector, kind, locations, name in zip(
                (D, B), kinds, (D_FIELD_LOCATIONS, B_FIELD_LOCATIONS), ("D", "B")):
            if kind is None:
                damped = apply_tiled_supergaussian_absorber(
                    vector, static, dynamic, duration, locations=locations)
            else:
                damped = tuple(sponge_advance(value, jnp.zeros_like(value), duration, rate, reference)
                               for value, rate, reference in zip(vector, kind[0], kind[1]))
            result.append(refresh_fields(damped, static, locations, name, metric, kind))
        return tuple(result)

    # Differentiate at zero duration, including the native halo/projector
    # tangent. This also avoids log(exp(-sigma*dt)) underflow in strong layers.
    _, tangent = jax.jvp(damp, (jnp.zeros_like(dynamic.dt),), (jnp.ones_like(dynamic.dt),))

    def field_energy(displacement, magnetic):
        density = compute_electromagnetic_energy_momentum(
            electromagnetic_moments(displacement, magnetic, static), state)[0]
        return volume_integral(volume * density, static, dynamic)

    return jax.jvp(field_energy, (D, B), tangent)[1]


@partial(jax.jit, static_argnames="static")
def sample_energy(particles, species, fields, static, dynamic):
    """Measure common-time energy, curvature/lapse power, and outward flux.

    D and B are Gaussian sqrt(gamma) times physical contravariant fields.
    Curl E/H are auxiliary constitutive fields, and are not used as Eulerian
    fields here. Particle u_i is covariant four-velocity, with rest energy.
    """
    from JAX_BSSN.evolution.spatial_derivatives import diff1_physical
    from PyPIC3D.relativity.dynamical_relativity import synchronize_particles

    bssn = dynamic.bssn_parameters
    state = bssn.state
    metric, D, B, absorber = fields[6], fields[0], fields[7][2], fields[7][4]
    inverse, volume, K = _physical_geometry(state, metric, static)

    moments = electromagnetic_moments(D, B, static)
    rho, momentum, stress = compute_electromagnetic_energy_momentum(moments, state)
    electromagnetic = volume_integral(volume * rho, static, dynamic)
    particle_energy = jnp.zeros_like(electromagnetic)
    if particles.active.size:
        common = synchronize_particles(particles, species, fields, static, dynamic)
        # Synchronization uses the actual filtered force. Measurements then
        # use raw geometry/sources so filtering remains visible in the
        # original continuum balance, rather than redefining physical energy.
        raw_static = static._replace(current_filter="none")
        Gamma = particle_lorentz_factor(common, metric, raw_static, dynamic)
        mass = (species.mass * species.weight)[None, None, None, :, None]
        particle_energy = jnp.sum(jnp.where(common.active, mass * Gamma, 0.0))
        rho_p, momentum_p, stress_p = compute_particle_energy_momentum(
            common, species, metric, raw_static, dynamic)
        rho, momentum, stress = rho + rho_p, momentum + momentum_p, stress + stress_p

    S_up = jnp.einsum("ij...,j...->i...", inverse, momentum)
    stress_up = jnp.einsum("ik...,jl...,kl...->ij...", inverse, inverse, stress)
    lapse_gradient = jnp.stack(tuple(diff1_physical(state.lapse, axis, bssn.evolution_parameters)
                                    for axis in range(3)))
    curvature_power = volume_integral(
        volume * state.lapse * jnp.einsum("ij...,ij...->...", K, stress_up), static, dynamic)
    lapse_power = volume_integral(
        -volume * jnp.sum(S_up * lapse_gradient, axis=0), static, dynamic)
    flux = volume * (state.lapse * S_up - state.shift * rho)
    boundary_power = surface_integral(flux, static, dynamic)
    absorbing_boundary_power = _absorbing_boundary_power(
        D, B, state, metric, volume, absorber, electromagnetic, static, dynamic)

    return EnergySample(bssn.time, electromagnetic, particle_energy, curvature_power,
                        lapse_power, boundary_power, absorbing_boundary_power)


def start_tracker(sample, reference=None):
    """Create an on-device baseline; host initialization validates its scale."""
    energy = sample.electromagnetic + sample.particles
    reference = energy if reference is None else jnp.asarray(reference, dtype=energy.dtype)
    zero = jnp.zeros_like(energy)
    return WorkEnergyTracker(sample, sample, reference, zero, zero, zero, zero, zero,
                             jnp.asarray(0, dtype=jnp.int32))


def advance_tracker(tracker, sample):
    """Integrate one synchronized interval, independently of output cadence."""
    previous = tracker.latest
    half_dt = (sample.time - previous.time) / 2
    curvature = half_dt * (previous.curvature_power + sample.curvature_power)
    lapse = half_dt * (previous.lapse_power + sample.lapse_power)
    boundary = half_dt * (previous.boundary_power + sample.boundary_power)
    absorbing = half_dt * (previous.absorbing_boundary_power + sample.absorbing_boundary_power)
    energy_change = ((sample.electromagnetic + sample.particles)
                     - (previous.electromagnetic + previous.particles))
    # One-interval form of the cumulative residual E - (E0 + W - T) reported
    # by tracker_values; outward transport therefore enters with a plus sign.
    return tracker._replace(
        latest=sample, curvature_work=tracker.curvature_work + curvature,
        lapse_work=tracker.lapse_work + lapse,
        boundary_work=tracker.boundary_work + boundary,
        absorbing_boundary_work=tracker.absorbing_boundary_work + absorbing,
        step_residual=energy_change - curvature - lapse + boundary, steps=tracker.steps + 1)


def baseline_tracker(tracker):
    """The same tracker viewed at its baseline: zero work, zero residual."""
    return start_tracker(tracker.baseline, tracker.reference)


def initialize_tracker(particles, species, fields, static, dynamic):
    """Validate the supported physical system and allocate its scalar carry.

    Called on the host after leapfrog initialization. The first coupled step
    re-establishes this baseline after its initial gauge/boundary preparation.
    """
    if static.solver != "dynamic_metric":
        raise ValueError("GR work-energy theorem requires solver='dynamic_metric'")
    if particles.active.size:
        active_species = jnp.any(particles.active, axis=(0, 1, 2, 4))
        if bool(jnp.any(active_species[:, None] & ~species.update_x)):
            raise ValueError("GR work-energy theorem does not support frozen particle components (constraint work)")
    shape = dynamic.bssn_parameters.state.lapse.shape
    if any(bc not in (0, 1, 2, 3) for bc in static.boundary_conditions):
        raise ValueError("Unsupported GR work-energy theorem field boundary")
    if any(n < 3 for n, bc in zip(shape, static.boundary_conditions) if bc not in (0, 1)):
        raise ValueError("GR work-energy theorem needs at least three C nodes on open axes")
    reference = static.gr_work_energy_reference
    # build_static_parameters checks this too, but drivers and tests that build
    # StaticParameters directly never pass through that path.
    if reference is not None and (not math.isfinite(reference) or reference <= 0):
        raise ValueError("gr_work_energy_reference must be finite and positive")
    sample = sample_energy(particles, species, fields, static, dynamic)
    energy = float(sample.electromagnetic + sample.particles)
    if not math.isfinite(energy):
        raise ValueError("GR work-energy theorem initial energy must be finite")
    if reference is None and energy <= 0:
        raise ValueError("Nonpositive initial energy requires a positive gr_work_energy_reference")
    return start_tracker(sample, reference)


def tracker_values(tracker):
    """Scalars for the existing time-series writer; no device transfers here."""
    sample = tracker.latest
    energy = sample.electromagnetic + sample.particles
    baseline_energy = tracker.baseline.electromagnetic + tracker.baseline.particles
    geometry_work = tracker.curvature_work + tracker.lapse_work
    budget = baseline_energy + geometry_work - tracker.boundary_work
    residual = energy - budget
    return dict(
        electromagnetic=sample.electromagnetic, particles=sample.particles,
        total=energy, budget=budget,
        curvature_power=sample.curvature_power, lapse_power=sample.lapse_power,
        geometry_power=sample.curvature_power + sample.lapse_power,
        boundary_power=sample.boundary_power,
        absorbing_boundary_power=sample.absorbing_boundary_power,
        curvature_work=tracker.curvature_work, lapse_work=tracker.lapse_work,
        geometry_work=geometry_work, boundary_work=tracker.boundary_work,
        absorbing_boundary_work=tracker.absorbing_boundary_work,
        residual=residual, step_residual=tracker.step_residual,
        relative_residual=jnp.abs(residual) / tracker.reference, reference=tracker.reference,
        baseline_time=tracker.baseline.time, baseline_energy=baseline_energy)
