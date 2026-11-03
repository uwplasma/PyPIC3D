"""BSSN geometry exchange and time levels for the dynamical PIC solver."""

import jax.numpy as jnp

from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE
from JAX_BSSN.evolution.time_evolve import enforce_algebraic_constraints
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.core import Metric, YeeMetric, D_FIELD_LOCATIONS, B_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import location_interpolate
from PyPIC3D.relativity.field_state import densitize_vector
from PyPIC3D.solvers.GR_yee.curved_maxwell import (
    compute_covariant_E, compute_covariant_H, update_D, update_B,
)
from PyPIC3D.utilities.field_helpers import add_external_fields


def bssn_to_tiles(values, static_parameters, *, boundary_conditions=None, extra_guard=0):
    """Sample global C-node arrays on tiles, with periodic or linear exteriors.

    Leading tensor indices are retained. Conducting axes include their final
    C endpoint; the tile width still counts cells, not both endpoints.
    """
    boundaries = (static_parameters.boundary_conditions if boundary_conditions is None
                  else boundary_conditions)
    shape = values.shape[-3:]
    leading = values.ndim - 3
    guard = static_parameters.guard_cells + extra_guard
    for direction, (size, width, bc) in enumerate(zip(shape, static_parameters.tile_shape, boundaries)):
        count = (size - (static_parameters.boundary_conditions[direction] == 1)) // width
        index = jnp.arange(count)[:, None] * width + jnp.arange(-guard, width + guard)[None, :]
        axis = leading + 2 * direction
        if bc == 0:
            values = jnp.take(values, index % size, axis=axis)
        else:
            sampled = jnp.take(values, jnp.clip(index, 0, size - 1), axis=axis)
            if size > 1:
                left = jnp.take(values, 1, axis=axis) - jnp.take(values, 0, axis=axis)
                right = jnp.take(values, size - 1, axis=axis) - jnp.take(values, size - 2, axis=axis)
                left = jnp.expand_dims(left, (axis, axis + 1))
                right = jnp.expand_dims(right, (axis, axis + 1))
                dimensions = [1] * sampled.ndim
                dimensions[axis:axis + 2] = index.shape
                sampled = (sampled + jnp.minimum(index, 0).reshape(dimensions) * left
                           + jnp.maximum(index - size + 1, 0).reshape(dimensions) * right)
            values = sampled
    order = tuple(range(leading)) + tuple(leading + i for i in (0, 2, 4, 1, 3, 5))
    return jnp.transpose(values, order)


def tiles_to_bssn(values, static_parameters):
    """Gather owned C samples into a global array, retaining tensor indices."""
    counts = values.shape[-6:-3]
    tile_indices, local_indices = [], []
    for count, width, bc in zip(counts, static_parameters.tile_shape, static_parameters.boundary_conditions):
        index = jnp.arange(count * width + (bc == 1))
        tile = jnp.minimum(index // width, count - 1)
        tile_indices.append(tile)
        local_indices.append(index - tile * width + static_parameters.guard_cells)
    tx, ty, tz = tile_indices
    ix, iy, iz = local_indices
    return values[(Ellipsis, tx[:, None, None], ty[None, :, None], tz[None, None, :],
                   ix[:, None, None], iy[None, :, None], iz[None, None, :])]


def build_bssn_metric(state, static_parameters, dynamic_parameters):
    """Average conformal geometry to Yee sites before deriving physical tensors."""
    W = bssn_to_tiles(state.conformal_factor, static_parameters, extra_guard=1)
    lapse = bssn_to_tiles(state.lapse, static_parameters, extra_guard=1)
    shift = bssn_to_tiles(state.shift, static_parameters, extra_guard=1)
    conformal = bssn_to_tiles(state.conformal_metric, static_parameters, extra_guard=1)

    def sample(location):
        # The extra halo supplies the upper neighbour of the outer metric guard.
        def transfer(value):
            return location_interpolate(value, ("C", "C", "C"), location)[..., 1:-1, 1:-1, 1:-1]

        w = jnp.maximum(transfer(W), W_FLOOR_VALUE)
        gamma = jnp.stack(tuple(jnp.stack(tuple(transfer(value) for value in row)) for row in conformal))
        gamma = jnp.moveaxis(gamma, (0, 1), (-2, -1))
        gamma = gamma * jnp.linalg.det(gamma)[..., None, None] ** (-1.0 / 3.0)
        physical = gamma / w[..., None, None] ** 2
        inverse = jnp.linalg.inv(gamma) * w[..., None, None] ** 2
        beta = jnp.stack(tuple(transfer(value) for value in shift), axis=-1)
        return Metric(transfer(lapse), beta, physical, inverse, w**-3)

    return YeeMetric(tuple(sample(location) for location in D_FIELD_LOCATIONS),
                     tuple(sample(location) for location in B_FIELD_LOCATIONS),
                     sample(("C", "C", "C")), sample(("V", "V", "V")))


def synchronize_particles(particles, species, fields, static_parameters, dynamic_parameters):
    """View u at the integer position time without changing the stored leapfrog."""
    if not particles.active.size:
        return particles
    half_dt = fields[7][3]
    push_D, push_B = add_external_fields(fields[0], fields[7][2], fields[5])
    common, _ = hybrid_boris_geodesic_push(
        particles, species, push_D, push_B, fields[6], static_parameters,
        dynamic_parameters._replace(dt=half_dt / 2), advance_positions=False)
    return common


def synchronized_magnetic_field(before, after, static_parameters, dynamic_parameters):
    """Reconstruct endpoint B from retained history using the native operator."""
    D, B = before[:2]
    previous_D, previous_B, _, _, absorber = before[7]
    metric = before[6]
    magnetic_absorber = None if absorber is None else absorber[1]
    half_D = tuple((value + old) / 2 for value, old in zip(D, previous_D))
    integer_B = update_B(
        compute_covariant_E(half_D, B, metric),
        tuple((value + old) / 2 for value, old in zip(B, previous_B)),
        metric, static_parameters, dynamic_parameters, dynamic_parameters.dt,
        absorber=magnetic_absorber)
    midpoint_D = tuple((value + new) / 2 for value, new in zip(D, after[0]))
    return update_B(compute_covariant_E(midpoint_D, after[1], metric), integer_B,
                    metric, static_parameters, dynamic_parameters, dynamic_parameters.dt,
                    absorber=magnetic_absorber)


def initialize_dynamical_relativity(particles, species, fields, static_parameters,
                                   dynamic_parameters, *, absorber=None):
    """Initialize on the host from common-time physical fields and momenta.

    The macro timestep is dt; the native PIC history spacing is h=dt/2.
    Runtime field slot 7 is (previous_D, previous_B, synchronized_B, h, absorber).
    Densitization is an input conversion, never part of a subsequent metric switch.
    """
    if any(bool(jnp.any(value != 0)) for vector in fields[5] for value in vector):
        raise ValueError("Dynamical relativity does not support prescribed external fields")
    if particles.active.size and any(bc != 0 for bc in (
            static_parameters.boundary_conditions + static_parameters.particle_boundary_conditions)):
        raise ValueError("Dynamical PIC currently requires periodic field and particle boundaries")
    if static_parameters.pml_active:
        raise ValueError("Dynamical relativity does not support PML")
    if static_parameters.metric not in ("numerical", "flat_cartesian"):
        raise ValueError("Dynamical relativity requires a Cartesian numerical metric")

    bssn = dynamic_parameters.bssn_parameters
    if static_parameters.current_filter != "none":
        from PyPIC3D.relativity.filtering import validate_gr_filter

        validate_gr_filter(static_parameters)
        if any(int(getattr(bssn.evolution_parameters, name)) != 0
               for name in ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc")):
            raise ValueError("Matched GR filtering requires all six BSSN boundaries to be periodic")
    state = enforce_algebraic_constraints(bssn.state)
    dynamic_parameters = dynamic_parameters._replace(bssn_parameters=bssn._replace(state=state))
    metric = build_bssn_metric(state, static_parameters, dynamic_parameters)
    D, B, _, rho, phi, external, _, _, overflow = fields
    D = densitize_vector(D, metric.D)
    B = densitize_vector(B, metric.B)
    external = (densitize_vector(external[0], metric.D), densitize_vector(external[1], metric.B))
    da, ba = (None, None) if absorber is None else absorber
    D = refresh_fields(D, static_parameters, D_FIELD_LOCATIONS, "D", metric, da)
    B = refresh_fields(B, static_parameters, B_FIELD_LOCATIONS, "B", metric, ba)
    h = dynamic_parameters.dt / 2
    native_parameters = dynamic_parameters._replace(dt=h)
    zero = tuple(jnp.zeros_like(value) for value in D)
    current = zero
    if particles.active.size:
        current = GR_direct_deposition(particles, species, zero, metric,
                                      static_parameters, native_parameters)
        push_D, push_B = add_external_fields(D, B, external)
        particles, _ = hybrid_boris_geodesic_push(
            particles, species, push_D, push_B, metric, static_parameters,
            native_parameters._replace(dt=-h / 2), advance_positions=False)

    E = compute_covariant_E(D, B, metric)
    H = compute_covariant_H(D, B, metric)
    previous_D = update_D(D, H, current, metric, static_parameters, native_parameters, -h, absorber=da)
    staggered_B = update_B(E, B, metric, static_parameters, native_parameters, -h / 2, absorber=ba)
    previous_B = update_B(E, B, metric, static_parameters, native_parameters, -1.5 * h, absorber=ba)
    history = (previous_D, previous_B, B, h, absorber)
    fields = (D, staggered_B, current, rho, phi, external, metric, history, overflow)
    if static_parameters.gr_work_energy_theorem:
        from PyPIC3D.diagnostics.gr_work_energy_theorem import initialize_tracker
        tracker = initialize_tracker(particles, species, fields, static_parameters, dynamic_parameters)
        dynamic_parameters = dynamic_parameters._replace(work_energy_tracker=tracker)
    return particles, fields, dynamic_parameters


def rephase_pic_state(particles, species, fields, new_metric, static_parameters,
                     dynamic_parameters, *, synchronized_particles=None):
    """Align staggered PIC data with a changed frozen metric generator.

    Common-time x, u, D and B are held fixed while evaluating both generators.
    Corrections at offsets -h/2, -h and -3h/2 preserve second-order Strang
    centering. The current correction includes the instantaneous metric change
    in coordinate velocity; it is not a new charge-conserving deposit. Direct
    currents retain the matched coupling filter in both generators. Particle
    gathers use that same filter without smoothing stored fields or geometry.
    """
    D, B, current, rho, phi, external, old_metric, history, overflow = fields
    previous_D, previous_B, common_B, h, absorber = history
    da, ba = (None, None) if absorber is None else absorber
    native_parameters = dynamic_parameters._replace(dt=h)
    common_particles = (synchronize_particles(particles, species, fields, static_parameters, dynamic_parameters)
                        if synchronized_particles is None else synchronized_particles)
    zero = tuple(jnp.zeros_like(value) for value in D)
    generators = []
    for metric in (old_metric, new_metric):
        source = zero
        pushed_u = common_particles.u
        if particles.active.size:
            source = GR_direct_deposition(common_particles, species, zero, metric,
                                         static_parameters, native_parameters)
            push_D, push_B = add_external_fields(D, common_B, external)
            pushed, _ = hybrid_boris_geodesic_push(
                common_particles, species, push_D, push_B, metric,
                static_parameters, native_parameters, advance_positions=False)
            pushed_u = pushed.u
        E = compute_covariant_E(D, common_B, metric)
        H = compute_covariant_H(D, common_B, metric)
        next_D = update_D(D, H, source, metric, static_parameters, native_parameters, h, absorber=da)
        next_B = update_B(E, common_B, metric, static_parameters, native_parameters, h, absorber=ba)
        generators.append((next_D, next_B, source, pushed_u))
    old, new = generators
    delta_D = tuple(n - o for n, o in zip(new[0], old[0]))
    delta_B = tuple(n - o for n, o in zip(new[1], old[1]))
    particles = particles._replace(u=particles.u - 0.5 * (new[3] - old[3]))
    B = tuple(value - 0.5 * delta for value, delta in zip(B, delta_B))
    previous_D = tuple(value - delta for value, delta in zip(previous_D, delta_D))
    previous_B = tuple(value - 1.5 * delta for value, delta in zip(previous_B, delta_B))
    current = tuple(value + (n - o) for value, n, o in zip(current, new[2], old[2]))

    D = refresh_fields(D, static_parameters, D_FIELD_LOCATIONS, "D", new_metric, da)
    B = refresh_fields(B, static_parameters, B_FIELD_LOCATIONS, "B", new_metric, ba)
    previous_D = refresh_fields(previous_D, static_parameters, D_FIELD_LOCATIONS, "D", new_metric, da)
    previous_B = refresh_fields(previous_B, static_parameters, B_FIELD_LOCATIONS, "B", new_metric, ba)
    common_B = refresh_fields(common_B, static_parameters, B_FIELD_LOCATIONS, "B", new_metric, ba)
    history = (previous_D, previous_B, common_B, h, absorber)
    return particles, (D, B, current, rho, phi, external, new_metric, history, overflow)
