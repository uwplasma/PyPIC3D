"""The shared particle-metric sampler for prescribed static metrics.

The lapse, shift and covariant spatial metric are interpolated with a C1
tensor-product cardinal cubic Hermite (Catmull-Rom) polynomial, whose nodal
slopes are the centered differences of the supplied grid data.  The inverse
metric and determinant are computed from the interpolated gamma, and every
derivative is the exact derivative of that same interpolant, so the geodesic
force is consistent with the metric used for the Lorentz force and the
position update.

This reconstruction is independent of the particle shape used to gather
electromagnetic fields.  Grid ``gamma_inv`` and ``sqrt_gamma`` are deliberately
not sampled here: they belong to the grid field operators.  Analytic metric
providers populate the grid, not particle samples.  There is no lower-order
or analytic fallback.

The four-point stencil needs three guard cells to cover midpoint samples
before tile migration.  Points outside the stencil return NaN, and invalid
metrics are reported through checkify rather than clamped or repaired.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from PyPIC3D.relativity.core import covariant_lorentz_factor

RECONSTRUCTION = "cardinal_cubic_hermite_consistent_v1"

SPHERICAL_METRICS = ("flat_spherical", "kerr_schild_spherical")


def on_spherical_axis(position):
    """Recognize the singular theta endpoints, including floating-point pi."""
    theta = position[..., 1]
    return (jnp.abs(theta) < 1e-14) | (jnp.abs(jnp.abs(theta) - jnp.pi) < 1e-14)


class ParticleMetric(NamedTuple):
    """
    3+1 metric at particle positions.

    ``grad_lapse[..., i]`` is ``d_i alpha``, ``grad_shift[..., j, i]`` is
    ``d_i beta^j`` and ``grad_gamma_inv[..., i, j, k]`` is ``d_i gamma^jk``.
    The derivative fields are ``None`` when sampled without derivatives.
    """

    lapse: object
    shift: object
    gamma: object
    gamma_inv: object
    sqrt_gamma: object
    grad_lapse: object = None
    grad_shift: object = None
    grad_gamma_inv: object = None


def hermite_weights(t):
    """Catmull-Rom weights of nodes i-1, i, i+1, i+2 at fraction t in [i, i+1]."""

    return jnp.stack(
        (
            -0.5 * t + t * t - 0.5 * t**3,
            1 - 2.5 * t * t + 1.5 * t**3,
            0.5 * t + 2 * t * t - 1.5 * t**3,
            -0.5 * t * t + 0.5 * t**3,
        ),
        axis=0,
    )


def interpolate_hermite(field, position, grid, active_axes, inactive_axis_indices):
    """
    Interpolate ``field[x, y, z, ...]`` to ``position[..., 3]``.

    Unresolved axes are sampled at ``inactive_axis_indices`` without
    interpolation.  Points whose four-node stencil leaves the grid return NaN.
    """

    if len(grid) != 3 or position.shape[-1] != 3:
        raise ValueError("Hermite sampling requires three coordinates and three grids")
    for axis in range(3):
        if field.shape[axis] != len(grid[axis]):
            raise ValueError("Metric array and coordinate grid shapes differ")
        if active_axes[axis] and len(grid[axis]) < 4:
            raise ValueError("Hermite sampling requires four nodes in each resolved direction")
        if not active_axes[axis] and not 0 <= inactive_axis_indices[axis] < len(grid[axis]):
            raise ValueError("Invalid unresolved-axis sample index")

    points = position.reshape((-1, 3))
    n_points = points.shape[0]
    stencil_indices = []
    stencil_weights = []
    inside = jnp.ones(n_points, bool)
    for axis in range(3):
        nodes = grid[axis]
        if active_axes[axis]:
            h = nodes[1] - nodes[0]
            cell = jnp.floor((points[:, axis] - nodes[0]) / h).astype(jnp.int32)
            inside = inside & (cell >= 1) & (cell + 2 < len(nodes))
            # Clipping keeps the gather in bounds; outside points become NaN below.
            cell = jnp.clip(cell, 1, len(nodes) - 3)
            t = (points[:, axis] - nodes[cell]) / h
            stencil_indices.append(cell[None, :] + jnp.arange(-1, 3)[:, None])
            stencil_weights.append(hermite_weights(t))
        else:
            stencil_indices.append(jnp.full((1, n_points), inactive_axis_indices[axis], dtype=jnp.int32))
            stencil_weights.append(jnp.ones((1, n_points), dtype=points.dtype))

    ix, iy, iz = stencil_indices
    stencil_values = field[ix[:, None, None, :], iy[None, :, None, :], iz[None, None, :, :]]
    result = jnp.einsum("ip,jp,kp,ijkp...->p...", *stencil_weights, stencil_values)
    result = jnp.where(inside.reshape((-1,) + (1,) * (result.ndim - 1)), result, jnp.nan)
    return result.reshape(position.shape[:-1] + field.shape[3:])


def interpolate_metric(
    metric,
    position,
    grid,
    metric_name,
    active_axes,
    inactive_axis_indices,
    *,
    derivatives=True,
):
    """
    Interpolate one tile of grid ``Metric`` data to particle positions.

    ``metric`` holds a single tile of ``YeeMetric.center`` data and ``grid``
    its C-grid coordinate axes (the base nodes, not half-cell locations).
    With ``derivatives=True`` the lapse, shift and inverse
    metric gradients are the exact derivatives of the Hermite interpolant,
    taken with one forward-mode JVP per resolved axis.  With derivatives
    disabled the same metric values are returned and gradient fields are None.
    Unresolved axes use the supplied node indices and have zero derivatives.
    """

    shape = position.shape[:-1]
    packed = jnp.concatenate(
        (
            metric.lapse[..., None],
            metric.shift,
            metric.gamma.reshape(metric.gamma.shape[:-2] + (9,)),
        ),
        axis=-1,
    )

    def interpolate(q):
        return interpolate_hermite(packed, q, grid, active_axes, inactive_axis_indices)

    values = interpolate(position)
    lapse = values[..., 0]
    shift = values[..., 1:4]
    gamma = values[..., 4:].reshape(shape + (3, 3))
    gamma_inv = jnp.linalg.inv(gamma)

    # sqrt(det gamma) keeps the coordinate orientation of the chart, which
    # flips sign in the reflected ghost cells across a polar axis.
    if metric_name in SPHERICAL_METRICS:
        orientation = jnp.sin(position[..., 1])
    elif metric_name == "flat_cylindrical":
        orientation = position[..., 0]
    else:
        orientation = jnp.ones(shape)
    sqrt_gamma = jnp.copysign(jnp.sqrt(jnp.linalg.det(gamma)), orientation)

    sampled = ParticleMetric(lapse, shift, gamma, gamma_inv, sqrt_gamma)
    if derivatives:
        gradient = jnp.stack(
            [
                jax.jvp(interpolate, (position,), (jnp.broadcast_to(jnp.eye(3, dtype=position.dtype)[axis], position.shape),))[1]
                if active_axes[axis]
                else jnp.zeros_like(values)
                for axis in range(3)
            ],
            axis=-1,
        )  # [..., component, derivative]
        grad_gamma = gradient[..., 4:, :].reshape(shape + (3, 3, 3))
        sampled = sampled._replace(
            grad_lapse=gradient[..., 0, :],
            grad_shift=gradient[..., 1:4, :],
            grad_gamma_inv=-jnp.einsum("...ij,...jlk,...lm->...kim", gamma_inv, grad_gamma, gamma_inv),
        )

    check_particle_samples(particle_metric_valid(sampled, position, metric_name), position, "particle metric")
    return sampled


def particle_active_axes(tile_counts, tile_shape):
    """Axes resolved by the global grid; a width-one tile is still physical when other tiles extend its axis."""

    return tuple(int(count) * int(width) > 1 for count, width in zip(tile_counts, tile_shape))


def safe_inactive_positions(position, active, grid, active_axes, guard_cells):
    """Move unused particle slots strictly inside the tile; live positions are unchanged."""

    interior = jnp.stack(
        tuple(
            axis[guard_cells] + (axis[1] - axis[0]) / 2 if resolved else axis[guard_cells]
            for axis, resolved in zip(grid, active_axes)
        )
    )
    return jnp.where(active[..., None], position, interior)


def positive_definite_3x3(gamma):
    """Sylvester criterion for a symmetric 3x3 tensor, without an eigensolver."""

    a, d, f = gamma[..., 0, 0], gamma[..., 1, 1], gamma[..., 2, 2]
    b = (gamma[..., 0, 1] + gamma[..., 1, 0]) / 2
    c = (gamma[..., 0, 2] + gamma[..., 2, 0]) / 2
    e = (gamma[..., 1, 2] + gamma[..., 2, 1]) / 2
    minor = a * d - b * b
    determinant = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)
    return (a > 0) & (minor > 0) & (determinant > 0) & jnp.isfinite(minor) & jnp.isfinite(determinant)


def particle_metric_valid(metric, position, metric_name):
    """Per-particle validity of a sampled metric; nothing is repaired."""

    symmetric = jnp.abs(metric.gamma - jnp.swapaxes(metric.gamma, -1, -2)) <= (
        32 * jnp.finfo(metric.gamma.dtype).eps * jnp.maximum(1.0, jnp.abs(metric.gamma))
    )
    valid = (
        jnp.isfinite(metric.lapse)
        & (metric.lapse > 0)
        & jnp.all(jnp.isfinite(metric.shift), axis=-1)
        & jnp.all(jnp.isfinite(metric.gamma), axis=(-2, -1))
        & jnp.all(jnp.isfinite(metric.gamma_inv), axis=(-2, -1))
        & jnp.all(symmetric, axis=(-2, -1))
        & positive_definite_3x3(metric.gamma)
    )
    if metric.grad_lapse is not None:
        valid &= (
            jnp.all(jnp.isfinite(metric.grad_lapse), axis=-1)
            & jnp.all(jnp.isfinite(metric.grad_shift), axis=(-2, -1))
            & jnp.all(jnp.isfinite(metric.grad_gamma_inv), axis=(-3, -2, -1))
        )

    if metric_name in SPHERICAL_METRICS:
        valid &= ~on_spherical_axis(position) & (position[..., 0] != 0)
    elif metric_name == "flat_cylindrical":
        valid &= position[..., 0] != 0
    return valid


def check_particle_samples(valid, position, stage, tile=None):
    """Report the first invalid particle through checkify (active only under checkify)."""

    if valid.size == 0:
        return
    slot = jnp.argmax((~valid).reshape(-1))
    tile = jnp.asarray((-1, -1, -1)) if tile is None else jnp.asarray(tile)
    checkify.debug_check(
        jnp.all(valid),
        stage + ": invalid particle sample; tile={tile}, flattened species/slot={slot}, position={position}",
        tile=tile,
        slot=slot,
        position=position.reshape(-1, 3)[slot],
    )


@partial(jax.jit, static_argnames="static_parameters")
def particle_lorentz_factor(particles, metric, static_parameters, dynamic_parameters):
    """
    Gamma = sqrt(1 + gamma^ij u_i u_j) for every tiled particle slot.

    ``particles.u`` holds covariant u_i, and the inverse metric is the same
    Hermite sample the pusher uses.  Inactive slots return 1.
    """

    g = int(static_parameters.guard_cells)
    ntx, nty, ntz = particles.active.shape[:3]
    active_axes = particle_active_axes((ntx, nty, ntz), static_parameters.tile_shape)

    def one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        x = x_tile.reshape(-1, 3)
        u = u_tile.reshape(-1, 3)
        active = active_tile.reshape(-1)
        center_grid = tuple(axis[tx, ty, tz] for axis in dynamic_parameters.grids.tiled_center_grid)
        x = safe_inactive_positions(x, active, center_grid, active_axes, g)
        sampled = interpolate_metric(
            jax.tree.map(lambda array: array[tx, ty, tz], metric.center),
            x,
            center_grid,
            static_parameters.metric,
            active_axes,
            (g, g, g),
            derivatives=False,
        )
        Gamma = covariant_lorentz_factor(jnp.where(active[:, None], u, 0.0), sampled.gamma_inv)
        return jnp.where(active, Gamma, 1.0).reshape(active_tile.shape)

    for _ in range(3):
        one_tile = jax.vmap(one_tile)
    tx, ty, tz = jnp.meshgrid(jnp.arange(ntx), jnp.arange(nty), jnp.arange(ntz), indexing="ij")
    return one_tile(particles.x, particles.u, particles.active, tx, ty, tz)
