from typing import NamedTuple

import jax
import jax.numpy as jnp


D_FIELD_LOCATIONS = (("V", "C", "C"), ("C", "V", "C"), ("C", "C", "V"))
B_FIELD_LOCATIONS = (("C", "V", "V"), ("V", "C", "V"), ("V", "V", "C"))


class Metric(NamedTuple):
    """3+1 metric sampled on one Yee location."""

    lapse: object
    shift: object
    gamma: object
    gamma_inv: object
    sqrt_gamma: object


class YeeMetric(NamedTuple):
    """
    Metric state on the grid locations used by the static-metric update.

    ``D`` and ``B`` are tuples with one metric per component location.
    ``center`` supplies lapse, shift and covariant gamma to the shared Hermite
    particle sampler; particle inverses and determinants are derived there.
    ``vertex`` is a shared nodal metric for diagnostics.
    """

    D: tuple
    B: tuple
    center: Metric
    vertex: Metric


def covariant_lorentz_factor(u_cov, gamma_inv):
    """
    Compute Gamma = sqrt(1 + gamma^ij u_i u_j).
    """

    u_sq = jnp.einsum("...i,...ij,...j->...", u_cov, gamma_inv, u_cov)
    return jnp.sqrt(1.0 + u_sq)


def contravariant_three_velocity(u_cov, gamma_inv):
    """
    Convert covariant spatial velocity/momentum to FIDO three-velocity v^i.
    """

    u_con = jnp.einsum("...ij,...j->...i", gamma_inv, u_cov)
    Gamma = covariant_lorentz_factor(u_cov, gamma_inv)
    return u_con / Gamma[..., jnp.newaxis]


def lower_vector(vector_con, gamma):
    """
    Lower a contravariant spatial vector with gamma_ij.
    """

    return jnp.einsum("...ij,...j->...i", gamma, vector_con)


def location_grid(center_grid, vertex_grid, location):
    """
    Select the tuple of grid axes associated with a C/V location triplet.
    """

    return tuple(
        center_grid[axis] if location[axis] == "C" else vertex_grid[axis]
        for axis in range(3)
    )


def analytic_metric_on_grid(grid, metric_at_position):
    """
    Evaluate an analytic spatial metric on a tiled grid.

    ``metric_at_position`` returns ``(lapse, shift, gamma, gamma_inv,
    sqrt_gamma)`` for one coordinate position.
    """

    x_grid, y_grid, z_grid = grid
    X, Y, Z = jnp.broadcast_arrays(
        x_grid[..., :, jnp.newaxis, jnp.newaxis],
        y_grid[..., jnp.newaxis, :, jnp.newaxis],
        z_grid[..., jnp.newaxis, jnp.newaxis, :],
    )
    grid_shape = X.shape
    positions = jnp.stack((X, Y, Z), axis=-1).reshape((-1, 3))

    lapse, shift, gamma, gamma_inv, sqrt_gamma = jax.vmap(metric_at_position)(positions)
    return Metric(
        lapse=lapse.reshape(grid_shape),
        shift=shift.reshape(grid_shape + (3,)),
        gamma=gamma.reshape(grid_shape + (3, 3)),
        gamma_inv=gamma_inv.reshape(grid_shape + (3, 3)),
        sqrt_gamma=sqrt_gamma.reshape(grid_shape),
    )


def build_yee_metric(dynamic_parameters, metric_at_position):
    """
    Evaluate ``metric_at_position`` on every D, B, center and vertex location.
    """

    center_grid = dynamic_parameters.grids.tiled_center_grid
    vertex_grid = dynamic_parameters.grids.tiled_vertex_grid

    def on_location(location):
        return analytic_metric_on_grid(
            location_grid(center_grid, vertex_grid, location), metric_at_position
        )

    return YeeMetric(
        D=tuple(on_location(location) for location in D_FIELD_LOCATIONS),
        B=tuple(on_location(location) for location in B_FIELD_LOCATIONS),
        center=analytic_metric_on_grid(center_grid, metric_at_position),
        vertex=analytic_metric_on_grid(vertex_grid, metric_at_position),
    )
