import jax.numpy as jnp

from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.ownership import owned_nodes
from PyPIC3D.relativity.field_interpolation import reconstruct_vector


def _shift_cross_component(beta, vector_components, component):
    beta_x = beta[..., 0]
    beta_y = beta[..., 1]
    beta_z = beta[..., 2]
    vector_x, vector_y, vector_z = vector_components

    if component == 0:
        return beta_y * vector_z - beta_z * vector_y
    if component == 1:
        return beta_z * vector_x - beta_x * vector_z
    return beta_x * vector_y - beta_y * vector_x


def compute_covariant_E(D_tiles, B_tiles, metric):
    """
    Compute covariant E_i from densitized D/B using FPIC Eq. (10).

    Densities are averaged to each target location and divided by the target
    sqrt_gamma, the metric-weighted transfer of the physical vector on a
    regular coordinate domain. All metric samples, including halos, must have
    finite positive sqrt_gamma.
    """

    E_cov = []
    for i, target_location in enumerate(D_FIELD_LOCATIONS):
        volume = metric.D[i].sqrt_gamma
        D_on_target = [value / volume for value in
                       reconstruct_vector(D_tiles, D_FIELD_LOCATIONS, target_location)]
        B_on_target = [value / volume for value in
                       reconstruct_vector(B_tiles, B_FIELD_LOCATIONS, target_location)]

        D_lower_i = 0.0
        for j in range(3):
            D_lower_i = D_lower_i + metric.D[i].gamma[..., i, j] * D_on_target[j]

        shift_cross = _shift_cross_component(metric.D[i].shift, tuple(B_on_target), i)
        E_cov.append(
            metric.D[i].lapse * D_lower_i
            + metric.D[i].sqrt_gamma * shift_cross
        )
    return tuple(E_cov)


def compute_covariant_H(D_tiles, B_tiles, metric):
    """
    Compute covariant H_i from densitized D/B using FPIC Eq. (9).

    Uses the same transfers and regular-domain requirements as
    ``compute_covariant_E``.
    """

    H_cov = []
    for i, target_location in enumerate(B_FIELD_LOCATIONS):
        volume = metric.B[i].sqrt_gamma
        B_on_target = [value / volume for value in
                       reconstruct_vector(B_tiles, B_FIELD_LOCATIONS, target_location)]
        D_on_target = [value / volume for value in
                       reconstruct_vector(D_tiles, D_FIELD_LOCATIONS, target_location)]

        B_lower_i = 0.0
        for j in range(3):
            B_lower_i = B_lower_i + metric.B[i].gamma[..., i, j] * B_on_target[j]

        shift_cross = _shift_cross_component(metric.B[i].shift, tuple(D_on_target), i)
        H_cov.append(
            metric.B[i].lapse * B_lower_i
            - metric.B[i].sqrt_gamma * shift_cross
        )
    return tuple(H_cov)


def update_D(D_tiles, H_tiles, J_tiles, metric, static_parameters, dynamic_parameters, dt):
    """Advance densitized D with densitized J and enforce the FIDO surface projection."""
    Dx, Dy, Dz = D_tiles
    Jx, Jy, Jz = J_tiles
    Hx, Hy, Hz = H_tiles
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    # Backward differences: spatial x/y/z are array axes 3/4/5 after tile axes.
    dHz_dy = (Hz - jnp.roll(Hz, 1, axis=4)) / dy
    dHy_dz = (Hy - jnp.roll(Hy, 1, axis=5)) / dz
    dHx_dz = (Hx - jnp.roll(Hx, 1, axis=5)) / dz
    dHz_dx = (Hz - jnp.roll(Hz, 1, axis=3)) / dx
    dHy_dx = (Hy - jnp.roll(Hy, 1, axis=3)) / dx
    dHx_dy = (Hx - jnp.roll(Hx, 1, axis=4)) / dy

    # Update owned nodes, including conducting upper C endpoints.
    Dx = jnp.where(
        owned_nodes(Dx.shape, D_FIELD_LOCATIONS[0], static_parameters),
        Dx + dt * ((dHz_dy - dHy_dz) - 4.0 * jnp.pi * Jx),
        Dx,
    )
    Dy = jnp.where(
        owned_nodes(Dy.shape, D_FIELD_LOCATIONS[1], static_parameters),
        Dy + dt * ((dHx_dz - dHz_dx) - 4.0 * jnp.pi * Jy),
        Dy,
    )
    Dz = jnp.where(
        owned_nodes(Dz.shape, D_FIELD_LOCATIONS[2], static_parameters),
        Dz + dt * ((dHy_dx - dHx_dy) - 4.0 * jnp.pi * Jz),
        Dz,
    )

    D_tiles = apply_tiled_supergaussian_absorber(
        (Dx, Dy, Dz), static_parameters, dynamic_parameters, dt,
        locations=D_FIELD_LOCATIONS,
    )
    return refresh_fields(D_tiles, static_parameters, D_FIELD_LOCATIONS, 'D', metric)


def update_B(E_tiles, B_tiles, metric, static_parameters, dynamic_parameters, dt):
    """Advance densitized B and remove its normal surface component."""
    Bx, By, Bz = B_tiles
    Ex, Ey, Ez = E_tiles
    dx, dy, dz = dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz

    # Forward differences: spatial x/y/z are array axes 3/4/5 after tile axes.
    dEz_dy = (jnp.roll(Ez, -1, axis=4) - Ez) / dy
    dEy_dz = (jnp.roll(Ey, -1, axis=5) - Ey) / dz
    dEx_dz = (jnp.roll(Ex, -1, axis=5) - Ex) / dz
    dEz_dx = (jnp.roll(Ez, -1, axis=3) - Ez) / dx
    dEy_dx = (jnp.roll(Ey, -1, axis=3) - Ey) / dx
    dEx_dy = (jnp.roll(Ex, -1, axis=4) - Ex) / dy

    # Update owned nodes, including conducting upper C endpoints.
    Bx = jnp.where(
        owned_nodes(Bx.shape, B_FIELD_LOCATIONS[0], static_parameters),
        Bx + dt * (-(dEz_dy - dEy_dz)),
        Bx,
    )
    By = jnp.where(
        owned_nodes(By.shape, B_FIELD_LOCATIONS[1], static_parameters),
        By + dt * (-(dEx_dz - dEz_dx)),
        By,
    )
    Bz = jnp.where(
        owned_nodes(Bz.shape, B_FIELD_LOCATIONS[2], static_parameters),
        Bz + dt * (-(dEy_dx - dEx_dy)),
        Bz,
    )

    B_tiles = apply_tiled_supergaussian_absorber(
        (Bx, By, Bz), static_parameters, dynamic_parameters, dt,
        locations=B_FIELD_LOCATIONS,
    )
    return refresh_fields(B_tiles, static_parameters, B_FIELD_LOCATIONS, 'B', metric)
