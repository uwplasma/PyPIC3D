import jax.numpy as jnp

from PyPIC3D.boundary_conditions.supergaussian import apply_tiled_supergaussian_absorber
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.boundary_conditions.absorbing_sponge import sponge_advance
from PyPIC3D.boundary_conditions.ghost_cells import owned_nodes
from PyPIC3D.relativity.field_interpolation import reconstruct_vector


def _upwind_shift_product(field, beta, source, axis):
    """Return target beta times the native field reconstructed upstream for -beta.

    Each shift cross-product transfer changes only this axis. C->V brackets
    current/upper samples; V->C brackets lower/current samples. The unlimited
    centered slope gives second-order reconstruction and needs two guards.
    """
    slope = .5 * (jnp.roll(field, -1, axis=axis+3)
                  - jnp.roll(field, 1, axis=axis+3))
    if source[axis] == "C":
        left = field + .5*slope
        right = jnp.roll(field - .5*slope, -1, axis=axis+3)
    else:
        left = jnp.roll(field + .5*slope, 1, axis=axis+3)
        right = field - .5*slope
    return beta * jnp.where(beta > 0, right, left)


def compute_covariant_E(D_tiles, B_tiles, metric):
    """Compute E_i from native densitized D/B and target-site geometry.

    The lapse-metric term averages D to the target, divides by its sqrt_gamma,
    lowers the index with target gamma, then multiplies target lapse. Only
    beta cross B uses upstream reconstruction, with transport velocity -beta.
    Metric samples and fields, including at least two guards, must be valid.
    """
    E_cov = []
    for i, target_location in enumerate(D_FIELD_LOCATIONS):
        sample = metric.D[i]
        D_on_target = [value / sample.sqrt_gamma for value in
                       reconstruct_vector(D_tiles, D_FIELD_LOCATIONS, target_location)]
        D_lower_i = 0.0
        for j in range(3):
            D_lower_i = D_lower_i + sample.gamma[..., i, j] * D_on_target[j]
        electric = sample.lapse * D_lower_i

        # E_shift = beta x B_density at D_i's native location.
        j, k = (i+1) % 3, (i+2) % 3
        electric += _upwind_shift_product(
            B_tiles[k], sample.shift[..., j], B_FIELD_LOCATIONS[k], j)
        electric -= _upwind_shift_product(
            B_tiles[j], sample.shift[..., k], B_FIELD_LOCATIONS[j], k)
        E_cov.append(electric)
    return tuple(E_cov)


def compute_covariant_H(D_tiles, B_tiles, metric):
    """Compute H_i with the same metric transfer and opposite shift-cross sign."""
    H_cov = []
    for i, target_location in enumerate(B_FIELD_LOCATIONS):
        sample = metric.B[i]
        B_on_target = [value / sample.sqrt_gamma for value in
                       reconstruct_vector(B_tiles, B_FIELD_LOCATIONS, target_location)]
        B_lower_i = 0.0
        for j in range(3):
            B_lower_i = B_lower_i + sample.gamma[..., i, j] * B_on_target[j]
        magnetic = sample.lapse * B_lower_i

        # H_shift = -beta x D_density at B_i's native location.
        j, k = (i+1) % 3, (i+2) % 3
        magnetic -= _upwind_shift_product(
            D_tiles[k], sample.shift[..., j], D_FIELD_LOCATIONS[k], j)
        magnetic += _upwind_shift_product(
            D_tiles[j], sample.shift[..., k], D_FIELD_LOCATIONS[j], k)
        H_cov.append(magnetic)
    return tuple(H_cov)


def update_D(D_tiles, H_tiles, J_tiles, metric, static_parameters, dynamic_parameters, dt, *, absorber=None):
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
        (Dx + dt * ((dHz_dy - dHy_dz) - 4.0 * jnp.pi * Jx) if absorber is None else
         sponge_advance(Dx, (dHz_dy - dHy_dz) - 4.0 * jnp.pi * Jx, dt, absorber[0][0], absorber[1][0])),
        Dx,
    )
    Dy = jnp.where(
        owned_nodes(Dy.shape, D_FIELD_LOCATIONS[1], static_parameters),
        (Dy + dt * ((dHx_dz - dHz_dx) - 4.0 * jnp.pi * Jy) if absorber is None else
         sponge_advance(Dy, (dHx_dz - dHz_dx) - 4.0 * jnp.pi * Jy, dt, absorber[0][1], absorber[1][1])),
        Dy,
    )
    Dz = jnp.where(
        owned_nodes(Dz.shape, D_FIELD_LOCATIONS[2], static_parameters),
        (Dz + dt * ((dHy_dx - dHx_dy) - 4.0 * jnp.pi * Jz) if absorber is None else
         sponge_advance(Dz, (dHy_dx - dHx_dy) - 4.0 * jnp.pi * Jz, dt, absorber[0][2], absorber[1][2])),
        Dz,
    )

    D_tiles = apply_tiled_supergaussian_absorber(
        (Dx, Dy, Dz), static_parameters, dynamic_parameters, dt,
        locations=D_FIELD_LOCATIONS,
    )
    return refresh_fields(D_tiles, static_parameters, D_FIELD_LOCATIONS, 'D', metric, absorber)


def update_B(E_tiles, B_tiles, metric, static_parameters, dynamic_parameters, dt, *, absorber=None):
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
        (Bx + dt * (-(dEz_dy - dEy_dz)) if absorber is None else
         sponge_advance(Bx, -(dEz_dy - dEy_dz), dt, absorber[0][0], absorber[1][0])),
        Bx,
    )
    By = jnp.where(
        owned_nodes(By.shape, B_FIELD_LOCATIONS[1], static_parameters),
        (By + dt * (-(dEx_dz - dEz_dx)) if absorber is None else
         sponge_advance(By, -(dEx_dz - dEz_dx), dt, absorber[0][1], absorber[1][1])),
        By,
    )
    Bz = jnp.where(
        owned_nodes(Bz.shape, B_FIELD_LOCATIONS[2], static_parameters),
        (Bz + dt * (-(dEy_dx - dEx_dy)) if absorber is None else
         sponge_advance(Bz, -(dEy_dx - dEx_dy), dt, absorber[0][2], absorber[1][2])),
        Bz,
    )

    B_tiles = apply_tiled_supergaussian_absorber(
        (Bx, By, Bz), static_parameters, dynamic_parameters, dt,
        locations=B_FIELD_LOCATIONS,
    )
    return refresh_fields(B_tiles, static_parameters, B_FIELD_LOCATIONS, 'B', metric, absorber)
