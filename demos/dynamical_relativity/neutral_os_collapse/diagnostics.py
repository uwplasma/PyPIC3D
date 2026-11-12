"""Cartesian-ray diagnostics for the neutral Oppenheimer-Snyder demo.

The evolved conformal factor is W. Saved ``chi`` is W**2, and the radial
components below are projections of the Cartesian metric, not a radial solve.
"""

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def extract_profiles(state, params):
    """Sample six axis rays and eight body diagonals at common radii.

    The first ray is +x and supplies the short names used by the plotters.
    Linear interpolation includes the true coordinate origin even when it
    lies between nodes. No spherical averaging changes the evolved data.
    """
    axes = jnp.asarray(((1., 0., 0.), (-1., 0., 0.), (0., 1., 0.),
                        (0., -1., 0.), (0., 0., 1.), (0., 0., -1.)))
    diagonals = jnp.asarray(((1., 1., 1.), (1., 1., -1.), (1., -1., 1.),
                             (1., -1., -1.), (-1., 1., 1.), (-1., 1., -1.),
                             (-1., -1., 1.), (-1., -1., -1.))) / jnp.sqrt(3.)
    directions = jnp.concatenate((axes, diagonals))
    origin = jnp.asarray((params.x_min, params.y_min, params.z_min))
    upper = origin + (jnp.asarray(state.lapse.shape) - 1) * params.dx
    radial_limit = jnp.min(jnp.minimum(-origin, upper))
    count = (min(state.lapse.shape) - 1) // 2 + 1
    r = jnp.linspace(0., radial_limit, count)
    coordinates = (directions[:, :, None] * r - origin[None, :, None]) / params.dx

    def sample(field):
        return jax.vmap(lambda points: map_coordinates(
            field, points, order=1, mode="nearest"))(coordinates)

    alpha = sample(state.lapse)
    W = sample(state.conformal_factor)
    shift = jax.vmap(sample)(state.shift)
    metric = jax.vmap(jax.vmap(sample))(state.conformal_metric)
    radial_metric = jnp.einsum("ai,ijar,aj->ar", directions, metric, directions)
    tangential_metric = (jnp.einsum("iiar->ar", metric) - radial_metric) / 2.
    beta = jnp.einsum("ai,iar->ar", directions, shift)

    projector = jnp.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    tangential_tensor = jnp.einsum("aij,jkar,akl->ilar", projector, metric, projector)
    tangential_anisotropy = tangential_tensor - jnp.einsum(
        "ar,aij->ijar", tangential_metric, projector)
    radial_tangent = jnp.einsum("aij,jkar,ak->iar", projector, metric, directions)
    shift_tangent = shift - jnp.einsum("ai,ar->iar", directions, beta)
    result = dict(r=r, ray_directions=directions,
                  ray_alpha=alpha, ray_beta=beta, ray_chi=W**2,
                  ray_conformal_grr=radial_metric, ray_conformal_gt=tangential_metric,
                  ray_transverse_anisotropy=jnp.sqrt(jnp.sum(tangential_anisotropy**2, axis=(0, 1)))
                    / (jnp.sqrt(2.) * jnp.abs(tangential_metric)),
                  ray_radial_tangent_ratio=jnp.sqrt(jnp.sum(radial_tangent**2, axis=0))
                    / jnp.abs(radial_metric),
                  ray_shift_tangent=jnp.sqrt(jnp.sum(shift_tangent**2, axis=0)))
    for name in ("alpha", "beta", "chi", "conformal_grr", "conformal_gt"):
        result[name] = result["ray_" + name][0]
    return result


def radial_derivative(field, dr):
    """Fourth-order derivative on an origin-inclusive uniform nodal ray.

    One-sided stencils at both ends avoid imposing half-cell parity or
    extrapolating through the puncture in the late-time trumpet geometry.
    """
    derivative = (jnp.roll(field, 2) - 8*jnp.roll(field, 1)
                  + 8*jnp.roll(field, -1) - jnp.roll(field, -2)) / (12*dr)
    derivative = derivative.at[0].set(
        (-25*field[0] + 48*field[1] - 36*field[2] + 16*field[3] - 3*field[4]) / (12*dr))
    derivative = derivative.at[1].set(
        (-3*field[0] - 10*field[1] + 18*field[2] - 6*field[3] + field[4]) / (12*dr))
    derivative = derivative.at[-2].set(
        (-field[-5] + 6*field[-4] - 18*field[-3] + 10*field[-2] + 3*field[-1]) / (12*dr))
    derivative = derivative.at[-1].set(
        (3*field[-5] - 16*field[-4] + 36*field[-3] - 48*field[-2] + 25*field[-1]) / (12*dr))
    return derivative


def reconstruct_metric(fields):
    """Return R, stationary g_RR, g_tt and their numerical exterior mask.

    A height-function time shift removes the radial spacetime cross term.
    It is valid for a stationary spherical final state; the caller reports
    measured symmetry and stationarity before interpreting this comparison.
    """
    r = jnp.asarray(fields["r"])
    alpha, beta = jnp.asarray(fields["alpha"]), jnp.asarray(fields["beta"])
    chi = jnp.asarray(fields["chi"])
    gamma_rr = jnp.asarray(fields["conformal_grr"]) / chi
    gamma_T = jnp.asarray(fields["conformal_gt"]) / chi
    R = r * jnp.sqrt(gamma_T)
    dR_dr = radial_derivative(R, r[1] - r[0])
    beta_r = gamma_rr * beta
    D = alpha**2 - beta_r * beta
    g_RR = (gamma_rr + beta_r**2 / D) / dR_dr**2
    outward = jnp.arange(r.size) >= jnp.maximum(2, jnp.argmin(R))
    # A horizon can leave a tiny positive D through cancellation. Do not turn
    # roundoff at its pole into an enormous apparently valid metric sample.
    resolved_D = 32*jnp.finfo(D.dtype).eps * (alpha**2 + jnp.abs(beta_r * beta))
    valid = (outward & (dR_dr > 0.) & (D > resolved_D) & (g_RR > 0.)
             & jnp.isfinite(R) & jnp.isfinite(g_RR) & jnp.isfinite(D))
    return dict(R=R, g_RR=jnp.where(valid, g_RR, jnp.nan),
                g_tt=jnp.where(valid, -D, jnp.nan), D=D, valid=valid)


def os_lapse_reference(mass, surface_areal_radius, count=1000):
    """Unit-initial-lapse OS lower limit, through its first zero (Eq. 97)."""
    eta_zero = jnp.arccos(2*jnp.exp(-1/6) - 1.)
    eta = jnp.linspace(0., eta_zero, count)
    radius_ratio = .5 * (1. + jnp.cos(eta))
    tau_over_mass = jnp.sqrt((surface_areal_radius / mass)**3 / 8.) * (eta + jnp.sin(eta))
    return tau_over_mass, 1. + 6*jnp.log(radius_ratio)
