"""BZ exterior constraints and integer-time physical diagnostic snapshots."""
from functools import partial
import math

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.diagnostics.static_metric import divergence, node_weights
from PyPIC3D.relativity.field_state import physical_vector
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import location_interpolate as _location_interpolate
from PyPIC3D.solvers.GR_yee.static_metric import compute_covariant_E, compute_covariant_H, update_B

if __package__:
    from .current_filter import smooth_conformal
    from .magnetization import measure_magnetization, collocate_magnetic_field
else:
    from current_filter import smooth_conformal
    from magnetization import measure_magnetization, collocate_magnetic_field


def conformal_charge(particles, species, template, static, dynamic, current_filter_passes):
    """Deposit charge and apply the same conformal filter as the current."""
    charge = compute_rho(particles, species, template, static, dynamic)
    if current_filter_passes:
        charge = smooth_conformal(charge, static, current_filter_passes)
    return charge


def exterior_mask(metric, static, dynamic, p, *, magnetic=False):
    """Owned exterior cells checked for Gauss/div B; tile seams stay included.

    Excludes the horizon interior, two cells at each physical boundary, and the
    sponge plus two cells.
    """
    g = static.guard_cells
    if magnetic:
        owned = jnp.zeros_like(metric.center.sqrt_gamma, dtype=bool).at[
            (slice(None),)*3+(slice(g,-g),slice(g,-g),slice(g,g+1))].set(True)
        grids = dynamic.grids.tiled_vertex_grid
    else:
        owned = node_weights(static, metric.center.sqrt_gamma) > 0
        grids = dynamic.grids.tiled_center_grid
    r = grids[0][..., :, None, None]
    theta = grids[1][..., None, :, None]
    interior = owned & (r >= p.r_min+2*p.dr) & (r <= p.r_max-2*p.dr)
    interior &= r >= p.horizon
    interior &= (r < p.sponge_start-2*p.dr)
    interior &= (theta >= p.theta_start+2*p.dtheta) & (theta <= p.theta_end-2*p.dtheta)
    return interior


def constraint_residuals(particles, species, fields, static, dynamic, p, *, current_filter_passes=0):
    """Conformal FD residuals, reported dimensionally and in cell-flux units."""
    metric = fields[6]
    charge = conformal_charge(particles, species, fields[3], static, dynamic, current_filter_passes)
    divD = divergence(fields[0], dynamic)
    divB = divergence(fields[1], dynamic, forward=True)
    return _constraint_residuals(divD, divB, charge, fields[1], metric, static, dynamic, p)


def _constraint_residuals(divD, divB, charge, magnetic_field, metric, static, dynamic, p):
    """Normalize constraints on the same exterior region for checks and output."""
    dm = exterior_mask(metric, static, dynamic, p)
    bm = exterior_mask(metric, static, dynamic, p, magnetic=True)
    volume = dynamic.dx*dynamic.dy*dynamic.dz
    def maximum(value, mask):
        return jnp.max(jnp.where(mask, jnp.abs(value), 0.))
    gauss_abs = maximum(divD-4*jnp.pi*charge, dm)
    magnetic_abs = maximum(divB, bm)
    dscale = jnp.maximum(1., jnp.maximum(maximum(divD*volume, dm), maximum(4*jnp.pi*charge*volume, dm)))
    areas = (dynamic.dy*dynamic.dz, dynamic.dx*dynamic.dz, dynamic.dx*dynamic.dy)
    bscale = jnp.maximum(1., jnp.max(jnp.array([
        maximum(b*a, bm) for b, a in zip(magnetic_field, areas)])))
    return dict(gauss=jnp.where(jnp.any(dm), gauss_abs*volume/dscale, jnp.nan),
                magnetic_divergence=jnp.where(jnp.any(bm), magnetic_abs*volume/bscale, jnp.nan),
                gauss_absolute=gauss_abs, magnetic_divergence_absolute=magnetic_abs,
                gauss_cells=jnp.sum(dm), magnetic_divergence_cells=jnp.sum(bm))


def validate_constraint_settings(gauss_tolerance, magnetic_divergence_tolerance,
                                 constraint_check_interval=100):
    for name, value in (('gauss_tolerance', gauss_tolerance),
                        ('magnetic_divergence_tolerance', magnetic_divergence_tolerance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if (isinstance(constraint_check_interval, bool)
            or not isinstance(constraint_check_interval, (int, np.integer))
            or constraint_check_interval <= 0):
        raise ValueError('constraint_check_interval must be a positive integer')


def check_constraints(residuals, *,
                      gauss_tolerance=1e-10, magnetic_divergence_tolerance=1e-10):
    """Only exterior residuals participate in constraint acceptance."""
    validate_constraint_settings(gauss_tolerance, magnetic_divergence_tolerance)
    exceeded = []
    for name, tolerance in (('gauss', gauss_tolerance),
                            ('magnetic_divergence', magnetic_divergence_tolerance)):
        value = float(residuals[name])
        if residuals.get(name+'_cells', 1) <= 0 or not math.isfinite(value):
            raise FloatingPointError(f'Nonfinite exterior {name} or empty exterior diagnostic region')
        if value >= tolerance:
            exceeded.append(f'{name}={value:.17g} >= {tolerance:.17g}')
    if exceeded:
        raise RuntimeError('Exterior divergence acceptance failed: '+', '.join(exceeded))


def assemble(array, g=3, theta_base=True):
    """Host-only output boundary; production stepping never assembles the mesh."""
    host = np.asarray(jax.device_get(array))
    return np.concatenate([host[i, 0, 0, g:-g, g:(-g+1 if theta_base else -g), g] for i in range(host.shape[0])], axis=0)


def diagnostics(particles, species, fields, p, static, dynamic, *, current_filter_passes=0):
    assemble_owned = partial(assemble, g=static.guard_cells)
    D, B, _, _, _, _, metric, previous, _ = fields
    # Reconstruct B at the integer D/position time using the production field stage.
    Dhalf = tuple((D[i]+previous[0][i])/2 for i in range(3))
    Bminus = tuple((B[i]+previous[1][i])/2 for i in range(3))
    # update_B returns refreshed densities
    B = update_B(compute_covariant_E(Dhalf, B, metric), Bminus, metric, static, dynamic, dynamic.dt)
    E = compute_covariant_E(D, B, metric)
    H = compute_covariant_H(D, B, metric)
    E = tuple(_location_interpolate(E[i], D_FIELD_LOCATIONS[i], ("C",)*3) for i in range(3))
    H = tuple(_location_interpolate(H[i], B_FIELD_LOCATIONS[i], ("C",)*3) for i in range(3))
    # the D^2/B^2, D.B and magnetization panels read the physical field
    physical_D = physical_vector(D, metric.D)
    physical_B = physical_vector(B, metric.B)
    bc = collocate_magnetic_field(physical_B)
    dc = jnp.stack([_location_interpolate(physical_D[i], D_FIELD_LOCATIONS[i], ("C",)*3)
                    for i in range(3)], axis=-1)
    d2 = jnp.einsum('...i,...ij,...j->...', dc, metric.center.gamma, dc)
    db = jnp.einsum('...i,...ij,...j->...', dc, metric.center.gamma, bc)
    mag = measure_magnetization(particles, species, physical_B, metric, static, dynamic)
    b2_safe = jnp.maximum(mag.magnetic_squared, jnp.finfo(d2.dtype).tiny)
    determinant = metric.center.sqrt_gamma
    denominator = determinant*bc[..., 0]
    omega = jnp.where(jnp.abs(denominator)>1e-12, -E[1]/denominator, jnp.nan)
    r = np.asarray(dynamic.grids.center[0][1:-1])
    theta = np.asarray(dynamic.grids.center[1][1:])
    # Uniform quadrature over one physical meridian, never both theta copies.
    sector = (theta>=p.theta_start-1e-12)&(theta<=p.theta_end+1e-12)
    flux = assemble_owned((E[1]*H[2]-E[2]*H[1])/(4*jnp.pi))
    luminosity = 2*np.pi*np.trapezoid(flux[:,sector],theta[sector],axis=1)
    divD = divergence(D, dynamic)
    divB = divergence(B, dynamic, forward=True)
    rho = conformal_charge(particles, species, fields[3], static, dynamic, current_filter_passes)
    constraints = assemble_owned(divD - 4*jnp.pi*rho)
    # Runtime acceptance uses evolved staggered B. Snapshot panels use the
    # integer-time reconstruction above; their magnetic residuals can differ.
    residuals = _constraint_residuals(
        divD, divergence(fields[1], dynamic, forward=True), rho,
        fields[1], metric, static, dynamic, p,
    )
    return dict(r=r, theta=theta, Hphi=assemble_owned(H[2]), omega=assemble_owned(omega),
                Br=assemble_owned(bc[..., 0]), Btheta=assemble_owned(bc[..., 1]),
                radial_flux=assemble_owned(determinant*bc[..., 0]),
                sigma=assemble_owned(mag.sigma), density=assemble_owned(jnp.sum(mag.number_density, axis=0)),
                D2_over_B2=assemble_owned(d2/b2_safe), DdotB_over_B2=assemble_owned(db/b2_safe),
                luminosity=luminosity, gauss=constraints, divB=assemble_owned(divB,theta_base=False),
                gauss_relative=np.asarray(residuals['gauss']),
                divB_relative=np.asarray(residuals['magnetic_divergence']),
                gauss_absolute=np.asarray(residuals['gauss_absolute']),
                divB_absolute=np.asarray(residuals['magnetic_divergence_absolute']),
                active_per_tile=np.asarray(jnp.sum(particles.active, axis=(-1, -2))).reshape(-1))


