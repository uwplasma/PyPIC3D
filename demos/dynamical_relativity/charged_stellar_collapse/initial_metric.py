"""Time-symmetric radial Hamiltonian initial data, ported from RadiShPICR.

Gaussian G=c=1 units. Cartesian sampling is not a Cartesian constraint solve.
Run this before initial_data.py; copy the printed species mass into the TOML.
"""
from pathlib import Path
from types import SimpleNamespace
import math
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

TOTAL_MASS = 1.0
SURFACE_AREAL_RADIUS = 10.0
CHARGE_TO_MASS = 0.2
NX = 64
HALF_WIDTH = 16.0
OUTPUT_DIR = Path(__file__).with_name("initial_data")

def _radial_laplacian(r, dr):
    """Match the Z4C derivative stencils, omitting the final boundary row."""

    num_cells = r.size
    rows = np.repeat(np.arange(num_cells - 2), 5)
    columns = rows + np.tile(np.arange(-2, 3), num_cells - 2)
    # Even parity on a half-cell-centered grid: -1 -> 0 and -2 -> 1.
    columns = np.where(columns < 0, -columns - 1, columns)
    d1 = np.asarray((1.0, -8.0, 0.0, 8.0, -1.0)) / (12.0 * dr)
    d2 = np.asarray((-1.0, 16.0, -30.0, 16.0, -1.0)) / (12.0 * dr**2)
    values = (d2 + 2.0 * d1 / r[:-2, None]).ravel()

    # The penultimate row uses the same one-sided derivatives as evolution.
    d1_outer = np.asarray((0.0, -1/12, 1/2, -3/2, 5/6, 1/4)) / dr
    d2_outer = np.asarray((1/12, -1/2, 7/6, -1/3, -5/4, 5/6)) / dr**2
    rows = np.concatenate((rows, np.full(6, num_cells - 2)))
    columns = np.concatenate((columns, np.arange(num_cells - 6, num_cells)))
    values = np.concatenate((values, d2_outer + 2.0 * d1_outer / r[-2]))

    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(num_cells - 1, num_cells),
    )

def _linear_shape_weight(delta):
    """Evaluate the one-dimensional linear CIC shape function."""

    return jnp.maximum(1.0 - jnp.abs(delta), 0.0)

def _quadratic_shape_weight(delta):
    """Evaluate the one-dimensional quadratic TSC shape function."""

    absolute_delta = jnp.abs(delta)
    center_weight = 0.75 - delta**2
    outer_weight = 0.5 * (1.5 - absolute_delta) ** 2

    return jnp.where(
        absolute_delta <= 0.5,
        center_weight,
        jnp.where(absolute_delta <= 1.5, outer_weight, 0.0),
    )

def _unbounded_raw_radial_shape_stencil(
    radial_positions,
    radial_grid,
    dr,
    shape_mode="nearest",
):
    """Return raw compact radial indices and weights before boundary handling."""

    floating_index = (radial_positions - radial_grid[0]) / dr

    if shape_mode == "nearest":
        anchor = jnp.floor(floating_index + 0.5).astype(jnp.int32)
        raw_indices = anchor[jnp.newaxis, :]
        # The anchor already makes a deterministic right-sided choice at an
        # exact half-cell tie, so every particle carries one full NGP weight.
        raw_weights = jnp.ones_like(raw_indices, dtype=radial_positions.dtype)
    elif shape_mode == "linear":
        anchor = jnp.floor(floating_index).astype(jnp.int32)
        offsets = jnp.asarray([0, 1], dtype=anchor.dtype)
        raw_indices = anchor[jnp.newaxis, :] + offsets[:, jnp.newaxis]
        delta = (
            floating_index[jnp.newaxis, :]
            - raw_indices.astype(radial_positions.dtype)
        )
        raw_weights = _linear_shape_weight(delta)
    else:
        anchor = jnp.rint(floating_index).astype(jnp.int32)
        offsets = jnp.asarray([-1, 0, 1], dtype=anchor.dtype)
        raw_indices = anchor[jnp.newaxis, :] + offsets[:, jnp.newaxis]
        delta = (
            floating_index[jnp.newaxis, :]
            - raw_indices.astype(radial_positions.dtype)
        )
        raw_weights = _quadratic_shape_weight(delta)

    return raw_indices, raw_weights

def apply_stencil_boundaries(
    raw_indices, raw_weights, num_cells, inner_open=False, inner_boundary_index=0,
):
    """Fold at the half-cell origin or discard open-boundary ghost shares.

    No retained weights are renormalized. Odd radial quantities change sign
    under origin reflection; open-inner weights carry their original sign.
    """

    reflected_indices = jnp.where(raw_indices < 0, -raw_indices - 1, raw_indices)
    grid_indices = jnp.where(inner_open, raw_indices, reflected_indices)
    first_index = jnp.where(inner_open, inner_boundary_index, 0)
    physical = (grid_indices >= first_index) & (grid_indices < num_cells)

    indices = jnp.clip(grid_indices, 0, num_cells - 1)
    even_weights = jnp.where(physical, raw_weights, 0.0)
    reflection_sign = jnp.where(raw_indices < 0, -1.0, 1.0)
    odd_weights = even_weights * jnp.where(inner_open, 1.0, reflection_sign)

    return indices, even_weights, odd_weights

def proper_radial_shell_volume(metric):
    inner_radius = jnp.maximum(metric.r - 0.5 * metric.dr, 0.0)
    outer_radius = metric.r + 0.5 * metric.dr
    coordinate_volume = (4.0 * jnp.pi / 3.0) * (
        outer_radius**3 - inner_radius**3
    )

    proper_volume_factor = (
        jnp.sqrt(metric.conformal_grr)
        * metric.conformal_gt
        / metric.chi**1.5
    )

    return coordinate_volume * proper_volume_factor

def proper_radial_shell_quadrature(metric):
    """Return proper-volume quadrature on both halves of every shell."""

    gauss_abscissae = jnp.asarray(
        (-jnp.sqrt(3.0 / 5.0), 0.0, jnp.sqrt(3.0 / 5.0)),
        dtype=metric.r.dtype,
    )
    gauss_weights = jnp.asarray(
        (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0),
        dtype=metric.r.dtype,
    )

    inner_radius = jnp.maximum(metric.r - 0.5 * metric.dr, 0.0)
    outer_radius = metric.r + 0.5 * metric.dr

    # The Ruyten transfer changes intervals at a grid center.  Integrate the
    # two polynomial pieces separately with the same cell-centered metric
    # factor used by the production finite-volume shell.
    segment_inner = jnp.stack((inner_radius, metric.r), axis=1)
    segment_outer = jnp.stack((metric.r, outer_radius), axis=1)
    segment_center = 0.5 * (segment_inner + segment_outer)
    segment_half_width = 0.5 * (segment_outer - segment_inner)

    quadrature_radius = (
        segment_center[:, :, jnp.newaxis]
        + segment_half_width[:, :, jnp.newaxis]
        * gauss_abscissae[jnp.newaxis, jnp.newaxis, :]
    )
    proper_volume_factor = (
        jnp.sqrt(metric.conformal_grr)
        * metric.conformal_gt
        / metric.chi**1.5
    )
    quadrature_volume = (
        4.0
        * jnp.pi
        * proper_volume_factor[:, jnp.newaxis, jnp.newaxis]
        * quadrature_radius**2
        * segment_half_width[:, :, jnp.newaxis]
        * gauss_weights[jnp.newaxis, jnp.newaxis, :]
    )

    return quadrature_radius.reshape(-1), quadrature_volume.reshape(-1)

def metric_correction_coefficients(metric, inner_open, shape_mode="quadratic"):
    """Match deposited proper volume with a limited Ruyten transfer.

    CIC uses |beta| < 1; quadratic TSC uses |beta| < 3/2. Limiting keeps
    weights nonnegative but can leave a shell-volume residual. The final
    shell is excluded from the recurrence, preserving outer truncation.
    """

    quadrature_radius, quadrature_volume = proper_radial_shell_quadrature(
        metric
    )
    raw_indices, raw_weights = _unbounded_raw_radial_shape_stencil(
        quadrature_radius,
        metric.r,
        metric.dr,
        shape_mode=shape_mode,
    )

    # Calibrate on the full metric grid. The moving absorbing boundary is
    # applied only to the final particle stencil, not to this recurrence.
    indices, retained_weights, _ = apply_stencil_boundaries(
        raw_indices, raw_weights, metric.r.size, inner_open,
    )
    uncorrected_shell_volume = jnp.zeros_like(metric.r).at[indices].add(
        retained_weights * quadrature_volume[jnp.newaxis, :]
    )

    floating_index = (quadrature_radius - metric.r[0]) / metric.dr
    lower_index = jnp.floor(floating_index).astype(jnp.int32)
    upper_fraction = floating_index - lower_index.astype(metric.r.dtype)
    valid_interval = jnp.logical_and(
        lower_index >= 0,
        lower_index < metric.r.shape[0] - 1,
    )
    interval_index = jnp.clip(lower_index, 0, metric.r.shape[0] - 2)
    interval_moment = jnp.zeros_like(metric.r[:-1]).at[interval_index].add(
        jnp.where(
            valid_interval,
            quadrature_volume * upper_fraction * (1.0 - upper_fraction),
            0.0,
        )
    )

    shell_volume_error = (
        proper_radial_shell_volume(metric)[:-1]
        - uncorrected_shell_volume[:-1]
    )
    beta_limit = (1.0 if shape_mode == "linear" else 1.5) * (
        1.0 - 8.0 * jnp.finfo(metric.r.dtype).eps
    )

    def solve_one_interval(previous_transfer, interval_data):
        volume_error, volume_moment = interval_data
        unconstrained_beta = (
            volume_error + previous_transfer
        ) / volume_moment
        beta = jnp.clip(
            unconstrained_beta,
            -beta_limit,
            beta_limit,
        )
        transfer = beta * volume_moment
        return transfer, beta

    _, beta = jax.lax.scan(
        solve_one_interval,
        jnp.asarray(0.0, dtype=metric.r.dtype),
        (shell_volume_error, interval_moment),
    )

    return beta

def apply_metric_transfer(radial_positions, metric, beta, raw_indices, raw_weights):
    """Transfer beta*f*(1-f) from the upper center to the lower center.

    The equal and opposite changes preserve each raw stencil's total weight.
    Mirror beta with odd parity on negative-radius intervals; leave the
    interval straddling the origin and exterior intervals uncorrected.
    """

    floating_index = (radial_positions - metric.r[0]) / metric.dr
    lower_index = jnp.floor(floating_index).astype(jnp.int32)
    upper_fraction = floating_index - lower_index.astype(metric.r.dtype)

    # Reflection maps interval i to -i-2 on a half-cell-centered grid.
    # The origin interval maps to -1 and has no correction coefficient.
    beta_index = jnp.where(lower_index >= 0, lower_index, -lower_index - 2)
    valid_interval = (beta_index >= 0) & (beta_index < beta.size)
    reflection_sign = jnp.where(lower_index >= 0, 1.0, -1.0)
    interval_beta = jnp.where(
        valid_interval,
        reflection_sign * beta[jnp.clip(beta_index, 0, beta.size - 1)],
        0.0,
    )
    transfer = interval_beta * upper_fraction * (1.0 - upper_fraction)

    corrected_weights = raw_weights + jnp.where(
        raw_indices == lower_index[jnp.newaxis, :],
        transfer[jnp.newaxis, :],
        0.0,
    )
    corrected_weights = corrected_weights - jnp.where(
        raw_indices == lower_index[jnp.newaxis, :] + 1,
        transfer[jnp.newaxis, :],
        0.0,
    )

    return corrected_weights

def seed_profile(count=120000):
    R = jnp.linspace(0., SURFACE_AREAL_RADIUS, count)
    density = 3*TOTAL_MASS/(4*jnp.pi*SURFACE_AREAL_RADIUS**3)
    integrand = 4*jnp.pi*R**2*density/jnp.sqrt(
        1-2*TOTAL_MASS*R**2/SURFACE_AREAL_RADIUS**3)
    cumulative = jnp.concatenate((jnp.zeros(1), jnp.cumsum(
        .5*(integrand[1:]+integrand[:-1])*jnp.diff(R))))
    return R, cumulative


def solve_initial_metric(num_cells=4000, radial_limit=100., tolerance=1.e-12,
                         max_iterations=100):
    """Return radial coordinates, psi, covariant electric field, mass factor.

    Shell-stratified quadrature retains fixed areal radii and charges while
    the completed map and a uniform dust mass multiplier are solved together.
    """
    dr = radial_limit/num_cells
    r = (jnp.arange(num_cells)+.5)*dr
    R_seed, cumulative = seed_profile()
    rest_mass = float(cumulative[-1])
    surface = .5*(SURFACE_AREAL_RADIUS-TOTAL_MASS+math.sqrt(
        SURFACE_AREAL_RADIUS*(SURFACE_AREAL_RADIUS-2*TOTAL_MASS)))
    root = jnp.sqrt(1-2*TOTAL_MASS*R_seed**2/SURFACE_AREAL_RADIUS**3)
    seed_r = surface*(R_seed/(1+root))/(SURFACE_AREAL_RADIUS/(
        1+math.sqrt(1-2*TOTAL_MASS/SURFACE_AREAL_RADIUS)))
    edges = jnp.minimum(jnp.arange(math.ceil(surface/dr)+1)*dr,surface)
    edge_mass = jnp.interp(jnp.interp(edges,seed_r,R_seed),R_seed,cumulative)
    counts = np.full(len(edges)-1,90000//(len(edges)-1),dtype=int)
    counts[:90000 % len(counts)] += 1
    mass_samples = jnp.concatenate([a+(jnp.arange(n)+.5)/n*(b-a)
        for a,b,n in zip(edge_mass[:-1],edge_mass[1:],counts)])
    weights = jnp.concatenate([jnp.full(n,(b-a)/n)
        for a,b,n in zip(edge_mass[:-1],edge_mass[1:],counts)])
    particle_R = jnp.interp(mass_samples,cumulative,R_seed)
    charge = CHARGE_TO_MASS*rest_mass
    rn = jnp.sqrt((1+TOTAL_MASS/(2*r[-2:]))**2-charge**2/(4*r[-2:]**2))
    B = TOTAL_MASS/(2*surface**3)
    psi = jnp.where(r<=surface,(1+TOTAL_MASS/(2*surface))**1.5/
        jnp.sqrt(1+B*r*r),1+TOTAL_MASS/(2*r))

    @jax.jit
    def sources(psi):
        metric = SimpleNamespace(r=r,dr=dr,chi=psi**-4,
                                 conformal_grr=jnp.ones_like(r),conformal_gt=jnp.ones_like(r))
        particle_r = jnp.interp(particle_R,jnp.concatenate((jnp.zeros(1),r*psi**2)),
                                jnp.concatenate((jnp.zeros(1),r)))
        indices, shape = _unbounded_raw_radial_shape_stencil(particle_r,r,dr,'quadratic')
        beta = metric_correction_coefficients(metric,True)
        shape = apply_metric_transfer(particle_r,metric,beta,indices,shape)
        indices, shape, _ = apply_stencil_boundaries(indices,shape,num_cells,True)
        volume = proper_radial_shell_volume(metric)
        shell_mass = jnp.zeros_like(r).at[indices].add(shape*weights[None,:])
        rho = shell_mass/volume
        flux = jnp.concatenate((jnp.zeros(1),jnp.cumsum(CHARGE_TO_MASS*shell_mass)))
        chi_faces = jnp.concatenate((metric.chi[:1],.5*(metric.chi[:-1]+metric.chi[1:]),metric.chi[-1:]))
        faces = jnp.arange(num_cells+1)*dr
        E_faces = jnp.where(faces>0,flux*jnp.sqrt(chi_faces)/jnp.maximum(faces,dr)**2,0.)
        E_r = .5*(E_faces[:-1]+E_faces[1:])
        return rho, psi**-4*E_r**2/(8*jnp.pi), E_r

    laplacian = _radial_laplacian(np.asarray(r),dr)
    exterior = sparse.csr_matrix((np.ones(2),(np.arange(2),np.arange(num_cells-2,num_cells))),shape=(2,num_cells))
    factor = 1.
    for iteration in range(max_iterations):
        if not bool(jnp.all(jnp.diff(r*psi**2)>0)) or not bool(jnp.all(psi>0)):
            raise RuntimeError('Hamiltonian iteration lost the monotone positive radial map')
        rho, rho_em, _ = sources(psi)
        dust = np.asarray(2*jnp.pi*psi**5*rho)
        electric = np.asarray(2*jnp.pi*psi**5*rho_em)
        operator = sparse.bmat([[laplacian,sparse.csr_matrix(dust[:-1,None])],
                               [exterior,sparse.csr_matrix((2,1))]],format='csr')
        solution = spsolve(operator,np.concatenate((-electric[:-1],np.asarray(rn))))
        change = max(float(jnp.max(jnp.abs(solution[:-1]-psi))),abs(solution[-1]-factor))
        psi = .5*(psi+solution[:-1])
        factor = .5*(factor+solution[-1])
        if change<tolerance:
            break
    else:
        raise RuntimeError('Hamiltonian solve failed to converge')
    if not math.isfinite(factor) or factor<=0:
        raise RuntimeError('Hamiltonian solve produced invalid rest mass')
    rho,rho_em,E_r = sources(psi)
    residual = laplacian@np.asarray(psi)+np.asarray(2*jnp.pi*psi**5*(factor*rho+rho_em))[:-1]
    print(f'Radial solve: {iteration+1} iterations, residual {np.max(np.abs(residual)):.6g}')
    return r,psi,E_r,float(factor)


def main():
    jax.config.update('jax_enable_x64',True)
    r,psi,E_r,factor = solve_initial_metric()
    R,cumulative = seed_profile()
    OUTPUT_DIR.mkdir(exist_ok=True)
    jnp.savez(OUTPUT_DIR/'radial_profile.npz',r=r,psi=psi,E_r=E_r,
              mass_factor=factor,seed_R=R,seed_mass=cumulative,
              total_mass=TOTAL_MASS,total_charge=CHARGE_TO_MASS*cumulative[-1])
    axis = -HALF_WIDTH+jnp.arange(NX)*(2*HALF_WIDTH/NX)
    x,y,z = jnp.meshgrid(axis,axis,axis,indexing='ij')
    radius = jnp.sqrt(x*x+y*y+z*z)
    W = jnp.interp(radius,r,psi)**-2
    jnp.save(OUTPUT_DIR/'conformal_factor.npy',W)
    jnp.save(OUTPUT_DIR/'conformal_metric.npy',jnp.eye(3)[:,:,None,None,None]*jnp.ones_like(W))
    jnp.save(OUTPUT_DIR/'lapse.npy',jnp.ones_like(W))
    print(f'Species mass = {factor:.17g}; charge = {CHARGE_TO_MASS}; weight = {float(cumulative[-1])/262144:.17g}')


if __name__ == '__main__':
    main()
