"""Synchronized physical energy, coordinate Killing flux, and constraints.

The C grid is nodal and the V grid is half-cell centered in PyPIC3D. Volume
quadrature therefore uses VVV, with exactly N^3 owned cells and no halos.
"""
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ownership import face_mask, owned_nodes
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import location_interpolate, metric_at_location
from .parameters import positions, kerr_radius


def collocate(vector, locations, target):
    return jnp.stack([location_interpolate(v, loc, target)
                      for v, loc in zip(vector, locations)], -1)


def local_fields(D, B, metric, target):
    """Density-weighted physical D/B and auxiliary covectors at one location."""
    sample = metric_at_location(metric, target)
    d = collocate(D, D_FIELD_LOCATIONS, target)/sample.sqrt_gamma[..., None]
    b = collocate(B, B_FIELD_LOCATIONS, target)/sample.sqrt_gamma[..., None]
    lower_d = jnp.einsum('...ij,...j->...i', sample.gamma, d)
    lower_b = jnp.einsum('...ij,...j->...i', sample.gamma, b)
    e = sample.lapse[..., None]*lower_d + sample.sqrt_gamma[..., None]*jnp.cross(sample.shift, b)
    h = sample.lapse[..., None]*lower_b - sample.sqrt_gamma[..., None]*jnp.cross(sample.shift, d)
    return d, b, lower_d, lower_b, e, h


def pec_residuals(D, B, solver):
    """Native FIDO projector residuals, including coupled D edge rows."""
    static, metric = solver.static, solver.metric
    result = {}
    for kind, vector, locations, samples in (
        ('D', D, D_FIELD_LOCATIONS, metric.D), ('B', B, B_FIELD_LOCATIONS, metric.B)
    ):
        maximum, scale = 0., 0.
        for i, loc in enumerate(locations):
            physical = vector[i]/samples[i].sqrt_gamma
            owned = owned_nodes(vector[i].shape, loc, static)
            scale = jnp.maximum(scale, jnp.max(jnp.where(owned, jnp.abs(physical), 0.)))
            reconstructed = collocate(vector, locations, loc)/samples[i].sqrt_gamma[..., None]
            incident = [axis for axis in range(3) if loc[axis] == 'C']
            masks = [face_mask(vector[i].shape, axis, static.tile_shape[axis], static.guard_cells)
                     for axis in incident]
            count = sum(mask.astype(jnp.int32) for mask in masks)
            for axis, wall in zip(incident, masks):
                expected = (samples[i].gamma_inv[..., i, axis]/samples[i].gamma_inv[..., axis, axis]
                            * reconstructed[..., axis]) if kind == 'D' else 0.
                expected = jnp.where(count > 1, 0., expected)
                residual = jnp.where(owned & wall, jnp.abs(physical-expected), 0.)
                maximum = jnp.maximum(maximum, jnp.max(residual))
        result[f'pec_{kind}_relative'] = maximum/jnp.maximum(scale, 1.e-300)
    return result


def measure(D, B, solver):
    """Input D and B must represent the same time."""
    p, static, dynamic, metric = solver.p, solver.static, solver.dynamic, solver.metric
    g, n, dx = static.guard_cells, p.cells, p.spacing
    cell = (0, 0, 0)+(slice(g, g+n),)*3
    target = ('V',)*3
    sample = metric.vertex
    d, b, lower_d, lower_b, e, h = local_fields(D, B, metric, target)
    density = jnp.sum(d*lower_d+b*lower_b, -1)*sample.sqrt_gamma/(8*jnp.pi)
    killing = jnp.sum(d*e+b*h, -1)*sample.sqrt_gamma/(8*jnp.pi)
    xyz = positions(dynamic, target)
    r = kerr_radius(xyz, p.a)
    active = jnp.ones_like(r, dtype=bool) if p.flat else r > p.core_radius
    exterior = jnp.ones_like(r, dtype=bool) if p.flat else r >= p.horizon
    fido_flux_density = jnp.cross(lower_d, lower_b)/(4*jnp.pi)
    radius = jnp.sqrt(jnp.sum(xyz*xyz, -1))
    radial_flux = jnp.sum(fido_flux_density*xyz, -1)/jnp.maximum(radius, 1.e-300)
    result = dict(
        fido_energy_total=jnp.sum(jnp.where(active, density, 0.)[cell])*dx**3,
        fido_energy_exterior=jnp.sum(jnp.where(exterior, density, 0.)[cell])*dx**3,
        killing_energy_exterior=jnp.sum(jnp.where(exterior, killing, 0.)[cell])*dx**3,
        radial_fido_flux_moment=jnp.sum(jnp.where(exterior, radial_flux, 0.)[cell])*dx**3,
    )
    # Oriented fluxes through the staircase surface separating r<r+ cells
    # from exterior cells. Positive horizon flux LEAVES the exterior into BH.
    # E x H is the coordinate-density Killing flux; do not multiply sqrt_gamma.
    outside = exterior[cell].astype(jnp.float64)
    wall_flux, horizon_flux = 0., 0.
    for axis, location in enumerate(B_FIELD_LOCATIONS):
        *_, eface, hface = local_fields(D, B, metric, location)
        flux = jnp.cross(eface, hface)[..., axis]/(4*jnp.pi)
        face = [slice(g, g+n)]*3
        face[axis] = slice(g, g+n+1)
        flux = flux[(0, 0, 0)+tuple(face)]
        wall_flux += (jnp.sum(jnp.take(flux, n, axis=axis))-
                      jnp.sum(jnp.take(flux, 0, axis=axis)))*dx**2
        middle = [slice(None)]*3
        middle[axis] = slice(1, n)
        horizon_flux += -jnp.sum(jnp.diff(outside, axis=axis)*flux[tuple(middle)])*dx**2
    result.update(killing_flux_wall_out=wall_flux, killing_flux_horizon_in=horizon_flux)

    # The electric divergence is at CCC; magnetic divergence is at VVV.
    # Exclude walls, horizon and their interpolation stencils, but not grid seams.
    for kind, vector, loc, forward in (('D', D, ('C',)*3, False), ('B', B, target, True)):
        q = positions(dynamic, loc)
        domain = jnp.all(jnp.abs(q) < p.half_width-2*dx, axis=-1)
        if not p.flat:
            domain &= kerr_radius(q, p.a) > p.horizon+2*3.**0.5*dx
        div = divergence(vector, dynamic, forward=forward)
        residual = jnp.max(jnp.where(domain, jnp.abs(div), 0.))
        scale = jnp.maximum(maximum_owned(vector, cell)/dx, 1.e-300)
        result[f'div_{kind}_absolute'] = residual
        result[f'div_{kind}_relative'] = residual/scale
        result[f'div_{kind}_cells'] = jnp.sum(domain)
    result.update(pec_residuals(D, B, solver))
    # Catch failures even if they occur in excluded interior/guard samples.
    result['finite_fields'] = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in (*D, *B)]))
    return result


def maximum_owned(vector, cell):
    return jnp.max(jnp.stack([jnp.max(jnp.abs(v[cell])) for v in vector]))


def snapshot(solver, state):
    D, B = solver.synchronized(state)
    result = measure(D, B, solver)
    # Constraints of evolved B live at the half-step, not at the reconstruction.
    raw = divergence(state[1], solver.dynamic, forward=True)
    q = positions(solver.dynamic, ('V',)*3)
    mask = jnp.all(jnp.abs(q) < solver.p.half_width-2*solver.p.spacing, axis=-1)
    if not solver.p.flat:
        mask &= kerr_radius(q, solver.p.a) > solver.p.horizon+2*3.**0.5*solver.p.spacing
    result['div_B_half_absolute'] = jnp.max(jnp.where(mask, jnp.abs(raw), 0.))
    result['finite_fields'] &= jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(v)) for field in state for v in field]))
    return result
