"""Finite-difference constraints and radial particle/charge budgets."""
from typing import NamedTuple
import jax.numpy as jnp
from PyPIC3D.utilities.grids import grid_domain_bounds
from PyPIC3D.boundary_conditions.grid_and_stencil import BC_ABSORBING, BC_CONDUCTING


def divergence(vector, dynamic, *, forward=False):
    """Coordinate divergence of native densities; no metric multiplication."""
    result = jnp.zeros_like(vector[0])
    for axis, (value, spacing) in enumerate(zip(vector, (dynamic.dx, dynamic.dy, dynamic.dz))):
        difference = (jnp.roll(value, -1, axis=axis+3)-value if forward else
                      value-jnp.roll(value, 1, axis=axis+3))
        result = result+difference/spacing
    return result


def node_weights(static, template):
    """Nodal quadrature on reflecting endpoints; ordinary ownership elsewhere."""
    weight = jnp.zeros_like(template)
    g = static.guard_cells
    weight = weight.at[(slice(None),)*3+(slice(g, -g),)*3].set(1.)
    for axis, bc in enumerate(static.particle_boundary_conditions):
        if bc != BC_CONDUCTING:
            continue
        low = [slice(None)]*6
        high = low.copy()
        low[axis], high[axis] = 0, -1
        low[axis+3], high[axis+3] = g, g+static.tile_shape[axis]
        inner = high.copy(); inner[axis+3] -= 1
        weight = weight.at[tuple(high)].set(.5*weight[tuple(inner)])
        weight = weight.at[tuple(low)].multiply(.5)
    return weight


class BoundaryDiagnostics(NamedTuple):
    """Axis-0 absorption budget and global push validity.

    Counts and charges are indexed by (lower/upper radial side, species).
    The budget deliberately measures radial absorbers only; it is not a
    surface budget for every absorbing axis of a general simulation.
    """
    absorbed_count: object
    absorbed_charge: object
    removed_grid_charge: object
    radial_current_outflow: object
    invalid_push: object


def step_diagnostics(old, new, current, species, static, dynamic):
    """Measure radial particle loss, removed cloud charge, and face flux.

    The radial coordinate is axis 0. ``current`` is the densitized J. Keep
    particle removal separate from current outflow: a lost particle's shape
    may still overlap the grid.
    """
    from PyPIC3D.deposition.rho import compute_rho
    g = static.guard_cells
    lower, upper = grid_domain_bounds(dynamic)[0]
    absorbing = static.particle_boundary_conditions[0] == BC_ABSORBING
    low = new.active & (new.x[..., 0] < lower) & absorbing
    high = new.active & (new.x[..., 0] > upper) & absorbing
    counts = jnp.stack([jnp.sum(a, axis=(0, 1, 2, 4)) for a in (low, high)])
    charges = counts*(species.charge*species.weight)[None, :]
    lost = new._replace(active=low | high)
    rho = compute_rho(lost, species, jnp.zeros_like(current[0]), static, dynamic)
    weights = node_weights(static, rho)
    removed = rho*weights*dynamic.dx*dynamic.dy*dynamic.dz
    flux = current[0]*dynamic.dy*dynamic.dz
    # Transverse nodal quadrature includes reflecting boundary nodes once.
    last = g+static.tile_shape[0]-1
    out = jnp.array([-jnp.sum(flux[0, :, :, g-1]*weights[0, :, :, g]),
                    jnp.sum(flux[-1, :, :, last]*weights[-1, :, :, last])])
    out = jnp.where(absorbing, out, 0.)
    displacement = jnp.abs(new.x-old.x)/jnp.array([dynamic.dx, dynamic.dy, dynamic.dz])
    invalid = jnp.any(old.active & (jnp.any(displacement > 1+1e-12, axis=-1) |
                       jnp.any(~jnp.isfinite(new.x), axis=-1) |
                       jnp.any(~jnp.isfinite(new.u), axis=-1)))
    return BoundaryDiagnostics(counts, charges, removed, out, invalid)
