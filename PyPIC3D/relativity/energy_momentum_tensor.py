"""Eulerian PIC and electromagnetic sources for the BSSN equations.

The returned tuple is ``(rho, S_i, S_ij)`` on the global BSSN grid, with
component axes first. These are physical densities, including particle rest
mass, in Gaussian electromagnetic units with c=1.
"""

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import BC_TYPE_PARTICLE, fold_tiled_ghost_cells
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    collapse_tiled_axis_stencil,
    prepare_particle_axis_stencil,
)
from PyPIC3D.deposition.shapes import get_first_order_weights, get_second_order_weights
from PyPIC3D.particles.particle_batching import (
    number_of_particle_batches,
    particle_batch_indices,
    prepare_particle_batches,
)
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS, covariant_lorentz_factor
from PyPIC3D.relativity.filtering import binomial_source_filter, filter_gr_grid, particle_metric
from PyPIC3D.relativity.interpolate_metric import (
    interpolate_metric,
    particle_active_axes,
    safe_inactive_positions,
)


@partial(jax.jit, static_argnames="static_parameters")
def electromagnetic_moments(displacement, magnetic, static_parameters):
    """Average native density products onto BSSN C nodes.

    D and B must be synchronized in time and have refreshed halos. Products
    are formed before averaging the eight shared Yee corners, retaining the
    energy of fields whose cell-centered mean vanishes.
    """
    from PyPIC3D.relativity.dynamical_relativity import tiles_to_bssn

    shape = tiles_to_bssn(displacement[0], static_parameters).shape
    dd = jnp.zeros((3, 3) + shape, dtype=displacement[0].dtype)
    bb = jnp.zeros_like(dd)
    db = jnp.zeros_like(dd)

    for corner in product((0, 1), repeat=3):
        d = jnp.stack(tuple(
            tiles_to_bssn(jnp.roll(
                value,
                tuple(c if loc == "V" else 0 for c, loc in zip(corner, location)),
                axis=(3, 4, 5),
            ), static_parameters)
            for value, location in zip(displacement, D_FIELD_LOCATIONS)
        ))
        b = jnp.stack(tuple(
            tiles_to_bssn(jnp.roll(
                value,
                tuple(c if loc == "V" else 0 for c, loc in zip(corner, location)),
                axis=(3, 4, 5),
            ), static_parameters)
            for value, location in zip(magnetic, B_FIELD_LOCATIONS)
        ))
        dd = dd + jnp.einsum("i...,j...->ij...", d, d) / 8.0
        bb = bb + jnp.einsum("i...,j...->ij...", b, b) / 8.0
        db = db + jnp.einsum("i...,j...->ij...", d, b) / 8.0
    return dd, bb, db


@jax.jit
def compute_electromagnetic_energy_momentum(moments, bssn_state):
    """Contract fixed density moments with the current BSSN stage geometry."""
    from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE

    dd, bb, db = moments
    W = jnp.maximum(bssn_state.conformal_factor, W_FLOOR_VALUE)
    conformal = bssn_state.conformal_metric
    determinant = jnp.linalg.det(jnp.moveaxis(conformal, (0, 1), (-2, -1)))
    conformal = conformal * determinant ** (-1.0 / 3.0)

    q = dd + bb
    contraction = jnp.einsum("ij...,ij...->...", conformal, q)
    rho = W**4 * contraction / (8.0 * jnp.pi)
    momentum = W**3 * jnp.stack((
        db[1, 2] - db[2, 1],
        db[2, 0] - db[0, 2],
        db[0, 1] - db[1, 0],
    )) / (4.0 * jnp.pi)
    stress = W**2 * (
        0.5 * conformal * contraction
        - jnp.einsum("ik...,jl...,kl...->ij...", conformal, conformal, q)
    ) / (4.0 * jnp.pi)
    return rho, momentum, stress


@partial(jax.jit, static_argnames="static_parameters")
def compute_particle_energy_momentum(
    particles, species_config, metric, static_parameters, dynamic_parameters,
):
    """Deposit synchronized particles as physical rho, S_i and S_ij.

    Positions and covariant momenta must refer to the same time. Deposit
    ``m*w*(Gamma, u_i, u_i*u_j/Gamma)/d^3x`` with the configured CIC/TSC
    shape, fold tile ghosts, then divide by the destination proper volume.
    Neutral and position-frozen species still contribute their rest mass.
    Gamma uses the same filtered particle metric as the pusher. Spatial
    source filtering is applied once, after adding PIC and EM contributions
    in compute_energy_momentum_tensor; destination volumes remain unfiltered.
    """
    from PyPIC3D.relativity.dynamical_relativity import tiles_to_bssn

    if not particles.active.size:
        rho = jnp.zeros_like(tiles_to_bssn(metric.center.sqrt_gamma, static_parameters))
        return rho, jnp.zeros((3,) + rho.shape, rho.dtype), jnp.zeros((3, 3) + rho.shape, rho.dtype)

    g = int(static_parameters.guard_cells)
    tile_shape = tuple(int(width) for width in static_parameters.tile_shape)
    local_shape = tuple(width + 2 * g for width in tile_shape)
    tile_counts = particles.active.shape[:3]
    active_axes = particle_active_axes(tile_counts, tile_shape)
    spacing = (dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)
    cell_volume = dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
    weighted_mass = species_config.mass * species_config.weight / cell_volume
    tiled_grid = dynamic_parameters.grids.tiled_center_grid
    stress_indices = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    particle_center = particle_metric(metric, static_parameters, dynamic_parameters)

    def deposit_one_tile(x_tile, u_tile, active_tile, tx, ty, tz):
        capacity, batch_size, active_indices, n_active = prepare_particle_batches(
            active_tile, static_parameters.particle_batch_size,
        )
        moments = jnp.zeros((10,) + local_shape, dtype=u_tile.dtype)
        if capacity == 0:
            return moments

        x = x_tile.reshape(-1, 3)
        u = u_tile.reshape(-1, 3)
        slots_per_species = active_tile.shape[-1]
        grid = tuple(axis[tx, ty, tz] for axis in tiled_grid)
        tile_metric = jax.tree.map(lambda array: array[tx, ty, tz], particle_center)
        n_batches = number_of_particle_batches(n_active, batch_size)

        def deposit_batch(batch_index, moments):
            indices, valid = particle_batch_indices(
                active_indices, n_active, batch_index, batch_size,
            )
            position = safe_inactive_positions(x[indices], valid, grid, active_axes, g)
            u_cov = jnp.where(valid[:, None], u[indices], 0.0)
            sampled = interpolate_metric(
                tile_metric, position, grid, static_parameters.metric,
                active_axes, (g, g, g), derivatives=False,
            )
            Gamma = covariant_lorentz_factor(u_cov, sampled.gamma_inv)
            mass = jnp.where(valid, weighted_mass[indices // slots_per_species], 0.0)
            particle_moments = jnp.stack((
                Gamma, u_cov[:, 0], u_cov[:, 1], u_cov[:, 2],
                *(u_cov[:, i] * u_cov[:, j] / Gamma for i, j in stress_indices),
            )) * mass[None, :]

            points, offsets = [], []
            for axis in range(3):
                _, _, offset, stencil = prepare_particle_axis_stencil(
                    position[:, axis], grid[axis], local_shape[axis],
                    static_parameters.shape_factor, 2,
                    wind=tile_shape[axis] * spacing[axis], ghost_cells=True,
                )
                offsets.append(offset)
                points.append(jnp.asarray(stencil))
            weights = (get_first_order_weights if static_parameters.shape_factor == 1
                       else get_second_order_weights)(*offsets, *spacing)
            collapsed = tuple(collapse_tiled_axis_stencil(
                points[axis], jnp.asarray(weights[axis]), local_shape[axis],
                not active_axes[axis], g,
            ) for axis in range(3))
            (ix, wx), (iy, wy), (iz, wz) = collapsed

            for i in range(ix.shape[0]):
                for j in range(iy.shape[0]):
                    for k in range(iz.shape[0]):
                        shape_weight = wx[i] * wy[j] * wz[k]
                        moments = moments.at[:, ix[i], iy[j], iz[k]].add(
                            particle_moments * shape_weight[None, :], mode="drop",
                        )
            return moments

        return jax.lax.fori_loop(0, n_batches, deposit_batch, moments)

    tile_indices = jnp.meshgrid(*(jnp.arange(count) for count in tile_counts), indexing="ij")
    deposit_tiles = deposit_one_tile
    for _ in range(3):
        deposit_tiles = jax.vmap(deposit_tiles)
    moments = deposit_tiles(particles.x, particles.u, particles.active, *tile_indices)
    moments = jnp.moveaxis(moments, 3, 0)

    # All ten moments live on C nodes. A tensor component is odd at a wall
    # exactly when that wall normal occurs an odd number of times in its indices.
    indices = ((), (0,), (1,), (2,)) + stress_indices
    parity = tuple(tuple((-1) ** component.count(axis) for axis in range(3))
                   for component in indices)
    moments = fold_tiled_ghost_cells(
        moments, static_parameters, g, bc_type=BC_TYPE_PARTICLE,
        location=(("C", "C", "C"),) * 10, reflecting_parity=parity,
    )
    moments = tiles_to_bssn(moments / metric.center.sqrt_gamma[None, ...], static_parameters)
    rho = moments[0]
    momentum = moments[1:4]
    stress = jnp.stack((
        jnp.stack((moments[4], moments[5], moments[6])),
        jnp.stack((moments[5], moments[7], moments[8])),
        jnp.stack((moments[6], moments[8], moments[9])),
    ))
    return rho, momentum, stress


@partial(jax.jit, static_argnames=("static_parameters", "source_filter"))
def compute_energy_momentum_tensor(
    particles, species_config, moments, bssn_state, static_parameters, dynamic_parameters,
    *, source_filter="none",
):
    """Build stage sources and filter their total coordinate densities once.

    The same synchronized particles and native EM products are re-evaluated
    in each RK geometry. Filtering sqrt(gamma)*(rho,S_i,S_ij), then dividing
    by the raw stage volume, preserves periodic integrated source components.
    Native EM products precede filtering so grid-scale field energy remains.
    ``source_filter='binomial'`` selects source-only electrovacuum smoothing,
    with replicated isolated edges, independently of matched PIC filtering.
    """
    if source_filter not in ("none", "binomial"):
        raise ValueError("source_filter must be 'none' or 'binomial'")
    if source_filter != "none":
        if static_parameters.current_filter != "none":
            raise ValueError("Source-only and matched filtering cannot be combined")
        if particles.active.size:
            raise ValueError("Source-only filtering requires empty particle storage")
        if static_parameters.metric not in ("flat_cartesian", "numerical"):
            raise ValueError("Source-only filtering requires a Cartesian metric")
    em = compute_electromagnetic_energy_momentum(moments, bssn_state)
    total = em
    if particles.active.size:
        from PyPIC3D.relativity.dynamical_relativity import build_bssn_metric

        metric = build_bssn_metric(bssn_state, static_parameters, dynamic_parameters)
        pic = compute_particle_energy_momentum(
            particles, species_config, metric, static_parameters, dynamic_parameters,
        )
        total = tuple(particle + field for particle, field in zip(pic, em))
    if static_parameters.current_filter == "none" and source_filter == "none":
        return total

    from JAX_BSSN.bssn.geometry import W_FLOOR_VALUE

    volume = jnp.maximum(bssn_state.conformal_factor, W_FLOOR_VALUE)**-3
    rho, momentum, stress = total
    components = jnp.stack((rho, *momentum,
                            stress[0, 0], stress[0, 1], stress[0, 2],
                            stress[1, 1], stress[1, 2], stress[2, 2]))
    if source_filter == "binomial":
        components = binomial_source_filter(volume * components, static_parameters.boundary_conditions) / volume
    else:
        components = filter_gr_grid(volume * components, static_parameters, dynamic_parameters) / volume
    stress = jnp.stack((
        jnp.stack((components[4], components[5], components[6])),
        jnp.stack((components[5], components[7], components[8])),
        jnp.stack((components[6], components[8], components[9])),
    ))
    return components[0], components[1:4], stress
