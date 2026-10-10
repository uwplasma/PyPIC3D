"""
Tests for shared Esirkepov deposition with static-metric Maxwell.

The property that matters is the discrete continuity equation in conformal
variables,

    ( sqrt(gamma) rho^{n+1} - sqrt(gamma) rho^n ) / dt
        + d-_i( sqrt(gamma) J^i ) = 0,

where ``d-`` is the backward difference that ``update_D`` uses for its
curl.  When that holds, and because the backward-difference divergence of a
backward-difference curl vanishes identically, the Gauss constraint

    d-_i( sqrt(gamma) D^i ) - 4 pi sqrt(gamma) rho

is preserved exactly in time with no divergence cleaning.
"""

import unittest

import numpy as np

from PyPIC3D.relativity.field_state import densitize_fields

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import update_tiled_vector_ghost_cells
from PyPIC3D.deposition.Esirkepov import Esirkepov_current
from PyPIC3D.deposition.rho import compute_rho
from PyPIC3D.initialization import (
    _encode_current_calculation,
    _validate_current_filter_contract,
    _validate_tiled_yee_configuration,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.metrics.flat import (
    initialize_flat_cartesian_metric,
    initialize_flat_spherical_metric,
)
from PyPIC3D.solvers.GR_yee.static_metric import update_D
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.kernel_fixtures import (
    empty_tiled_scalar,
    empty_tiled_vector,
    kernel_parameters,
)


def _unit_species():
    return SpeciesConfig(
        charge=jnp.asarray([1.0]),
        mass=jnp.asarray([1.0]),
        weight=jnp.asarray([1.0]),
        update_x=jnp.asarray([[True, True, True]]),
    )


def _interior(static_parameters):
    g = int(static_parameters.guard_cells)
    return slice(g, -g)


def _backward_divergence(vector_tiles, static_parameters, dynamic_parameters):
    """
    ``d-_i V^i`` at the cell centre, matching ``update_D``.

    The D components live on the faces, so the backward difference of a
    face-centred density lands on the centre -- exactly where ``compute_rho``
    deposits.  The slice idiom is the one used in ``static_metric.py``.
    """

    g = int(static_parameters.guard_cells)
    active = slice(g, -g)
    backward = slice(g - 1, -g - 1)
    Vx, Vy, Vz = vector_tiles
    return (
        (Vx[:, :, :, active, active, active] - Vx[:, :, :, backward, active, active])
        / dynamic_parameters.dx
        + (Vy[:, :, :, active, active, active] - Vy[:, :, :, active, backward, active])
        / dynamic_parameters.dy
        + (Vz[:, :, :, active, active, active] - Vz[:, :, :, active, active, backward])
        / dynamic_parameters.dz
    )


def _tiled_single_particle(x, tile_grid_shape):
    """One particle owned by tile (0, 0, 0), broadcast across the tile grid."""

    position = jnp.asarray(x, dtype=float).reshape((1, 1, 1, 1, 1, 3))
    position = jnp.broadcast_to(position, tile_grid_shape + (1, 1, 3))
    active = jnp.zeros(tile_grid_shape + (1, 1), dtype=bool).at[0, 0, 0].set(True)
    return position, active


def _continuity_residual(
    metric_name,
    x_old,
    x_new,
    *,
    shape_factor=1,
    N=(8, 8, 8),
    wind=(8.0, 8.0, 8.0),
    mins=(None, None, None),
    tile_shape=None,
):
    """Return (max |residual|, max |div J|) for one displacement."""

    Nx, Ny, Nz = N
    tile_shape = tile_shape or (Nx, Ny, Nz)
    static_parameters, dynamic_parameters = kernel_parameters(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        x_wind=wind[0],
        y_wind=wind[1],
        z_wind=wind[2],
        x_min=mins[0],
        y_min=mins[1],
        z_min=mins[2],
        dt=0.1,
        tile_shape=tile_shape,
        shape_factor=shape_factor,
        solver="static_metric",
        metric=metric_name,
        current_deposition="esirkepov",
        current_filter="none",
        particle_pusher="hybrid_boris_geodesic",
    )
    tile_grid_shape = (
        Nx // tile_shape[0],
        Ny // tile_shape[1],
        Nz // tile_shape[2],
    )

    old_position, active = _tiled_single_particle(x_old, tile_grid_shape)
    new_position, _ = _tiled_single_particle(x_new, tile_grid_shape)
    velocity = jnp.zeros_like(old_position)
    species = _unit_species()
    particles_old = TiledParticles(x=old_position, u=velocity, active=active)
    particles_new = TiledParticles(x=new_position, u=velocity, active=active)

    J = Esirkepov_current(
        particles_old, particles_new, species,
        empty_tiled_vector(static_parameters, dynamic_parameters),
        static_parameters, dynamic_parameters,
    )

    rho_template = empty_tiled_scalar(static_parameters, dynamic_parameters)
    rho_old = compute_rho(particles_old, species, rho_template, static_parameters, dynamic_parameters)
    rho_new = compute_rho(particles_new, species, rho_template, static_parameters, dynamic_parameters)

    interior = _interior(static_parameters)
    d_rho_dt = (
        rho_new[:, :, :, interior, interior, interior]
        - rho_old[:, :, :, interior, interior, interior]
    ) / dynamic_parameters.dt
    div_J = _backward_divergence(J, static_parameters, dynamic_parameters)

    residual = d_rho_dt + div_J
    return float(jnp.max(jnp.abs(residual))), float(jnp.max(jnp.abs(div_J)))


class GRESirkepovContinuityFixtures:
    """The deposition satisfies discrete charge continuity exactly."""

    # Each particle sits at least three cells clear of every domain edge.  The
    # continuity identity is local, so a stencil that straddles the boundary
    # folds charge around the periodic wrap and the residual stops telescoping
    # -- a property of the probe, not of the deposit.  The curvilinear charts
    # also keep the domain away from r = 0 and sin(theta) = 0, where sqrt(gamma)
    # degenerates.
    CASES = (
        ("flat_cartesian", (0.30, -0.70, 1.10), (0.62, -0.41, 1.33), (8.0, 8.0, 8.0), (None, None, None)),
        ("flat_cylindrical", (5.30, 4.70, 4.10), (5.62, 4.91, 4.33), (8.0, 8.0, 8.0), (2.0, 0.3, 0.2)),
        ("flat_spherical", (5.30, 1.10, 1.00), (5.62, 1.18, 1.06), (8.0, 1.6, 1.6), (2.0, 0.4, 0.3)),
        ("kerr_schild_cartesian", (5.30, 5.70, 6.10), (5.62, 5.99, 6.33), (8.0, 8.0, 8.0), (2.0, 2.0, 2.0)),
        ("kerr_schild_spherical", (7.30, 1.10, 1.00), (7.62, 1.18, 1.06), (8.0, 1.6, 1.6), (4.0, 0.4, 0.3)),
    )


class GRESirkepovConventionsFixtures:
    """Coordinate-current invariants and the sqrt(gamma) convention."""

    def _flat_setup(self):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=8,
            Nz=8,
            x_wind=8.0,
            y_wind=8.0,
            z_wind=8.0,
            dt=0.1,
            tile_shape=(8, 8, 8),
            shape_factor=1,
            metric="flat_cartesian",
            current_deposition="esirkepov",
            current_filter="none",
        )
        return static_parameters, dynamic_parameters


class GaussConstraintPreservationFixtures:
    """The Gauss constraint is conserved by the field update and the deposit."""

    def _loop_setup(self, current_deposition):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=8,
            Nz=8,
            x_wind=8.0,
            y_wind=8.0,
            z_wind=8.0,
            dt=0.02,
            tile_shape=(8, 8, 8),
            shape_factor=1,
            solver="static_metric",
            metric="flat_cartesian",
            current_deposition=current_deposition,
            current_filter="none",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        B = empty_tiled_vector(static_parameters, dynamic_parameters)
        J = empty_tiled_vector(static_parameters, dynamic_parameters)
        rho = jnp.zeros_like(J[0])
        phi = jnp.zeros_like(J[0])
        fields = densitize_fields((D, B, J, rho, phi, (D, B), metric, (D, B), jnp.asarray(False)))

        position = jnp.asarray([(0.30, -0.70, 1.10)]).reshape((1, 1, 1, 1, 1, 3))
        velocity = jnp.asarray([(0.45, -0.30, 0.20)]).reshape((1, 1, 1, 1, 1, 3))
        active = jnp.ones((1, 1, 1, 1, 1), dtype=bool)
        particles = TiledParticles(x=position, u=velocity, active=active)
        return static_parameters, dynamic_parameters, metric, fields, particles

    def _gauss_residual(self, D, particles, species, static_parameters, dynamic_parameters):
        """``D`` is the runtime density ``sqrt(gamma) D^i``."""
        rho = compute_rho(
            particles,
            species,
            empty_tiled_scalar(static_parameters, dynamic_parameters),
            static_parameters,
            dynamic_parameters,
        )
        interior = _interior(static_parameters)
        conformal_rho = rho[:, :, :, interior, interior, interior]
        return (
            _backward_divergence(D, static_parameters, dynamic_parameters)
            - 4.0 * jnp.pi * conformal_rho
        )

    def _constraint_drift(self, current_deposition, steps=4):
        static_parameters, dynamic_parameters, _, fields, particles = self._loop_setup(current_deposition)
        species = _unit_species()

        # jit the step, closing over static_parameters the way
        # __main__.run_PyPIC3D does.  Without this the lax.cond deposition
        # selector re-traces both kernels on every iteration, which dominates
        # the runtime of this test.
        step = jax.jit(
            lambda particles, species, fields, dynamic: time_loop_static_metric(
                particles, species, fields, static_parameters, dynamic
            )
        )

        initial = self._gauss_residual(fields[0], particles, species, static_parameters, dynamic_parameters)
        scale = float(jnp.max(jnp.abs(initial)))

        drift = 0.0
        for _ in range(steps):
            particles, fields = step(particles, species, fields, dynamic_parameters)
            residual = self._gauss_residual(fields[0], particles, species, static_parameters, dynamic_parameters)
            drift = max(drift, float(jnp.max(jnp.abs(residual - initial))))
        return drift, scale


class DepositionDispatchFixtures:
    """The lax.cond selector works under jit, the way the driver runs it."""

    def _one_jitted_step(self, current_deposition):
        static_parameters, dynamic_parameters = kernel_parameters(
            Nx=8,
            Ny=8,
            Nz=8,
            x_wind=8.0,
            y_wind=8.0,
            z_wind=8.0,
            dt=0.02,
            tile_shape=(8, 8, 8),
            shape_factor=1,
            solver="static_metric",
            metric="flat_cartesian",
            current_deposition=current_deposition,
            current_filter="none",
            particle_pusher="hybrid_boris_geodesic",
        )
        metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
        D = empty_tiled_vector(static_parameters, dynamic_parameters)
        B = empty_tiled_vector(static_parameters, dynamic_parameters)
        J = empty_tiled_vector(static_parameters, dynamic_parameters)
        rho = jnp.zeros_like(J[0])
        fields = densitize_fields((D, B, J, rho, jnp.zeros_like(J[0]), (D, B), metric, (D, B), jnp.asarray(False)))
        particles = TiledParticles(
            x=jnp.asarray([(0.30, -0.70, 1.10)]).reshape((1, 1, 1, 1, 1, 3)),
            u=jnp.asarray([(0.45, -0.30, 0.20)]).reshape((1, 1, 1, 1, 1, 3)),
            active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
        )

        # close over static_parameters exactly as __main__.run_PyPIC3D does
        step = jax.jit(
            lambda particles, species, fields, dynamic: time_loop_static_metric(
                particles, species, fields, static_parameters, dynamic
            )
        )
        particles, fields = step(particles, _unit_species(), fields, dynamic_parameters)
        return fields


class GRESirkepovConfigurationFixtures:
    """The scheme is reachable from configuration and correctly constrained."""

    def _static_config(self, **overrides):
        config = {
            "solver": "static_metric",
            "current_calculation": "esirkepov",
            "particle_pusher": "hybrid_boris_geodesic",
            "filter_j": "none",
            "particle_tile_nx": 4,
            "particle_tile_ny": 4,
            "particle_tile_nz": 4,
        }
        config.update(overrides)
        return config
