"""Energy-tracker checks through the coupled BSSN/Maxwell/PIC time loop."""

import unittest
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed")
    raise

from PyPIC3D.diagnostics.gr_work_energy_theorem import (
    tracker_values,
    initialize_tracker,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    initialize_dynamical_relativity,
    synchronize_particles,
    tiles_to_bssn,
)
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity
from tests.support.dynamical_relativity_runtime_fixtures import coupled_state
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters

jax.config.update("jax_enable_x64", True)


def maxwell_state(n=24, *, dt=None, shift=0., pulse=False, enabled=True, backreaction=False):
    """Smooth right-moving Maxwell data on a flat, frozen Cartesian geometry."""
    dx = 4. / n
    dt = .1 * dx if dt is None else dt
    static, dynamic = kernel_parameters(
        Nx=n, Ny=1, Nz=1, x_wind=4., y_wind=1., z_wind=1.,
        tile_shape=(n, 1, 1), dt=dt, guard_cells=3,
        solver="dynamic_metric", metric="flat_cartesian",
        particle_pusher="hybrid_boris_geodesic", current_deposition="GR_direct",
        current_filter="none", boundary_conditions=(3 if pulse else 0, 0, 0),
    )
    static = static._replace(gr_work_energy_theorem=enabled,
                             metric_backreaction=backreaction)
    static, bssn = build_bssn_parameters(static, dynamic, evolution_parameters={"nu": 0.})
    state = bssn.state._replace(shift=bssn.state.shift.at[0].set(shift))
    bssn = bssn._replace(state=state)
    dynamic = dynamic._replace(bssn_parameters=bssn)

    zero = empty_tiled_vector(static, dynamic)
    x_D = dynamic.grids.tiled_center_grid[0][..., :, None, None]
    x_B = dynamic.grids.tiled_vertex_grid[0][..., :, None, None]

    def profile(x):
        return (jnp.exp(-((x - .3) / .45)**2) if pulse
                else .2 * jnp.sin(.5 * jnp.pi * x))

    D = (zero[0], jnp.broadcast_to(profile(x_D), zero[0].shape), zero[2])
    B = (zero[0], zero[1], jnp.broadcast_to(profile(x_B), zero[0].shape))
    x = jnp.zeros((1, 1, 1, 0, 0, 3))
    particles = TiledParticles(x, x, jnp.zeros(x.shape[:-1], dtype=bool))
    scalar = jnp.zeros((0,))
    species = SpeciesConfig(scalar, scalar, scalar, jnp.zeros((0, 3), dtype=bool))
    fields = (D, B, zero, zero[0], zero[0], (zero, zero), None,
              (D, B), jnp.asarray(False))
    particles, fields, dynamic = initialize_dynamical_relativity(
        particles, species, fields, static, dynamic)
    return static, dynamic, particles, species, fields


@partial(jax.jit, static_argnames=("static", "steps"))
def evolve_scan(static, species, state, steps):
    """Keep diagnostics in the production scan carry, without host samples."""
    def step(carry, unused):
        particles, fields, dynamic = carry
        return time_loop_dynamical_relativity(
            particles, species, fields, static, dynamic), None

    return jax.lax.scan(step, state, None, length=steps)[0]


