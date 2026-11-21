"""
Validation and convergence tests for the prescribed-metric (``static_metric``) solver.

Three things are checked here that the existing ``static_metric_convergence_test``
does not cover:

* the analytic 3+1 metric data is compared against *closed-form* general
  relativity results (Kerr-Schild four-metric, Kerr circular-orbit frequencies,
  flat-space straight lines seen from curvilinear charts) rather than against
  itself;
* every substep of the algorithm -- metric sampling, geodesic/Boris push,
  constitutive relations, both curl updates, current deposition and the full
  time loop -- is convergence tested separately in *space* and in *time*;
* the tests are run over a spread of charts: flat Cartesian, flat cylindrical,
  flat spherical, Schwarzschild and Kerr in Cartesian and spherical Kerr-Schild
  form, a Gaussian-normal (synchronous) chart with a fully non-diagonal spatial
  metric, and a generic stationary chart with a varying lapse, shift and
  three-metric.

The reference solutions come from an independent RK4 integrator that evaluates
the metric and its derivatives *analytically* at the particle position, so the
production grid sampling is never used to produce the answer it is checked
against.
"""

import itertools
from functools import lru_cache, partial
import math
import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.pusher.hybrid_boris_geodesic import (
    hybrid_boris_geodesic_push,
    magnetic_boris_rotation,
)
from PyPIC3D.pusher.particle_push import seed_leapfrog_velocity
from PyPIC3D.relativity.core import (
    B_FIELD_LOCATIONS,
    D_FIELD_LOCATIONS,
    Metric,
    YeeMetric,
    analytic_metric_on_grid,
    build_yee_metric,
    contravariant_three_velocity,
    covariant_lorentz_factor,
)
from PyPIC3D.relativity.metrics.flat import (
    _flat_cartesian_metric_at_position,
    _flat_cylindrical_metric_at_position,
    _flat_spherical_metric_at_position,
)
from PyPIC3D.relativity.metrics.kerr_schild import (
    _kerr_schild_cartesian_metric_at_position,
    _kerr_schild_spherical_metric_at_position,
)
from PyPIC3D.relativity.interpolate_metric import interpolate_metric
from PyPIC3D.solvers.GR_yee.curved_maxwell import (
    compute_covariant_E,
    compute_covariant_H,
    update_B,
    update_D,
)
from PyPIC3D.relativity.field_state import densitize_vector, physical_fields, physical_vector
from PyPIC3D.solvers.GR_yee.time_loop import time_loop_static_metric
from tests.kernel_fixtures import empty_tiled_scalar, empty_tiled_vector, kernel_parameters


LEVI_CIVITA = jnp.zeros((3, 3, 3))
for _i, _j, _k in itertools.permutations(range(3)):
    LEVI_CIVITA = LEVI_CIVITA.at[_i, _j, _k].set(
        1 if (_i, _j, _k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1
    )


# ---------------------------------------------------------------------------
# Extra analytic charts used only by the tests
# ---------------------------------------------------------------------------

def _pack_metric(lapse, shift, gamma):
    return lapse, shift, gamma, jnp.linalg.inv(gamma), jnp.sqrt(jnp.linalg.det(gamma))


def gaussian_normal_metric_at_position(position, amp=0.30, cross=0.12, k=0.5):
    """
    Gaussian-normal (synchronous) chart: ``alpha = 1``, ``beta^i = 0``.

    The spatial metric is fully non-diagonal and varies along all three axes,
    so every Christoffel symbol is populated while the lapse and shift terms of
    the geodesic force vanish identically.
    """
    x, y, z = position
    dtype = x.dtype
    gxx = 1.0 + amp * jnp.sin(k * x) * jnp.cos(k * y)
    gyy = 1.0 + amp * jnp.cos(k * y) * jnp.sin(k * z)
    gzz = 1.0 + amp * jnp.sin(k * z) * jnp.cos(k * x)
    gxy = cross * jnp.sin(k * x) * jnp.sin(k * y)
    gxz = cross * jnp.cos(k * y) * jnp.cos(k * z)
    gyz = cross * jnp.sin(k * z) * jnp.cos(k * x)
    gamma = jnp.array(
        [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]], dtype=dtype
    )
    return _pack_metric(jnp.asarray(1.0, dtype=dtype), jnp.zeros(3, dtype=dtype), gamma)


def generic_lapse_shift_metric_at_position(
    position, amp=0.30, cross=0.12, k=0.5, lapse_amp=0.20, shift_amp=0.10
):
    """
    Generic stationary chart with a varying lapse, shift and three-metric.

    Nothing about this chart is special, which is the point: it exercises every
    term of the geodesic force and of the constitutive relations at once.
    """
    x, y, z = position
    dtype = x.dtype
    _, _, gamma, _, _ = gaussian_normal_metric_at_position(position, amp, cross, k)
    lapse = 1.0 + lapse_amp * jnp.sin(k * x) * jnp.cos(k * z)
    shift = shift_amp * jnp.array(
        [jnp.cos(k * y), jnp.sin(k * z), jnp.cos(k * x)], dtype=dtype
    )
    return _pack_metric(lapse, shift, gamma)


def schwarzschild_cartesian_metric_at_position(position):
    return _kerr_schild_cartesian_metric_at_position(position, mass=1.0, spin=0.0)


def kerr_cartesian_metric_at_position(spin):
    def metric_at_position(position):
        return _kerr_schild_cartesian_metric_at_position(position, mass=1.0, spin=spin)

    return metric_at_position


def kerr_spherical_metric_at_position(spin):
    def metric_at_position(position):
        return _kerr_schild_spherical_metric_at_position(position, mass=1.0, spin=spin)

    return metric_at_position


# ---------------------------------------------------------------------------
# Grid-free reference integrator (FPIC Eqs. 13-16 / Entity Eq. 19)
# ---------------------------------------------------------------------------

def reference_rhs(metric_at_position, q_over_m=0.0, D=(0.0, 0.0, 0.0), B=(0.0, 0.0, 0.0)):
    """
    ``d/dt (x^i, u_i)`` with the metric evaluated analytically at the particle.
    """
    D = jnp.asarray(D, dtype=float)
    B = jnp.asarray(B, dtype=float)

    def differentiated(position):
        lapse, shift, gamma, gamma_inv, sqrt_gamma = metric_at_position(position)
        return (lapse, shift, gamma_inv), (lapse, shift, gamma, gamma_inv, sqrt_gamma)

    jacobian = jax.jacfwd(differentiated, has_aux=True)

    def rhs(state):
        position, u_cov = state[:3], state[3:]
        (d_lapse, d_shift, d_gamma_inv), (
            lapse,
            shift,
            gamma,
            gamma_inv,
            sqrt_gamma,
        ) = jacobian(position)
        Gamma = jnp.sqrt(1.0 + jnp.einsum("i,ij,j->", u_cov, gamma_inv, u_cov))
        u_con = gamma_inv @ u_cov
        dx_dt = lapse * u_con / Gamma - shift
        du_dt = (
            -Gamma * d_lapse
            + jnp.einsum("k,ki->i", u_cov, d_shift)
            - (lapse / (2.0 * Gamma))
            * jnp.einsum("l,m,lmi->i", u_cov, u_cov, d_gamma_inv)
            + q_over_m * lapse * (gamma @ D)
            + q_over_m * lapse / Gamma * sqrt_gamma * jnp.cross(u_con, B)
        )
        return jnp.concatenate((dx_dt, du_dt))

    return rhs


@partial(jax.jit, static_argnames=("rhs", "n_steps"))
def rk4_integrate(rhs, state, dt, n_steps):
    def step(current, _):
        k1 = rhs(current)
        k2 = rhs(current + 0.5 * dt * k1)
        k3 = rhs(current + 0.5 * dt * k2)
        k4 = rhs(current + dt * k3)
        return current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    result, _ = jax.lax.scan(step, state, None, length=n_steps)
    return result


@lru_cache(maxsize=32)
def _reference_rhs_cached(metric_at_position, q_over_m, D, B):
    return jax.jit(reference_rhs(metric_at_position, q_over_m, D, B))


@lru_cache(maxsize=64)
def _reference_trajectory_cached(metric_at_position, x0, u0, T, q_over_m, D, B, n):
    rhs = _reference_rhs_cached(metric_at_position, q_over_m, D, B)
    state = jnp.concatenate((jnp.asarray(x0, dtype=float), jnp.asarray(u0, dtype=float)))
    return rk4_integrate(rhs, state, T / n, n)


def reference_trajectory(metric_at_position, x0, u0, T, q_over_m=0.0, D=(0, 0, 0), B=(0, 0, 0), n=40000):
    # Only the independent analytic reference is cached. Every production
    # trajectory and every convergence/refinement assertion still executes.
    return _reference_trajectory_cached(
        metric_at_position, tuple(map(float, x0)), tuple(map(float, u0)),
        float(T), float(q_over_m), tuple(map(float, D)), tuple(map(float, B)), int(n))


def tearDownModule():
    _reference_trajectory_cached.cache_clear()
    _reference_rhs_cached.cache_clear()
    _pusher_fixture.cache_clear()
    _vacuum_fixture.cache_clear()


def hamiltonian(metric_at_position, state):
    """``H = alpha * Gamma - beta^i u_i = -u_t``, conserved for stationary charts."""
    lapse, shift, _gamma, gamma_inv, _sqrt_gamma = metric_at_position(state[:3])
    Gamma = jnp.sqrt(1.0 + jnp.einsum("i,ij,j->", state[3:], gamma_inv, state[3:]))
    return lapse * Gamma - jnp.dot(shift, state[3:])


# ---------------------------------------------------------------------------
# Production-kernel drivers
# ---------------------------------------------------------------------------

def parameters(N, wind, mins, dt, metric_name="flat_cartesian", shape_factor=1):
    Nx, Ny, Nz = N
    return kernel_parameters(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        x_wind=wind[0],
        y_wind=wind[1],
        z_wind=wind[2],
        x_min=mins[0],
        y_min=mins[1],
        z_min=mins[2],
        dt=dt,
        tile_shape=(Nx, Ny, Nz),
        shape_factor=shape_factor,
        solver="static_metric",
        current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic",
        metric=metric_name,
    )


def location_grid(location, dynamic_parameters):
    center = dynamic_parameters.grids.tiled_center_grid
    vertex = dynamic_parameters.grids.tiled_vertex_grid
    return tuple(center[axis] if location[axis] == "C" else vertex[axis] for axis in range(3))


def location_positions(location, dynamic_parameters):
    x_grid, y_grid, z_grid = location_grid(location, dynamic_parameters)
    X = x_grid[..., :, None, None]
    Y = y_grid[..., None, :, None]
    Z = z_grid[..., None, None, :]
    X, Y, Z = jnp.broadcast_arrays(X, Y, Z)
    return jnp.stack((X, Y, Z), axis=-1)


def build_center_metric(dynamic_parameters, metric_at_position):
    """
    The pusher interpolates ``center`` and divides the field densities by the
    ``D``/``B`` samples, which are the same center values here.
    """
    center = analytic_metric_on_grid(
        dynamic_parameters.grids.tiled_center_grid, metric_at_position
    )
    return YeeMetric(D=(center,) * 3, B=(center,) * 3, center=center, vertex=center)


build_full_metric = build_yee_metric


def sample_center_metric(metric, positions, static_parameters, dynamic_parameters):
    grid = tuple(axis[0, 0, 0] for axis in dynamic_parameters.grids.tiled_center_grid)
    g = int(static_parameters.guard_cells)
    return interpolate_metric(
        jax.tree.map(lambda array: array[0, 0, 0], metric.center),
        positions,
        grid,
        static_parameters.metric,
        (True, True, True),
        (g, g, g),
        derivatives=False,
    )


def one_particle(x, u):
    return TiledParticles(
        x=jnp.asarray(x, dtype=float).reshape((1, 1, 1, 1, 1, 3)),
        u=jnp.asarray(u, dtype=float).reshape((1, 1, 1, 1, 1, 3)),
        active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
    )


def one_species(charge, mass=1.0, weight=1.0):
    return SpeciesConfig(
        charge=jnp.asarray([charge]),
        mass=jnp.asarray([mass]),
        weight=jnp.asarray([weight]),
        update_x=jnp.asarray([[True, True, True]]),
    )


def constant_vector(static_parameters, dynamic_parameters, values):
    empty = empty_tiled_vector(static_parameters, dynamic_parameters)
    return tuple(empty[i].at[...].set(values[i]) for i in range(3))


@lru_cache(maxsize=1)
def _pusher_fixture(metric_at_position, N, wind, mins, metric_name, charge,
                    D_values, B_values, shape_factor):
    # Geometry and prescribed fields do not depend on dt. Keep only the most
    # recent grid: the largest convergence fixtures are intentionally large.
    static, dynamic = parameters(
        (N, N, N), wind, mins, 1.0, metric_name=metric_name, shape_factor=shape_factor)
    metric = build_center_metric(dynamic, metric_at_position)
    D = densitize_vector(constant_vector(static, dynamic, D_values), metric.D)
    B = densitize_vector(constant_vector(static, dynamic, B_values), metric.B)
    return static, dynamic, metric, D, B, one_species(charge)


@partial(jax.jit, static_argnames="static_parameters")
def _advance_pusher(particles, species, D, B, metric, static_parameters, dynamic_parameters, n_steps):
    # Arrays are runtime arguments, not constants embedded into a new scan for
    # every refinement. A dynamic loop count reuses the same executable for
    # the temporal study without changing the number of integration steps.
    def step(_index, current):
        advanced, _ = hybrid_boris_geodesic_push(
            current, species, D, B, metric, static_parameters, dynamic_parameters)
        return advanced
    return jax.lax.fori_loop(0, n_steps, step, particles)


def run_production_pusher(
    metric_at_position,
    x0,
    u0,
    T,
    dt,
    N,
    wind,
    mins,
    metric_name="flat_cartesian",
    charge=0.0,
    D_values=(0.0, 0.0, 0.0),
    B_values=(0.0, 0.0, 0.0),
    offset_initial_half_step=True,
    shape_factor=1,
):
    """
    Advance one particle with the production pusher and return ``(x, u)`` at ``T``.

    ``particles.u`` is a leapfrog half-step quantity, so the physical ``u(0)`` is
    pushed back by ``dt/2`` before the run and the final ``u`` is pushed forward
    by ``dt/2``, using the same operator, before it is compared.
    """
    static_parameters, dynamic_parameters, metric, D, B, species = _pusher_fixture(
        metric_at_position, int(N), tuple(map(float, wind)), tuple(map(float, mins)), metric_name, float(charge),
        tuple(map(float, D_values)), tuple(map(float, B_values)), int(shape_factor))
    dynamic_parameters = dynamic_parameters._replace(dt=jnp.asarray(dt))

    def half_step(x, u, half_dt):
        stepped, _ = hybrid_boris_geodesic_push(
            one_particle(x, u),
            species,
            D,
            B,
            metric,
            static_parameters,
            dynamic_parameters._replace(dt=jnp.asarray(half_dt)),
        )
        return stepped.u[0, 0, 0, 0, 0]

    if offset_initial_half_step:
        # the production helper, so this measures what a real run does
        u_start = seed_leapfrog_velocity(
            one_particle(x0, u0), species, D, B, static_parameters, dynamic_parameters,
            metric=metric,
        ).u[0, 0, 0, 0, 0]
    else:
        u_start = jnp.asarray(u0, dtype=float)

    particles = _advance_pusher(
        one_particle(x0, u_start), species, D, B, metric,
        static_parameters, dynamic_parameters, int(round(T / dt)))
    x_final = particles.x[0, 0, 0, 0, 0]
    u_final = particles.u[0, 0, 0, 0, 0]
    return x_final, half_step(x_final, u_final, 0.5 * dt)


def convergence_order(coarse_error, fine_error, ratio=2.0):
    return math.log(coarse_error / fine_error) / math.log(ratio)


def interior_rms(array, guard_cells, skip=0):
    window = (slice(None),) * 3 + (slice(guard_cells + skip, -guard_cells - skip),) * 3
    return float(jnp.sqrt(jnp.mean(array[window] ** 2)))


# ---------------------------------------------------------------------------
# Chart catalogue used by the convergence tests
# ---------------------------------------------------------------------------

PUSHER_CASES = (
    dict(
        name="flat cylindrical",
        metric=_flat_cylindrical_metric_at_position,
        metric_name="flat_cylindrical",
        x0=(3.0, 0.0, 0.0),
        u0=(0.2, 0.9, 0.1),
        T=1.5,
        wind=(1.5, 0.8, 0.8),
        mins=(2.5, -0.3, -0.3),
    ),
    dict(
        name="flat spherical",
        metric=_flat_spherical_metric_at_position,
        metric_name="flat_spherical",
        x0=(3.0, 1.4, 0.0),
        u0=(0.2, 0.6, 0.5),
        T=1.5,
        wind=(1.5, 0.8, 0.8),
        mins=(2.5, 1.1, -0.3),
    ),
    dict(
        name="Schwarzschild Kerr-Schild Cartesian",
        metric=schwarzschild_cartesian_metric_at_position,
        metric_name="kerr_schild_cartesian",
        x0=(10.0, 0.0, 0.3),
        u0=(0.09539392014169457, 0.3651483716701107, 0.0),
        T=10.0,
        wind=(8.0, 9.0, 4.0),
        mins=(4.0, -2.0, -2.0),
    ),
    dict(
        name="Kerr a=0.9 Kerr-Schild Cartesian",
        metric=kerr_cartesian_metric_at_position(0.9),
        metric_name="kerr_schild_cartesian",
        x0=(8.0, 0.0, 0.4),
        u0=(0.10, 0.42, 0.02),
        T=10.0,
        wind=(8.0, 9.0, 4.0),
        mins=(1.5, -2.0, -2.0),
    ),
    dict(
        name="Kerr a=0.9 Kerr-Schild spherical",
        metric=kerr_spherical_metric_at_position(0.9),
        metric_name="kerr_schild_spherical",
        x0=(8.0, 1.35, 0.0),
        u0=(0.10, 0.35, 3.4),
        T=10.0,
        wind=(3.0, 0.7, 1.9),
        mins=(6.0, 1.1, -0.4),
    ),
    dict(
        name="Gaussian-normal chart",
        metric=gaussian_normal_metric_at_position,
        metric_name="flat_cartesian",
        x0=(0.3, -0.2, 0.1),
        u0=(0.35, 0.22, -0.18),
        T=3.0,
        wind=(3.5, 3.5, 3.5),
        mins=(-1.5, -1.5, -1.5),
    ),
    dict(
        name="generic lapse+shift chart with B field",
        metric=generic_lapse_shift_metric_at_position,
        metric_name="flat_cartesian",
        x0=(0.3, -0.2, 0.1),
        u0=(0.35, 0.22, -0.18),
        T=3.0,
        wind=(3.5, 3.5, 3.5),
        mins=(-1.5, -1.5, -1.5),
        charge=1.0,
        D_values=(0.02, 0.01, 0.0),
        B_values=(0.0, 0.0, 0.4),
    ),
)

FIELD_CASES = (
    ("flat cylindrical", _flat_cylindrical_metric_at_position, (1.6, 1.2, 1.2), (2.2, -0.6, -0.6), "flat_cylindrical"),
    ("flat spherical", _flat_spherical_metric_at_position, (1.6, 1.0, 1.2), (2.2, 1.0, -0.6), "flat_spherical"),
    ("Schwarzschild KS-cart", schwarzschild_cartesian_metric_at_position, (4.0, 4.0, 4.0), (5.0, -2.0, -2.0), "kerr_schild_cartesian"),
    ("Kerr a=0.9 KS-cart", kerr_cartesian_metric_at_position(0.9), (4.0, 4.0, 4.0), (4.0, -2.0, -2.0), "kerr_schild_cartesian"),
    ("Kerr a=0.9 KS-spherical", kerr_spherical_metric_at_position(0.9), (3.0, 1.0, 1.6), (6.0, 1.0, -0.8), "kerr_schild_spherical"),
    ("Gaussian-normal", gaussian_normal_metric_at_position, (4.0, 4.0, 4.0), (-2.0, -2.0, -2.0), "flat_cartesian"),
    ("generic lapse+shift", generic_lapse_shift_metric_at_position, (4.0, 4.0, 4.0), (-2.0, -2.0, -2.0), "flat_cartesian"),
)


def smooth_D_field(position, k=0.7):
    x, y, z = position
    return jnp.array(
        [
            jnp.sin(k * x + 0.3) * jnp.cos(0.6 * k * y) * jnp.cos(0.4 * k * z),
            jnp.cos(0.5 * k * x) * jnp.sin(k * y + 1.1) * jnp.cos(0.7 * k * z),
            jnp.cos(0.3 * k * x) * jnp.cos(0.8 * k * y) * jnp.sin(k * z + 2.0),
        ]
    )


def smooth_B_field(position, k=0.7):
    x, y, z = position
    return jnp.array(
        [
            jnp.cos(k * x + 0.9) * jnp.sin(0.5 * k * y) * jnp.cos(0.9 * k * z),
            jnp.sin(0.7 * k * x) * jnp.cos(k * y + 2.2) * jnp.cos(0.3 * k * z),
            jnp.cos(0.6 * k * x) * jnp.sin(0.4 * k * y) * jnp.cos(k * z + 0.4),
        ]
    )


def exact_covariant_E(metric_at_position):
    def field(position):
        lapse, shift, gamma, _gamma_inv, sqrt_gamma = metric_at_position(position)
        return lapse * (gamma @ smooth_D_field(position)) + sqrt_gamma * jnp.einsum(
            "ijk,j,k->i", LEVI_CIVITA, shift, smooth_B_field(position)
        )

    return field


def exact_covariant_H(metric_at_position):
    def field(position):
        lapse, shift, gamma, _gamma_inv, sqrt_gamma = metric_at_position(position)
        return lapse * (gamma @ smooth_B_field(position)) - sqrt_gamma * jnp.einsum(
            "ijk,j,k->i", LEVI_CIVITA, shift, smooth_D_field(position)
        )

    return field


def exact_curl_over_sqrt_gamma(metric_at_position, covariant_field):
    def field(position):
        jacobian = jax.jacfwd(covariant_field)(position)
        sqrt_gamma = metric_at_position(position)[4]
        return jnp.einsum("ijk,kj->i", LEVI_CIVITA, jacobian) / sqrt_gamma

    return field


def sample_component(dynamic_parameters, field, location, component):
    positions = location_positions(location, dynamic_parameters)
    shape = positions.shape[:-1]
    return jax.vmap(field)(positions.reshape(-1, 3))[:, component].reshape(shape)


def fill_staggered_vector(dynamic_parameters, field, locations):
    return tuple(
        sample_component(dynamic_parameters, field, location, component)
        for component, location in enumerate(locations)
    )


# ===========================================================================
# 4. Current deposition
# ===========================================================================

def deposition_velocity_field(position, k=0.6):
    x, y, z = position
    return jnp.array(
        [
            0.30 * jnp.sin(k * x + 0.4) * jnp.cos(0.5 * k * y),
            0.25 * jnp.cos(0.6 * k * y + 1.1) * jnp.cos(0.4 * k * z),
            0.20 * jnp.sin(0.7 * k * z + 2.0) * jnp.cos(0.3 * k * x),
        ]
    )


def exact_source_velocity(metric_at_position, position):
    """``alpha v^i - beta^i`` from FPIC's ``J = alpha j - rho beta``."""
    lapse, shift, _gamma, gamma_inv, _sqrt_gamma = metric_at_position(position)
    u_cov = deposition_velocity_field(position)
    Gamma = jnp.sqrt(1.0 + u_cov @ (gamma_inv @ u_cov))
    return lapse * (gamma_inv @ u_cov) / Gamma - shift


def coordinate_lattice(mins, wind, N, per_cell):
    axes = []
    for axis in range(3):
        spacing = wind[axis] / N
        base = mins[axis] + spacing * jnp.arange(N)
        offsets = spacing * (jnp.arange(per_cell) + 0.5) / per_cell
        axes.append((base[:, None] + offsets[None, :]).reshape(-1))
    X, Y, Z = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack((X, Y, Z), axis=-1).reshape(-1, 3)


# ===========================================================================
# 5. Full time loop
# ===========================================================================

def empty_particle_state():
    return (
        TiledParticles(
            x=jnp.zeros((1, 1, 1, 0, 0, 3)),
            u=jnp.zeros((1, 1, 1, 0, 0, 3)),
            active=jnp.zeros((1, 1, 1, 0, 0), dtype=bool),
        ),
        SpeciesConfig(
            charge=jnp.zeros((0,)),
            mass=jnp.zeros((0,)),
            weight=jnp.zeros((0,)),
            update_x=jnp.zeros((0, 3), dtype=bool),
        ),
    )


def consistent_initial_field_state(static_parameters, dynamic_parameters, metric, D_field, B_field, dt):
    """
    Seed both leapfrog chains from one physical state.

    The solver keeps the densities ``sqrt(gamma) D`` at integer steps,
    ``sqrt(gamma) B`` at half steps, and one extra copy of each, so ``D^{-1}``
    and ``B^{-3/2}`` are produced by stepping the same discrete operators
    backwards from ``t = 0``.
    """
    D_0 = densitize_vector(fill_staggered_vector(dynamic_parameters, D_field, D_FIELD_LOCATIONS), metric.D)
    B_0 = densitize_vector(fill_staggered_vector(dynamic_parameters, B_field, B_FIELD_LOCATIONS), metric.B)
    E_0 = compute_covariant_E(D_0, B_0, metric)
    H_0 = compute_covariant_H(D_0, B_0, metric)
    zero_current = empty_tiled_vector(static_parameters, dynamic_parameters)
    B_minus_half = update_B(
        E_0, B_0, metric, static_parameters, dynamic_parameters, -0.5 * dt
    )
    B_minus_three_half = update_B(
        E_0, B_0, metric, static_parameters, dynamic_parameters, -1.5 * dt
    )
    D_minus_one = update_D(
        D_0, H_0, zero_current, metric, static_parameters, dynamic_parameters, -dt
    )
    scalar = empty_tiled_scalar(static_parameters, dynamic_parameters)
    external = (
        empty_tiled_vector(static_parameters, dynamic_parameters),
        empty_tiled_vector(static_parameters, dynamic_parameters),
    )
    return (
        D_0,
        B_minus_half,
        zero_current,
        scalar,
        scalar,
        external,
        metric,
        (D_minus_one, B_minus_three_half),
        jnp.asarray(False),
    )


@lru_cache(maxsize=1)
def _vacuum_fixture(metric_at_position, metric_name, N, wind, mins):
    static_parameters, dynamic_parameters = parameters(
        (N, N, N), wind, mins, 1.0, metric_name=metric_name
    )
    metric = build_full_metric(dynamic_parameters, metric_at_position)
    return static_parameters, dynamic_parameters, metric


@partial(jax.jit, static_argnames="static_parameters")
def _advance_time_loop(particles, species, fields, static_parameters, dynamic_parameters, steps):
    def step(_index, carry):
        current_particles, current_fields = carry
        return time_loop_static_metric(
            current_particles, species, current_fields, static_parameters, dynamic_parameters)
    return jax.lax.fori_loop(0, steps, step, (particles, fields))


def evolve_vacuum(metric_at_position, metric_name, N, wind, mins, dt, T, D_field, B_field):
    static_parameters, dynamic_parameters, metric = _vacuum_fixture(
        metric_at_position, metric_name, int(N), tuple(map(float, wind)), tuple(map(float, mins)))
    dynamic_parameters = dynamic_parameters._replace(dt=jnp.asarray(dt))
    fields = consistent_initial_field_state(
        static_parameters, dynamic_parameters, metric, D_field, B_field, dt
    )
    particles, species = empty_particle_state()
    particles, fields = _advance_time_loop(
        particles, species, fields, static_parameters, dynamic_parameters, int(round(T / dt)))
    return static_parameters, dynamic_parameters, physical_fields(fields)


PERIODIC_LOOP_CASES = (
    ("flat Cartesian", _flat_cartesian_metric_at_position, "flat_cartesian"),
    ("Gaussian-normal", gaussian_normal_metric_at_position, "flat_cartesian"),
    ("generic lapse+shift", generic_lapse_shift_metric_at_position, "flat_cartesian"),
)
PERIOD = 4.0 * math.pi  # one period of the k = 0.5 test charts


def periodic_D_field(position, k=0.5):
    x, y, z = position
    return jnp.array(
        [
            jnp.sin(k * y) * jnp.cos(k * z),
            jnp.sin(k * z) * jnp.cos(k * x),
            jnp.sin(k * x) * jnp.cos(k * y),
        ]
    )


def periodic_B_field(position, k=0.5):
    x, y, z = position
    return jnp.array(
        [
            jnp.cos(k * y) * jnp.sin(k * z),
            jnp.cos(k * z) * jnp.sin(k * x),
            jnp.cos(k * x) * jnp.sin(k * y),
        ]
    )


class TestFieldSolverConvergenceFixtures:
    LEVELS = (12, 24, 48)
    def _field_errors(self, metric_at_position, wind, mins, metric_name, N):
        dt = 1.0e-6
        static_parameters, dynamic_parameters = parameters(
            (N, N, N), wind, mins, dt, metric_name=metric_name
        )
        metric = build_full_metric(dynamic_parameters, metric_at_position)
        guard = int(static_parameters.guard_cells)
        D_tiles = fill_staggered_vector(dynamic_parameters, smooth_D_field, D_FIELD_LOCATIONS)
        B_tiles = fill_staggered_vector(dynamic_parameters, smooth_B_field, B_FIELD_LOCATIONS)
        D_density = densitize_vector(D_tiles, metric.D)
        B_density = densitize_vector(B_tiles, metric.B)
        zero_current = empty_tiled_vector(static_parameters, dynamic_parameters)

        E_computed = compute_covariant_E(D_density, B_density, metric)
        H_computed = compute_covariant_H(D_density, B_density, metric)
        E_exact = exact_covariant_E(metric_at_position)
        H_exact = exact_covariant_H(metric_at_position)

        error_E = max(
            interior_rms(
                E_computed[i] - sample_component(dynamic_parameters, E_exact, D_FIELD_LOCATIONS[i], i),
                guard,
            )
            for i in range(3)
        )
        error_H = max(
            interior_rms(
                H_computed[i] - sample_component(dynamic_parameters, H_exact, B_FIELD_LOCATIONS[i], i),
                guard,
            )
            for i in range(3)
        )

        B_next = physical_vector(
            update_B(E_computed, B_density, metric, static_parameters, dynamic_parameters, dt),
            metric.B,
        )
        D_next = physical_vector(
            update_D(
                D_density, H_computed, zero_current, metric, static_parameters, dynamic_parameters, dt
            ),
            metric.D,
        )
        curl_E = exact_curl_over_sqrt_gamma(metric_at_position, E_exact)
        curl_H = exact_curl_over_sqrt_gamma(metric_at_position, H_exact)
        error_B_update = max(
            interior_rms(
                (B_next[i] - B_tiles[i]) / dt
                + sample_component(dynamic_parameters, curl_E, B_FIELD_LOCATIONS[i], i),
                guard,
            )
            for i in range(3)
        )
        error_D_update = max(
            interior_rms(
                (D_next[i] - D_tiles[i]) / dt
                - sample_component(dynamic_parameters, curl_H, D_FIELD_LOCATIONS[i], i),
                guard,
            )
            for i in range(3)
        )
        return error_E, error_H, error_B_update, error_D_update


class TestCurrentDepositionConvergenceFixtures:
    LEVELS = (12, 24, 48)
    PER_CELL = 2
