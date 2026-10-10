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
from PyPIC3D.solvers.GR_yee.static_metric import (
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
# 1. The analytic 3+1 data against closed-form general relativity
# ===========================================================================

class TestAnalyticMetricsAgainstClosedForm(unittest.TestCase):
    def test_kerr_schild_cartesian_reproduces_the_closed_form_four_metric(self):
        """``(alpha, beta^i, gamma_ij)`` must rebuild ``eta_mu_nu + 2 H l_mu l_nu``."""

        def closed_form(position, mass, spin):
            x, y, z = position
            rho_squared = x * x + y * y + z * z
            r_squared = 0.5 * (
                rho_squared
                - spin**2
                + jnp.sqrt((rho_squared - spin**2) ** 2 + 4.0 * spin**2 * z * z)
            )
            r = jnp.sqrt(r_squared)
            denominator = r_squared + spin**2
            ell = jnp.array(
                [
                    1.0,
                    (r * x + spin * y) / denominator,
                    (r * y - spin * x) / denominator,
                    z / r,
                ]
            )
            H = mass * r**3 / (r**4 + spin**2 * z * z)
            return jnp.diag(jnp.array([-1.0, 1.0, 1.0, 1.0])) + 2.0 * H * jnp.outer(ell, ell)

        for spin in (0.0, 0.5, 0.9, 0.998):
            for position in ((5.0, 2.0, 1.0), (3.0, -1.5, 0.8), (-4.0, 1.0, -2.0)):
                point = jnp.asarray(position, dtype=float)
                lapse, shift, gamma, _gamma_inv, _sqrt_gamma = (
                    _kerr_schild_cartesian_metric_at_position(point, mass=1.0, spin=spin)
                )
                shift_lower = gamma @ shift
                rebuilt = jnp.zeros((4, 4))
                rebuilt = rebuilt.at[0, 0].set(-(lapse**2) + jnp.dot(shift_lower, shift))
                rebuilt = rebuilt.at[0, 1:].set(shift_lower)
                rebuilt = rebuilt.at[1:, 0].set(shift_lower)
                rebuilt = rebuilt.at[1:, 1:].set(gamma)
                error = float(jnp.max(jnp.abs(rebuilt - closed_form(point, 1.0, spin))))
                self.assertLess(error, 1.0e-12, f"spin={spin} position={position} error={error}")

    def test_every_metric_is_algebraically_self_consistent(self):
        """``gamma . gamma_inv = I`` and ``sqrt_gamma**2 = det(gamma)``."""
        cases = (
            (_flat_cylindrical_metric_at_position, (2.7, 0.4, 0.1)),
            (_flat_spherical_metric_at_position, (3.1, 1.2, 0.3)),
            (schwarzschild_cartesian_metric_at_position, (5.0, 2.0, 1.0)),
            (kerr_cartesian_metric_at_position(0.9), (5.0, 2.0, 1.0)),
            (kerr_cartesian_metric_at_position(0.998), (5.0, 2.0, 1.0)),
            (kerr_spherical_metric_at_position(0.9), (7.0, 1.3, 0.4)),
            (kerr_spherical_metric_at_position(0.998), (2.2, 0.7, 0.4)),
            (gaussian_normal_metric_at_position, (0.4, -0.3, 0.9)),
            (generic_lapse_shift_metric_at_position, (0.4, -0.3, 0.9)),
        )
        for metric_at_position, position in cases:
            point = jnp.asarray(position, dtype=float)
            _lapse, _shift, gamma, gamma_inv, sqrt_gamma = metric_at_position(point)
            identity_error = float(jnp.max(jnp.abs(gamma @ gamma_inv - jnp.eye(3))))
            determinant = jnp.linalg.det(gamma)
            determinant_error = float(jnp.abs(sqrt_gamma**2 - determinant) / jnp.abs(determinant))
            self.assertLess(identity_error, 1.0e-12)
            self.assertLess(determinant_error, 1.0e-12)
            self.assertGreater(float(determinant), 0.0)

    def test_flat_curvilinear_charts_reproduce_a_straight_line(self):
        """
        A force-free particle in flat space travels in a straight line.  Running
        the *production pusher* in cylindrical and spherical charts must return
        the image of that Cartesian straight line.
        """
        velocity = jnp.array([0.25, 0.30, -0.15])
        lorentz = 1.0 / math.sqrt(1.0 - float(velocity @ velocity))
        u_cartesian = lorentz * velocity
        start_cartesian = jnp.array([3.0, 0.55, 0.30])
        T = 1.2
        end_cartesian = start_cartesian + velocity * T

        def to_cylindrical(x):
            return jnp.array([jnp.hypot(x[0], x[1]), jnp.arctan2(x[1], x[0]), x[2]])

        def from_cylindrical(q):
            return jnp.array([q[0] * jnp.cos(q[1]), q[0] * jnp.sin(q[1]), q[2]])

        def to_spherical(x):
            radius = jnp.linalg.norm(x)
            return jnp.array([radius, jnp.arccos(x[2] / radius), jnp.arctan2(x[1], x[0])])

        def from_spherical(q):
            return jnp.array(
                [
                    q[0] * jnp.sin(q[1]) * jnp.cos(q[2]),
                    q[0] * jnp.sin(q[1]) * jnp.sin(q[2]),
                    q[0] * jnp.cos(q[1]),
                ]
            )

        def lower_into_chart(inverse_map, q, u_cart):
            return jax.jacfwd(inverse_map)(q).T @ u_cart

        charts = (
            ("flat_cylindrical", _flat_cylindrical_metric_at_position, to_cylindrical, from_cylindrical),
            ("flat_spherical", _flat_spherical_metric_at_position, to_spherical, from_spherical),
        )
        for metric_name, metric_at_position, forward, inverse in charts:
            q0 = forward(start_cartesian)
            u0 = lower_into_chart(inverse, q0, u_cartesian)
            q_expected = forward(end_cartesian)
            u_expected = lower_into_chart(inverse, q_expected, u_cartesian)
            mins = tuple(float(q0[axis]) - 0.6 for axis in range(3))
            wind = (2.0, 1.6, 1.6)
            q_final, u_final = run_production_pusher(
                metric_at_position,
                tuple(float(value) for value in q0),
                tuple(float(value) for value in u0),
                T,
                T / 600.0,
                48,
                wind,
                mins,
                metric_name=metric_name,
            )
            position_error = float(jnp.linalg.norm(q_final - q_expected))
            momentum_error = float(jnp.linalg.norm(u_final - u_expected))
            # The residual is the O(h^2) grid sampling of the metric, not the
            # scheme; TestPusherConvergence measures that order directly.
            self.assertLess(position_error, 1.0e-4, f"{metric_name}: {position_error}")
            self.assertLess(momentum_error, 1.0e-4, f"{metric_name}: {momentum_error}")

    def test_schwarzschild_and_kerr_circular_orbits_have_the_right_frequency(self):
        """
        A Kerr equatorial circular orbit at ``r`` has ``Omega = M^(1/2)/(r^(3/2) + a M^(1/2))``
        in Kerr-Schild time.  The production pusher must keep ``r`` fixed and
        return to the starting azimuth after ``2 pi / Omega``.
        """
        mass, radius = 1.0, 8.0
        for spin in (0.0, 0.9):
            root = math.sqrt(mass * radius)
            denominator = radius * math.sqrt(radius**2 - 3.0 * mass * radius + 2.0 * spin * root)
            energy = (radius**2 - 2.0 * mass * radius + spin * root) / denominator
            angular_momentum = root * (radius**2 - 2.0 * spin * root + spin**2) / denominator
            delta = radius**2 - 2.0 * mass * radius + spin**2
            u_r = (2.0 * mass * radius * energy - spin * angular_momentum) / delta
            omega = math.sqrt(mass) / (radius**1.5 + spin * math.sqrt(mass))
            period = 2.0 * math.pi / omega

            x_final, u_final = run_production_pusher(
                kerr_spherical_metric_at_position(spin),
                (radius, 0.5 * math.pi, 0.0),
                (u_r, 0.0, angular_momentum),
                0.98 * period,
                0.05,
                48,
                (6.0, 0.8, 7.3),
                (5.0, 1.15, -0.3),
                metric_name="kerr_schild_spherical",
            )
            radius_drift = abs(float(x_final[0]) - radius) / radius
            theta_drift = abs(float(x_final[1]) - 0.5 * math.pi)
            # run_production_pusher rounds the step count; compare at that
            # actual time, not the unachieved requested endpoint.
            actual_time = round((0.98 * period) / 0.05) * 0.05
            azimuth_error = abs(float(x_final[2]) - omega * actual_time)
            momentum_error = abs(float(u_final[2]) - angular_momentum) / abs(angular_momentum)
            self.assertLess(radius_drift, 2.0e-3, f"spin={spin} radius drift {radius_drift}")
            self.assertLess(theta_drift, 1.0e-4, f"spin={spin} theta drift {theta_drift}")
            self.assertLess(azimuth_error, 5.0e-3, f"spin={spin} azimuth error {azimuth_error}")
            self.assertLess(momentum_error, 1.0e-12, f"spin={spin} u_phi drift {momentum_error}")


# ===========================================================================
# 2. Particle pusher: convergence in space and in time, per chart
# ===========================================================================

class TestPusherConvergence(unittest.TestCase):
    SPATIAL_LEVELS = (16, 32, 64)
    TEMPORAL_STEPS = (200, 400, 800, 1600)

    def test_pusher_accuracy_for_exactly_reconstructed_cylindrical_metric(self):
        """Cylindrical gamma is quadratic: spatial refinement has no error to remove."""
        case = PUSHER_CASES[0]
        reference = reference_trajectory(case["metric"], case["x0"], case["u0"], case["T"])
        positions = []
        for N in (16, 32):
            x, _ = run_production_pusher(
                case["metric"], case["x0"], case["u0"], case["T"], case["T"]/4000,
                N, case["wind"], case["mins"], metric_name=case["metric_name"])
            self.assertLess(float(jnp.linalg.norm(x-reference[:3])), 2e-9)
            positions.append(x)
        self.assertLess(float(jnp.linalg.norm(positions[0]-positions[1])), 1e-12)

    def test_pusher_spatial_convergence_above_temporal_and_reference_floors(self):
        """
        Hermite values are generally third order and derivatives second order.
        Integrated trajectory errors can converge faster through cancellation;
        require at least second order without imposing a false upper bound.
        Explicit dt and RK4 refinement keep the finest spatial error resolved.
        """
        report = []
        for case in PUSHER_CASES[1:]:
            reference = reference_trajectory(
                case["metric"],
                case["x0"],
                case["u0"],
                case["T"],
                case.get("charge", 0.0),
                case.get("D_values", (0, 0, 0)),
                case.get("B_values", (0, 0, 0)),
            )
            dt = case["T"] / 16000.0
            errors = []
            for N in self.SPATIAL_LEVELS:
                x_final, _u_final = run_production_pusher(
                    case["metric"],
                    case["x0"],
                    case["u0"],
                    case["T"],
                    dt,
                    N,
                    case["wind"],
                    case["mins"],
                    metric_name=case["metric_name"],
                    charge=case.get("charge", 0.0),
                    D_values=case.get("D_values", (0, 0, 0)),
                    B_values=case.get("B_values", (0, 0, 0)),
                )
                errors.append(float(jnp.linalg.norm(x_final - reference[:3])))
            refined, _ = run_production_pusher(
                case["metric"], case["x0"], case["u0"], case["T"], dt/2,
                self.SPATIAL_LEVELS[-1], case["wind"], case["mins"],
                metric_name=case["metric_name"], charge=case.get("charge", 0.0),
                D_values=case.get("D_values", (0, 0, 0)),
                B_values=case.get("B_values", (0, 0, 0)))
            coarse_reference = reference_trajectory(
                case["metric"], case["x0"], case["u0"], case["T"],
                case.get("charge", 0.0), case.get("D_values", (0, 0, 0)),
                case.get("B_values", (0, 0, 0)), n=20000)
            temporal_delta = float(jnp.linalg.norm(refined-x_final))
            reference_delta = float(jnp.linalg.norm(coarse_reference[:3]-reference[:3]))
            orders = [
                convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
            ]
            report.append(f"{case['name']}: errors={errors} orders={orders} "
                          f"temporal_delta={temporal_delta} reference_delta={reference_delta}")
            self.assertLess(temporal_delta, .05*errors[-1], "\n".join(report))
            self.assertLess(reference_delta, .01*errors[-1], "\n".join(report))
            for order in orders:
                self.assertGreater(order, 1.8, "\n".join(report))
            jax.clear_caches()

    def test_pusher_converges_second_order_in_time_for_every_chart(self):
        """
        Time-step self-convergence on a fixed reconstructed metric isolates
        Strang/Boris/leapfrog temporal truncation from spatial reconstruction.
        """
        report = []
        for case in PUSHER_CASES:
            N = 24
            solutions = []
            for steps in self.TEMPORAL_STEPS:
                solutions.append(
                    run_production_pusher(
                        case["metric"],
                        case["x0"],
                        case["u0"],
                        case["T"],
                        case["T"] / steps,
                        N,
                        case["wind"],
                        case["mins"],
                        metric_name=case["metric_name"],
                        charge=case.get("charge", 0.0),
                        D_values=case.get("D_values", (0, 0, 0)),
                        B_values=case.get("B_values", (0, 0, 0)),
                    )
                )
            finest = solutions[-1]
            errors = [
                float(jnp.linalg.norm(solution[0] - finest[0]))
                + float(jnp.linalg.norm(solution[1] - finest[1]))
                for solution in solutions[:-1]
            ]
            orders = [
                convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
            ]
            report.append(f"{case['name']}: errors={errors} orders={orders}")
            for order in orders:
                self.assertGreater(order, 1.7, "\n".join(report))

    def test_pusher_is_second_order_with_second_order_particle_shapes(self):
        """Second-order electromagnetic particle shapes retain GR spatial accuracy."""
        case = PUSHER_CASES[5]  # Gaussian-normal chart
        reference = reference_trajectory(case["metric"], case["x0"], case["u0"], case["T"])
        errors = []
        for N in self.SPATIAL_LEVELS:
            x_final, _u_final = run_production_pusher(
                case["metric"],
                case["x0"],
                case["u0"],
                case["T"],
                case["T"] / 4000.0,
                N,
                case["wind"],
                case["mins"],
                metric_name=case["metric_name"],
                shape_factor=2,
            )
            errors.append(float(jnp.linalg.norm(x_final - reference[:3])))
        orders = [convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)]
        for order in orders:
            self.assertGreater(order, 1.8, f"errors={errors} orders={orders}")


# ===========================================================================
# 3. Field solver: constitutive relations and both curl updates
# ===========================================================================

class TestFieldSolverConvergence(unittest.TestCase):
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

    def test_constitutive_relations_and_curl_updates_are_second_order(self):
        """
        ``E_i = alpha gamma_ij D^j + sqrt(gamma) eps_ijk beta^j B^k`` (FPIC Eq. 10),
        ``H_i = alpha gamma_ij B^j - sqrt(gamma) eps_ijk beta^j D^k`` (FPIC Eq. 9),
        ``dt B^i = -curl(E)^i / sqrt(gamma)`` (Eq. 8) and
        ``dt D^i = curl(H)^i / sqrt(gamma) - 4 pi J^i`` (Eq. 7).
        """
        report = []
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES:
            measurements = [
                self._field_errors(metric_at_position, wind, mins, metric_name, N)
                for N in self.LEVELS
            ]
            for index, label in enumerate(("E_i", "H_i", "dB/dt", "dD/dt")):
                errors = [measurement[index] for measurement in measurements]
                if max(errors) < 1.0e-13:
                    continue  # exact for this chart (diagonal metric, zero shift)
                orders = [
                    convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
                ]
                report.append(f"{name} {label}: errors={errors} orders={orders}")
                for order in orders:
                    self.assertGreater(order, 1.85, "\n".join(report))
                    self.assertLess(order, 2.3, "\n".join(report))

    def test_constitutive_relations_are_exact_when_no_interpolation_is_needed(self):
        """
        For a diagonal three-metric with zero shift, ``E_i`` and ``H_i`` need no
        off-location interpolation and must be reproduced to round-off.
        """
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES[:2]:
            error_E, error_H, _dB, _dD = self._field_errors(
                metric_at_position, wind, mins, metric_name, 16
            )
            self.assertLess(error_E, 1.0e-13, f"{name}: {error_E}")
            self.assertLess(error_H, 1.0e-13, f"{name}: {error_H}")


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


class TestCurrentDepositionConvergence(unittest.TestCase):
    LEVELS = (12, 24, 48)
    PER_CELL = 2

    def test_deposited_current_converges_second_order_for_every_chart(self):
        report = []
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES:
            errors = []
            for N in self.LEVELS:
                static_parameters, dynamic_parameters = parameters(
                    (N, N, N), wind, mins, 0.01, metric_name=metric_name
                )
                metric = build_full_metric(dynamic_parameters, metric_at_position)
                guard = int(static_parameters.guard_cells)
                positions = coordinate_lattice(mins, wind, N, self.PER_CELL)
                momenta = jax.vmap(deposition_velocity_field)(positions)
                count = positions.shape[0]
                particles = TiledParticles(
                    x=positions.reshape(1, 1, 1, 1, count, 3),
                    u=momenta.reshape(1, 1, 1, 1, count, 3),
                    active=jnp.ones((1, 1, 1, 1, count), dtype=bool),
                )
                J = physical_vector(GR_direct_deposition(
                    particles,
                    one_species(1.0),
                    empty_tiled_vector(static_parameters, dynamic_parameters),
                    metric,
                    static_parameters,
                    dynamic_parameters,
                ), metric.D)
                cell_volume = float(dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz)
                lattice_density = self.PER_CELL**3 / cell_volume
                component_errors = []
                for i in range(3):
                    grid_positions = location_positions(D_FIELD_LOCATIONS[i], dynamic_parameters)
                    shape = grid_positions.shape[:-1]
                    flat = grid_positions.reshape(-1, 3)
                    source = jax.vmap(
                        lambda point: exact_source_velocity(metric_at_position, point)
                    )(flat)[:, i].reshape(shape)
                    sqrt_gamma = jax.vmap(lambda point: metric_at_position(point)[4])(flat).reshape(shape)
                    exact = lattice_density * source / sqrt_gamma
                    component_errors.append(
                        interior_rms(J[i] - exact, guard, skip=3)
                        / interior_rms(exact, guard, skip=3)
                    )
                errors.append(max(component_errors))
            orders = [convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)]
            report.append(f"{name}: errors={errors} orders={orders}")
            for order in orders:
                self.assertGreater(order, 1.55, "\n".join(report))

    def test_total_deposited_current_equals_the_single_particle_source(self):
        """
        The shape function is a partition of unity, so
        ``sum_g sqrt(gamma)_g J^i_g dV`` must equal ``q w (alpha v^i - beta^i)``
        *exactly*, with the metric taken where the kernel itself takes it -- from
        the grid.  Any deviation is a lost or double-counted contribution, not a
        discretisation error.
        """
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES:
            N = 16
            static_parameters, dynamic_parameters = parameters(
                (N, N, N), wind, mins, 0.01, metric_name=metric_name
            )
            metric = build_full_metric(dynamic_parameters, metric_at_position)
            guard = int(static_parameters.guard_cells)
            position = jnp.asarray(
                [mins[axis] + (0.5137, 0.4231, 0.6117)[axis] * wind[axis] for axis in range(3)]
            )
            momentum = deposition_velocity_field(position)
            particles = TiledParticles(
                x=position.reshape(1, 1, 1, 1, 1, 3),
                u=momentum.reshape(1, 1, 1, 1, 1, 3),
                active=jnp.ones((1, 1, 1, 1, 1), dtype=bool),
            )
            J = GR_direct_deposition(
                particles,
                one_species(2.0, weight=3.0),
                empty_tiled_vector(static_parameters, dynamic_parameters),
                metric,
                static_parameters,
                dynamic_parameters,
            )
            cell_volume = dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz
            window = (slice(None),) * 3 + (slice(guard, -guard),) * 3
            totals = jnp.asarray(
                [
                    jnp.sum(J[i][window]) * cell_volume
                    for i in range(3)
                ]
            )
            sampled = sample_center_metric(
                metric, position.reshape(1, 3), static_parameters, dynamic_parameters
            )
            velocity = contravariant_three_velocity(momentum.reshape(1, 3), sampled.gamma_inv)
            expected = 6.0 * (sampled.lapse[:, None] * velocity - sampled.shift)[0]
            error = float(jnp.max(jnp.abs(totals - expected)) / jnp.max(jnp.abs(expected)))
            self.assertLess(error, 1.0e-12, f"{name}: relative error {error}")

    def test_metric_sampled_source_velocity_is_third_order(self):
        """
        Hermite primitive values and their consistent inverse give a generally
        third-order source velocity ``alpha v^i - beta^i``. Sampling many
        off-node positions avoids special sub-cell superconvergence.
        """
        key = jax.random.PRNGKey(11)
        report = []
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES[2:]:
            errors = []
            for N in (12, 24, 48):
                static_parameters, dynamic_parameters = parameters(
                    (N, N, N), wind, mins, 0.01, metric_name=metric_name
                )
                metric = build_center_metric(dynamic_parameters, metric_at_position)
                fractions = jax.random.uniform(key, (400, 3), minval=0.2, maxval=0.8)
                positions = jnp.asarray(mins) + fractions * jnp.asarray(wind)
                sampled = sample_center_metric(
                    metric, positions, static_parameters, dynamic_parameters
                )
                momenta = jax.vmap(deposition_velocity_field)(positions)
                velocity = contravariant_three_velocity(momenta, sampled.gamma_inv)
                computed = sampled.lapse[:, None] * velocity - sampled.shift
                exact = jax.vmap(
                    lambda point: exact_source_velocity(metric_at_position, point)
                )(positions)
                errors.append(
                    float(jnp.sqrt(jnp.mean(jnp.sum((computed - exact) ** 2, axis=1))))
                )
            orders = [
                convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
            ]
            report.append(f"{name}: errors={errors} orders={orders}")
            for order in orders:
                self.assertGreater(order, 2.65, "\n".join(report))
                self.assertLess(order, 3.35, "\n".join(report))

    def test_hermite_metric_values_and_derivatives_have_distinct_orders(self):
        """Smooth supplied data exercise lapse, shift and non-diagonal gamma in 3D."""
        provider = generic_lapse_shift_metric_at_position
        positions = jax.random.uniform(jax.random.PRNGKey(913), (400, 3), minval=-1.2, maxval=1.2)
        exact = jax.vmap(provider)(positions)
        gradients = jax.vmap(jax.jacfwd(lambda q: (provider(q)[0], provider(q)[1], provider(q)[3])))(positions)
        errors = {name: [] for name in ("lapse", "shift", "gamma", "inverse", "grad_lapse", "grad_shift", "grad_inverse")}
        for N in (12, 24, 48):
            s, d = parameters((N, N, N), (4., 4., 4.), (-2., -2., -2.), .01)
            m = build_center_metric(d, provider)
            tile = jax.tree.map(lambda a: a[0, 0, 0], m.center)
            grid = tuple(a[0, 0, 0] for a in d.grids.tiled_center_grid)
            sampled = interpolate_metric(tile, positions, grid, "numerical", (True, True, True), (3, 3, 3))
            computed = sampled[:4] + sampled[5:]
            expected = exact[:4] + (gradients[0], gradients[1], jnp.moveaxis(gradients[2], -1, -3))
            for name, a, b in zip(errors, computed, expected):
                errors[name].append(float(jnp.sqrt(jnp.mean((a-b)**2))))
        for name, values in errors.items():
            expected_order = 2 if name.startswith("grad_") else 3
            orders = [convergence_order(a, b) for a, b in zip(values, values[1:])]
            with self.subTest(quantity=name):
                for order in orders:
                    self.assertGreater(order, expected_order-.35, (values, orders))
                    self.assertLess(order, expected_order+.35, (values, orders))


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


class TestTimeLoopConvergence(unittest.TestCase):
    def test_flat_vacuum_plane_wave_is_second_order_in_space(self):
        """
        ``D^z = sin(k(x - t))``, ``B^y = -D^z`` is an exact vacuum solution of
        Eqs. (7)-(8) with ``alpha = 1``, ``beta = 0``.  Propagating it a full
        transit through the complete time loop pins down every sign in the
        Maxwell update.
        """
        L = 2.0 * math.pi
        T = 2.0 * math.pi
        errors = []
        for N in (16, 32, 64):
            dt = T / 2048.0
            static_parameters, dynamic_parameters = parameters(
                (N, 4, 4), (L, L, L), (0.0, 0.0, 0.0), dt, metric_name="flat_cartesian"
            )
            metric = build_full_metric(dynamic_parameters, _flat_cartesian_metric_at_position)
            guard = int(static_parameters.guard_cells)

            def travelling_D(position, t):
                return jnp.array([0.0, 0.0, jnp.sin(position[0] - t)])

            def travelling_B(position, t):
                return jnp.array([0.0, -jnp.sin(position[0] - t), 0.0])

            fields = consistent_initial_field_state(
                static_parameters,
                dynamic_parameters,
                metric,
                lambda p: travelling_D(p, 0.0),
                lambda p: travelling_B(p, 0.0),
                dt,
            )
            particles, species = empty_particle_state()

            steps = int(round(T / dt))
            particles, fields = _advance_time_loop(
                particles, species, fields, static_parameters, dynamic_parameters, steps)
            positions = location_positions(D_FIELD_LOCATIONS[2], dynamic_parameters)
            exact = jnp.sin(positions[..., 0] - steps * dt)
            errors.append(interior_rms(physical_fields(fields)[0][2] - exact, guard))
        orders = [convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)]
        for order in orders:
            self.assertGreater(order, 1.85, f"errors={errors} orders={orders}")
            self.assertLess(order, 2.2, f"errors={errors} orders={orders}")

    def test_time_loop_is_second_order_in_time_for_curved_charts(self):
        report = []
        T = 1.0
        for name, metric_at_position, metric_name in PERIODIC_LOOP_CASES:
            solutions = []
            step_counts = (50, 100, 200, 400)
            for steps in step_counts:
                _sp, _dp, fields = evolve_vacuum(
                    metric_at_position,
                    metric_name,
                    16,
                    (PERIOD,) * 3,
                    (0.0,) * 3,
                    T / steps,
                    T,
                    periodic_D_field,
                    periodic_B_field,
                )
                solutions.append(fields[0])
            finest = solutions[-1]
            errors = [
                max(
                    interior_rms(solution[component] - finest[component], 2)
                    for component in range(3)
                )
                for solution in solutions[:-1]
            ]
            orders = [convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)]
            report.append(f"{name}: errors={errors} orders={orders}")
            for order in orders:
                self.assertGreater(order, 1.85, "\n".join(report))


# ===========================================================================
# 6. Invariants and constraints
# ===========================================================================

class TestInvariantsAndConstraints(unittest.TestCase):
    def test_generalized_boris_rotation_preserves_the_lorentz_factor(self):
        """
        The magnetic half of the push is a rotation in the metric ``gamma_ij``
        (Aperture Eqs. 17-19, FPIC Eq. 16), so
        ``Gamma = sqrt(1 + gamma^ij u_i u_j)`` must be invariant to round-off for
        any time step, not just small ones.
        """
        charts = (
            ("flat cylindrical", _flat_cylindrical_metric_at_position, (2.7, 0.4, 0.1)),
            ("flat spherical", _flat_spherical_metric_at_position, (3.1, 1.2, 0.3)),
            ("Kerr a=0.9 KS-cart", kerr_cartesian_metric_at_position(0.9), (5.0, 2.0, 1.0)),
            ("Kerr a=0.9 KS-sph", kerr_spherical_metric_at_position(0.9), (7.0, 1.3, 0.4)),
            ("Gaussian-normal", gaussian_normal_metric_at_position, (0.4, -0.3, 0.9)),
            ("generic lapse+shift", generic_lapse_shift_metric_at_position, (0.4, -0.3, 0.9)),
        )
        key = jax.random.PRNGKey(3)
        for name, metric_at_position, position in charts:
            lapse, shift, gamma, gamma_inv, sqrt_gamma = metric_at_position(
                jnp.asarray(position, dtype=float)
            )
            sampled = Metric(
                lapse=jnp.asarray([lapse]),
                shift=shift[None, :],
                gamma=gamma[None, :, :],
                gamma_inv=gamma_inv[None, :, :],
                sqrt_gamma=jnp.asarray([sqrt_gamma]),
            )
            worst = 0.0
            for trial in range(25):
                key_u, key_B = jax.random.split(jax.random.fold_in(key, trial))
                u_cov = 1.5 * jax.random.normal(key_u, (1, 3))
                B_con = 2.0 * jax.random.normal(key_B, (1, 3))
                for dt in (0.01, 0.5, 5.0, 50.0):
                    rotated = magnetic_boris_rotation(
                        u_cov, B_con, sampled, jnp.asarray([1.3]), dt
                    )
                    before = float(covariant_lorentz_factor(u_cov, sampled.gamma_inv)[0])
                    after = float(covariant_lorentz_factor(rotated, sampled.gamma_inv)[0])
                    worst = max(worst, abs(after - before) / before)
            self.assertLess(worst, 1.0e-12, f"{name}: {worst}")

    def test_magnetic_constraint_is_preserved_to_round_off(self):
        """
        ``d_i (sqrt(gamma)_i B^i)`` is the divergence the Yee update annihilates
        identically on the native densities; the metric weighting in ``E`` must
        not break that.
        """
        key = jax.random.PRNGKey(0)
        for name, metric_at_position, wind, mins, metric_name in FIELD_CASES:
            N = 16
            static_parameters, dynamic_parameters = parameters(
                (N, N, N), wind, mins, 0.01, metric_name=metric_name
            )
            metric = build_full_metric(dynamic_parameters, metric_at_position)
            guard = int(static_parameters.guard_cells)
            shape = empty_tiled_scalar(static_parameters, dynamic_parameters).shape
            B_tiles = densitize_vector(tuple(
                jax.random.normal(jax.random.fold_in(key, i), shape) for i in range(3)
            ), metric.B)
            D_tiles = densitize_vector(tuple(
                jax.random.normal(jax.random.fold_in(key, 10 + i), shape) for i in range(3)
            ), metric.D)
            E_tiles = compute_covariant_E(D_tiles, B_tiles, metric)

            before = divergence(B_tiles, dynamic_parameters, forward=True)
            evolved = B_tiles
            for _ in range(20):
                evolved = update_B(
                    E_tiles, evolved, metric, static_parameters, dynamic_parameters,
                    dynamic_parameters.dt,
                )
            after = divergence(evolved, dynamic_parameters, forward=True)
            window = (slice(None),) * 3 + (slice(guard + 1, -guard - 1),) * 3
            drift = float(jnp.max(jnp.abs((after - before)[window])))
            scale = float(jnp.max(jnp.abs(before[window])))
            self.assertLess(drift / scale, 1.0e-11, f"{name}: drift={drift} scale={scale}")

    def test_geodesic_invariants_converge_second_order_in_grid_spacing(self):
        """
        ``H = alpha Gamma - beta^i u_i`` (stationarity) and ``L_z = x u_y - y u_x``
        (axisymmetry) must be conserved.  They are evaluated at matching times,
        ``(x^{n+1/2}, u^{n+1/2})``, because the pusher stores ``u`` on half steps.
        """
        cases = (
            (
                "Schwarzschild KS-cart",
                schwarzschild_cartesian_metric_at_position,
                "kerr_schild_cartesian",
                (10.0, 0.0, 0.3),
                (0.09539392014169457, 0.3651483716701107, 0.0),
                10.0,
                (8.0, 9.0, 4.0),
                (4.0, -2.0, -2.0),
            ),
            (
                "Kerr a=0.9 KS-cart",
                kerr_cartesian_metric_at_position(0.9),
                "kerr_schild_cartesian",
                (8.0, 0.0, 0.4),
                (0.10, 0.42, 0.02),
                10.0,
                (8.0, 9.0, 4.0),
                (1.5, -2.0, -2.0),
            ),
        )
        report = []
        for name, metric_at_position, metric_name, x0, u0, T, wind, mins in cases:
            state0 = jnp.concatenate((jnp.asarray(x0), jnp.asarray(u0)))
            H0 = float(hamiltonian(metric_at_position, state0))
            L0 = x0[0] * u0[1] - x0[1] * u0[0]
            energy_errors = []
            momentum_errors = []
            for N in (12, 24, 48):
                dt = 0.01
                static_parameters, dynamic_parameters = parameters(
                    (N, N, N), wind, mins, dt, metric_name=metric_name
                )
                metric = build_center_metric(dynamic_parameters, metric_at_position)
                D = empty_tiled_vector(static_parameters, dynamic_parameters)
                B = empty_tiled_vector(static_parameters, dynamic_parameters)
                species = one_species(0.0)
                started, _ = hybrid_boris_geodesic_push(
                    one_particle(x0, u0),
                    species,
                    D,
                    B,
                    metric,
                    static_parameters,
                    dynamic_parameters._replace(dt=jnp.asarray(-0.5 * dt)),
                )
                particles = one_particle(x0, started.u[0, 0, 0, 0, 0])

                def body(current, _):
                    advanced, centered = hybrid_boris_geodesic_push(
                        current, species, D, B, metric, static_parameters, dynamic_parameters
                    )
                    return advanced, (centered.x[0, 0, 0, 0, 0], centered.u[0, 0, 0, 0, 0])

                _final, (X, U) = jax.lax.scan(body, particles, None, length=int(round(T / dt)))
                H = jax.vmap(lambda s: hamiltonian(metric_at_position, s))(
                    jnp.concatenate((X, U), axis=1)
                )
                Lz = X[:, 0] * U[:, 1] - X[:, 1] * U[:, 0]
                energy_errors.append(float(jnp.max(jnp.abs(H - H0))) / abs(H0))
                momentum_errors.append(float(jnp.max(jnp.abs(Lz - L0))) / abs(L0))
            energy_orders = [
                convergence_order(energy_errors[i], energy_errors[i + 1])
                for i in range(len(energy_errors) - 1)
            ]
            momentum_orders = [
                convergence_order(momentum_errors[i], momentum_errors[i + 1])
                for i in range(len(momentum_errors) - 1)
            ]
            report.append(
                f"{name}: dH={energy_errors} p={energy_orders} dL={momentum_errors} p={momentum_orders}"
            )
            self.assertGreater(energy_orders[-1], 1.7, "\n".join(report))
            self.assertGreater(momentum_orders[-1], 1.7, "\n".join(report))


# ===========================================================================
# 7. Leapfrog start
# ===========================================================================

class TestLeapfrogVelocitySeed(unittest.TestCase):
    """
    ``seed_leapfrog_velocity`` is what makes the staggered scheme second order.
    Without it the initial state carries an O(dt) error that the pusher cannot
    recover from, and the measured rate collapses to one.
    """

    STEP_COUNTS = (200, 400, 800, 1600)

    def _final_positions(self, case, seeded, N):
        positions = []
        for steps in self.STEP_COUNTS:
            x_final, _u_final = run_production_pusher(
                case["metric"],
                case["x0"],
                case["u0"],
                case["T"],
                case["T"] / steps,
                N,
                case["wind"],
                case["mins"],
                metric_name=case["metric_name"],
                charge=case.get("charge", 0.0),
                D_values=case.get("D_values", (0, 0, 0)),
                B_values=case.get("B_values", (0, 0, 0)),
                offset_initial_half_step=seeded,
            )
            positions.append(x_final)
        return positions

    def test_seeding_restores_second_order_in_a_curved_chart(self):
        """
        Generic lapse+shift chart.  Time-step self-convergence on a fixed grid,
        so the O(h^2) metric-sampling floor cancels and only the effect of the
        initial state is left.
        """
        case = PUSHER_CASES[6]  # generic lapse+shift chart, charged, with a B field
        report = []
        rates = {}
        for seeded in (True, False):
            positions = self._final_positions(case, seeded, 32)
            finest = positions[-1]
            errors = [float(jnp.linalg.norm(x - finest)) for x in positions[:-1]]
            orders = [
                convergence_order(errors[i], errors[i + 1]) for i in range(len(errors) - 1)
            ]
            rates[seeded] = (errors, orders)
            report.append(f"seeded={seeded}: errors={errors} orders={orders}")

        seeded_errors, seeded_orders = rates[True]
        unseeded_errors, unseeded_orders = rates[False]
        for order in seeded_orders:
            self.assertGreater(order, 1.8, "\n".join(report))
        # self-convergence against the finest run biases the estimator upwards,
        # so the unseeded rate is only required to stay clearly short of second
        # order -- the decisive statement is the size of the error itself
        self.assertLess(max(unseeded_orders), 1.8, "\n".join(report))
        self.assertGreater(
            unseeded_errors[0] / seeded_errors[0], 100.0, "\n".join(report)
        )


if __name__ == "__main__":
    unittest.main()
