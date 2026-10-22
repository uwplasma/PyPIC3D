"""An exact circularly polarized Einstein--Maxwell wave.

Rosen plane-wave construction: https://arxiv.org/pdf/1202.0540, section II.
With u=t-z+0.7, ds^2=-dt^2+a(u)^2(dx^2+dy^2)+dz^2 and a=cos(u),
R_mn=-2(a''/a) (du)_m (du)_n = 2 (du)_m (du)_n.
The transverse orthonormal E and B have unit magnitude, E dot B=0,
and B=z_hat cross E. Thus 8*pi*T_mn=2 (du)_m (du)_n and R=0.
The densitized fields are D=a(cos(theta),sin(theta),0) and
B=a(-sin(theta),cos(theta),0), theta=8*pi*u. Their covariant E/H
components equal these densities, so Maxwell's curl equations reduce to
partial_t=partial_u and partial_z=-partial_u; both divergences vanish.

This is an analytical, genuinely curved solution, not a manufactured source
or a finest-grid reference. Space and time are refined together.
"""
import unittest

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from PyPIC3D.relativity.dynamical_relativity import tiles_to_bssn
from PyPIC3D.solvers.GR_yee.time_loop_dynamical_relativity import time_loop_dynamical_relativity

def native_runtime(shape, params, *, boundaries=(0, 0, 0), bssn=None,
                   displacement=None, magnetic=None, densitized=False,
                   gravity_G=1.0, absorber=None, initialize=True):
    """Single-tile native PyPIC fixture; shape counts physical BSSN C nodes."""
    from PyPIC3D.utilities.parameters import build_static_parameters, build_dynamic_parameters
    from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
    from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
    from PyPIC3D.relativity.dynamical_relativity import (
        bssn_to_tiles, build_bssn_metric, initialize_dynamical_relativity)
    from PyPIC3D.relativity.field_state import physical_vector

    widths = tuple(n - (bc == 1) for n, bc in zip(shape, boundaries))
    static = build_static_parameters(dict(
        solver="dynamic_metric", Nx=widths[0], Ny=widths[1], Nz=widths[2],
        tile_shape=widths, boundary_conditions=boundaries, shape_factor=1,
        particle_pusher="hybrid_boris_geodesic", current_deposition="GR_direct",
        current_filter="none", metric="numerical", guard_cells=3))
    origins = (params.x_min, params.y_min, params.z_min)
    center = tuple(origin + params.dx * jnp.arange(-1, width + 1) for origin, width in zip(origins, widths))
    vertex = tuple(axis + params.dx / 2 for axis in center)
    tc = tuple((origin + params.dx * jnp.arange(-3, width + 3)).reshape(1, 1, 1, -1)
               for origin, width in zip(origins, widths))
    tv = tuple(axis + params.dx / 2 for axis in tc)
    dynamic = build_dynamic_parameters(dict(
        dt=params.dt, dx=params.dx, dy=params.dx, dz=params.dx,
        Nx=widths[0], Ny=widths[1], Nz=widths[2],
        x_wind=params.dx * widths[0], y_wind=params.dx * widths[1], z_wind=params.dx * widths[2],
        C=1., eps=1., mu=1., kb=1., alpha=1.,
        grids=dict(center=center, vertex=vertex, tiled_center_grid=tc, tiled_vertex_grid=tv)))
    bssn = flat_bssn_variables(shape) if bssn is None else bssn
    dynamic = dynamic._replace(bssn_parameters=build_bssn_parameters(
        static, dynamic, state=bssn, evolution_parameters=params, gravity_G=gravity_G))
    zero = jnp.zeros((3,) + shape)
    displacement = zero if displacement is None else displacement
    magnetic = zero if magnetic is None else magnetic
    D = tuple(bssn_to_tiles(value, static) for value in displacement)
    B = tuple(bssn_to_tiles(value, static) for value in magnetic)
    if densitized:
        metric = build_bssn_metric(bssn, static, dynamic)
        D, B = physical_vector(D, metric.D), physical_vector(B, metric.B)
    z = tuple(jnp.zeros_like(value) for value in D)
    fields = (D, B, z, z[0], z[0], (z, z), None, (z, z), jnp.asarray(False))
    x = jnp.zeros((1, 1, 1, 0, 0, 3), dtype=jnp.float64)
    particles = TiledParticles(x, x, jnp.zeros(x.shape[:-1], dtype=bool))
    empty = jnp.zeros((0,), dtype=jnp.float64)
    species = SpeciesConfig(empty, empty, empty, jnp.zeros((0, 3), dtype=bool))
    if initialize:
        particles, fields, dynamic = initialize_dynamical_relativity(
            particles, species, fields, static, dynamic, absorber=absorber)
    return particles, species, fields, static, dynamic



def analytical_wave(z, time):
    """Return BSSN geometry and densitized D/B at the supplied coordinates."""
    u = time - z + 0.7
    a = jnp.cos(u)
    q = -jnp.tan(u)  # a'/a; K_ij = -partial_t gamma_ij / 2.
    zero = jnp.zeros_like(a)
    metric = jnp.stack((
        jnp.stack((a**(2/3), zero, zero)),
        jnp.stack((zero, a**(2/3), zero)),
        jnp.stack((zero, zero, a**(-4/3))),
    ))
    traceless = metric * jnp.stack((-q/3, -q/3, 2*q/3))[:, None]
    state = BSSNVariables(
        conformal_metric=metric, conformal_factor=a**(-2/3),
        traceless_K=traceless, trace_K=-2*q,
        conformal_connection=jnp.stack((zero, zero, 4*q*a**(4/3)/3)),
        lapse=jnp.ones_like(a), shift=jnp.stack((zero, zero, zero)),
    )
    theta = 8*jnp.pi*u
    D = jnp.stack((a*jnp.cos(theta), a*jnp.sin(theta), zero))
    B = jnp.stack((-a*jnp.sin(theta), a*jnp.cos(theta), zero))
    return state, D, B


class TestEinsteinMaxwellWave(unittest.TestCase):
    def test_coupled_wave_converges_to_analytical_solution(self):
        errors = []
        for n in (64, 128, 256):
            dx = 1.0/n
            dt = 0.2*dx
            steps = n//2
            final_time = steps*dt
            z = (-0.5 + dx*jnp.arange(n))[None, None, :]
            z_B = z + dx/2  # Both transverse B components live at upper z vertices.
            initial, D, _ = analytical_wave(z, 0.0)
            _, _, B = analytical_wave(z_B, 0.0)
            params = BSSNParameters(dx=dx, dt=dt, z_min=-0.5)
            particles, species, fields, static, dynamic = native_runtime(
                (1, 1, n), params, boundaries=(0, 0, 3), bssn=initial,
                displacement=D, magnetic=B, densitized=True, gravity_G=1.0,
            )

            # Keep exact geometry only at the boundaries. The interior evolves
            # through BSSN with the numerical EM stress-energy at every stage.
            def boundary(state, time):
                exact, _, _ = analytical_wave(z, time)
                return jax.tree.map(
                    lambda value, reference: jnp.where(jnp.abs(z) >= 0.4, reference, value),
                    state, exact,
                )

            def gauge(state, time):
                return (jnp.ones_like(state.lapse), jnp.zeros_like(state.shift),
                        jnp.zeros_like(state.lapse), jnp.zeros_like(state.shift))

            @jax.jit
            def evolve(particles, fields, dynamic):
                def step(index, carry):
                    p, f, d = carry
                    return time_loop_dynamical_relativity(
                        p, species, f, static, d,
                        prescribed_gauge=gauge, gravity_boundary=boundary,
                    )
                return jax.lax.fori_loop(0, steps, step, (particles, fields, dynamic))

            particles, fields, dynamic = evolve(particles, fields, dynamic)
            actual = dynamic.bssn_parameters.state
            exact, exact_D, _ = analytical_wave(z, final_time)
            _, _, exact_B = analytical_wave(z_B, final_time)
            actual_D = jnp.stack([tiles_to_bssn(v, static) for v in fields[0]])
            actual_B = jnp.stack([tiles_to_bssn(v, static) for v in fields[7][2]])
            gamma = actual.conformal_metric / actual.conformal_factor**2
            exact_gamma = exact.conformal_metric / exact.conformal_factor**2

            # This fixed physical window is separated from the constant EM
            # boundaries by more than the propagation distance T=0.1.
            resolution_errors = []
            for numerical, reference, coordinate in (
                (actual_D, exact_D, z), (actual_B, exact_B, z_B),
                (gamma, exact_gamma, z), (actual.trace_K, exact.trace_K, z),
            ):
                self.assertTrue(bool(jnp.all(jnp.isfinite(numerical))))
                mask = jnp.abs(coordinate) < 0.2
                squared_error = jnp.sum(jnp.where(mask, (numerical-reference)**2, 0.0))
                squared_reference = jnp.sum(jnp.where(mask, reference**2, 0.0))
                resolution_errors.append(float(jnp.sqrt(squared_error/squared_reference)))
            errors.append(resolution_errors)

            # Include the metric halo and retained negative-time samples in
            # the focusing bound. Curvature is nonzero: R_tt=2 everywhere.
            u_max = final_time + 0.5 + 3*dx + 0.7
            self.assertLess(u_max, float(jnp.pi/2))
            self.assertLess(float(jnp.min(exact_gamma[0, 0])), 0.5)
            self.assertAlmostEqual(float(dynamic.bssn_parameters.time), final_time, places=12)

        errors = jnp.asarray(errors)
        orders = jnp.log2(errors[:-1]/errors[1:])
        message = f"columns=D,B,gamma,K; errors={errors}; orders={orders}"
        self.assertTrue(bool(jnp.all(errors[1:] < errors[:-1])), message)
        self.assertTrue(bool(jnp.all(orders > 1.7)), message)
        self.assertTrue(bool(jnp.all(errors[-1] < 0.01)), message)


if __name__ == "__main__":
    unittest.main()
