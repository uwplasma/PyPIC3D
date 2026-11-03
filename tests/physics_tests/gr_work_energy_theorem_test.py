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
from tests.code_tests.dynamical_relativity_test import coupled_state
from tests.kernel_fixtures import empty_tiled_vector, kernel_parameters

jax.config.update("jax_enable_x64", True)


def maxwell_state(n=24, *, dt=None, shift=0., pulse=False, enabled=True):
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
    static = static._replace(gr_work_energy_theorem=enabled)
    bssn = build_bssn_parameters(static, dynamic, gravity_G=0.)
    state = bssn.state._replace(shift=bssn.state.shift.at[0].set(shift))
    bssn = bssn._replace(state=state,
                         evolution_parameters=bssn.evolution_parameters._replace(nu=0.))
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


class GRWorkEnergyEvolutionTests(unittest.TestCase):
    def assert_tree_close(self, actual, expected, *, atol=2.e-12):
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            self.assertTrue(bool(jnp.allclose(left, right, rtol=2.e-12, atol=atol)),
                            f"unequal values with shape {left.shape}")

    def test_periodic_maxwell_residual_is_synchronized_energy_drift(self):
        static, dynamic, particles, species, fields = maxwell_state(n=16)
        particles, fields, dynamic = evolve_scan(
            static, species, (particles, fields, dynamic), 12)
        tracker = dynamic.work_energy_tracker
        values = tracker_values(tracker)
        drift = tracker.latest.electromagnetic - tracker.baseline.electromagnetic
        self.assertAlmostEqual(float(values["residual"]), float(drift), places=14)
        self.assertEqual(float(tracker.curvature_work), 0.)
        self.assertEqual(float(tracker.lapse_work), 0.)
        self.assertEqual(float(tracker.boundary_work), 0.)
        self.assertEqual(float(tracker.latest.particles), 0.)
        self.assertEqual(int(tracker.steps), 12)
        self.assertTrue(bool(jnp.isfinite(values["relative_residual"])))
        # Reconstructed continuum energy has a truncation error, not an exact
        # Yee modified-energy invariant. This small smooth run resolves it.
        self.assertLess(float(values["relative_residual"]), 2.e-3)

    def test_diagnostic_does_not_change_evolution_or_depend_on_sampling_cadence(self):
        static, dynamic, particles, species, fields = maxwell_state(n=16)
        initial = (particles, fields, dynamic)
        enabled = evolve_scan(static, species, initial, 12)
        disabled = evolve_scan(
            static._replace(gr_work_energy_theorem=False), species,
            (particles, fields, dynamic._replace(work_energy_tracker=None)), 12)
        self.assert_tree_close(enabled[:2], disabled[:2])
        self.assert_tree_close(enabled[2].bssn_parameters, disabled[2].bssn_parameters)

        chunked = initial
        for _ in range(3):
            chunked = evolve_scan(static, species, chunked, 4)
            # A writer samples scalar values, but never updates their carry.
            values = tracker_values(chunked[2].work_energy_tracker)
            self.assertTrue(bool(jnp.isfinite(values["total"])))
        self.assert_tree_close(enabled, chunked)

    def test_particle_field_exchange_has_opposite_signs_and_total_closure(self):
        errors = []
        for dt in (.04, .02):
            static, dynamic, particles, species, fields = coupled_state(
                particles=True, dt=dt)
            static = static._replace(gr_work_energy_theorem=True)
            dynamic = dynamic._replace(work_energy_tracker=initialize_tracker(
                particles, species, fields, static, dynamic))
            common = synchronize_particles(particles, species, fields, static, dynamic)
            gamma = jnp.sqrt(1. + jnp.sum(common.u**2, axis=-1))
            electric = jnp.asarray([component[0, 0, 0, 3, 3, 3] for component in fields[0]])
            power = jnp.sum(common.active * species.charge[None, None, None, :, None]
                            * species.weight[None, None, None, :, None]
                            * jnp.sum(common.u * electric, axis=-1) / gamma)
            self.assertGreater(float(power), 0.)
            _, _, final = evolve_scan(static, species, (particles, fields, dynamic), round(.16 / dt))
            tracker = final.work_energy_tracker
            particle_gain = tracker.latest.particles - tracker.baseline.particles
            field_gain = tracker.latest.electromagnetic - tracker.baseline.electromagnetic
            self.assertGreater(float(particle_gain), 0.)
            self.assertLess(float(field_gain), 0.)
            self.assertLess(abs(float(particle_gain / (.16 * power)) - 1.), .1)
            errors.append(float(jnp.abs(particle_gain + field_gain) / particle_gain))
            rest = jnp.sum(common.active * species.mass[None, None, None, :, None]
                           * species.weight[None, None, None, :, None])
            self.assertGreater(float(tracker.baseline.particles), float(rest))
        self.assertLess(errors[-1], 2.e-3, str(errors))
        self.assertGreater(errors[0] / errors[1], 3., str(errors))

    def test_outgoing_pulse_shift_flux_and_residual_refine(self):
        errors = []
        energy_errors, transport_errors, field_errors = [], [], []
        for n in (48, 96, 192):
            # Both characteristic coordinate speeds are positive (2.5, .5),
            # making the constant upper exterior an outflow boundary. Zero
            # shift would freeze a last B plane and reflect this Yee pulse.
            static, dynamic, particles, species, fields = maxwell_state(
                n=n, pulse=True, shift=-1.5)
            _, final_fields, final = evolve_scan(static, species, (particles, fields, dynamic), 2 * n)
            tracker = final.work_energy_tracker
            values = tracker_values(tracker)
            initial = tracker.baseline.electromagnetic
            loss = initial - tracker.latest.electromagnetic
            self.assertGreater(float(loss / initial), .5)
            self.assertGreater(float(tracker.boundary_work / initial), .5)
            self.assertEqual(float(tracker.curvature_work + tracker.lapse_work), 0.)
            errors.append(float(values["relative_residual"]))

            def exact_energy(time):
                center = .3 + 2.5 * time
                limits = jnp.sqrt(2.) * (jnp.asarray([-2., 2.]) - center) / .45
                return .45 * jnp.sqrt(jnp.pi / 2.) * (erf(limits[1]) - erf(limits[0])) / (8 * jnp.pi)

            exact_initial = exact_energy(0.)
            exact_final = exact_energy(tracker.latest.time)
            energy_errors.append(float(jnp.abs(tracker.latest.electromagnetic - exact_final) / exact_initial))
            transport_errors.append(float(jnp.abs(tracker.boundary_work - exact_initial + exact_final) / exact_initial))
            x = dynamic.grids.center[0][1:-1]
            xb = dynamic.grids.vertex[0][1:-1]
            exact_D = jnp.exp(-((x - .3 - 2.5 * tracker.latest.time) / .45)**2)
            exact_B = jnp.exp(-((xb - .3 - 2.5 * tracker.latest.time) / .45)**2)
            numerical_D = tiles_to_bssn(final_fields[0][1], static)[:, 0, 0]
            numerical_B = tiles_to_bssn(final_fields[7][2][2], static)[:, 0, 0]
            error = dynamic.dx * jnp.sum((numerical_D - exact_D)**2 + (numerical_B - exact_B)**2)
            field_errors.append(float(jnp.sqrt(error / (8 * jnp.pi * exact_initial))))
        orders = jnp.log2(jnp.asarray(errors[:-1]) / jnp.asarray(errors[1:]))
        evidence = (f"residual={errors}, orders={orders}; energy={energy_errors}; "
                    f"transport={transport_errors}; field_L2={field_errors}")
        # The constant exterior produces a boundary truncation layer: the
        # native-field error also converges slower than second order here.
        # Preserve that solver behavior and check independent solution errors
        # alongside the tracker, rather than demanding an RK4 energy rate.
        field_orders = jnp.log2(jnp.asarray(field_errors[:-1]) / jnp.asarray(field_errors[1:]))
        energy_orders = jnp.log2(jnp.asarray(energy_errors[:-1]) / jnp.asarray(energy_errors[1:]))
        self.assertTrue(bool(jnp.all(orders > .8)), evidence)
        self.assertTrue(bool(jnp.all(field_orders > .8)), evidence)
        self.assertTrue(bool(jnp.all(energy_orders > 1.5)), evidence)
        self.assertTrue(bool(jnp.all(jnp.diff(jnp.asarray(transport_errors)) < 0.)), evidence)
        self.assertLess(errors[-1], .002, evidence)

    def test_initial_baseline_uses_prepared_geometry(self):
        static, dynamic, particles, species, fields = coupled_state()
        static = static._replace(gr_work_energy_theorem=True)
        dynamic = dynamic._replace(work_energy_tracker=initialize_tracker(
            particles, species, fields, static, dynamic))
        initial_energy = dynamic.work_energy_tracker.baseline.electromagnetic

        def boundary(state, time):
            return state._replace(conformal_factor=jnp.full_like(state.conformal_factor, 1.2))

        @jax.jit
        def advance(particles, fields, dynamic):
            return time_loop_dynamical_relativity(
                particles, species, fields, static, dynamic, gravity_boundary=boundary)

        _, _, final = advance(particles, fields, dynamic)
        baseline = final.work_energy_tracker.baseline
        # At fixed native densities, EM rho scales as W^4 and volume as
        # W^-3. A prepared W=1.2 therefore changes E0 by exactly 1.2.
        self.assertAlmostEqual(float(baseline.electromagnetic / initial_energy), 1.2, places=12)
        self.assertEqual(float(baseline.time), 0.)
        self.assertLess(abs(float(tracker_values(final.work_energy_tracker)["residual"])), 2.e-13)


if __name__ == "__main__":
    unittest.main()
