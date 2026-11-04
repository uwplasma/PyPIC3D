"""Continuum GR energy budgets on synchronized native Maxwell/PIC states."""

import unittest

import jax
import jax.numpy as jnp
import toml

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_CONSTANT, BC_CONDUCTING
from PyPIC3D.diagnostics.gr_work_energy_theorem import (
    EnergySample,
    advance_tracker,
    tracker_values,
    initialize_tracker,
    start_tracker,
    sample_energy,
    surface_integral,
    volume_integral,
)
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.utilities.parameters import build_static_parameters, dynamic_parameters_for_output
from PyPIC3D.utilities.toml_helpers import update_parameters_from_toml
from tests.kernel_fixtures import kernel_parameters

try:
    from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
    from PyPIC3D.relativity.dynamical_relativity import (
        build_bssn_metric,
        initialize_dynamical_relativity,
        tiles_to_bssn,
    )
except ModuleNotFoundError as error:
    if error.name != "JAX_BSSN":
        raise
    build_bssn_parameters = None


jax.config.update("jax_enable_x64", True)


def parameters(**kwargs):
    values = dict(
        Nx=8, Ny=1, Nz=1, x_wind=8., y_wind=1., z_wind=1.,
        x_min=0., y_min=0., z_min=0., tile_shape=(8, 1, 1),
        guard_cells=3, solver="dynamic_metric", metric="numerical",
        particle_pusher="hybrid_boris_geodesic", current_deposition="esirkepov",
        current_filter="none", particle_batch_size=4,
    )
    values.update(kwargs)
    return kernel_parameters(**values)


def native_runtime(static, dynamic, *, gamma=None, curvature=None):
    """Initialize genuine native histories from common-time uniform EM data."""
    bssn = build_bssn_parameters(static, dynamic, gravity_G=0.)
    state = bssn.state
    if gamma is not None:
        W = jnp.linalg.det(gamma) ** (-1. / 6.)
        conformal = gamma * W**2
        state = state._replace(
            conformal_factor=jnp.full_like(state.conformal_factor, W),
            conformal_metric=conformal[:, :, None, None, None]
            * jnp.ones_like(state.conformal_factor),
        )
    if curvature is not None:
        trace = jnp.einsum("ij,ij->", jnp.linalg.inv(gamma), curvature)
        traceless = W**2 * (curvature - gamma * trace / 3.)
        state = state._replace(
            trace_K=jnp.full_like(state.trace_K, trace),
            traceless_K=traceless[:, :, None, None, None]
            * jnp.ones_like(state.conformal_factor),
        )
    bssn = bssn._replace(state=state, evolution_parameters=bssn.evolution_parameters._replace(nu=0.))
    dynamic = dynamic._replace(bssn_parameters=bssn)
    metric = build_bssn_metric(state, static, dynamic)
    electric = tuple(jnp.full_like(metric.D[i].lapse, (.5, -.3, .2)[i]) for i in range(3))
    magnetic = tuple(jnp.full_like(metric.B[i].lapse, (.1, .4, -.2)[i]) for i in range(3))
    zero = tuple(jnp.zeros_like(value) for value in electric)
    counts = metric.center.lapse.shape[:3]
    x = jnp.zeros(counts + (0, 0, 3))
    particles = TiledParticles(x, x, jnp.zeros(x.shape[:-1], dtype=bool))
    vector = jnp.zeros((0,))
    species = SpeciesConfig(vector, vector, vector, jnp.zeros((0, 3), dtype=bool))
    fields = (electric, magnetic, zero, zero[0], zero[0], (zero, zero), None,
              (electric, magnetic), jnp.asarray(False))
    particles, fields, dynamic = initialize_dynamical_relativity(
        particles, species, fields, static, dynamic)
    return particles, species, fields, dynamic


class WorkEnergyAssertions(unittest.TestCase):
    def assertClose(self, actual, expected, *, rtol=2.e-12, atol=2.e-12):
        self.assertTrue(bool(jnp.allclose(actual, expected, rtol=rtol, atol=atol)),
                        f"actual={actual}, expected={expected}")


class WorkEnergyArithmeticTests(WorkEnergyAssertions):
    def test_toml_configuration_preserves_opt_in_and_reference(self):
        from PyPIC3D.initialization import default_parameters

        template, _ = parameters()
        for section in ("simulation_parameters", "static_parameters"):
            for enabled in (False, True):
                with self.subTest(section=section, enabled=enabled):
                    plotting, static, dynamic = default_parameters()
                    self.assertFalse(static["gr_work_energy_theorem"])
                    self.assertIsNone(static["gr_work_energy_reference"])
                    static.update(template._asdict())
                    config = toml.loads(
                        f'[{section}]\nsolver = "dynamic_metric"\n'
                        f'gr_work_energy_theorem = {str(enabled).lower()}\n'
                        'gr_work_energy_reference = 2.5\n')
                    static, _, _ = update_parameters_from_toml(config, static, dynamic, plotting)
                    runtime = build_static_parameters(static)
                    self.assertEqual(runtime.gr_work_energy_theorem, enabled)
                    self.assertEqual(runtime.gr_work_energy_reference, 2.5)

    def test_parameter_output_excludes_runtime_budget_carry(self):
        _, dynamic = parameters()
        initial = EnergySample(*map(jnp.asarray, (0., 5., 0., 0., 0., 0., 0.)))
        dynamic = dynamic._replace(work_energy_tracker=start_tracker(initial))
        output = dynamic_parameters_for_output(dynamic)
        self.assertNotIn("work_energy_tracker", output)
        self.assertNotIn("grids", output)
        self.assertClose(output["dt"], dynamic.dt)
        # The host configuration remains a plain TOML record, not a checkpoint.
        restored = toml.loads(toml.dumps({"dynamic_parameters": output}))["dynamic_parameters"]
        self.assertEqual(restored, output)

    def test_linear_power_trapezoid_and_signed_step_residual(self):
        initial = EnergySample(*map(jnp.asarray, (0., 10., 2., 1., -.5, .2, 0.)))
        tracker = start_tracker(initial, reference=3.)
        # Integrated geometric work is 3 - 0.5; outward transport is 0.6.
        final = EnergySample(*map(jnp.asarray, (2., 12.4, 2., 2., 0., .4, 0.)))
        tracker = jax.jit(advance_tracker)(tracker, final)
        self.assertClose(tracker.curvature_work, 3.)
        self.assertClose(tracker.lapse_work, -.5)
        self.assertClose(tracker.boundary_work, .6)
        self.assertClose(tracker.step_residual, .5)
        self.assertClose(tracker.reference, 3.)
        self.assertEqual(int(tracker.steps), 1)

        loss = final._replace(time=jnp.asarray(3.), electromagnetic=jnp.asarray(10.))
        tracker = advance_tracker(tracker, loss)
        self.assertLess(float(tracker.step_residual), 0.)

    def test_full_domain_volume_and_face_quadrature_are_exact_for_linear_data(self):
        static, dynamic = parameters(
            Nx=8, Ny=4, Nz=1, x_wind=4., y_wind=2., z_wind=.7,
            x_min=.2, y_min=-.3, tile_shape=(8, 4, 1),
            boundary_conditions=(BC_CONSTANT, BC_CONDUCTING, 0),
        )
        x = .2 + jnp.arange(8)[:, None, None] * .5
        y = -.3 + jnp.arange(5)[None, :, None] * .5
        one = jnp.ones((8, 5, 1))
        integrand = one + 2*x + 3*y
        volume = 4 * 2 * .7
        self.assertClose(volume_integral(integrand, static, dynamic),
                         volume * (1 + 2*(.2 + 2.) + 3*(-.3 + 1.)))
        flux = jnp.stack((2*x*one, -.4*y*one, jnp.zeros_like(one)))
        self.assertClose(surface_integral(flux, static, dynamic), 1.6 * volume)

    def test_periodic_face_pairs_cancel(self):
        static, dynamic = parameters()
        flux = jnp.arange(24.).reshape(3, 8, 1, 1)
        self.assertClose(surface_integral(flux, static, dynamic), 0.)

    def test_absorbing_boundary_work_does_not_erase_continuum_residual(self):
        initial = EnergySample(*map(jnp.asarray, (0., 5., 0., 0., 0., 0., -2.)))
        final = initial._replace(time=jnp.asarray(1.), electromagnetic=jnp.asarray(3.))
        tracker = advance_tracker(start_tracker(initial), final)
        self.assertClose(tracker.absorbing_boundary_work, -2.)
        self.assertClose(tracker.step_residual, -2.)
        self.assertClose(tracker.reference, 5.)
        values = tracker_values(tracker)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(value))) for value in values.values()))


@unittest.skipIf(build_bssn_parameters is None, "requires the optional JAX BSSN dependency")
class WorkEnergySamplingTests(WorkEnergyAssertions):
    def test_multitile_ownership_excludes_poisoned_ghosts_and_counts_pec_endpoints_once(self):
        origins, lengths = (.2, -.3, .4), (4., 3., 1.4)
        counts, widths = (2, 2, 2), (4, 3, 2)
        for boundaries in ((0, 0, 0), (1, 1, 1), (BC_CONSTANT, 1, 0), (1, 0, BC_CONSTANT)):
            with self.subTest(boundaries=boundaries):
                static, dynamic = parameters(
                    Nx=8, Ny=6, Nz=4, x_wind=lengths[0], y_wind=lengths[1], z_wind=lengths[2],
                    x_min=origins[0], y_min=origins[1], z_min=origins[2],
                    tile_shape=(8, 6, 4), boundary_conditions=boundaries)
                # Synthetic tiles test ownership independently of mesh/device
                # allocation. Internal duplicate endpoints remain poisoned.
                static = static._replace(tile_shape=widths, field_mesh=None)
                g = static.guard_cells
                spacing = (dynamic.dx, dynamic.dy, dynamic.dz)
                slopes = tuple(s if bc else 0. for s, bc in zip((.3, -.2, .5), boundaries))
                tiled = jnp.full((2,) + counts + tuple(width + 2*g for width in widths), jnp.nan)
                for tx in range(2):
                    for ty in range(2):
                        for tz in range(2):
                            tile = (tx, ty, tz)
                            owned = tuple(width + (bc == BC_CONDUCTING and t == count-1)
                                          for t, count, width, bc in zip(tile, counts, widths, boundaries))
                            x, y, z = tuple(origin + dx*(t*width + jnp.arange(n))
                                            for origin, dx, t, width, n in zip(origins, spacing, tile, widths, owned))
                            linear = (4. + slopes[0]*x[:, None, None] + slopes[1]*y[None, :, None]
                                      + slopes[2]*z[None, None, :])
                            data = jnp.stack((jnp.ones_like(linear), linear))
                            interior = (slice(None),) + tile + tuple(slice(g, g+n) for n in owned)
                            tiled = tiled.at[interior].set(data)

                gathered = jax.jit(lambda values: tiles_to_bssn(values, static))(tiled)
                shape = tuple(count*width + (bc == BC_CONDUCTING)
                              for count, width, bc in zip(counts, widths, boundaries))
                x, y, z = tuple(origin + dx*jnp.arange(n)
                                for origin, dx, n in zip(origins, spacing, shape))
                expected = (4. + slopes[0]*x[:, None, None] + slopes[1]*y[None, :, None]
                            + slopes[2]*z[None, None, :])
                self.assertEqual(gathered.shape, (2,) + shape)
                self.assertClose(gathered[0], jnp.ones(shape))
                self.assertClose(gathered[1], expected)
                volume = lengths[0]*lengths[1]*lengths[2]
                self.assertClose(volume_integral(gathered[0], static, dynamic), volume)
                self.assertClose(volume_integral(gathered[1], static, dynamic),
                                 volume*(4. + sum(s*(origin + length/2)
                                                 for s, origin, length in zip(slopes, origins, lengths))))

    def test_offdiagonal_metric_and_curvature_contractions(self):
        static, dynamic = parameters()
        gamma = jnp.asarray(((1.4, .2, .1), (.2, .9, -.12), (.1, -.12, 1.2)))
        curvature = jnp.asarray(((.03, -.02, .01), (-.02, -.04, .015), (.01, .015, .02)))
        particles, species, fields, dynamic = native_runtime(
            static, dynamic, gamma=gamma, curvature=curvature)
        sample = jax.jit(lambda p, f, d: sample_energy(p, species, f, static, d))(
            particles, fields, dynamic)
        electric, magnetic = jnp.asarray((.5, -.3, .2)), jnp.asarray((.1, .4, -.2))
        e_cov, b_cov = gamma @ electric, gamma @ magnetic
        rho = (electric @ e_cov + magnetic @ b_cov) / (8*jnp.pi)
        stress = rho * gamma - (jnp.outer(e_cov, e_cov) + jnp.outer(b_cov, b_cov)) / (4*jnp.pi)
        inverse = jnp.linalg.inv(gamma)
        proper_volume = 8 * jnp.sqrt(jnp.linalg.det(gamma))
        self.assertClose(sample.electromagnetic, rho * proper_volume)
        self.assertClose(sample.curvature_power,
                         jnp.sum(curvature * (inverse @ stress @ inverse)) * proper_volume)
        self.assertClose(sample.lapse_power, 0.)
        self.assertClose(sample.boundary_power, 0.)

    def test_lapse_power_uses_raised_momentum_and_negative_sign(self):
        static, dynamic = parameters(Nx=16, x_wind=8., tile_shape=(16, 1, 1))
        gamma = jnp.asarray(((1.4, .2, .1), (.2, .9, -.12), (.1, -.12, 1.2)))
        particles, species, fields, dynamic = native_runtime(static, dynamic, gamma=gamma)
        wave_number = 2*jnp.pi / 8.
        x = jnp.arange(16)[:, None, None] * dynamic.dx
        bssn = dynamic.bssn_parameters
        state = bssn.state._replace(lapse=1 + .2*jnp.sin(wave_number*x))
        dynamic = dynamic._replace(bssn_parameters=bssn._replace(state=state))
        metric = build_bssn_metric(state, static, dynamic)
        tiled_x = dynamic.grids.tiled_center_grid[0][..., :, None, None]
        zero = tuple(jnp.zeros_like(value) for value in fields[0])
        electric = (zero[0], .7*jnp.cos(wave_number*tiled_x)*metric.D[1].sqrt_gamma, zero[2])
        magnetic = (zero[0], zero[1], .3*metric.B[2].sqrt_gamma)
        history = (electric, magnetic, magnetic, fields[7][3], None)
        fields = (electric, magnetic, zero, fields[3], fields[4], fields[5], metric, history, fields[8])
        sample = sample_energy(particles, species, fields, static, dynamic)
        theta = wave_number * dynamic.dx
        discrete_k = (8*jnp.sin(theta) - jnp.sin(2*theta)) / (6*dynamic.dx)
        expected = (-8 * jnp.linalg.det(gamma) * jnp.linalg.inv(gamma)[0, 0]
                    * .7 * .3 * .2 * discrete_k / (8*jnp.pi))
        self.assertClose(sample.lapse_power, expected)
        self.assertLess(float(sample.lapse_power), 0.)
        self.assertClose(sample.curvature_power, 0.)

    def test_synchronized_magnetic_history_is_used(self):
        static, dynamic = parameters()
        particles, species, fields, dynamic = native_runtime(static, dynamic)
        first = sample_energy(particles, species, fields, static, dynamic)
        # The stored half-step B deliberately disagrees with the retained B(t).
        changed = fields[:1] + (tuple(100*value for value in fields[1]),) + fields[2:]
        second = sample_energy(particles, species, changed, static, dynamic)
        self.assertClose(second.electromagnetic, first.electromagnetic)

    def test_native_absorber_reports_signed_damping_and_reference_power(self):
        static, dynamic = parameters()
        particles, species, fields, dynamic = native_runtime(static, dynamic)
        initial = sample_energy(particles, species, fields, static, dynamic)
        sigma = .12
        for reference_factor in (0., 2.):
            with self.subTest(reference_factor=reference_factor):
                absorber = tuple((tuple(jnp.full_like(value, sigma) for value in vector),
                                  tuple(reference_factor*value for value in vector),
                                  tuple(jnp.zeros_like(value, dtype=bool) for value in vector))
                                 for vector in (fields[0], fields[7][2]))
                history = fields[7][:4] + (absorber,)
                damped = fields[:7] + (history, fields[8])
                sample = sample_energy(particles, species, damped, static, dynamic)
                self.assertClose(sample.absorbing_boundary_power,
                                 2*sigma*(reference_factor-1)*initial.electromagnetic)

    def test_supergaussian_power_uses_native_damping_profile(self):
        static, dynamic = parameters(
            supergaussian_active=True, supergaussian_layers=((0, -1, 8, 2., .12),))
        particles, species, fields, dynamic = native_runtime(static, dynamic)
        sample = sample_energy(particles, species, fields, static, dynamic)
        mean_sigma = jnp.mean(.12*((8-jnp.arange(8))/8)**2)
        self.assertClose(sample.absorbing_boundary_power, -2*mean_sigma*sample.electromagnetic)

    def test_particle_energy_includes_rest_mass_weight_and_offdiagonal_metric(self):
        static, dynamic = parameters()
        gamma = jnp.asarray(((1.4, .2, .1), (.2, .9, -.12), (.1, -.12, 1.2)))
        _, _, fields, dynamic = native_runtime(static, dynamic, gamma=gamma)
        x = jnp.asarray((1.25, 0., 0.)).reshape(1, 1, 1, 1, 1, 3)
        u = jnp.asarray((.3, -.2, .1)).reshape(x.shape)
        particles = TiledParticles(x, u, jnp.ones(x.shape[:-1], dtype=bool))
        species = SpeciesConfig(jnp.asarray((0.,)), jnp.asarray((2.,)), jnp.asarray((.3,)),
                                jnp.ones((1, 3), dtype=bool))
        sample = sample_energy(particles, species, fields, static, dynamic)
        momentum = u.reshape(3)
        gamma_particle = jnp.sqrt(1 + momentum @ jnp.linalg.inv(gamma) @ momentum)
        self.assertClose(sample.particles, .6*gamma_particle)
        self.assertGreater(float(sample.particles), .6)

    def test_only_active_frozen_species_invalidates_the_total_budget(self):
        static, dynamic = parameters()
        _, _, fields, dynamic = native_runtime(static, dynamic)
        static = static._replace(gr_work_energy_theorem=True)
        x = jnp.asarray((1.25, 0., 0.)).reshape(1, 1, 1, 1, 1, 3)
        particles = TiledParticles(x, jnp.zeros_like(x), jnp.ones(x.shape[:-1], dtype=bool))
        species = SpeciesConfig(jnp.asarray((0.,)), jnp.asarray((2.,)), jnp.asarray((.3,)),
                                jnp.zeros((1, 3), dtype=bool))
        with self.assertRaisesRegex(ValueError, "frozen particle components"):
            initialize_tracker(particles, species, fields, static, dynamic)
        inactive = particles._replace(active=jnp.zeros_like(particles.active))
        tracker = initialize_tracker(inactive, species, fields, static, dynamic)
        self.assertClose(tracker.baseline.particles, 0.)

    def test_prescribed_flrw_volume_work_and_second_order_budget(self):
        static, dynamic = parameters()
        particles, species, fields, dynamic = native_runtime(static, dynamic)
        initial_state = dynamic.bssn_parameters.state
        hubble = .3

        def at_time(time):
            a = jnp.exp(hubble*time)
            state = initial_state._replace(
                conformal_factor=jnp.ones_like(initial_state.conformal_factor)/a,
                trace_K=jnp.full_like(initial_state.trace_K, -3*hubble),
            )
            parameters_at_time = dynamic._replace(bssn_parameters=dynamic.bssn_parameters._replace(
                state=state, time=time))
            metric = build_bssn_metric(state, static, parameters_at_time)
            current_fields = fields[:6] + (metric,) + fields[7:]
            return sample_energy(particles, species, current_fields, static, parameters_at_time)

        initial = at_time(jnp.asarray(0.))
        last = at_time(jnp.asarray(1.))
        self.assertClose(last.electromagnetic, initial.electromagnetic*jnp.exp(-hubble))
        self.assertClose(last.curvature_power, -hubble*last.electromagnetic)
        self.assertClose(last.lapse_power, 0.)
        self.assertClose(last.boundary_power, 0.)

        def integrate(times):
            def accumulate(tracker, time):
                tracker = advance_tracker(tracker, at_time(time))
                return tracker, tracker.step_residual
            return jax.lax.scan(accumulate, start_tracker(initial), times)[0]

        residuals = []
        for steps in (8, 16, 32):
            tracker = jax.jit(integrate)(jnp.arange(1, steps+1, dtype=jnp.float64)/steps)
            residual = (tracker.latest.electromagnetic - tracker.baseline.electromagnetic
                        - tracker.curvature_work)
            self.assertGreater(float(residual), 0.)
            residuals.append(residual)
        self.assertClose(residuals[0]/residuals[1], 4., rtol=2.e-4)
        self.assertClose(residuals[1]/residuals[2], 4., rtol=2.e-4)

    def test_uniform_native_energy_excludes_all_halo_widths(self):
        samples = []
        for guard in (3, 5):
            static, dynamic = parameters(Nx=16, x_wind=16., tile_shape=(16, 1, 1), guard_cells=guard)
            particles, species, fields, dynamic = native_runtime(static, dynamic)
            samples.append(sample_energy(particles, species, fields, static, dynamic))
        self.assertClose(samples[0].electromagnetic, samples[1].electromagnetic)

    def test_empty_energy_requires_positive_explicit_normalization(self):
        static, dynamic = parameters()
        particles, species, fields, dynamic = native_runtime(static, dynamic)
        zero = tuple(jnp.zeros_like(value) for value in fields[0])
        fields = (zero, zero, zero) + fields[3:7] + ((zero, zero, zero, fields[7][3], None), fields[8])
        static = static._replace(gr_work_energy_theorem=True)
        with self.assertRaises(ValueError):
            initialize_tracker(particles, species, fields, static, dynamic)
        carry = initialize_tracker(
            particles, species, fields, static._replace(gr_work_energy_reference=2.), dynamic)
        self.assertClose(carry.reference, 2.)
        for reference in (0., -1.):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                initialize_tracker(
                    particles, species, fields, static._replace(gr_work_energy_reference=reference), dynamic)


if __name__ == "__main__":
    unittest.main()
