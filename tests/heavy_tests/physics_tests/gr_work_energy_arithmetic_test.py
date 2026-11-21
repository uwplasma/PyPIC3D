from tests.support.gr_work_energy_theorem_runtime_fixtures import (
    BC_CONDUCTING,
    BC_CONSTANT,
    EnergySample,
    SpeciesConfig,
    TiledParticles,
    WorkEnergyAssertions,
    advance_tracker,
    build_bssn_metric,
    build_bssn_parameters,
    jax,
    jnp,
    native_runtime,
    parameters,
    sample_energy,
    start_tracker,
    surface_integral,
    tracker_values,
    unittest,
    volume_integral,
)


@unittest.skipIf(build_bssn_parameters is None, "requires the optional JAX BSSN dependency")
class WorkEnergySamplingTests(WorkEnergyAssertions):

    def test_lapse_power_uses_raised_momentum_and_negative_sign(self):
        static, dynamic = parameters(Nx=16, x_wind=8., tile_shape=(16, 1, 1))
        gamma = jnp.asarray(((1.4, .2, .1), (.2, .9, -.12), (.1, -.12, 1.2)))
        particles, species, fields, static, dynamic = native_runtime(static, dynamic, gamma=gamma)
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


    def test_supergaussian_power_uses_native_damping_profile(self):
        static, dynamic = parameters(
            supergaussian_active=True, supergaussian_layers=((0, -1, 8, 2., .12),))
        particles, species, fields, static, dynamic = native_runtime(static, dynamic)
        sample = sample_energy(particles, species, fields, static, dynamic)
        mean_sigma = jnp.mean(.12*((8-jnp.arange(8))/8)**2)
        self.assertClose(sample.absorbing_boundary_power, -2*mean_sigma*sample.electromagnetic)


if __name__ == "__main__":
    unittest.main()
