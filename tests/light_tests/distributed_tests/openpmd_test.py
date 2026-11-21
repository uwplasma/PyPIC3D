"""Focused cross-device regression tests."""

from tests.support.openpmd_fixtures import (
    SimpleNamespace,
    jax,
    particle_parameters_from_values,
)
from tests.support.openpmd_fixtures import (
    FakeIteration,
    FakeSeries,
    OpenPMDDiagnosticsTestsFixtures,
    SpeciesConfig,
    TiledParticles,
    _parameter_values,
    _zero_field,
    async_writer,
    build_yee_metric,
    field_tiles_from_global,
    jnp,
    kernel_parameters,
    kernel_parameters_from_values,
    np,
    openPMD,
    particle_lorentz_factor,
    patch,
    unittest,
    vector_tiles_from_global,
)


class OpenPMDDiagnosticsTests(OpenPMDDiagnosticsTestsFixtures, unittest.TestCase):


    def test_static_metric_snapshot_writes_covariant_gamma_momentum_and_stored_position(self):
        static, dynamic = kernel_parameters(
            Nx=8, Ny=8, Nz=1, x_min=1., y_min=.5, z_min=0.,
            x_wind=1., y_wind=1., z_wind=1., dt=.005,
            solver='static_metric', particle_pusher='hybrid_boris_geodesic',
            metric='numerical', tile_shape=(4, 8, 1),
        )
        gamma_metric = jnp.array([[2., .2, .1], [.2, 3., -.1], [.1, -.1, 1.5]])
        # a uniform metric is reproduced exactly by the Hermite interpolant

        def provider(position):
            return 1., jnp.zeros(3), gamma_metric, jnp.linalg.inv(gamma_metric), jnp.sqrt(jnp.linalg.det(gamma_metric))

        metric = build_yee_metric(dynamic, provider)
        # one live particle in each of the two x tiles, plus an inactive slot
        x = jnp.array([[[1.2, .8, 0.], [1.3, .9, 0.]], [[1.7, 1.1, 0.], [0., 0., 0.]]]).reshape(2, 1, 1, 1, 2, 3)
        u = jnp.array([[[.6, .2, -.1], [0., 0., 0.]], [[-2., 1., 3.], [5., 5., 5.]]]).reshape(2, 1, 1, 1, 2, 3)
        active = jnp.array([[True, False], [True, False]]).reshape(2, 1, 1, 1, 2)
        particles = TiledParticles(x, u, active)
        species = SpeciesConfig(jnp.array([-1.]), jnp.array([2.]), jnp.array([3.]), jnp.ones((1, 3), bool))

        gamma = particle_lorentz_factor(particles, metric, static, dynamic)
        snapshot = async_writer.make_tiled_particle_snapshot(
            particles, step=0, time=0., species_names=('electrons',), species_config=species, gamma=gamma)
        iteration = FakeIteration()
        openPMD.write_tiled_particle_snapshot_to_iteration(iteration, snapshot, static, dynamic)

        group = iteration.particles['electrons']
        x_live = np.array([[1.2, .8, 0.], [1.7, 1.1, 0.]])
        u_live = np.array([[.6, .2, -.1], [-2., 1., 3.]])
        expected_gamma = np.sqrt(1. + np.einsum('ni,ij,nj->n', u_live, np.linalg.inv(gamma_metric), u_live))
        written_gamma = np.concatenate([data for _, _, data in group['gamma'].chunks])
        np.testing.assert_allclose(written_gamma, expected_gamma, rtol=1e-6)
        self.assertTrue(np.all(written_gamma > 1.))
        for axis, component in enumerate(('x', 'y', 'z')):
            momentum = np.concatenate([data for _, _, data in group['momentum'][component].chunks])
            position = np.concatenate([data for _, _, data in group['position'][component].chunks])
            np.testing.assert_allclose(momentum, 2. * u_live[:, axis], rtol=1e-6)
            np.testing.assert_allclose(position, x_live[:, axis], rtol=1e-6)


    def test_host_particle_snapshot_preserves_chunks_and_empty_species_without_jax_arrays(self):
        static, dynamic = particle_parameters_from_values(
            _parameter_values(), dynamic_values={'C': 10.})
        static = static._replace(tile_shape=(2, 2, 1))
        x = np.zeros((2, 1, 1, 2, 2, 3), dtype=np.float64)
        x[0, 0, 0, 0, 0, 0] = -1.5
        x[1, 0, 0, 0, 1, 0] = .5 + 2.**-40
        active = np.zeros(x.shape[:-1], dtype=bool)
        active[0, 0, 0, 0, 0] = active[1, 0, 0, 0, 1] = True
        index = (slice(None),)*6
        snapshot = SimpleNamespace(
            species_names=('live particles', 'empty species'),
            species_charge=np.array([-1., 1.]), species_mass=np.array([2., 3.]),
            species_weight=np.array([4., 5.]), x_shards=[(index, x)],
            u_shards=[(index, np.zeros_like(x))], active_shards=[(index[:-1], active)])
        iteration = FakeIteration()
        with jax.enable_x64(False), \
             patch.object(jnp, 'asarray', side_effect=AssertionError('host snapshot sent to JAX')), \
             patch.object(jax, 'device_put', side_effect=AssertionError('unexpected device transfer')):
            openPMD.write_tiled_particle_snapshot_to_iteration(iteration, snapshot, static, dynamic)
        record = iteration.particles['live_particles']['position']['x']
        self.assertEqual([(offset, extent) for offset, extent, _ in record.chunks],
                         [((0,), (1,)), ((1,), (1,))])
        actual = np.concatenate([data for _, _, data in record.chunks])
        np.testing.assert_array_equal(actual, [-1.5, .5 + 2.**-40])
        self.assertEqual(actual.dtype, np.dtype('float64'))
        empty = iteration.particles['empty_species']['position']['x']
        self.assertEqual(empty.dataset_shape, (0,))
        self.assertEqual(empty.chunks, [])
        self.assertEqual(record.unit_SI, 1.)


if __name__ == "__main__":
    unittest.main()
