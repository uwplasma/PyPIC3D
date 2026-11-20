from tests.support.gr_work_energy_theorem_runtime_fixtures import (
    EnergySample,
    SpeciesConfig,
    TiledParticles,
    WorkEnergyAssertions,
    build_bssn_parameters,
    build_static_parameters,
    dynamic_parameters_for_output,
    initialize_tracker,
    jnp,
    native_runtime,
    parameters,
    sample_energy,
    start_tracker,
    toml,
    unittest,
    update_parameters_from_toml,
)


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


@unittest.skipIf(build_bssn_parameters is None, "requires the optional JAX BSSN dependency")
class WorkEnergySamplingTests(WorkEnergyAssertions):
    def test_synchronized_magnetic_history_is_used(self):
        static, dynamic = parameters()
        particles, species, fields, static, dynamic = native_runtime(static, dynamic)
        first = sample_energy(particles, species, fields, static, dynamic)
        # The stored half-step B deliberately disagrees with the retained B(t).
        changed = fields[:1] + (tuple(100*value for value in fields[1]),) + fields[2:]
        second = sample_energy(particles, species, changed, static, dynamic)
        self.assertClose(second.electromagnetic, first.electromagnetic)

    def test_only_active_frozen_species_invalidates_the_total_budget(self):
        static, dynamic = parameters()
        _, _, fields, static, dynamic = native_runtime(static, dynamic)
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

    def test_empty_energy_requires_positive_explicit_normalization(self):
        static, dynamic = parameters()
        particles, species, fields, static, dynamic = native_runtime(static, dynamic)
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
