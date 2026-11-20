from tests.support.collisions_runtime_fixtures import (
    load_collisions_from_toml,
    make_config,
    make_particles,
    run,
    unittest,
)


class TestCollisions(unittest.TestCase):
    def test_ionization_creates_pairs_in_free_slots(self):
        config = make_config()
        particles = make_particles(scale=1.0e7)
        particles = particles._replace(active=particles.active.at[0, 0, 0, 1].set(False))
        ion = (("mcc", 0, 1.0e30, 0.0, 6.6e-26, (("ionization", 15.0, (0.0, 15.0, 1e9), (0.0, 1e-19, 1e-19), 0, 1),)),)
        after, overflow = run(ion, particles, config)
        new_e = int(after.active[..., 0, :].sum() - particles.active[..., 0, :].sum())
        self.assertEqual(int(after.active[..., 1, :].sum()), int(particles.active[..., 0, :].sum()))
        self.assertEqual(new_e, 7)
        # every electron ionizes; ions fit, but only 7 free electron slots exist
        self.assertTrue(bool(overflow))

    def test_toml_loader(self):
        config = {"collisions": {"ee": {"type": "binary", "species": ["e", "e"], "coulomb_log": 12.0},
                                 "en": {"type": "background_mcc", "species": "e", "background_density": 1e20,
                                        "background_mass": 6.6e-26, "elastic": {"cross_section": 1e-19},
                                        "ionization": {"cross_section": [[15.0, 0.0], [100.0, 1e-20]],
                                                       "energy": 15.0, "product_species": ["e", "i"]}}}}
        specs = load_collisions_from_toml(config, ("e", "i"))
        self.assertEqual(specs[0], ("binary", 0, 0, 12.0))
        self.assertEqual([p[0] for p in specs[1][5]], ["elastic", "ionization"])
        self.assertEqual(specs[1][5][1][4:], (0, 1))

    def ionization_config(self, product_species=("e", "i")):
        return {"collisions": {"en": {"type": "background_mcc", "species": "e", "background_density": 1e20,
                                      "background_mass": 6.6e-26,
                                      "ionization": {"cross_section": [[15.0, 0.0], [100.0, 1e-20]],
                                                     "energy": 15.0, "product_species": list(product_species)}}}}

    def test_toml_loader_requires_ionization_products_to_share_the_projectile_weight(self):
        names = ("e", "i")
        config = self.ionization_config()
        # One event makes one real pair per real projectile, and weight is per species,
        # so unequal weights would deposit a net charge that was never there.
        self.assertEqual(load_collisions_from_toml(config, names, [1e11, 1e11])[0][5][0][4:], (0, 1))
        self.assertEqual(load_collisions_from_toml(config, names, [1e15 / 15, 2e15 / 30])[0][5][0][4:], (0, 1))
        with self.assertRaisesRegex(ValueError, "must name two species sharing the weight"):
            load_collisions_from_toml(config, names, [1e11, 5e10])
        with self.assertRaisesRegex(ValueError, "must name two species sharing the weight"):
            load_collisions_from_toml(self.ionization_config(()), names, [1e11, 1e11])
        # Processes that create nothing are unaffected by the product-species rules.
        elastic = {"collisions": {"en": {"type": "background_mcc", "species": "e", "background_density": 1e20,
                                         "background_mass": 6.6e-26, "elastic": {"cross_section": 1e-19}}}}
        self.assertEqual(load_collisions_from_toml(elastic, names, [1e11, 5e10])[0][5][0][4:], (-1, -1))


if __name__ == "__main__":
    unittest.main()
