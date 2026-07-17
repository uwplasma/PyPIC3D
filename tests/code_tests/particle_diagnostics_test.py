import unittest

import jax
import jax.numpy as jnp

from PyPIC3D.diagnostics.output_adapters import particles_for_output
from tests.initial_particles import build_tiled_particles, species_names, tiled_species


jax.config.update("jax_enable_x64", True)


class TestTiledParticleDiagnostics(unittest.TestCase):
    def _world(self):
        return {
            "Nx": 4,
            "Ny": 2,
            "Nz": 1,
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "dt": 0.2,
            "x_wind": 4.0,
            "y_wind": 2.0,
            "z_wind": 1.0,
        }

    def _simulation_parameters(self):
        return {
            "particle_tile_nx": 2,
            "particle_tile_ny": 1,
            "particle_tile_nz": 1,
        }

    def _species(self):
        world = self._world()
        ions = tiled_species(
            name="ions",
            N_particles=4,
            charge=2.0,
            mass=3.0,
            weight=4.0,
            T=1.0,
            x1=jnp.array([-1.5, -0.5, 0.5, 1.5]),
            x2=jnp.array([-0.5, -0.5, 0.5, 0.5]),
            x3=jnp.array([0.0, 0.0, 0.0, 0.0]),
            v1=jnp.array([0.1, 0.2, 0.3, 0.4]),
            v2=jnp.array([1.0, 1.1, 1.2, 1.3]),
            v3=jnp.array([2.0, 2.1, 2.2, 2.3]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            active_mask=jnp.array([True, False, True, True]),
            dt=world["dt"],
        )
        electrons = tiled_species(
            name="electrons",
            N_particles=3,
            charge=-1.0,
            mass=0.5,
            weight=8.0,
            T=1.0,
            x1=jnp.array([-1.25, 0.25, 1.25]),
            x2=jnp.array([0.25, -0.25, 0.25]),
            x3=jnp.array([0.0, 0.0, 0.0]),
            v1=jnp.array([-0.1, -0.2, -0.3]),
            v2=jnp.array([-1.0, -1.1, -1.2]),
            v3=jnp.array([-2.0, -2.1, -2.2]),
            xwind=world["x_wind"],
            ywind=world["y_wind"],
            zwind=world["z_wind"],
            dx=world["dx"],
            dy=world["dy"],
            dz=world["dz"],
            active_mask=jnp.array([False, True, True]),
            dt=world["dt"],
        )
        return [ions, electrons]

    def test_flatten_tiled_particles_matches_active_original_species(self):
        species_list = self._species()
        tiled_particles, species_config = build_tiled_particles(species_list, self._world(), self._simulation_parameters())

        flattened_species = particles_for_output(tiled_particles, species_config=species_config)

        self.assertEqual(len(flattened_species), len(species_list))
        for species_index, (original, flattened) in enumerate(zip(species_list, flattened_species)):
            active = original["active"]

            original_x = original["x"][active]
            original_u = original["u"][active]
            flattened_x = flattened.x
            flattened_u = flattened.u

            self.assertEqual(flattened.species_index, species_index)
            self.assertTrue(jnp.allclose(flattened_x, original_x))
            self.assertTrue(jnp.allclose(flattened_u, original_u))
            self.assertTrue(jnp.allclose(flattened.charge, original["charge"] * jnp.ones(jnp.sum(active))))
            self.assertTrue(jnp.allclose(flattened.mass, original["mass"] * jnp.ones(jnp.sum(active))))
            self.assertTrue(jnp.allclose(flattened.weight, original["weight"] * jnp.ones(jnp.sum(active))))

    def test_inactive_tiled_slots_are_not_flattened(self):
        species_list = self._species()
        tiled_particles, species_config = build_tiled_particles(species_list, self._world(), self._simulation_parameters())

        flattened_species = particles_for_output(tiled_particles, species_config=species_config)

        self.assertEqual(flattened_species[0].x.shape[0], 3)
        self.assertEqual(flattened_species[1].x.shape[0], 2)
        self.assertFalse(jnp.any(jnp.isclose(flattened_species[0].x[:, 0], -0.5)))
        self.assertFalse(jnp.any(jnp.isclose(flattened_species[1].x[:, 0], -1.25)))

    def test_absorbed_particles_do_not_appear_in_flattened_output(self):
        species_list = self._species()
        tiled_particles, species_config = build_tiled_particles(species_list, self._world(), self._simulation_parameters())

        tiled_particles = tiled_particles._replace(
            active=tiled_particles.active.at[1, 1, 0, 0, 1].set(False)
        )
        flattened_species = particles_for_output(tiled_particles, species_config=species_config)

        self.assertEqual(flattened_species[0].x.shape[0], 2)
        self.assertFalse(jnp.any(jnp.isclose(flattened_species[0].x[:, 0], 1.5)))

    def test_flattened_diagnostic_positions_match_original_half_step_positions(self):
        species_list = self._species()
        world = self._world()
        world["particle_boundary_conditions"] = {
            "x": 0,
            "y": 0,
            "z": 0,
        }
        tiled_particles, species_config = build_tiled_particles(species_list, world, self._simulation_parameters())
        static_parameters = {"particle_boundary_conditions": (0, 0, 0)}

        flattened_species = particles_for_output(
            tiled_particles,
            species_config=species_config,
            static_parameters=static_parameters,
            dynamic_parameters=world,
        )

        for original, flattened in zip(species_list, flattened_species):
            active = original["active"]
            original_position = original["x"][active] - 0.5 * original["u"][active] * world["dt"]
            flattened_position = flattened.x_diagnostic

            self.assertTrue(jnp.allclose(flattened_position, original_position))

    def test_species_names_are_preserved_when_metadata_is_available(self):
        species_list = self._species()
        tiled_particles, species_config = build_tiled_particles(species_list, self._world(), self._simulation_parameters())
        names = species_names(species_list)

        flattened_species = particles_for_output(tiled_particles, species_config=species_config, species_names=names)

        self.assertEqual(flattened_species[0].name, "ions")
        self.assertEqual(flattened_species[1].name, "electrons")


if __name__ == "__main__":
    unittest.main()
