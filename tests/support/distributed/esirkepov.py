from tests.support.esirkepov_fixtures import (
    TiledEsirkepovCurrentFixtures as SharedFixtures,
    TiledParticles,
    jax,
    jnp,
)


class TiledEsirkepovCurrentFixtures(SharedFixtures):
    def _one_tile_particles_from_tiled(self, particles):
        n_species = particles.active.shape[3]

        x_by_species = []
        u_by_species = []
        max_slots = 1
        for species_index in range(n_species):
            active = jnp.asarray(jax.device_get(particles.active[:, :, :, species_index, :])).reshape(-1)
            x = jnp.asarray(jax.device_get(particles.x[:, :, :, species_index, :, :])).reshape(-1, 3)[active]
            u = jnp.asarray(jax.device_get(particles.u[:, :, :, species_index, :, :])).reshape(-1, 3)[active]
            x_by_species.append(x)
            u_by_species.append(u)
            max_slots = max(max_slots, int(x.shape[0]))

        x_one_tile = jnp.zeros((1, 1, 1, n_species, max_slots, 3), dtype=particles.x.dtype)
        u_one_tile = jnp.zeros((1, 1, 1, n_species, max_slots, 3), dtype=particles.u.dtype)
        active_one_tile = jnp.zeros((1, 1, 1, n_species, max_slots), dtype=bool)

        for species_index in range(n_species):
            n_active = int(x_by_species[species_index].shape[0])
            if n_active == 0:
                continue
            x_one_tile = x_one_tile.at[0, 0, 0, species_index, :n_active].set(jnp.asarray(x_by_species[species_index]))
            u_one_tile = u_one_tile.at[0, 0, 0, species_index, :n_active].set(jnp.asarray(u_by_species[species_index]))
            active_one_tile = active_one_tile.at[0, 0, 0, species_index, :n_active].set(True)

        return TiledParticles(
            x=x_one_tile,
            u=u_one_tile,
            active=active_one_tile,
        )

    def _tile_shape_for_parameters(self, parameter_set):
        return (
            2 if parameter_set["Nx"] > 1 else 1,
            2 if parameter_set["Ny"] > 1 else 1,
            2 if parameter_set["Nz"] > 1 else 1,
        )

    def _assert_tiled_current_matches_reference(self, parameter_set, x_old, u, tile_shape=None):
        dynamic_values = {"C": 1.0, "eps": 1.0, "alpha": 1.0}
        if tile_shape is None:
            tile_shape = self._tile_shape_for_parameters(parameter_set)

        tiled_particles, species_config = self._particles_from_arrays(parameter_set, tile_shape, x_old, u)
        J_tiles, J_from_tiles = self._assembled_esirkepov_current(
            parameter_set,
            tiled_particles,
            species_config,
            dynamic_values,
            tile_shape,
        )
        _, J_reference = self._assembled_esirkepov_current(
            parameter_set,
            self._one_tile_particles_from_tiled(tiled_particles),
            species_config,
            dynamic_values,
            self._one_tile_shape_for_parameters(parameter_set),
        )

        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(
                jnp.allclose(tiled_component, reference_component, rtol=1.0e-12, atol=1.0e-12),
                f"max diff {jnp.max(jnp.abs(tiled_component - reference_component))}",
            )
