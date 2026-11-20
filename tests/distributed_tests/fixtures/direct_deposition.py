from tests.support.direct_deposition_fixtures import (
    DirectDepositionFixtures as SharedFixtures,
    TiledParticles,
    jnp,
    np,
)


class DirectDepositionFixtures(SharedFixtures):
    def _one_tile_particles_from_tiled(self, particles):
        n_species = particles.active.shape[3]
        active = np.asarray(particles.active).transpose(3, 0, 1, 2, 4).reshape(n_species, -1)
        x = np.asarray(particles.x).transpose(3, 0, 1, 2, 4, 5).reshape(n_species, -1, 3)
        u = np.asarray(particles.u).transpose(3, 0, 1, 2, 4, 5).reshape(n_species, -1, 3)
        # Reuse a small capacity across these fixtures instead of compiling a
        # different reference kernel for each tile topology's inactive padding.
        capacity = max(8, int(active.sum(axis=1).max()))
        flat_x = np.zeros((1, 1, 1, n_species, capacity, 3), dtype=x.dtype)
        flat_u = np.zeros_like(flat_x)
        flat_active = np.zeros(flat_x.shape[:-1], dtype=bool)
        for species in range(n_species):
            count = int(active[species].sum())
            flat_x[0, 0, 0, species, :count] = x[species, active[species]]
            flat_u[0, 0, 0, species, :count] = u[species, active[species]]
            flat_active[0, 0, 0, species, :count] = True
        return TiledParticles(jnp.asarray(flat_x), jnp.asarray(flat_u), jnp.asarray(flat_active))

    def _compare_tiled_to_one_tile(self, particles, species_config, parameter_set, simulation_parameters, filter="none", alpha=1.0):
        dynamic_values = {"C": 3.0e8, "alpha": alpha}
        J_tiles, J_from_tiles = self._assembled_tiled_current(
            particles, species_config, parameter_set, simulation_parameters, dynamic_values, filter=filter
        )
        _, J_reference = self._assembled_tiled_current(
            self._one_tile_particles_from_tiled(particles),
            species_config,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            dynamic_values,
            filter=filter,
        )

        for tile_component in J_tiles:
            self.assertEqual(tile_component.ndim, 6)
            # ensure that the tiled current density components have 6 dimensions (tile_x, tile_y, tile_z, tile_nx, tile_ny, tile_nz)
        for reference_component, tiled_component in zip(J_reference, J_from_tiles):
            self.assertTrue(jnp.allclose(tiled_component, reference_component, rtol=5.0e-15, atol=5.0e-15))
