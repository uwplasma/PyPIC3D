from tests.support.pusher_fixtures import (
    TiledParticlePusherFixtures as SharedFixtures,
    build_tiled_particles,
    jnp,
    kernel_parameters_from_values,
    particle_parameters_from_tile_values,
    particle_push,
    tile_vector_field,
)


class TiledParticlePusherFixtures(SharedFixtures):
    def _copy_parameters_for_tile_shape(self, parameter_set, tile_shape, g):
        tiled_parameters = dict(parameter_set)
        tiled_parameters["grids"] = dict(parameter_set["grids"])
        return self._with_tiled_grids(tiled_parameters, tile_shape, g=g)

    def _simulation_parameters_for_tile_shape(self, tile_shape):
        return {
            "particle_tile_nx": tile_shape[0],
            "particle_tile_ny": tile_shape[1],
            "particle_tile_nz": tile_shape[2],
        }

    def _push_tiled_species(self, species, parameter_set, tile_shape, E, B, dynamic_values, relativistic=True, particle_pusher="boris"):
        particle_static, particle_dynamic = particle_parameters_from_tile_values(
            parameter_set,
            self._simulation_parameters_for_tile_shape(tile_shape),
            dynamic_values=dynamic_values,
        )
        tiled_particles, species_config = build_tiled_particles(
            [species],
            particle_static,
            particle_dynamic,
        )
        g = int(parameter_set["guard_cells"])
        static_parameters, dynamic_parameters = kernel_parameters_from_values(parameter_set, dynamic_values)
        static_parameters = static_parameters._replace(
            relativistic=bool(relativistic),
            particle_pusher=particle_pusher,
        )
        return particle_push(
            tiled_particles,
            species_config,
            tile_vector_field(E, parameter_set, tile_shape, num_guard_cells=g),
            tile_vector_field(B, parameter_set, tile_shape, num_guard_cells=g),
            static_parameters,
            dynamic_parameters,
        )

    def _flatten_active_by_position(self, tiled_particles):
        active = tiled_particles.active.reshape(-1)
        x = tiled_particles.x.reshape(-1, 3)[active]
        u = tiled_particles.u.reshape(-1, 3)[active]
        order = jnp.lexsort((x[:, 2], x[:, 1], x[:, 0]))
        return x[order], u[order]
