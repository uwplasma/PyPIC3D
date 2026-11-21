"""One-tile numerical and physical checks."""


from tests.support.static_metric_fixtures import (
    ConstitutiveFieldsFixtures,
    D_FIELD_LOCATIONS,
    GRDirectDepositionFixtures,
    GR_direct_deposition,
    HybridPusherFixtures,
    J_from_rhov,
    Metric,
    ParticleMetric,
    SpeciesConfig,
    StaticMetricInitializationFixtures,
    TiledParticles,
    _constant_tiled_vector,
    _metric_locations_with_grids,
    _replace_lapse_shift,
    _single_particle_state,
    active_interior,
    compute_covariant_E,
    compute_covariant_H,
    coordinate_velocity,
    densitize_vector,
    empty_tiled_vector,
    geodesic_acceleration,
    hybrid_boris_geodesic_push,
    hybrid_pusher,
    initialize_flat_cartesian_metric,
    initialize_flat_cylindrical_metric,
    initialize_flat_spherical_metric,
    initialize_kerr_schild_spherical_metric,
    interpolate_metric,
    jax,
    jnp,
    kernel_parameters,
    location_grid,
    magnetic_boris_rotation,
    physical_vector,
    unittest,
)


class TestGRDirectDeposition(GRDirectDepositionFixtures, unittest.TestCase):


    def test_GR_direct_deposition_is_adjoint_to_staggered_field_gather(self):
        positions = jnp.asarray(
            (
                ((0.1, 0.2, 0.3), (3.9, 1.7, 2.2)),
                ((4.1, 2.8, 1.4), (7.9, 3.8, 3.7)),
            )
        ).reshape((1, 1, 1, 1, 4, 3))
        u_cov = jnp.asarray(
            (
                ((0.31, -0.17, 0.09), (-0.22, 0.28, -0.13)),
                ((0.19, 0.11, -0.24), (-0.27, -0.16, 0.21)),
            )
        ).reshape((1, 1, 1, 1, 4, 3))
        active = jnp.ones((1, 1, 1, 1, 4), dtype=bool)
        species = SpeciesConfig(
            charge=jnp.asarray([-0.7]),
            mass=jnp.asarray([1.0]),
            weight=jnp.asarray([1.3]),
            update_x=jnp.asarray([[True, True, True]]),
        )

        for shape_factor in (1, 2):
            static_parameters, dynamic_parameters = kernel_parameters(
            guard_cells=3,
                Nx=8,
                Ny=4,
                Nz=4,
                x_wind=8.0,
                y_wind=4.0,
                z_wind=4.0,
                x_min=0.0,
                y_min=0.0,
                z_min=0.0,
                dt=0.1,
                tile_shape=(8, 4, 4),
                shape_factor=shape_factor,
                current_filter="none",
                solver="static_metric",
                current_deposition="GR_direct",
                particle_pusher="hybrid_boris_geodesic",
            )
            metric = initialize_flat_cartesian_metric(static_parameters, dynamic_parameters)
            metric = _replace_lapse_shift(metric, lapse=0.73, shift=(0.12, -0.08, 0.05))
            particles = TiledParticles(x=positions, u=u_cov, active=active)

            center_grid = dynamic_parameters.grids.tiled_center_grid
            vertex_grid = dynamic_parameters.grids.tiled_vertex_grid
            D_grids = tuple(
                location_grid(center_grid, vertex_grid, location)
                for location in D_FIELD_LOCATIONS
            )
            D = []
            for component, grid in enumerate(D_grids):
                x_grid, y_grid, z_grid = grid
                field = (
                    jnp.sin(2.0 * jnp.pi * x_grid[..., :, jnp.newaxis, jnp.newaxis] / 8.0)
                    + (0.3 + 0.1 * component)
                    * jnp.cos(2.0 * jnp.pi * y_grid[..., jnp.newaxis, :, jnp.newaxis] / 4.0)
                    + (0.2 - 0.05 * component)
                    * jnp.sin(2.0 * jnp.pi * z_grid[..., jnp.newaxis, jnp.newaxis, :] / 4.0)
                )
                D.append(field)
            D = tuple(D)

            J = GR_direct_deposition(
                particles,
                species,
                empty_tiled_vector(static_parameters, dynamic_parameters),
                metric,
                static_parameters,
                dynamic_parameters,
            )

            g = int(static_parameters.guard_cells)
            interior = (
                slice(None),
                slice(None),
                slice(None),
                slice(g, -g),
                slice(g, -g),
                slice(g, -g),
            )
            # J is the native density sqrt(gamma) J^i, so it already carries the volume factor.
            grid_work = sum(jnp.sum(D[i][interior] * J[i][interior]) for i in range(3))
            grid_work *= dynamic_parameters.dx * dynamic_parameters.dy * dynamic_parameters.dz

            gathered_D = []
            for tx in range(1):
                gathered_D.append(
                    hybrid_pusher.gather_vector(
                        tuple(D[i][tx, 0, 0] for i in range(3)),
                        D_FIELD_LOCATIONS,
                        particles.x[tx, 0, 0],
                        tuple(axis[tx, 0, 0] for axis in center_grid),
                        tuple(axis[tx, 0, 0] for axis in vertex_grid),
                        static_parameters,
                        (True, True, True),
                        (g, g, g),
                    )
                )
            gathered_D = jnp.stack(gathered_D, axis=0).reshape(particles.x.shape)

            Gamma = jnp.sqrt(1.0 + jnp.sum(particles.u**2, axis=-1))
            source_velocity = 0.73 * particles.u / Gamma[..., jnp.newaxis]
            source_velocity -= jnp.asarray((0.12, -0.08, 0.05))
            weighted_charge = species.charge[0] * species.weight[0]
            particle_work = weighted_charge * jnp.sum(
                particles.active[..., jnp.newaxis]
                * source_velocity
                * gathered_D
            )

            scale = jnp.maximum(jnp.abs(grid_work), jnp.abs(particle_work))
            relative_residual = jnp.abs(grid_work - particle_work) / scale
            self.assertLess(relative_residual, 1.0e-12)


if __name__ == "__main__":
    unittest.main()
