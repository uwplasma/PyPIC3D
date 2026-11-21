"""One-tile numerical and physical checks."""

from tests.support.rho_fixtures import (
    BC_CONDUCTING,
    BC_PERIODIC,
    TiledRhoFixtures,
    jnp,
    kernel_parameters_from_values,
    node_weights,
    particle_species,
    unittest,
)


class TestTiledRho(TiledRhoFixtures, unittest.TestCase):
    def test_tsc_charge_is_conserved_at_both_reflecting_particle_walls(self):
        parameter_set = self._build_parameter_values(
            shape_factor=2,
            particle_boundary_conditions={
                "x": BC_PERIODIC,
                "y": BC_PERIODIC,
                "z": BC_CONDUCTING,
            },
        )
        charge = 2.0
        macro_weight = 0.25
        particles = [
            particle_species(
                name="positive",
                charge=charge,
                mass=1.0,
                weight=macro_weight,
                x1=jnp.array([-0.3, 0.3, -0.3, 0.3]),
                x2=jnp.zeros(4),
                # TSC stencils cross both physical walls, unevenly, so the two
                # wall nodes receive different reflected charge.
                x3=jnp.array([-0.99, -0.76, 0.6, 0.9]),
            )
        ]

        rho_tiles, _ = self._deposit_and_assemble(
            particles,
            parameter_set,
            self._one_tile_parameters(parameter_set),
            {"alpha": 1.0},
        )
        g = int(parameter_set["guard_cells"])
        static_parameters, _ = kernel_parameters_from_values(
            self._parameters_with_tiled_grids(parameter_set, self._one_tile_parameters(parameter_set)),
            {"alpha": 1.0},
        )
        # reflected nodal charge integrates with endpoint trapezoid weights,
        # including the owned upper wall node
        weights = node_weights(static_parameters, rho_tiles)
        deposited_charge = (
            jnp.sum(weights * rho_tiles)
            * parameter_set["dx"]
            * parameter_set["dy"]
            * parameter_set["dz"]
        )
        expected_charge = len(particles[0]["x"]) * charge * macro_weight

        self.assertAlmostEqual(float(deposited_charge), expected_charge, places=12)
        self.assertGreaterEqual(float(jnp.min(rho_tiles[:, :, :, g:-g, g:-g, g:-g+1])), -1.0e-14)


if __name__ == "__main__":
    unittest.main()
