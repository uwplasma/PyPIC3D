from tests.support.utils_runtime_fixtures import (
    T_to_vth,
    jnp,
    unittest,
    vth_to_T,
)


class TestUtilsFunctions(unittest.TestCase):
    def setUp(self):
        self.parameter_set = {
            'Nx': 4,
            'Ny': 4,
            'Nz': 4,
            'dx': 0.1,
            'dy': 0.1,
            'dz': 0.1,
            'dt': 0.01,
            'x_wind': 1.0,
            'y_wind': 1.0,
            'z_wind': 1.0
        }
        self.plasma_parameters = {
            "Theoretical Plasma Frequency": 1.0,
            "Debye Length": 0.01,
            "Thermal Velocity": 1.0,
            "Number of Electrons": 10,
            "dx per debye length": 2.0
        }

    def test_temperature_thermal_velocity_conversion_round_trip(self):
        temperature = 8.0
        mass = 2.0
        kb = 4.0

        thermal_velocity = T_to_vth(temperature, mass, kb)

        self.assertTrue(jnp.allclose(thermal_velocity, 4.0))
        self.assertTrue(jnp.allclose(vth_to_T(thermal_velocity, mass, kb), temperature))


if __name__ == "__main__":
    unittest.main()
