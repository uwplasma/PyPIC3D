import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import jax
import numpy as np
import toml

from PyPIC3D import __main__ as pypic_main
from PyPIC3D.__main__ import run_PyPIC3D


jax.config.update("jax_enable_x64", True)


class TiledEsirkepovIntegrationTest(unittest.TestCase):
    def test_short_output_enabled_tiled_esirkepov_run_writes_openpmd_without_vtk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            vx_path = os.path.join(tmpdir, "vx.npy")
            np.save(x_path, np.array([-1.5, -0.5, 0.5, 1.5]))
            np.save(zeros_path, np.zeros(4))
            np.save(vx_path, np.array([0.10, -0.05, 0.07, -0.02]))

            config = {
                "simulation_parameters": {
                    "name": "tiled esirkepov output smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 8,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 4.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 0.01,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "esirkepov",
                    "filter_j": "none",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {
                    "plotting_interval": 1,
                    "plot_openpmd_fields": True,
                    "plot_openpmd_particles": True,
                    "plot_phasespace": True,
                },
                "particle1": {
                    "name": "electrons",
                    "N_particles": 4,
                    "charge": -1.0,
                    "mass": 2.0,
                    "weight": 0.5,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": vx_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }

            _Nt, _plotting, _simulation, _plasma, _constants, _particles, fields, world, _species_config = run_PyPIC3D(config)
            _E, _B, J, *_rest = fields

            self.assertEqual(J[0].shape[-3:], (6, 5, 5))
            self.assertEqual(tuple(int(width) for width in world["tile_shape"]), (2, 1, 1))
            for filename in (
                "total_energy.txt",
                "electric_field_energy.txt",
                "magnetic_field_energy.txt",
                "kinetic_energy.txt",
                "total_momentum.txt",
            ):
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "data", filename)))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "data", "fields.h5")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "data", "particles.h5")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "data", "vector_field_slice")))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "data", "particles", "electrons.000000000.vtu")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "data", "phase_space", "x", "electrons_phase_space.000000000.npy")))

    def test_main_prints_final_energy_for_tiled_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_path = os.path.join(tmpdir, "x.npy")
            zeros_path = os.path.join(tmpdir, "zeros.npy")
            config_path = os.path.join(tmpdir, "final_energy.toml")
            np.save(x_path, np.array([0.0]))
            np.save(zeros_path, np.zeros(1))

            config = {
                "simulation_parameters": {
                    "name": "final energy smoke",
                    "output_dir": tmpdir,
                    "solver": "electrodynamic_yee",
                    "Nx": 4,
                    "Ny": 1,
                    "Nz": 1,
                    "x_wind": 1.0,
                    "y_wind": 1.0,
                    "z_wind": 1.0,
                    "dt": 1.0e-10,
                    "Nt": 1,
                    "shape_factor": 1,
                    "particle_tile_nx": 2,
                    "particle_tile_ny": 1,
                    "particle_tile_nz": 1,
                    "current_calculation": "j_from_rhov",
                    "filter_j": "none",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {"plotting_interval": 1},
                "particle1": {
                    "name": "electrons",
                    "N_particles": 1,
                    "charge": -1.0,
                    "mass": 1.0,
                    "weight": 1.0,
                    "temperature": 1.0,
                    "initial_x": x_path,
                    "initial_y": zeros_path,
                    "initial_z": zeros_path,
                    "initial_vx": zeros_path,
                    "initial_vy": zeros_path,
                    "initial_vz": zeros_path,
                },
            }
            with open(config_path, "w") as f:
                toml.dump(config, f)

            with patch("sys.argv", ["PyPIC3D", "--config", config_path]):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    pypic_main.main()

            self.assertIn("Final Kinetic Energy", stdout.getvalue())
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "data", "output.toml")))


if __name__ == "__main__":
    unittest.main()
