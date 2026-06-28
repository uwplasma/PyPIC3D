import os
import tempfile
import unittest

import jax
import numpy as np

from PyPIC3D.__main__ import run_PyPIC3D


jax.config.update("jax_enable_x64", True)


class TiledEsirkepovIntegrationTest(unittest.TestCase):
    def test_short_output_enabled_tiled_esirkepov_run_keeps_live_state_tiled(self):
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
                    "solver": "tiled_yee",
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
                    "fast_backend": "default",
                    "particle_pusher": "boris",
                    "relativistic": False,
                },
                "plotting": {
                    "plotting_interval": 1,
                    "plot_vtk_vectors": True,
                    "plot_vtk_particles": True,
                    "plot_openpmd_fields": True,
                    "plot_openpmd_particles": True,
                    "plot_phasespace": False,
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

            _Nt, _plotting, _simulation, _plasma, _constants, _particles, fields, world = run_PyPIC3D(config)
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
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "data", "vector_field_slice", "vector_field_slice_000000000.vtk")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "data", "particles", "electrons.000000000.vtu")))


if __name__ == "__main__":
    unittest.main()
