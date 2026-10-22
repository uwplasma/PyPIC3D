"""Geometry transfer and leapfrog initialization for native BSSN coupling."""

import unittest
import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

try:
    import JAX_BSSN
except ModuleNotFoundError as error:
    if error.name == "JAX_BSSN":
        raise unittest.SkipTest("Optional JAX_BSSN dependency is not installed") from error
    raise

from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.deposition.GR_direct_deposition import GR_direct_deposition
from PyPIC3D.pusher.hybrid_boris_geodesic import hybrid_boris_geodesic_push
from PyPIC3D.relativity.bssn_parameters import build_bssn_parameters
from PyPIC3D.relativity.dynamical_relativity import (
    bssn_to_tiles, tiles_to_bssn, build_bssn_metric,
    initialize_dynamical_relativity, synchronize_particles, rephase_pic_state,
)
from tests.kernel_fixtures import kernel_parameters


def runtime(boundaries=(0, 0, 0)):
    static, dynamic = kernel_parameters(
        Nx=8, Ny=1, Nz=1, x_wind=2., y_wind=.25, z_wind=.25,
        x_min=0., y_min=0., z_min=0., dt=.01, guard_cells=3,
        solver="dynamic_metric", metric="numerical", current_deposition="GR_direct",
        particle_pusher="hybrid_boris_geodesic", boundary_conditions=boundaries)
    parameters = build_bssn_parameters(static, dynamic, evolution_parameters={"nu": 0.})
    dynamic = dynamic._replace(bssn_parameters=parameters)
    scalar = jnp.zeros((1, 1, 1, 14, 7, 7))
    zero = (scalar,) * 3
    electric = (scalar + .04, scalar, scalar)
    fields = (electric, zero, zero, scalar, scalar, (zero, zero), None, (zero, zero), jnp.asarray(False))
    position = jnp.array([.7, 0., 0.]).reshape(1, 1, 1, 1, 1, 3)
    momentum = jnp.array([.2, .05, 0.]).reshape(position.shape)
    particles = TiledParticles(position, momentum, jnp.ones(position.shape[:-1], dtype=bool))
    species = SpeciesConfig(jnp.array([.4]), jnp.array([1.]), jnp.array([.01]), jnp.ones((1, 3), dtype=bool))
    return static, dynamic, particles, species, fields


class BSSNParametersTest(unittest.TestCase):
    def test_flat_defaults_and_dynamic_controls(self):
        static, dynamic, *_ = runtime()
        parameters = dynamic.bssn_parameters
        self.assertEqual(parameters.state.conformal_factor.shape, (8, 1, 1))
        self.assertEqual(int(parameters.evolution_parameters.zero_shift), 1)
        self.assertEqual(int(parameters.evolution_parameters.gauge), 0)
        self.assertEqual(float(parameters.evolution_parameters.x_min), float(dynamic.grids.center[0][1]))
        for leaf in jax.tree.leaves(parameters):
            self.assertTrue(isinstance(leaf, jax.Array))
        with self.assertRaisesRegex(ValueError, "C=1"):
            build_bssn_parameters(static, dynamic._replace(C=jnp.asarray(2.)))

    def test_component_leading_tile_roundtrip(self):
        static, *_ = runtime()
        static = static._replace(tile_shape=(4, 2, 1), field_mesh=None)
        values = jnp.arange(2 * 3 * 8 * 4).reshape(2, 3, 8, 4, 1)
        tiles = bssn_to_tiles(values, static)
        self.assertEqual(tiles.shape, (2, 3, 2, 2, 1, 10, 8, 7))
        self.assertTrue(bool(jnp.array_equal(tiles_to_bssn(tiles, static), values)))
        self.assertTrue(bool(jnp.array_equal(tiles[..., 0, 0, 0, 2, 3, 3], values[..., -1, 0, 0])))

    def test_gravity_boundaries_and_caller_state_are_independent_of_em(self):
        static, dynamic, *_ = runtime()
        initial = dynamic.bssn_parameters
        state = initial.state._replace(trace_K=jnp.full_like(initial.state.trace_K, .03))
        evolution = initial.evolution_parameters._replace(xl_bc=1, xr_bc=1, gauge=1, zero_shift=0, eta=.7)
        parameters = build_bssn_parameters(static, dynamic, state=state,
                                           evolution_parameters=evolution, gravity_G=.2, time=3.)
        self.assertIs(parameters.state, state)
        self.assertEqual(static.boundary_conditions, (0, 0, 0))
        self.assertEqual(int(parameters.evolution_parameters.xl_bc), 1)
        self.assertEqual(int(parameters.evolution_parameters.xr_bc), 1)
        for actual, expected in zip(parameters.evolution_parameters, evolution):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))
        self.assertEqual(float(parameters.gravity_G), .2)
        self.assertEqual(float(parameters.time), 3.)

    def test_ordinary_pypic_imports_do_not_require_numerical_relativity(self):
        source = '''
import sys
import importlib.abc
class NoNR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("JAX_BSSN"):
            raise AssertionError("Ordinary PyPIC3D imported optional numerical relativity")
sys.meta_path.insert(0, NoNR())
import PyPIC3D.initialization
import PyPIC3D.utilities.parameters
import PyPIC3D.solvers.GR_yee.time_loop
assert not any(name.startswith("JAX_BSSN") for name in sys.modules)
'''
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_conducting_endpoint_and_affine_metric_guards(self):
        static, dynamic, *_ = runtime((1, 0, 0))
        state = dynamic.bssn_parameters.state
        self.assertEqual(state.conformal_factor.shape, (9, 1, 1))
        x = jnp.arange(9, dtype=jnp.float64)[:, None, None]
        values = 1. + .01 * x
        self.assertTrue(bool(jnp.array_equal(tiles_to_bssn(bssn_to_tiles(values, static), static), values)))
        state = state._replace(lapse=values)
        metric = jax.jit(lambda q: build_bssn_metric(q, static, dynamic))(state)
        expected = 1. + .01 * (jnp.arange(14) - 3 + .5)
        self.assertTrue(bool(jnp.allclose(metric.D[0].lapse[0, 0, 0, :, 3, 3], expected, atol=1.e-14)))
        self.assertTrue(bool(jnp.array_equal(metric.center.gamma, jnp.broadcast_to(jnp.eye(3), metric.center.gamma.shape))))

    def test_initial_current_and_common_particle_view(self):
        static, dynamic, particles, species, fields = runtime()
        stored, fields, dynamic = initialize_dynamical_relativity(particles, species, fields, static, dynamic)
        self.assertTrue(bool(jnp.array_equal(stored.x, particles.x)))
        self.assertTrue(bool(jnp.any(fields[2][0] != 0)))
        self.assertEqual(float(fields[7][3]), .005)
        common = synchronize_particles(stored, species, fields, static, dynamic)
        self.assertTrue(bool(jnp.allclose(common.u, particles.u, atol=1.e-14, rtol=0.)))
        self.assertTrue(bool(jnp.array_equal(common.x, particles.x)))

    def test_momentum_only_push_preserves_positions(self):
        static, dynamic, particles, species, fields = runtime()
        metric = build_bssn_metric(dynamic.bssn_parameters.state, static, dynamic)
        pushed, centered = hybrid_boris_geodesic_push(
            particles, species, fields[0], fields[1], metric, static, dynamic,
            advance_positions=False)
        self.assertTrue(bool(jnp.array_equal(pushed.x, particles.x)))
        self.assertTrue(bool(jnp.array_equal(centered.x, particles.x)))
        self.assertTrue(bool(jnp.any(pushed.u != particles.u)))

    def test_initialization_rejects_unsupported_sources_and_geometry(self):
        static, dynamic, particles, species, fields = runtime()
        prescribed = fields[:5] + ((fields[0], fields[1]),) + fields[6:]
        with self.assertRaisesRegex(ValueError, "external fields"):
            initialize_dynamical_relativity(particles, species, prescribed, static, dynamic)
        for incompatible, message in (
            (static._replace(particle_boundary_conditions=(2, 0, 0)), "periodic"),
            (static._replace(boundary_conditions=(3, 0, 0)), "periodic"),
            (static._replace(metric="flat_spherical"), "Cartesian"),
            (static._replace(pml_active=True), "PML"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                initialize_dynamical_relativity(particles, species, fields, incompatible, dynamic)

    def test_metric_rephase_preserves_common_fields_and_identity(self):
        static, dynamic, particles, species, fields = runtime()
        particles, fields, dynamic = initialize_dynamical_relativity(particles, species, fields, static, dynamic)
        same_particles, same_fields = rephase_pic_state(particles, species, fields, fields[6], static, dynamic)
        for actual, expected in zip(jax.tree.leaves((same_particles, same_fields)), jax.tree.leaves((particles, fields))):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))
        state = dynamic.bssn_parameters.state
        metric = build_bssn_metric(state._replace(lapse=1.1 * state.lapse), static, dynamic)
        changed_particles, changed_fields = rephase_pic_state(particles, species, fields, metric, static, dynamic)
        self.assertTrue(bool(jnp.array_equal(changed_particles.x, particles.x)))
        self.assertTrue(bool(jnp.any(changed_particles.u != particles.u)))
        for actual, expected in zip(changed_fields[0] + changed_fields[7][2], fields[0] + fields[7][2]):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))

    def test_filtered_current_is_consistent_at_initialization_and_metric_switch(self):
        static, dynamic, common, species, fields = runtime()
        static = static._replace(current_filter="bilinear")
        particles, fields, dynamic = initialize_dynamical_relativity(common, species, fields, static, dynamic)
        direct = GR_direct_deposition(common, species, fields[2], fields[6], static, dynamic)
        unfiltered = GR_direct_deposition(common, species, fields[2], fields[6],
                                         static._replace(current_filter="none"), dynamic)
        self.assertTrue(any(bool(jnp.any(a != b)) for a, b in zip(direct, unfiltered)))
        for actual, expected in zip(fields[2], direct):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))
        same = rephase_pic_state(particles, species, fields, fields[6], static, dynamic)
        for actual, expected in zip(jax.tree.leaves(same), jax.tree.leaves((particles, fields))):
            self.assertTrue(bool(jnp.array_equal(actual, expected)))

    def test_native_output_uses_updated_time_and_correct_staggering(self):
        import openpmd_api as io
        from PyPIC3D.diagnostics.async_writer import (
            create_async_tiled_openpmd_field_writer, enqueue_openpmd_field_output,
        )
        from PyPIC3D.diagnostics.output_adapters import build_field_output_map

        static, dynamic, particles, species, fields = runtime()
        particles, fields, dynamic = initialize_dynamical_relativity(
            particles, species, fields, static, dynamic)
        dynamic = dynamic._replace(bssn_parameters=dynamic.bssn_parameters._replace(time=jnp.asarray(.7)))
        output = build_field_output_map(fields, particles, species, static, dynamic)
        with tempfile.TemporaryDirectory() as directory:
            writer = create_async_tiled_openpmd_field_writer(static, dynamic, directory)
            try:
                enqueue_openpmd_field_output(writer, output, dynamic, 0, 0)
                enqueue_openpmd_field_output(writer, output, dynamic, 2, 999)
            finally:
                writer.close()
            series = io.Series(str(Path(directory) / "fields.h5"), io.Access.read_only)
            self.assertEqual(series.iterations[0].meshes["J"].time_offset, 0.)
            iteration = series.iterations[2]
            self.assertAlmostEqual(iteration.time, .7)
            self.assertAlmostEqual(iteration.meshes["J"].time_offset, -.0025)
            self.assertEqual(iteration.meshes["B"].time_offset, 0.)
            self.assertEqual(iteration.meshes["E"].get_attribute("normalization"), "Gaussian, c=1")
            self.assertEqual(tuple(iteration.meshes["E"]["x"].position), (.5, 0., 0.))
            self.assertEqual(tuple(iteration.meshes["B"]["x"].position), (0., .5, .5))
            self.assertIn("lapse", iteration.meshes)
            values = iteration.meshes["E"]["x"].load_chunk()
            series.flush()
            values = jnp.array(values)
            jax.block_until_ready(values)
            series.close()
        self.assertEqual(values.shape, (8, 1, 1))
        self.assertTrue(bool(jnp.allclose(values, .04)))

    def test_dynamic_configuration_rejects_ignored_or_incompatible_inputs(self):
        from PyPIC3D.initialization import initialize_simulation

        cases = (
            ({"particle_pusher": "boris"}, {}, "hybrid_boris_geodesic"),
            ({"C": 2.}, {}, "C=1"),
            ({}, {"previous_field1": {"type": 0}}, "common-time D/B"),
            ({}, {"field1": {"type": 6}}, "seeds current"),
            ({}, {"particle1": {"x_bc": "reflecting"}}, "periodic"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for options, blocks, message in cases:
                config = {"simulation_parameters": {
                    "solver": "dynamic_metric", "output_dir": directory,
                    "Nx": 8, "Ny": 1, "Nz": 1,
                    "x_wind": 2., "y_wind": .25, "z_wind": .25,
                    "Nt": 1, "dt": .01, **options,
                }, **blocks}
                with self.subTest(options=options, blocks=blocks):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, message):
                            initialize_simulation(config)

    def test_initial_field_and_particle_dumps_use_initial_bssn_time(self):
        import openpmd_api as pmd
        from PyPIC3D.diagnostics.openPMD import (
            write_openpmd_initial_fields, write_openpmd_initial_particles,
        )
        from PyPIC3D.diagnostics.output_adapters import build_field_output_map

        static, dynamic, common_particles, species, fields = runtime()
        dynamic = dynamic._replace(bssn_parameters=dynamic.bssn_parameters._replace(time=jnp.asarray(2.5)))
        particles, fields, dynamic = initialize_dynamical_relativity(
            common_particles, species, fields, static, dynamic)
        output = build_field_output_map(fields, particles, species, static, dynamic)
        with tempfile.TemporaryDirectory() as directory:
            write_openpmd_initial_fields(output, static, dynamic, directory)
            write_openpmd_initial_particles(
                common_particles, static, dynamic, directory,
                species_config=species, species_names=("test",), metric=fields[6])
            for path in ("initial_fields/initial_fields.h5", "initial_particles/test_initial_particles.h5"):
                series = pmd.Series(str(Path(directory) / "data" / path), pmd.Access.read_only)
                iteration = series.iterations[0]
                self.assertAlmostEqual(iteration.time, 2.5)
                if "J" in iteration.meshes:
                    self.assertEqual(iteration.meshes["J"].time_offset, 0.)
                series.close()


if __name__ == "__main__":
    unittest.main()
