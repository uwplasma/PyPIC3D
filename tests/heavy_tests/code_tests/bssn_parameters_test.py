from tests.support.bssn_parameters_runtime_fixtures import (
    GR_direct_deposition,
    Path,
    bssn_to_tiles,
    build_bssn_metric,
    build_bssn_parameters,
    contextlib,
    hybrid_boris_geodesic_push,
    initialize_dynamical_relativity,
    io,
    jax,
    jnp,
    rephase_pic_state,
    runtime,
    subprocess,
    synchronize_particles,
    sys,
    tempfile,
    tiles_to_bssn,
    unittest,
)


class BSSNParametersTest(unittest.TestCase):


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


if __name__ == "__main__":
    unittest.main()
