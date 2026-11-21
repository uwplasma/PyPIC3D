"""One-tile numerical and physical checks."""

from tests.support.dark_photon_fixtures import (
    D_FIELD_LOCATIONS,
    DarkPhotonFixtures,
    build_tiled_particles,
    compute_dark_B,
    contextlib,
    dark_divergence,
    dark_gradient,
    dark_photon_push_fields,
    evolve,
    filter_electric_field_for_particles,
    initialize_dark_photon_fields,
    initialize_fields,
    initialize_simulation,
    interior,
    io,
    jax,
    jnp,
    kernel_parameters,
    np,
    os,
    parameters,
    particle_species,
    synchronized_dark_fields,
    tempfile,
    time_loop_dark_photon,
    time_loop_electrodynamic,
    unittest,
    update_dark_E,
    update_tiled_ghost_cells,
    update_tiled_ghost_cells,
    wave,
    wave_errors,
)


class TestDarkPhoton(DarkPhotonFixtures, unittest.TestCase):


    def test_uniform_mass_oscillation_and_massless_limit(self):
        for mass in (0., 0.7):
            s, d = parameters(n=1, dt=1/1024, dark_mu=mass)
            E, A, J, phi, _ = initialize_fields(s, d)
            A = (jnp.ones_like(A[0]), A[1], A[2])
            state = initialize_dark_photon_fields(s, d, E, A, phi)
            result = evolve(state, s, d, 1024)
            omega = float(d.C)*mass
            np.testing.assert_allclose(result[0][0][interior(s)], omega*np.sin(omega), atol=2e-7)
            np.testing.assert_allclose(result[1][0][interior(s)], np.cos(omega*(1-float(d.dt)/2)), atol=2e-7)
        s, d = parameters(dark_mu=0., dt=1/128)
        state = initialize_dark_photon_fields(s, d, *wave(s, d, False, discrete=True))
        self.assertLess(max(wave_errors(s, d, evolve(state, s, d, 128), False, discrete=True)), 3e-5)


if __name__ == "__main__":
    unittest.main()
