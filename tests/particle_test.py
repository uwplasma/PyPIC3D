import unittest
import jax
import numpy as np
from PyPIC3D.particle import initial_particles, cold_start_init, periodic_boundary_condition, euler_update, update_position, total_KE, total_momentum, particle_species

import jax.numpy as jnp

class TestParticleMethods(unittest.TestCase):

    def setUp(self):
        self.N_particles = 100
        self.x_wind = 10.0
        self.y_wind = 10.0
        self.z_wind = 10.0
        self.mass = 1.0
        self.T = 300.0
        self.kb = 1.38e-23
        self.key1, self.key2, self.key3 = jax.random.split(jax.random.PRNGKey(0), 3)
        self.dt = 0.01

    def test_initial_particles(self):
        x, y, z, vx, vy, vz = initial_particles(self.N_particles, self.x_wind, self.y_wind, self.z_wind, self.mass, self.T, self.kb, self.key1, self.key2, self.key3)
        self.assertEqual(x.shape[0], self.N_particles)
        self.assertEqual(y.shape[0], self.N_particles)
        self.assertEqual(z.shape[0], self.N_particles)
        self.assertEqual(vx.shape[0], self.N_particles)
        self.assertEqual(vy.shape[0], self.N_particles)
        self.assertEqual(vz.shape[0], self.N_particles)

    def test_cold_start_init(self):
        start = 0.0
        x, y, z, vx, vy, vz = cold_start_init(start, self.N_particles, self.x_wind, self.y_wind, self.z_wind, self.mass, self.T, self.kb, self.key1, self.key2, self.key3)
        self.assertTrue(jnp.all(x == start))
        self.assertTrue(jnp.all(y == start))
        self.assertTrue(jnp.all(z == start))

    def test_periodic_boundary_condition(self):
        x = jnp.array([5.0, -6.0, 0.0])
        y = jnp.array([5.0, -6.0, 0.0])
        z = jnp.array([5.0, -6.0, 0.0])
        x, y, z = periodic_boundary_condition(self.x_wind, self.y_wind, self.z_wind, x, y, z)
        self.assertTrue(jnp.all(x == jnp.array([-5.0, 5.0, 0.0])))
        self.assertTrue(jnp.all(y == jnp.array([-5.0, 5.0, 0.0])))
        self.assertTrue(jnp.all(z == jnp.array([-5.0, 5.0, 0.0])))

    def test_euler_update(self):
        s = jnp.array([1.0, 2.0, 3.0])
        v = jnp.array([0.1, 0.2, 0.3])
        s_updated = euler_update(s, v, self.dt)
        self.assertTrue(jnp.allclose(s_updated, jnp.array([1.001, 2.002, 3.003])))

    def test_update_position(self):
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([1.0, 2.0, 3.0])
        z = jnp.array([1.0, 2.0, 3.0])
        vx = jnp.array([0.1, 0.2, 0.3])
        vy = jnp.array([0.1, 0.2, 0.3])
        vz = jnp.array([0.1, 0.2, 0.3])
        x, y, z = update_position(x, y, z, vx, vy, vz, self.dt, self.x_wind, self.y_wind, self.z_wind)
        self.assertTrue(jnp.allclose(x, jnp.array([1.001, 2.002, 3.003])))
        self.assertTrue(jnp.allclose(y, jnp.array([1.001, 2.002, 3.003])))
        self.assertTrue(jnp.allclose(z, jnp.array([1.001, 2.002, 3.003])))

    def test_total_KE(self):
        species = particle_species("test", self.N_particles, 1.0, self.mass, jnp.ones(self.N_particles), jnp.ones(self.N_particles), jnp.ones(self.N_particles), jnp.zeros(self.N_particles), jnp.zeros(self.N_particles), jnp.zeros(self.N_particles))
        ke = total_KE([species])
        self.assertAlmostEqual(ke, 0.5 * self.mass * self.N_particles * 3)

    def test_total_momentum(self):
        vx = jnp.ones(self.N_particles)
        vy = jnp.ones(self.N_particles)
        vz = jnp.ones(self.N_particles)
        momentum = total_momentum(self.mass, vx, vy, vz)
        self.assertAlmostEqual(momentum, self.mass * self.N_particles * jnp.sqrt(3))

    def test_particle_species(self):
        species = particle_species("test", self.N_particles, 1.0, self.mass, jnp.ones(self.N_particles), jnp.ones(self.N_particles), jnp.ones(self.N_particles), jnp.zeros(self.N_particles), jnp.zeros(self.N_particles), jnp.zeros(self.N_particles))
        self.assertEqual(species.get_name(), "test")
        self.assertEqual(species.get_charge(), 1.0)
        self.assertEqual(species.get_number_of_particles(), self.N_particles)
        self.assertTrue(jnp.all(species.get_velocity()[0] == jnp.ones(self.N_particles)))
        self.assertTrue(jnp.all(species.get_position()[0] == jnp.zeros(self.N_particles)))
        self.assertEqual(species.get_mass(), self.mass)
        species.set_velocity(jnp.zeros(self.N_particles), jnp.zeros(self.N_particles), jnp.zeros(self.N_particles))
        self.assertTrue(jnp.all(species.get_velocity()[0] == jnp.zeros(self.N_particles)))
        species.set_position(jnp.ones(self.N_particles), jnp.ones(self.N_particles), jnp.ones(self.N_particles))
        self.assertTrue(jnp.all(species.get_position()[0] == jnp.ones(self.N_particles)))
        species.set_mass(2.0)
        self.assertEqual(species.get_mass(), 2.0)
        self.assertAlmostEqual(species.kinetic_energy(), 0.0)
        self.assertAlmostEqual(species.momentum(), 0.0)

if __name__ == '__main__':
    unittest.main()