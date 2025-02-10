import unittest
import jax
import jax.numpy as jnp
import sys
import os

# # Add the parent directory to the sys.path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PyPIC3D.particle import (
    initial_particles, total_KE, total_momentum, particle_species
)

jax.config.update("jax_enable_x64", True)

class TestParticleMethods(unittest.TestCase):
    def setUp(self):
        """
        Set up the initial conditions for the particle test.

        Attributes:
            N_particles (int): Number of particles.
            x_wind (float): Wind velocity in the x direction.
            y_wind (float): Wind velocity in the y direction.
            z_wind (float): Wind velocity in the z direction.
            mass (float): Mass of each particle.
            T (float): Temperature in Kelvin.
            kb (float): Boltzmann constant.
            key1 (jax.random.PRNGKey): First random key for JAX.
            key2 (jax.random.PRNGKey): Second random key for JAX.
            key3 (jax.random.PRNGKey): Third random key for JAX.
            dt (float): Time step for the simulation.
        """
        self.N_particles = 3
        self.x_wind = 10.0
        self.y_wind = 10.0
        self.z_wind = 10.0
        self.mass = 1.0
        self.T = 300.0
        self.kb = 1.38e-23
        self.key1, self.key2, self.key3 = jax.random.split(jax.random.PRNGKey(0), 3)
        self.dt = 1.0

    def test_initial_particles(self):
        """
        Test the initial_particles function to ensure it generates the correct number of particles.

        This test verifies that the initial_particles function returns arrays of the correct shape
        for the positions (x, y, z) and velocities (vx, vy, vz) of the particles.

        The function checks:
        - The number of particles in the x, y, and z position arrays is equal to N_particles.
        - The number of particles in the vx, vy, and vz velocity arrays is equal to N_particles.

        Attributes:
            N_particles (int): Number of particles to generate.
            x_wind (float): Wind component in the x direction.
            y_wind (float): Wind component in the y direction.
            z_wind (float): Wind component in the z direction.
            mass (float): Mass of each particle.
            T (float): Temperature.
            kb (float): Boltzmann constant.
            key1, key2, key3: Keys for random number generation.
        """
        x, y, z, vx, vy, vz = initial_particles(
            N_particles=self.N_particles,
            minx=-self.x_wind/2,
            maxx=self.x_wind/2,
            miny=-self.y_wind/2,
            maxy=self.y_wind/2,
            minz=-self.z_wind/2,
            maxz=self.z_wind/2,
            mass=self.mass,
            T=self.T,
            kb=self.kb,
            key1=self.key1,
            key2=self.key2,
            key3=self.key3,
        )
        self.assertEqual(x.shape[0], self.N_particles)
        self.assertEqual(y.shape[0], self.N_particles)
        self.assertEqual(z.shape[0], self.N_particles)
        self.assertEqual(vx.shape[0], self.N_particles)
        self.assertEqual(vy.shape[0], self.N_particles)
        self.assertEqual(vz.shape[0], self.N_particles)

    def test_periodic_boundary_condition(self):
        """
        Test the periodic boundary condition for particle species.
        This test initializes particle positions and velocities, creates a particle species
        with these properties, and applies the periodic boundary condition. It then checks
        if the positions of the particles are correctly wrapped around the boundaries.
        The test verifies that the positions of the particles after applying the periodic
        boundary condition are as expected.
        Asserts:
            - The x positions of the particles are [-5.0, 5.0, 0.0].
            - The y positions of the particles are [-5.0, 5.0, 0.0].
            - The z positions of the particles are [-5.0, 5.0, 0.0].
        """

        x1 = jnp.array([5.50, -6.0, 0.0])
        x2 = jnp.array([5.50, -6.0, 0.0])
        x3 = jnp.array([5.50, -6.0, 0.0])
    
        species = particle_species(

            name = "test",
            N_particles = self.N_particles,
            charge = 1.0,
            mass = self.mass,
            weight = 1.0,
            T = self.T,
            v1 = jnp.array([1.0, 2.0, 3.0]),
            v2 = jnp.array([1.0, 2.0, 3.0]),
            v3 = jnp.array([1.0, 2.0, 3.0]),
            x1 = x1,
            x2 = x2,
            x3 = x3,
            dx = 1.0,
            dy = 1.0,
            dz = 1.0,
            xwind = self.x_wind,
            ywind = self.y_wind,
            zwind = self.z_wind,
            subcells = (x1, x1, x2, x2, x3, x3),
        )

        species.periodic_boundary_condition(self.x_wind, self.y_wind, self.z_wind)
        x, y, z = species.get_position()
        self.assertTrue(jnp.all(x == jnp.array([-5.0, 5.0, 0.0])))
        self.assertTrue(jnp.all(y == jnp.array([-5.0, 5.0, 0.0])))
        self.assertTrue(jnp.all(z == jnp.array([-5.0, 5.0, 0.0])))

    def test_update_position(self):
        """
        Test the update_position method of the particle_species class.
        This test initializes a particle_species instance with predefined positions
        and velocities, updates the positions using the update_position method, and
        verifies that the new positions are as expected.
        The test checks:
        - The initial positions of the particles.
        - The velocities of the particles.
        - The updated positions after calling update_position.
        - The correctness of the updated positions using jnp.allclose.
        Asserts:
        - The updated x positions are close to the expected values.
        - The updated y positions are close to the expected values.
        - The updated z positions are close to the expected values.
        """
        x1=jnp.array([0.1, 0.2, 0.3])
        x2=jnp.array([0.1, 0.2, 0.3])
        x3=jnp.array([0.1, 0.2, 0.3])
    
        species = particle_species(
            name="test",
            N_particles=self.N_particles,
            charge=1.0,
            mass=self.mass,
            weight=1.0,
            T=self.T,
            v1=jnp.array([1.0, 2.0, 3.0]),
            v2=jnp.array([1.0, 2.0, 3.0]),
            v3=jnp.array([1.0, 2.0, 3.0]),
            x1=x1,
            x2=x2,
            x3=x3,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            xwind=self.x_wind,
            ywind=self.y_wind,
            zwind=self.z_wind,
            subcells=(x1, x1, x2, x2, x3, x3),
        )
        species.update_position(self.dt, self.x_wind, self.y_wind, self.z_wind)
        x, y, z = species.get_position()
        self.assertTrue(jnp.allclose(x, jnp.array([1.1, 2.2, 3.3])))
        self.assertTrue(jnp.allclose(y, jnp.array([1.1, 2.2, 3.3])))
        self.assertTrue(jnp.allclose(z, jnp.array([1.1, 2.2, 3.3])))

    def test_total_KE(self):
        """
        Test the total kinetic energy calculation for a particle species.
        This test initializes a particle species with a given number of particles,
        mass, and temperature. It sets the velocities of all particles to 1.0 in
        each direction and calculates the total kinetic energy using the 
        `total_KE` function. The expected kinetic energy is compared to the 
        calculated value using `assertAlmostEqual`.
        The expected kinetic energy is calculated as:
            KE = 0.5 * mass * N_particles * 3
        where:
            - mass is the mass of each particle
            - N_particles is the number of particles
            - 3 accounts for the three velocity components (v1, v2, v3)
        Asserts:
            The calculated kinetic energy is almost equal to the expected value.
        """

        x1=jnp.zeros(self.N_particles)
        x2=jnp.zeros(self.N_particles)
        x3=jnp.zeros(self.N_particles)
    
        species = particle_species(
            name="test",
            N_particles=self.N_particles,
            charge=1.0,
            mass=self.mass,
            weight=1.0,
            T=self.T,
            v1 = jnp.ones(self.N_particles),
            v2 = jnp.ones(self.N_particles),
            v3 = jnp.ones(self.N_particles),
            x1=x1,
            x2=x2,
            x3=x3,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            xwind=self.x_wind,
            ywind=self.y_wind,
            zwind=self.z_wind,
            subcells=(x1, x1, x2, x2, x3, x3),
        )

        ke = total_KE([species])
        self.assertAlmostEqual(ke, 0.5 * self.mass * self.N_particles * 3)

    def test_total_momentum(self):
        """
        Test the total_momentum function.

        This test verifies that the total_momentum function correctly calculates
        the total momentum of a system of particles. It initializes the velocities
        (vx, vy, vz) of all particles to 1 and checks if the computed momentum
        matches the expected value, which is mass * N_particles * sqrt(3).

        The expected momentum is calculated based on the formula:
            momentum = mass * N_particles * sqrt(vx^2 + vy^2 + vz^2)
                     = mass * N_particles * sqrt(1^2 + 1^2 + 1^2)
                     = mass * N_particles * sqrt(3)

        Asserts:
            self.assertAlmostEqual: Checks if the calculated momentum is almost
            equal to the expected momentum value.
        """
        vx = jnp.ones(self.N_particles)
        vy = jnp.ones(self.N_particles)
        vz = jnp.ones(self.N_particles)
        momentum = total_momentum(self.mass, vx, vy, vz)
        self.assertAlmostEqual(momentum, self.mass * self.N_particles * jnp.sqrt(3))

    def test_particle_species(self):
        """
        Test the particle_species class.
        This test verifies the following functionalities of the particle_species class:
        - Initialization of particle_species with given parameters.
        - Retrieval of particle species name.
        - Retrieval of particle charge.
        - Retrieval of the number of particles.
        - Retrieval of particle velocities and positions.
        - Setting of particle velocities and positions.
        - Setting of particle mass.
        - Calculation of kinetic energy and momentum.
        The test performs the following checks:
        - The name of the species is correctly set and retrieved.
        - The charge of the species is correctly set and retrieved.
        - The number of particles is correctly set and retrieved.
        - The initial velocities and positions are correctly set and retrieved.
        - The velocities and positions can be updated and retrieved correctly.
        - The mass can be updated and retrieved correctly.
        - The kinetic energy and momentum are calculated correctly.
        """

        x1=jnp.zeros(self.N_particles)
        x2=jnp.zeros(self.N_particles)
        x3=jnp.zeros(self.N_particles)
    
        species = particle_species(
            name="test",
            N_particles=self.N_particles,
            charge=1.0,
            mass=self.mass,
            weight=1.0,
            T=self.T,
            v1 = jnp.ones(self.N_particles),
            v2 = jnp.ones(self.N_particles),
            v3 = jnp.ones(self.N_particles),
            x1=x1,
            x2=x2,
            x3=x3,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            xwind=self.x_wind,
            ywind=self.y_wind,
            zwind=self.z_wind,
            subcells=(x1, x1, x2, x2, x3, x3),
        )

        
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