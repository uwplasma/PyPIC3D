import unittest
import jax
import jax.numpy as jnp
import sys
import os


from PyPIC3D.particle import (
    initial_particles, particle_species
)

from PyPIC3D.J import J_from_rhov

from PyPIC3D.rho import compute_rho

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
            N_per_cell=0.1,
            N_particles=self.N_particles,
            minx=-self.x_wind/2,
            maxx=self.x_wind/2,
            miny=-self.y_wind/2,
            maxy=self.y_wind/2,
            minz=-self.z_wind/2,
            maxz=self.z_wind/2,
            mass=self.mass,
            Tx=self.T,
            Ty=self.T,
            Tz=self.T,
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

        species.periodic_boundary_condition()
        x, y, z = species.get_position()
        self.assertTrue(jnp.all(x == jnp.array([-4.5, 4.0, 0.0])))
        self.assertTrue(jnp.all(y == jnp.array([-4.5, 4.0, 0.0])))
        self.assertTrue(jnp.all(z == jnp.array([-4.5, 4.0, 0.0])))

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
            dt=self.dt
        )

        species.update_position()
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

        ke = species.kinetic_energy()
        self.assertAlmostEqual(ke, 0.5 * self.mass * self.N_particles * 3)

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

    def test_J_from_rhov(self):
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        z = jnp.array([0.0])
        vx = jnp.array([0.5])
        vy = jnp.array([0.5])
        vz = jnp.array([0.5])
        # define particle position and velocity

        dx = self.x_wind / 10
        dy = self.y_wind / 10
        dz = self.z_wind / 10
        # uniform spatial resolution in xyz

        grid = jnp.arange(-self.x_wind/2, self.x_wind/2, dx), jnp.arange(-self.y_wind/2, self.y_wind/2, dy), jnp.arange(-self.z_wind/2, self.z_wind/2, dz)
        # grid for the simulation
        
        num_J = (jnp.zeros((10,10,10)), jnp.zeros((10,10,10)), jnp.zeros((10,10,10)))
        # numerical current density arrays

        Jx = jnp.zeros((10,10,10))
        Jx = Jx.at[5,5,5].set(0.5 / (dx*dy*dz))  # non zero value at the center of the grid
        Jy = jnp.zeros((10,10,10))
        Jy = Jy.at[5,5,5].set(0.5 / (dx*dy*dz))  # non zero value at the center of the grid
        Jz = jnp.zeros((10,10,10))
        Jz = Jz.at[5,5,5].set(0.5 / (dx*dy*dz))  # non zero value at the center of the grid
        J_exp = (Jx, Jy, Jz)
        # build expected J arrays with non-zero values at the center of the grid

        species = particle_species(
            name="test",
            N_particles=1,
            charge=1.0,
            mass=self.mass,
            weight=1.0,
            T=self.T,
            v1 = vx,
            v2 = vy,
            v3 = vz,
            x1=x,
            x2=y,
            x3=z,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            xwind=self.x_wind,
            ywind=self.y_wind,
            zwind=self.z_wind,
            subcells=(x, x, y, y, z, z),
        )

        constants = {'C': 3e8, 'alpha' : 1.0}
        world = {'dx': dx, 'dy': dy, 'dz': dz, 'Nx': 10, 'Ny': 10, 'Nz': 10}
        # define constants and world parameters

        num_J = J_from_rhov([species], num_J, constants, world, grid)

        self.assertTrue(jnp.allclose(num_J[0], J_exp[0]))
        self.assertTrue(jnp.allclose(num_J[1], J_exp[1]))
        self.assertTrue(jnp.allclose(num_J[2], J_exp[2]))

    def test_rho(self):
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        z = jnp.array([0.0])
        vx = jnp.array([0.5])
        vy = jnp.array([0.5])
        vz = jnp.array([0.5])
        # define particle position and velocity

        dx = self.x_wind / 10
        dy = self.y_wind / 10
        dz = self.z_wind / 10
        # uniform spatial resolution in xyz

        grid = jnp.arange(-self.x_wind/2, self.x_wind/2, dx), jnp.arange(-self.y_wind/2, self.y_wind/2, dy), jnp.arange(-self.z_wind/2, self.z_wind/2, dz)
        # grid for the simulation

        num_rho = jnp.zeros((10,10,10))
        # numerical charge density array

        exp_rho = jnp.zeros((10,10,10))
        exp_rho = exp_rho.at[5,5,5].set(1.0 / (dx*dy*dz))  # non zero value at the center of the grid
        # build expected rho array with non-zero values at the center of the grid

        species = particle_species(
            name="test",
            N_particles=1,
            charge=1.0,
            mass=self.mass,
            weight=1.0,
            T=self.T,
            v1 = vx,
            v2 = vy,
            v3 = vz,
            x1=x,
            x2=y,
            x3=z,
            dx=1.0,
            dy=1.0,
            dz=1.0,
            xwind=self.x_wind,
            ywind=self.y_wind,
            zwind=self.z_wind,
            subcells=(x, x, y, y, z, z),
        )

        constants = {'C': 3e8, 'alpha' : 1.0}
        world = {'dx': dx, 'dy': dy, 'dz': dz, 'Nx': 10, 'Ny': 10, 'Nz': 10, 'x_wind': self.x_wind, 'y_wind': self.y_wind, 'z_wind': self.z_wind}
        # define constants and world parameters

        num_rho = compute_rho([species], num_rho, world, constants)
        # compute rho

        self.assertTrue(jnp.allclose(num_rho, exp_rho))
        # check if computed rho matches expected rho

if __name__ == '__main__':
    unittest.main()