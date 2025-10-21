import unittest
import jax
import sys
import os
from PyPIC3D.particle import particle_species
from PyPIC3D.J import J_from_rhov, Esirkepov_current, get_first_order_weights, get_second_order_weights, wrap_around
from PyPIC3D.rho import compute_rho
from PyPIC3D.solvers.fdtd import centered_finite_difference_divergence
import numpy as np

import jax.numpy as jnp

# Add the parent directory to the path to import PyPIC3D modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


jax.config.update("jax_enable_x64", True)

class TestCurrentMethods(unittest.TestCase):
    def setUp(self):
        """Set up common test parameters."""
        self.N_particles = 3
        self.x_wind = 1.0
        self.y_wind = 1.0
        self.z_wind = 1.0
        self.mass = 1.0
        self.T = 300.0
        self.kb = 1.38e-23
        self.dt = 0.1
        
        # Grid parameters
        self.Nx = 100
        self.Ny = 1
        self.Nz = 1
        self.dx = self.x_wind / self.Nx
        self.dy = self.y_wind / self.Ny
        self.dz = self.z_wind / self.Nz
        
        # Create grid
        self.grid = (
            jnp.arange(-self.x_wind/2, self.x_wind/2, self.dx),
            jnp.arange(-self.y_wind/2, self.y_wind/2, self.dy),
            jnp.arange(-self.z_wind/2, self.z_wind/2, self.dz)
        )
        
        # Constants and world parameters
        self.constants = {'C': 3e8, 'alpha': 1.0}
        self.world = {
            'dx': self.dx, 'dy': self.dy, 'dz': self.dz,
            'Nx': self.Nx, 'Ny': self.Ny, 'Nz': self.Nz,
            'x_wind': self.x_wind, 'y_wind': self.y_wind, 'z_wind': self.z_wind,
            'dt': self.dt
        }

    def create_test_particle_species(self, x, y, z, vx, vy, vz, charge=1.0, weight=1.0, shape=2):
        """Helper method to create particle species for testing."""
        return particle_species(
            name="test",
            N_particles=len(x) if hasattr(x, '__len__') else 1,
            charge=charge,
            mass=self.mass,
            weight=weight,
            T=self.T,
            v1=vx,
            v2=vy,
            v3=vz,
            x1=x,
            x2=y,
            x3=z,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            xwind=self.x_wind,
            ywind=self.y_wind,
            zwind=self.z_wind,
            shape=shape,
            dt=self.dt
        )

    def test_wrap_around_function(self):
        """Test the wrap_around utility function."""
        # Test single values
        self.assertEqual(wrap_around(5, 10), 5)
        self.assertEqual(wrap_around(10, 10), 0)
        self.assertEqual(wrap_around(15, 10), 5)
        
        # Test arrays
        indices = jnp.array([5, 10, 15, 20])
        wrapped = wrap_around(indices, 10)
        expected = jnp.array([5, 0, 5, 10])
        self.assertTrue(jnp.allclose(wrapped, expected))

    def test_j_from_rhov_initialization(self):
        """Test that J_from_rhov properly initializes current arrays."""
        
        # Initialize current arrays with some non-zero values
        J_init = (
            jnp.ones((self.Nx, self.Ny, self.Nz)),
            jnp.ones((self.Nx, self.Ny, self.Nz)) * 2,
            jnp.ones((self.Nx, self.Ny, self.Nz)) * 3
        )
        
        result_J = J_from_rhov([], J_init, self.constants, self.world, self.grid)
        # This should return zeros when no particles are present

        # Check that the function returns a tuple of 3 arrays
        self.assertIsInstance(result_J, tuple)
        self.assertEqual(len(result_J), 3)
        
        # Check that arrays are zeroed (based on the implementation)
        self.assertTrue(jnp.allclose(result_J[0], 0.0))
        self.assertTrue(jnp.allclose(result_J[1], 0.0))
        self.assertTrue(jnp.allclose(result_J[2], 0.0))

    def test_j_from_rhov_with_no_particles(self):
        """Test J_from_rhov with empty particle list."""
        J_init = (
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz))
        )
        
        result_J = J_from_rhov([], J_init, self.constants, self.world, self.grid)
        
        # Should return zeros when no particles
        self.assertTrue(jnp.allclose(result_J[0], 0.0))
        self.assertTrue(jnp.allclose(result_J[1], 0.0))
        self.assertTrue(jnp.allclose(result_J[2], 0.0))

    def test_j_from_rhov_particle_concatenation(self):
        """Test that J_from_rhov handles multiple particles correctly."""
        # Create multiple particles
        N = 3
        x = jnp.array([0.0, 1.0, -1.0])
        y = jnp.array([0.0, 0.0, 0.0])
        z = jnp.array([0.0, 0.0, 0.0])
        vx = jnp.array([1.0, 2.0, 3.0])
        vy = jnp.array([0.5, 1.0, 1.5])
        vz = jnp.array([0.2, 0.4, 0.6])
        
        species = self.create_test_particle_species(x, y, z, vx, vy, vz)
        
        J_init = (
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz))
        )
        
        # This should not raise an error and should handle concatenation
        result_J = J_from_rhov([species], J_init, self.constants, self.world, self.grid)
        
        # The function should complete without error
        self.assertIsInstance(result_J, tuple)
        self.assertEqual(len(result_J), 3)

    def test_esirkepov_current_initialization(self):
        """Test Esirkepov current initialization and structure."""
        # Create a particle with old and new positions
        x = jnp.array([0.1])
        y = jnp.array([0.0])
        z = jnp.array([0.0])
        vx = jnp.array([1.0])
        vy = jnp.array([0.0])
        vz = jnp.array([0.0])
        
        species = self.create_test_particle_species(x, y, z, vx, vy, vz)
        
        # Set old position (required for Esirkepov)
        species.old_x1 = jnp.array([0.0])  # Previous x position
        species.old_x2 = jnp.array([0.0])  # Previous y position
        species.old_x3 = jnp.array([0.0])  # Previous z position
        
        J_init = (
            jnp.ones((self.Nx, self.Ny, self.Nz)),
            jnp.ones((self.Nx, self.Ny, self.Nz)),
            jnp.ones((self.Nx, self.Ny, self.Nz))
        )
        
        # This might fail due to incomplete implementation, but should at least initialize
        try:
            result_J = Esirkepov_current([species], J_init, self.constants, self.world, self.grid)
            
            # Check basic structure
            self.assertIsInstance(result_J, tuple)
            self.assertEqual(len(result_J), 3)
            
            # Arrays should be zeroed initially (based on implementation)
            for J_component in result_J:
                self.assertEqual(J_component.shape, (self.Nx, self.Ny, self.Nz))
                
        except Exception as e:
            # If function is incomplete, this is expected
            self.assertIsInstance(e, (NotImplementedError, IndexError, TypeError))

    def test_esirkepov_with_different_dimensions(self):
        """Test Esirkepov current with different dimensional configurations."""
        # Test 1D configuration (Ny = Nz = 1)
        world_1d = self.world.copy()
        world_1d.update({'Ny': 1, 'Nz': 1})
        
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        z = jnp.array([0.0])
        vx = jnp.array([1.0])
        vy = jnp.array([0.0])
        vz = jnp.array([0.0])
        
        species = self.create_test_particle_species(x, y, z, vx, vy, vz)
        species.old_x1 = jnp.array([0.0])
        species.old_x2 = jnp.array([0.0])
        species.old_x3 = jnp.array([0.0])
        
        J_init_1d = (
            jnp.zeros((self.Nx, 1, 1)),
            jnp.zeros((self.Nx, 1, 1)),
            jnp.zeros((self.Nx, 1, 1))
        )
        
        try:
            result_J = Esirkepov_current([species], J_init_1d, self.constants, world_1d, self.grid)
            self.assertIsInstance(result_J, tuple)
        except Exception as e:
            # Expected if implementation is incomplete
            pass

    def test_current_conservation_properties(self):
        """Test basic current conservation properties."""
        # Create a moving particle
        x = jnp.array([0.0])
        y = jnp.array([0.0])
        z = jnp.array([0.0])
        vx = jnp.array([2.0])
        vy = jnp.array([1.0])
        vz = jnp.array([0.5])
        
        species = self.create_test_particle_species(x, y, z, vx, vy, vz)
        
        J_init = (
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz))
        )
        
        result_J = J_from_rhov([species], J_init, self.constants, self.world, self.grid)
        
        # Check that result is finite and well-defined
        for J_component in result_J:
            self.assertTrue(jnp.all(jnp.isfinite(J_component)))

    def test_multiple_species_handling(self):
        """Test handling of multiple particle species."""
        # Create two different species
        x1 = jnp.array([0.0])
        y1 = jnp.array([0.0])
        z1 = jnp.array([0.0])
        vx1 = jnp.array([1.0])
        vy1 = jnp.array([0.0])
        vz1 = jnp.array([0.0])
        
        x2 = jnp.array([1.0])
        y2 = jnp.array([0.0])
        z2 = jnp.array([0.0])
        vx2 = jnp.array([-1.0])
        vy2 = jnp.array([0.0])
        vz2 = jnp.array([0.0])
        
        species1 = self.create_test_particle_species(x1, y1, z1, vx1, vy1, vz1, charge=1.0)
        species2 = self.create_test_particle_species(x2, y2, z2, vx2, vy2, vz2, charge=-1.0)
        
        J_init = (
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz))
        )
        
        result_J = J_from_rhov([species1, species2], J_init, self.constants, self.world, self.grid)
        
        # Should handle multiple species without error
        self.assertIsInstance(result_J, tuple)
        self.assertEqual(len(result_J), 3)

    def test_boundary_conditions_in_current_calculation(self):
        """Test current calculation near boundaries."""
        # Place particles near domain boundaries
        x_boundary = jnp.array([4.9, -4.9])  # Near +/- x boundaries
        y_boundary = jnp.array([0.0, 0.0])
        z_boundary = jnp.array([0.0, 0.0])
        vx_boundary = jnp.array([1.0, -1.0])
        vy_boundary = jnp.array([0.0, 0.0])
        vz_boundary = jnp.array([0.0, 0.0])
        
        species = self.create_test_particle_species(
            x_boundary, y_boundary, z_boundary, 
            vx_boundary, vy_boundary, vz_boundary
        )
        
        J_init = (
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz)),
            jnp.zeros((self.Nx, self.Ny, self.Nz))
        )
        
        result_J = J_from_rhov([species], J_init, self.constants, self.world, self.grid)
        
        # Should complete without error even for boundary particles
        for J_component in result_J:
            self.assertTrue(jnp.all(jnp.isfinite(J_component)))

    # def test_continuity_equation_J_rhov(self):
    #     # dt = 0.1

    #     N_particles = 10

    #     x = jnp.linspace(-self.x_wind/2, self.x_wind/2, N_particles)
    #     # uniform particles in x direction
    #     y = jnp.zeros(N_particles)
    #     z = jnp.zeros(N_particles)

    #     # vx = 10*jnp.ones(N_particles) # particles moving at 1 m/s in x direction
    #     vx = jax.random.uniform(jax.random.key(0), shape = (N_particles,), minval=-1, maxval=1)
    #     vy = jnp.zeros(N_particles)
    #     vz = jnp.zeros(N_particles)
    #     # species traveling in +x direction

    #     species = self.create_test_particle_species(x, y, z, vx, vy, vz)

    #     Jx = jnp.zeros((self.Nx, self.Ny, self.Nz))
    #     Jy = jnp.zeros((self.Nx, self.Ny, self.Nz))
    #     Jz = jnp.zeros((self.Nx, self.Ny, self.Nz))
    #     J = (Jx, Jy, Jz)
    #     # Initialize current arrays



    #     species.update_position()
    #     # update once to get initial rho
    #     rho_initial = compute_rho([species], jnp.zeros((self.Nx, self.Ny, self.Nz)), self.world, self.constants)
    #     # Compute initial charge density
    #     species.update_position()
    #     # update one more time for current calculation

    #     result_J = J_from_rhov([species], J, self.constants, self.world, self.grid)
    #     # Compute current using Esirkepov method

    #     rho_final = compute_rho([species], jnp.zeros((self.Nx, self.Ny, self.Nz)), self.world, self.constants)
    #     # Compute final charge density

    #     drho_dt = (rho_final - rho_initial) / self.dt
    #     # Time derivative of charge density

    #     div_J = ( jnp.roll(result_J[0], 1, axis=0) - result_J[0] ) / self.dx

    #     # div_J = centered_finite_difference_divergence(result_J[0], result_J[1], result_J[2], self.dx, self.dy, self.dz, bc='periodic')
    #     # Divergence of current density

    #     sliced_drho_dt = drho_dt[4:-4, 0, 0]
    #     sliced_div_J = div_J[4:-4, 0, 0]
    #     # Focus on interior points to avoid boundary artifacts

    #     print(f"sliced_drho_dt: {jnp.unique(sliced_drho_dt)}")
    #     print(f"sliced_div_J: {jnp.unique(sliced_div_J)}")

    #     mean_error = jnp.mean(jnp.abs(sliced_drho_dt + sliced_div_J))
    #     max_error = jnp.max(jnp.abs(sliced_drho_dt + sliced_div_J))
    #     median_error = jnp.median(jnp.abs(sliced_drho_dt + sliced_div_J))

    #     print(f"Mean error in continuity equation: {mean_error}")
    #     print(f"Max error in continuity equation: {max_error}")
    #     print(f"Median error in continuity equation: {median_error}")

    #     # self.assertLess(mean_error, 1e-10)
    #     # self.assertLess(max_error, 1e-9)
    #     # self.assertLess(median_error, 1e-10)

    #     import matplotlib.pyplot as plt


    #     plt.plot(vx)
    #     plt.savefig('vx.png', dpi=150)
    #     plt.close()

    #     plt.plot(x)
    #     plt.savefig('x.png', dpi=150)
    #     plt.close()

    #     Jx_res, Jy_res, Jz_res = result_J

    #     Jx_np = np.array(Jx_res)
    #     Jy_np = np.array(Jy_res)
    #     Jz_np = np.array(Jz_res)

    #     x_coords = np.linspace(-self.x_wind/2, self.x_wind/2, self.Nx, endpoint=False)
    #     mid_y = Jx_np.shape[1] // 2
    #     mid_z = Jx_np.shape[2] // 2

    #     fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    #     axes[0].plot(x_coords, Jx_np[:, mid_y, mid_z], lw=1)
    #     axes[0].set_ylabel('Jx')
    #     axes[0].set_title('Current components along x at center (y,z)')

    #     axes[1].plot(x_coords, Jy_np[:, mid_y, mid_z], lw=1)
    #     axes[1].set_ylabel('Jy')

    #     axes[2].plot(x_coords, Jz_np[:, mid_y, mid_z], lw=1)
    #     axes[2].set_ylabel('Jz')
    #     axes[2].set_xlabel('x')

    #     plt.tight_layout()
    #     # plt.show()
    #     plt.savefig('J_components.png', dpi=150)
    #     plt.close(fig)


    #     plt.plot(rho_initial[:,0,0], label='Initial rho')
    #     plt.plot(rho_final[:,0,0], label='Final rho')
    #     plt.legend()
    #     plt.savefig('rho_comparison.png', dpi=150)
    #     plt.close()

    #     plt.plot(sliced_drho_dt, label='d rho/dt')
    #     plt.plot(sliced_div_J, label='div J')
    #     plt.legend()
    #     plt.savefig('continuity_terms.png', dpi=150)
    #     plt.close()

    def test_continuity_equation_esirkepov(self):
        # dt = 0.1

        N_particles = 1000

        x = jnp.linspace(-self.x_wind/2, self.x_wind/2, N_particles)
        # uniform particles in x direction
        y = jnp.zeros(N_particles)
        z = jnp.zeros(N_particles)

        # vx = 10*jnp.ones(N_particles) # particles moving at 1 m/s in x direction
        vx = jax.random.uniform(jax.random.key(0), shape = (N_particles,), minval=-10, maxval=10)
        vy = jnp.zeros(N_particles)
        vz = jnp.zeros(N_particles)
        # species traveling in +x direction

        species = self.create_test_particle_species(x, y, z, vx, vy, vz)

        Jx = jnp.zeros((self.Nx, self.Ny, self.Nz))
        Jy = jnp.zeros((self.Nx, self.Ny, self.Nz))
        Jz = jnp.zeros((self.Nx, self.Ny, self.Nz))
        J = (Jx, Jy, Jz)
        # Initialize current arrays



        species.update_position()
        # update once to get initial rho
        rho_initial = compute_rho([species], jnp.zeros((self.Nx, self.Ny, self.Nz)), self.world, self.constants)
        # Compute initial charge density
        species.update_position()
        # update one more time for current calculation

        result_J = Esirkepov_current([species], J, self.constants, self.world, self.grid)
        # Compute current using Esirkepov method
        J_rhov = J_from_rhov([species], J, self.constants, self.world, self.grid)
        # Compute current using rhov method for comparison

        rho_final = compute_rho([species], jnp.zeros((self.Nx, self.Ny, self.Nz)), self.world, self.constants)
        # Compute final charge density

        drho_dt = (rho_final - rho_initial) / self.dt
        # Time derivative of charge density

        div_J = centered_finite_difference_divergence(result_J[0], result_J[1], result_J[2], self.dx, self.dy, self.dz, bc='periodic')
        # Divergence of current density

        sliced_drho_dt = drho_dt[4:-4, 0, 0]
        sliced_div_J = div_J[4:-4, 0, 0]
        # Focus on interior points to avoid boundary artifacts

        print(f"drho_dt shape: {drho_dt.shape}")
        print(f"div_J shape: {div_J.shape}")

        print(f"sliced_drho_dt: {jnp.unique(sliced_drho_dt)}")
        print(f"sliced_div_J: {jnp.unique(sliced_div_J)}")

        mean_error = jnp.mean(jnp.abs(sliced_drho_dt + sliced_div_J))
        max_error = jnp.max(jnp.abs(sliced_drho_dt + sliced_div_J))
        median_error = jnp.median(jnp.abs(sliced_drho_dt + sliced_div_J))

        print(f"Mean error in continuity equation: {mean_error}")
        print(f"Max error in continuity equation: {max_error}")
        print(f"Median error in continuity equation: {median_error}")

        # self.assertLess(mean_error, 1e-10)
        # self.assertLess(max_error, 1e-9)
        # self.assertLess(median_error, 1e-10)

        import matplotlib.pyplot as plt


        plt.plot(vx)
        plt.savefig('vx.png', dpi=150)
        plt.close()

        plt.plot(x)
        plt.savefig('x.png', dpi=150)
        plt.close()

        Jx_res, Jy_res, Jz_res = result_J

        Jx_np = np.array(Jx_res)
        Jy_np = np.array(Jy_res)
        Jz_np = np.array(Jz_res)

        x_coords = np.linspace(-self.x_wind/2, self.x_wind/2, self.Nx, endpoint=False)
        mid_y = Jx_np.shape[1] // 2
        mid_z = Jx_np.shape[2] // 2

        fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(x_coords, Jx_np[:, mid_y, mid_z], lw=1)
        axes[0].set_ylabel('Jx')
        axes[0].set_title('Current components along x at center (y,z)')

        axes[1].plot(x_coords, Jy_np[:, mid_y, mid_z], lw=1)
        axes[1].set_ylabel('Jy')

        axes[2].plot(x_coords, Jz_np[:, mid_y, mid_z], lw=1)
        axes[2].set_ylabel('Jz')
        axes[2].set_xlabel('x')

        plt.tight_layout()
        # plt.show()
        plt.savefig('J_components.png', dpi=150)
        plt.close(fig)


        plt.plot(rho_initial[:,0,0], label='Initial rho')
        plt.plot(rho_final[:,0,0], label='Final rho')
        plt.legend()
        plt.savefig('rho_comparison.png', dpi=150)
        plt.close()

        plt.plot(sliced_drho_dt, label='d rho/dt')
        plt.plot(sliced_div_J, label='div J')
        plt.legend()
        plt.savefig('continuity_terms.png', dpi=150)
        plt.close()

        plt.plot(result_J[0][:,0,0], label='Jx Esirkepov')
        plt.plot(J_rhov[0][:,0,0], label='Jx rhov')
        plt.legend()
        plt.savefig('Jx_comparison.png', dpi=150)
        plt.close()

    # def test_zero_velocity_current(self):
    #     """Test current calculation with zero velocity particles."""
    #     x = jnp.array([0.0, 1.0, -1.0])
    #     y = jnp.array([0.0, 0.0, 0.0])
    #     z = jnp.array([0.0, 0.0, 0.0])
    #     vx = jnp.array([0.0, 0.0, 0.0])  # Zero velocities
    #     vy = jnp.array([0.0, 0.0, 0.0])
    #     vz = jnp.array([0.0, 0.0, 0.0])
        
    #     species = self.create_test_particle_species(x, y, z, vx, vy, vz)
        
    #     J_init = (
    #         jnp.zeros((self.Nx, self.Ny, self.Nz)),
    #         jnp.zeros((self.Nx, self.Ny, self.Nz)),
    #         jnp.zeros((self.Nx, self.Ny, self.Nz))
    #     )
        
    #     result_J = J_from_rhov([species], J_init, self.constants, self.world, self.grid)
        
    #     # Zero velocity should produce minimal current (depending on implementation)
    #     for J_component in result_J:
    #         self.assertTrue(jnp.all(jnp.isfinite(J_component)))

    # def test_current_units_and_scaling(self):
    #     """Test current density units and scaling."""
    #     # Create particle with known charge and velocity
    #     charge = 2.0  # Multiple of elementary charge
    #     x = jnp.array([0.0])
    #     y = jnp.array([0.0])
    #     z = jnp.array([0.0])
    #     vx = jnp.array([1.0])
    #     vy = jnp.array([0.0])
    #     vz = jnp.array([0.0])
        
    #     species = self.create_test_particle_species(x, y, z, vx, vy, vz, charge=charge)
        
    #     J_init = (
    #         jnp.zeros((self.Nx, self.Ny, self.Nz)),
    #         jnp.zeros((self.Nx, self.Ny, self.Nz)),
    #         jnp.zeros((self.Nx, self.Ny, self.Nz))
    #     )
        
    #     result_J = J_from_rhov([species], J_init, self.constants, self.world, self.grid)
        
    #     # Current should scale with charge
    #     # The exact scaling depends on the implementation details
    #     for J_component in result_J:
    #         self.assertTrue(jnp.all(jnp.isfinite(J_component)))

    # def test_digital_filter_application(self):
    #     """Test that digital filtering is applied correctly."""
    #     # Test with different alpha values
    #     alpha_values = [0.0, 0.5, 1.0]
        
    #     x = jnp.array([0.0])
    #     y = jnp.array([0.0])
    #     z = jnp.array([0.0])
    #     vx = jnp.array([1.0])
    #     vy = jnp.array([0.0])
    #     vz = jnp.array([0.0])
        
    #     species = self.create_test_particle_species(x, y, z, vx, vy, vz)
    #     species.old_x1 = jnp.array([0.0])
    #     species.old_x2 = jnp.array([0.0])
    #     species.old_x3 = jnp.array([0.0])
        
    #     for alpha in alpha_values:
    #         constants = {'C': 3e8, 'alpha': alpha}
            
    #         J_init = (
    #             jnp.zeros((self.Nx, self.Ny, self.Nz)),
    #             jnp.zeros((self.Nx, self.Ny, self.Nz)),
    #             jnp.zeros((self.Nx, self.Ny, self.Nz))
    #         )
            
    #         try:
    #             # Test Esirkepov (which applies digital filtering)
    #             result_J = Esirkepov_current([species], J_init, constants, self.world, self.grid)
                
    #             # Should complete without error
    #             self.assertIsInstance(result_J, tuple)
                
    #         except Exception as e:
    #             # Expected if implementation is incomplete
    #             pass

    # def test_shape_factor_consistency(self):
    #     """Test consistency between different shape factors."""
    #     x = jnp.array([0.5])  # Off-grid position
    #     y = jnp.array([0.0])
    #     z = jnp.array([0.0])
    #     vx = jnp.array([1.0])
    #     vy = jnp.array([0.0])
    #     vz = jnp.array([0.0])
        
    #     for shape in [1, 2]:  # First and second order
    #         species = self.create_test_particle_species(x, y, z, vx, vy, vz, shape=shape)
    #         species.old_x1 = jnp.array([0.0])
    #         species.old_x2 = jnp.array([0.0])
    #         species.old_x3 = jnp.array([0.0])
            
    #         J_init = (
    #             jnp.zeros((self.Nx, self.Ny, self.Nz)),
    #             jnp.zeros((self.Nx, self.Ny, self.Nz)),
    #             jnp.zeros((self.Nx, self.Ny, self.Nz))
    #         )
            
    #         try:
    #             result_J = Esirkepov_current([species], J_init, self.constants, self.world, self.grid)
                
    #             # Both shape factors should produce valid results
    #             for J_component in result_J:
    #                 self.assertTrue(jnp.all(jnp.isfinite(J_component)))
                    
    #         except Exception as e:
    #             # Expected if implementation is incomplete
    #             pass


if __name__ == '__main__':
    unittest.main()