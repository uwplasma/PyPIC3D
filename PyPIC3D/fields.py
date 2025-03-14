import jax
from jax import jit
from jax import lax
import functools
from functools import partial
#from memory_profiler import profile
# import external libraries

from PyPIC3D.pstd import spectral_poisson_solve, spectral_gradient
from PyPIC3D.fdtd import centered_finite_difference_gradient
from PyPIC3D.rho import compute_rho
from PyPIC3D.sor import solve_poisson_sor
# import internal libraries

def initialize_fields(Nx, Ny, Nz):
    """
    Initializes the electric and magnetic field arrays, as well as the electric potential and charge density arrays.

    Args:
        Nx (int): Number of grid points in the x-direction.
        Ny (int): Number of grid points in the y-direction.
        Nz (int): Number of grid points in the z-direction.

    Returns:
        Ex (ndarray): Electric field array in the x-direction.
        Ey (ndarray): Electric field array in the y-direction.
        Ez (ndarray): Electric field array in the z-direction.
        Bx (ndarray): Magnetic field array in the x-direction.
        By (ndarray): Magnetic field array in the y-direction.
        Bz (ndarray): Magnetic field array in the z-direction.
        phi (ndarray): Electric potential array.
        rho (ndarray): Charge density array.
    """
    # get the number of grid points in each direction
    Ex = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ey = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Ez = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric field arrays as 0
    Bx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    By = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Bz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the magnetic field arrays as 0

    Jx = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Jy = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    Jz = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the current density arrays as 0

    phi = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    rho = jax.numpy.zeros(shape = (Nx, Ny, Nz) )
    # initialize the electric potential and charge density arrays as 0

    return Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, phi, rho

@partial(jit, static_argnums=(4, 5))
def solve_poisson(rho, constants, world, phi, solver, bc='periodic', M = None):
    """
    Solve the Poisson equation for electrostatic potential.

    Args:
        rho (ndarray): Charge density.
        eps (float): Permittivity.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.
        phi (ndarray): Initial guess for the electrostatic potential.
        bc (str): Boundary condition.
        M (ndarray, optional): Preconditioner matrix for the conjugate gradient solver.

    Returns:
        phi (ndarray): Solution to the Poisson equation.
    """

    if solver == 'spectral':
        phi = spectral_poisson_solve(rho, constants, world)
    elif solver == 'fdtd':
        eps = constants['eps']
        dx = world['dx']
        dy = world['dy']
        dz = world['dz']
        # lapl = functools.partial(centered_finite_difference_laplacian, dx=dx, dy=dy, dz=dz, bc=bc)
        # lapl = jit(lapl)
        # #define the laplacian operator using finite difference method
        # #phi = conjugate_grad(lapl, -rho/eps, phi, tol=1e-9, maxiter=5000, M=M)

        # phi = conjugated_gradients(lapl, -rho/eps, phi, tol=1e-9, maxiter=1000)
        sor = functools.partial(solve_poisson_sor, dx=dx, dy=dy, dz=dz, eps=eps, omega=0.15, tol=1e-12, max_iter=15000)
        phi = sor(phi, rho)
        #phi = jax.scipy.sparse.linalg.cg(lapl, -rho/eps, x0=phi, tol=1e-6, maxiter=40000, M=M)[0]
        #phi = solve_poisson_sor(phi, rho, dx, dy, dz, eps, omega=0.25, tol=1e-6, max_iter=100000)
    return phi

#@profile
@partial(jit, static_argnums=(6, 7))
#@jit
def calculateE(world, particles, constants, rho, phi, M, solver, bc):
    """
    Calculate the electric field components (Ex, Ey, Ez) and electric potential (phi)
    based on the given parameters.

    Args:
        E (tuple): Tuple containing the electric field components (Ex, Ey, Ez).
        world (dict): Dictionary containing the simulation world parameters such as
                    grid spacing (dx, dy, dz) and window dimensions (x_wind, y_wind, z_wind).
        particles (array): Array containing particle positions and properties.
        constants (dict): Dictionary containing physical constants such as permittivity (eps).
        rho (array): Charge density array.
        phi (array): Electric potential array.
        M (int): Parameter for the solver.
        solver (str): Type of solver to use ('spectral' or other).
        bc (str): Boundary condition type.
        verbose (bool): Flag to enable verbose output.

    Returns:
        tuple: Updated electric field components (Ex, Ey, Ez), electric potential (phi), 
            and charge density (rho).
    """


    dx = world['dx']
    dy = world['dy']
    dz = world['dz']

    rho = compute_rho(particles, rho, world)
    # calculate the charge density based on the particle positions

    #if_verbose_print(verbose, f"Calculating Charge Density, Max Value: {jnp.max(jnp.abs(rho))}" )

    phi = phi.at[:,:,:].set(solve_poisson(rho, constants, world, phi=phi, solver=solver, bc=bc, M=M))
    # solve the Poisson equation to get the electric potential

    #if_verbose_print(verbose, f"Calculating Electric Potential, Max Value: {jnp.max(phi)}",  )
    #if_verbose_print(verbose, f"Potential Error: {compute_pe(phi, rho, constants, world, solver, bc='periodic')}%", )

    Ex, Ey, Ez = lax.cond(
        solver == 'spectral',
        lambda _: spectral_gradient(phi, world),
        lambda _: centered_finite_difference_gradient(phi, dx, dy, dz, bc),
        operand=None
    )

    # compute the gradient of the electric potential to get the electric field
    Ex = -Ex
    Ey = -Ey
    Ez = -Ez
    # multiply by -1 to get the correct direction of the electric field

    return (Ex, Ey, Ez), phi, rho


@partial(jit, static_argnums=(7))
def update_E(grid, staggered_grid, E, B, J, world, constants, curl_func):
    """
    Update the electric field components (Ex, Ey, Ez) based on the given parameters.

    Args:
        grid (object): The grid object containing the simulation grid.
        staggered_grid (object): The staggered grid object for the simulation.
        E (tuple): A tuple containing the electric field components (Ex, Ey, Ez).
        B (tuple): A tuple containing the magnetic field components (Bx, By, Bz).
        J (tuple): A tuple containing the current density components (Jx, Jy, Jz).
        world (dict): A dictionary containing the world parameters such as 'dx', 'dy', 'dz', and 'dt'.
        constants (dict): A dictionary containing the physical constants such as 'C' (speed of light) and 'eps' (permittivity).
        curl_func (function): A function to calculate the curl of the magnetic field.

    Returns:
        tuple: Updated electric field components (Ex, Ey, Ez).
    """

    Ex, Ey, Ez = E
    Bx, By, Bz = B
    Jx, Jy, Jz = J

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    dt = world['dt']
    C = constants['C']
    eps = constants['eps']
    mu = constants['mu']

    curlx, curly, curlz = curl_func(Bx, By, Bz)
    # calculate the curl of the magnetic field

    # Ex = Ex.at[:,:,:].add( ( C**2 * curlx - Jx / eps ) * dt )
    # Ey = Ey.at[:,:,:].add( ( C**2 * curly - Jy / eps ) * dt )
    # Ez = Ez.at[:,:,:].add( ( C**2 * curlz - Jz / eps ) * dt )
    # jax.debug.print("Mean Curl B Magnitude: {}", jax.numpy.mean(jax.numpy.sqrt(curlx**2 + curly**2 + curlz**2)))
    # jax.debug.print("Mean Current Density Magnitude: {}", jax.numpy.mean(jax.numpy.sqrt(Jx**2 + Jy**2 + Jz**2)))

    # jax.debug.print("Speed of light**2 (C): {}", C**2)
    # jax.debug.print("Permittivity (eps): {}", eps)
    # jax.debug.print("curl x: {}", jax.numpy.mean( jax.numpy.abs(curlx)))
    # jax.debug.print("dt factor: {}", dt)

    # jax.debug.print("Jx: {}", jax.numpy.mean( jax.numpy.abs(Jx)))
    # jax.debug.print("Jx/eps: {}", jax.numpy.mean( jax.numpy.abs(Jx/eps)))

    # jax.debug.print("E update factor1: {}", jax.numpy.mean( jax.numpy.abs( ( C**2 * curlx ) * dt) ) )
    # jax.debug.print("E update factor2: {}", jax.numpy.mean( jax.numpy.abs( ( Jx / eps ) * dt) ) )
    # jax.debug.print("E update factor: {}", jax.numpy.mean( jax.numpy.abs( ( C**2 * curlx - Jx / eps ) * dt) ) )

    Ex = Ex + ( C**2 * curlx - Jx / eps ) * dt
    Ey = Ey + ( C**2 * curly - Jy / eps ) * dt
    Ez = Ez + ( C**2 * curlz - Jz / eps ) * dt

    return (Ex, Ey, Ez)


@partial(jit, static_argnums=(6))
def update_B(grid, staggered_grid, E, B, world, constants, curl_func):
    """
    Update the magnetic field components (Bx, By, Bz) using the curl of the electric field.

    Args:
        grid (ndarray): The grid on which the fields are defined.
        staggered_grid (ndarray): The staggered grid for field calculations.
        E (tuple): The electric field components (Ex, Ey, Ez).
        B (tuple): The magnetic field components (Bx, By, Bz).
        world (dict): Dictionary containing simulation parameters such as 'dx', 'dy', 'dz', and 'dt'.
        constants (dict): Dictionary containing physical constants.
        curl_func (function): Function to calculate the curl of the electric field.

    Returns:
        tuple: Updated magnetic field components (Bx, By, Bz).
    """

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    dt = world['dt']

    Ex, Ey, Ez = E
    Bx, By, Bz = B

    curlx, curly, curlz = curl_func(Ex, Ey, Ez)
    # calculate the curl of the electric field

    # Bx = Bx.at[:,:,:].add(-1*dt*curlx)
    # By = By.at[:,:,:].add(-1*dt*curly)
    # Bz = Bz.at[:,:,:].add(-1*dt*curlz)

    # curl_magnitude = jax.numpy.sqrt(curlx**2 + curly**2 + curlz**2)
    # jax.debug.print("Mean Curl E Magnitude: {}", jax.numpy.mean(curl_magnitude))

    # jax.debug.print("B update factor: {}", jax.numpy.mean( jax.numpy.abs(-1*dt*curlx)) )

    Bx = Bx - dt*curlx
    By = By - dt*curly
    Bz = Bz - dt*curlz

    return (Bx, By, Bz)