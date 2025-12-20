import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax
# import external libraries

from PyPIC3D.utils import digital_filter, wrap_around

@jit
def J_from_rhov(particles, J, constants, world, grid):
    """
    Compute the current density from the charge density and particle velocities.

    Args:
        particles (list): List of particle species, each with methods to get charge, subcell position, resolution, and index.
        rho (ndarray): Charge density array.
        J (tuple): Current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
        constants (dict): Dictionary containing physical constants.

    Returns:
        tuple: Updated current density arrays (Jx, Jy, Jz) for the x, y, and z directions respectively.
    """

    C = constants['C']
    # speed of light

    dx = world['dx']
    dy = world['dy']
    dz = world['dz']
    Nx = world['Nx']
    Ny = world['Ny']
    Nz = world['Nz']
    # get the world parameters

    Jx, Jy, Jz = J
    # unpack the values of J
    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)
    # initialize the current arrays as 0
    J = (Jx, Jy, Jz)
    # initialize the current density as a tuple

    if particles:
        # if there are particles in the simulation

        x_mid = [ (species.get_old_position()[0] + species.get_forward_position()[0]) / 2 for species in particles]
        y_mid = [ (species.get_old_position()[1] + species.get_forward_position()[1]) / 2 for species in particles]
        z_mid = [ (species.get_old_position()[2] + species.get_forward_position()[2]) / 2 for species in particles]
        # # midpoint rule to get x_t+1/2 position
        # # v is already at t+1/2 from the Boris push
        # yields J at t+1/2
        total_x = jnp.concatenate( x_mid )
        total_y = jnp.concatenate( y_mid )
        total_z = jnp.concatenate( z_mid )
        # use the mid-point position for current deposition

        total_dqvx = jnp.concatenate( [species.get_charge() / (dx*dy*dz) * species.get_velocity()[0] for species in particles] )
        total_dqvy = jnp.concatenate( [species.get_charge() / (dx*dy*dz) * species.get_velocity()[1] for species in particles] )
        total_dqvz = jnp.concatenate( [species.get_charge() / (dx*dy*dz) * species.get_velocity()[2] for species in particles] )
        # concatenate all the particle data for easier processing

        shape_factor = particles[0].get_shape()
        # assume all species have the same shape factor

        x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (total_x - grid[0][0]) / dx).astype(int),
            lambda _: jnp.round( (total_x - grid[0][0]) / dx).astype(int),
            operand=None
        )

        y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (total_y - grid[1][0]) / dy).astype(int),
            lambda _: jnp.round( (total_y - grid[1][0]) / dy).astype(int),
            operand=None
        )

        z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (total_z - grid[2][0]) / dz).astype(int),
            lambda _: jnp.round( (total_z - grid[2][0]) / dz).astype(int),
            operand=None
        )
        # calculate the nearest grid point based on shape factor


        deltax = (total_x - grid[0][0]) - (x0 * dx)
        deltay = (total_y - grid[1][0]) - (y0 * dy)
        deltaz = (total_z - grid[2][0]) - (z0 * dz)
        # Calculate the difference between the particle position and the nearest grid point

        x0 = wrap_around(x0, Nx)
        y0 = wrap_around(y0, Ny)
        z0 = wrap_around(z0, Nz)
        # wrap around the grid points for periodic boundary conditions

        x1 = wrap_around(x0+1, Nx)
        y1 = wrap_around(y0+1, Ny)
        z1 = wrap_around(z0+1, Nz)
        # calculate the right grid point

        x_minus1 = x0 - 1
        y_minus1 = y0 - 1
        z_minus1 = z0 - 1
        # calculate the left grid point

        xpts = [x_minus1, x0, x1]
        ypts = [y_minus1, y0, y1]
        zpts = [z_minus1, z0, z1]
        # place all the points in a list

        x_weights, y_weights, z_weights = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights( deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None
        )

        for i in range(len(x_weights)):
            for j in range(len(y_weights)):
                for k in range(len(z_weights)):
                    Jx = Jx.at[xpts[i], ypts[j], zpts[k]].add( (total_dqvx) * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
                    Jy = Jy.at[xpts[i], ypts[j], zpts[k]].add( (total_dqvy) * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
                    Jz = Jz.at[xpts[i], ypts[j], zpts[k]].add( (total_dqvz) * x_weights[i] * y_weights[j] * z_weights[k], mode='drop')
        # Add the particle current to the current density arrays

        
        alpha = constants['alpha']
        Jx = digital_filter(Jx, alpha)
        Jy = digital_filter(Jy, alpha)
        Jz = digital_filter(Jz, alpha)
        J = (Jx, Jy, Jz)
        # apply a digital filter to the current density arrays

    return J


def _unwrapped_cell_index(x, xmin, dx, shape_factor):
    # matches your existing convention: linear -> floor, quadratic -> round
    return jax.lax.cond(
        shape_factor == 1,
        lambda _: jnp.floor((x - xmin) / dx).astype(jnp.int32),
        lambda _: jnp.round((x - xmin) / dx).astype(jnp.int32),
        operand=None,
    )

def _roll_old_weights_to_new_frame(old_w_list, shift):
    """
    old_w_list: list of 5 arrays, each (Np,)
    shift: (Np,) integer = old_i0 - new_i0 (expected in {-1,0,1} for Esirkepov)
    Returns a list of 5 arrays rolled per particle so old weights align with new-cell frame.
    """
    old_w = jnp.stack(old_w_list, axis=0)  # (5, Np)

    def roll_one_particle(w5, s):
        return jnp.roll(w5, -s, axis=0)

    rolled = jax.vmap(roll_one_particle, in_axes=(1, 0), out_axes=1)(old_w, shift)  # (5,Np)
    return [rolled[i, :] for i in range(5)]

def _stencil_points(i0, N):
    """
    i0: (Np,) unwrapped base index at new position
    Returns pts_u: (S,Np) unwrapped and pts: (S,Np) wrapped, where S=5 for your weights.
    """
    offsets = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.int32)[:, None]  # (5,1)
    pts_u = i0[None, :] + offsets                                     # (5,Np)
    pts = wrap_around(pts_u, N) if N != 1 else jnp.zeros_like(pts_u)
    return pts_u, pts

def pad_to_5(w3):
    # w3 is [w(-1), w(0), w(+1)] in your convention
    z = jnp.zeros_like(w3[0])
    return [z, w3[0], w3[1], w3[2], z]


def Esirkepov_current(particles, J, constants, world, grid):
    """
    Local per-particle Esirkepov deposition that works for 1D/2D/3D by setting inactive dims to size 1.
    J is a tuple (Jx,Jy,Jz) arrays shaped (Nx,Ny,Nz).
    """
    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    dx, dy, dz, dt = world["dx"], world["dy"], world["dz"], world["dt"]
    xmin, ymin, zmin = grid[0][0], grid[1][0], grid[2][0]

    xmin = xmin - dx/2
    ymin = ymin - dy/2
    zmin = zmin - dz/2
    # adjust grid minimums for staggered grid

    # zero current arrays
    Jx = Jx.at[:, :, :].set(0)
    Jy = Jy.at[:, :, :].set(0)
    Jz = Jz.at[:, :, :].set(0)

    x_active = (Nx != 1)
    y_active = (Ny != 1)
    z_active = (Nz != 1)
    # determine which axis are null

    for species in particles:
        q = species.get_charge()
        old_x, old_y, old_z = species.get_old_position()
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        shape_factor = species.get_shape()
        N_particles = species.get_number_of_particles()
        # get the particle properties
        
        x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (x - xmin) / dx).astype(int),
            lambda _: jnp.round( (x - xmin) / dx).astype(int),
            operand=None
        )
        y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (y - ymin) / dy).astype(int),
            lambda _: jnp.round( (y - ymin) / dy).astype(int),
            operand=None
        )
        z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (z - zmin) / dz).astype(int),
            lambda _: jnp.round( (z - zmin) / dz).astype(int),
            operand=None
        ) # calculate the nearest grid point based on shape factor for new positions

        old_x0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_x - xmin) / dx).astype(int),
            lambda _: jnp.round( (old_x - xmin) / dx).astype(int),
            operand=None
        )
        old_y0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_y - ymin) / dy).astype(int),
            lambda _: jnp.round( (old_y - ymin) / dy).astype(int),
            operand=None
        )
        old_z0 = jax.lax.cond(
            shape_factor == 1,
            lambda _: jnp.floor( (old_z - zmin) / dz).astype(int),
            lambda _: jnp.round( (old_z - zmin) / dz).astype(int),
            operand=None
        ) # calculate the nearest grid point based on shape factor for old positions

        deltax = (x - xmin) - x0 * dx
        deltay = (y - ymin) - y0 * dy
        deltaz = (z - zmin) - z0 * dz
        # get the difference between the particle position and the nearest grid point
        old_deltax = (old_x - xmin) - old_x0 * dx
        old_deltay = (old_y - ymin) - old_y0 * dy
        old_deltaz = (old_z - zmin) - old_z0 * dz
        # get the difference between the particle position and the nearest grid point

        x0 = wrap_around(x0, Nx)
        y0 = wrap_around(y0, Ny)
        z0 = wrap_around(z0, Nz)
        # wrap around the grid points for periodic boundary conditions
        x1 = wrap_around(x0+1, Nx)
        y1 = wrap_around(y0+1, Ny)
        z1 = wrap_around(z0+1, Nz)
        # calculate the right grid point
        x2 = wrap_around(x0+2, Nx)
        y2 = wrap_around(y0+2, Ny)
        z2 = wrap_around(z0+2, Nz)
        # calculate the second right grid point
        x_minus1 = x0 - 1
        y_minus1 = y0 - 1
        z_minus1 = z0 - 1
        # calculate the left grid point
        x_minus2 = x0 - 2
        y_minus2 = y0 - 2
        z_minus2 = z0 - 2
        # calculate the second left grid point

        xpts = [x_minus2, x_minus1, x0, x1, x2]
        ypts = [y_minus2, y_minus1, y0, y1, y2]
        zpts = [z_minus2, z_minus1, z0, z1, z2]
        # place all the points in a list

        xw, yw, zw = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None,
        )
        # get the weights for the new positions
        oxw, oyw, ozw = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            operand=None,
        ) # get the weights for the old positions

        tmp = jnp.zeros_like(xw[0])
        # build the temporary zero array for padding

        xw = [tmp, xw[0], xw[1], xw[2], tmp]
        yw = [tmp, yw[0], yw[1], yw[2], tmp]
        zw = [tmp, zw[0], zw[1], zw[2], tmp]
        # pad the weights to 5 points for consistency

        oxw = [tmp, oxw[0], oxw[1], oxw[2], tmp]
        oyw = [tmp, oyw[0], oyw[1], oyw[2], tmp]
        ozw = [tmp, ozw[0], ozw[1], ozw[2], tmp]
        # pad the old weights to 5 points for consistency

        # align old weights into new-cell frame (roll by shift = old_i0 - new_i0)
        shift_x = ((old_x0 - x0 + Nx//2) % Nx) - Nx//2
        shift_y = ((old_y0 - y0 + Ny//2) % Ny) - Ny//2
        shift_z = ((old_z0 - z0 + Nz//2) % Nz) - Nz//2
        oxw = _roll_old_weights_to_new_frame(oxw, shift_x)
        oyw = _roll_old_weights_to_new_frame(oyw, shift_y)
        ozw = _roll_old_weights_to_new_frame(ozw, shift_z)

        # --- build Esirkepov W on compact stencil ---
        if x_active and y_active and z_active:
            Wx_, Wy_, Wz_ = get_3D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles)
        elif (x_active and y_active and (not z_active)) or (x_active and z_active and (not y_active)) or (y_active and z_active and (not x_active)):
            null_dim = lax.cond(
                not x_active,
                lambda _: 0,
                lambda _: lax.cond(
                    not y_active,
                    lambda _: 1,
                    lambda _: 2,
                    operand=None,
                ),
                operand=None,
            )
            # determine which dimension is inactive

            Wx_, Wy_, Wz_ = get_2D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, null_dim=null_dim)
        elif x_active and (not y_active) and (not z_active):
            # 1D in x: Esirkepov reduces to 1D continuity;
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=0)
        elif y_active and (not x_active) and (not z_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=1)
        elif z_active and (not x_active) and (not y_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, N_particles, dim=2)

        dJx = jax.lax.cond(
            x_active,
            lambda _: (q / (dy * dz)) / dt * jnp.ones(N_particles),
            lambda _: q * vx / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        dJy = jax.lax.cond(
            y_active,
            lambda _: (q / (dx * dz)) / dt * jnp.ones(N_particles),
            lambda _: q * vy / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        dJz = jax.lax.cond(
            z_active,
            lambda _: (q / (dx * dy)) / dt * jnp.ones(N_particles),
            lambda _: q * vz / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )
        # calculate prefactors for current deposition

        # local “difference RHS”
        Fx = dJx * Wx_   # (Sx,Sy,Sz,Np)
        Fy = dJy * Wy_
        Fz = dJz * Wz_

        Jx_loc = jnp.zeros_like(Fx)
        Jy_loc = jnp.zeros_like(Fy)
        Jz_loc = jnp.zeros_like(Fz)

        # Using Backward Finite Difference approach for prefix sum #################################
        # Jx currents
        Jx_loc = jnp.cumsum(Fx, axis=0)
        Jx_loc = Jx_loc.at[4, :, :, :].set(0)
        # J(5) = 0 for Esirkepov because the sum of the differences of the stencil weights over all cells is 1 - 1 = 0
        # Jy currents
        Jy_loc = jnp.cumsum(Fy, axis=1)
        Jy_loc = Jy_loc.at[:, 4, :, :].set(0)
        # J(5) = 0 for Esirkepov because the sum of the differences of the stencil weights over all cells is 1 - 1 = 0
        # Jz currents
        Jz_loc = jnp.cumsum(Fz, axis=2)
        Jz_loc = Jz_loc.at[:, :, 4, :].set(0)
        # J(5) = 0 for Esirkepov because the sum of the differences of the stencil weights over all cells is 1 - 1 = 0
        # This assumes 5 cells in each dimension for the stencil, but 6 faces (so 5 differences).
        # This should give periodic wrap around J(1) = J(6) = 0 as required.
        ################################################################################################
        if x_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jx = Jx.at[xpts[i], ypts[j], zpts[k]].add(Jx_loc[i, j, k, :], mode="drop")
                        # deposit Jx using Esirkepov weights
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jx = Jx.at[xpts[i], ypts[j], zpts[k]].add(Fx[i, j, k, :], mode="drop")
                        # deposit Jx using midpoint weights for inactive dimension

        if y_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jy = Jy.at[xpts[i], ypts[j], zpts[k]].add(Jy_loc[i, j, k, :], mode="drop")
                        # deposit Jy using Esirkepov weights
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jy = Jy.at[xpts[i], ypts[j], zpts[k]].add(Fy[i, j, k, :], mode="drop")
                        # deposit Jy using midpoint weights for inactive dimension

        if z_active:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jz = Jz.at[xpts[i], ypts[j], zpts[k]].add(Jz_loc[i, j, k, :], mode="drop")
                        # deposit Jz using Esirkepov weights
        
        else:
            for i in range(5):
                for j in range(5):
                    for k in range(5):
                        Jz = Jz.at[xpts[i], ypts[j], zpts[k]].add(Fz[i, j, k, :], mode="drop")
                        # deposit Jz using midpoint weights for inactive dimension


    return (Jx, Jy, Jz)
        
        
def get_3D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=None):

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros_like( Wx_)
    Wz_ = jnp.zeros_like( Wx_)


    for i in range(len(x_weights)):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[i,j,k,:].set( (x_weights[i] - old_x_weights[i]) * ( 1/3 * (y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k])     \
                                                    +  1/6 * (y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k]) ) )

                Wy_ = Wy_.at[i,j,k,:].set( (y_weights[j] - old_y_weights[j]) * ( 1/3 * (x_weights[i] * z_weights[k] + old_x_weights[i] * old_z_weights[k])     \
                                                    +  1/6 * (x_weights[i] * old_z_weights[k] + old_x_weights[i] * z_weights[k]) ) )

                Wz_ = Wz_.at[i,j,k,:].set( (z_weights[k] - old_z_weights[k]) * ( 1/3 * (x_weights[i] * y_weights[j] + old_x_weights[i] * old_y_weights[j])     \
                                                    +  1/6 * (x_weights[i] * old_y_weights[j] + old_x_weights[i] * y_weights[j]) ) )

    return Wx_, Wy_, Wz_

def get_2D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, null_dim=2):
    d_Sx = []
    d_Sy = []
    d_Sz = []

    d_Sx = [ x_weights[i] - old_x_weights[i] for i in range(len(x_weights)) ]
    d_Sy = [ y_weights[i] - old_y_weights[i] for i in range(len(y_weights)) ]
    d_Sz = [ z_weights[i] - old_z_weights[i] for i in range(len(z_weights)) ]

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros_like( Wx_)
    Wz_ = jnp.zeros_like( Wx_)
    # initialize the weight arrays

    # XY Plane
    def xy_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            for j in range(len(y_weights)):
                Wx_ = Wx_.at[i,j,2,:].set( 1/2 * d_Sx[i] * ( y_weights[j] + old_y_weights[j] ) )
                Wy_ = Wy_.at[i,j,2,:].set( 1/2 * d_Sy[j] * ( x_weights[i] + old_x_weights[i] ) )
                Wz_ = Wz_.at[i,j,2,:].set( 1/3 * ( x_weights[i] * y_weights[j] + old_x_weights[i] * old_y_weights[j] )     \
                                        +  1/6 * ( x_weights[i] * old_y_weights[j] + old_x_weights[i] * y_weights[j] ) )
        # Weights if the 2D plane is in the XY plane

        return Wx_, Wy_, Wz_
    

    # XZ Plane
    def xz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[i,2,k,:].set( 1/2 * d_Sx[i] * ( z_weights[k] + old_z_weights[k] ) )
                Wy_ = Wy_.at[i,2,k,:].set( 1/3 * ( x_weights[i] * z_weights[k] + old_x_weights[i] * old_z_weights[k] )     \
                                        +  1/6 * ( x_weights[i] * old_z_weights[k] + old_x_weights[i] * z_weights[k] ) )
                Wz_ = Wz_.at[i,2,k,:].set( 1/2 * d_Sz[k] * ( x_weights[i] + old_x_weights[i] ) )
        # Weights if the 2D plane is in the XZ plane
        return Wx_, Wy_, Wz_
    

    # YZ Plane
    def yz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for j in range(len(y_weights)):
            for k in range(len(z_weights)):
                Wx_ = Wx_.at[2,j,k,:].set( 1/3 * ( y_weights[j] * z_weights[k] + old_y_weights[j] * old_z_weights[k] )     \
                                        +  1/6 * ( y_weights[j] * old_z_weights[k] + old_y_weights[j] * z_weights[k] ) )
                Wy_ = Wy_.at[2,j,k,:].set( 1/2 * d_Sy[j] * ( z_weights[k] + old_z_weights[k] ) )
                Wz_ = Wz_.at[2,j,k,:].set( 1/2 * d_Sz[k] * ( y_weights[j] + old_y_weights[j] ) )
        # Weights if the 2D plane is in the YZ plane
        return Wx_, Wy_, Wz_
    

    Wx_, Wy_, Wz_ = lax.cond(
        null_dim == 0,
        lambda _: yz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
        lambda _: lax.cond(
            null_dim == 1,
            lambda _: xz_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            lambda _: xy_plane(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            operand=None
        ),
        operand=None
    )

    return Wx_, Wy_, Wz_


def get_1D_esirkepov_weights(x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights, N_particles, dim=0):

    Wx_ = jnp.zeros( (len(x_weights),len(y_weights),len(z_weights), N_particles) )
    Wy_ = jnp.zeros_like( Wx_)
    Wz_ = jnp.zeros_like( Wx_)
    # initialize the weight arrays

    def x_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for i in range(len(x_weights)):
            Wx_ = Wx_.at[i, 2, 2, :].set( (x_weights[i] - old_x_weights[i]) )
            # get the weights for x direction
            Wy_ = Wy_.at[:, 2, 2, :].set( (x_weights[i] + old_x_weights[i]) / 2 )
            Wz_ = Wz_.at[:, 2, 2, :].set( (x_weights[i] + old_x_weights[i]) / 2 )
            # use a midpoint average for inactive directions
        # weights if x direction is active
        return Wx_, Wy_, Wz_

    def y_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for j in range(len(y_weights)):
            Wy_ = Wy_.at[2, j, 2, :].set( (y_weights[j] - old_y_weights[j]) )
            # weights for y direction
            Wx_ = Wx_.at[2, :, 2, :].set( (y_weights[j] + old_y_weights[j]) / 2 )
            Wz_ = Wz_.at[2, :, 2, :].set( (y_weights[j] + old_y_weights[j]) / 2 )
            # use a midpoint average for inactive directions
        # weights if y direction is active
        return Wx_, Wy_, Wz_
    
    def z_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for k in range(len(z_weights)):
            Wz_ = Wz_.at[2, 2, k, :].set( (z_weights[k] - old_z_weights[k]) )
            # weights for z direction
            Wx_ = Wx_.at[2, 2, :, :].set( (z_weights[k] + old_z_weights[k]) / 2 )
            Wy_ = Wy_.at[2, 2, :, :].set( (z_weights[k] + old_z_weights[k]) / 2 )
            # use a midpoint average for inactive directions
        # weights if z direction is active
        return Wx_, Wy_, Wz_
    
    Wx_, Wy_, Wz_ = lax.cond(
        dim == 0,
        lambda _: x_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
        lambda _: lax.cond(
            dim == 1,
            lambda _: y_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            lambda _: z_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights),
            operand=None,
        ),
        operand=None,
    )
    # determine which dimension is active and calculate weights accordingly


    return Wx_, Wy_, Wz_

@jit
def get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """
    Calculate the second-order weights for particle current distribution.

    Args:
        deltax, deltay, deltaz (float): Particle position offsets from grid points.
        dx, dy, dz (float): Grid spacings in x, y, and z directions.

    Returns:
        tuple: Weights for x, y, and z directions.
    """
    Sx0 = (3/4) - (deltax/dx)**2
    Sy0 = (3/4) - (deltay/dy)**2
    Sz0 = (3/4) - (deltaz/dz)**2

    Sx1 = (1/2) * ((1/2) - (deltax/dx))**2
    Sy1 = (1/2) * ((1/2) - (deltay/dy))**2
    Sz1 = (1/2) * ((1/2) - (deltaz/dz))**2

    Sx_minus1 = (1/2) * ((1/2) + (deltax/dx))**2
    Sy_minus1 = (1/2) * ((1/2) + (deltay/dy))**2
    Sz_minus1 = (1/2) * ((1/2) + (deltaz/dz))**2
    # second order weights

    x_weights = [Sx_minus1, Sx0, Sx1]
    y_weights = [Sy_minus1, Sy0, Sy1]
    z_weights = [Sz_minus1, Sz0, Sz1]

    return x_weights, y_weights, z_weights

@jit
def get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz):
    """
    Calculate the first-order weights for particle current distribution.

    Args:
        deltax, deltay, deltaz (float): Particle position offsets from grid points.
        dx, dy, dz (float): Grid spacings in x, y, and z directions.

    Returns:
        tuple: Weights for x, y, and z directions.
    """
    Sx0 = jnp.asarray(1 - deltax / dx)
    Sy0 = jnp.asarray(1 - deltay / dy)
    Sz0 = jnp.asarray(1 - deltaz / dz)

    Sx1 = jnp.asarray(deltax / dx)
    Sy1 = jnp.asarray(deltay / dy)
    Sz1 = jnp.asarray(deltaz / dz)

    Sx_minus1 = jnp.zeros_like(Sx0)
    Sy_minus1 = jnp.zeros_like(Sy0)
    Sz_minus1 = jnp.zeros_like(Sz0)
    # No second-order weights for first-order weighting

    x_weights = [Sx_minus1, Sx0, Sx1]
    y_weights = [Sy_minus1, Sy0, Sy1]
    z_weights = [Sz_minus1, Sz0, Sz1]

    return x_weights, y_weights, z_weights