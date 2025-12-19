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

        total_x = jnp.concatenate( [species.get_position()[0] for species in particles] )
        total_y = jnp.concatenate( [species.get_position()[1] for species in particles] )
        total_z = jnp.concatenate( [species.get_position()[2] for species in particles] )

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

    for species in particles:
        q = species.get_charge()

        old_x, old_y, old_z = species.get_old_position()
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        shape_factor = species.get_shape()

        Np = x.shape[0]

        # --- unwrapped cell indices at new and old positions ---
        x0 = _unwrapped_cell_index(x, xmin, dx, shape_factor)
        y0 = _unwrapped_cell_index(y, ymin, dy, shape_factor)
        z0 = _unwrapped_cell_index(z, zmin, dz, shape_factor)

        old_x0 = _unwrapped_cell_index(old_x, xmin, dx, shape_factor)
        old_y0 = _unwrapped_cell_index(old_y, ymin, dy, shape_factor)
        old_z0 = _unwrapped_cell_index(old_z, zmin, dz, shape_factor)

        # --- subcell offsets (must use UNWRAPPED index) ---
        deltax = (x - xmin) - x0 * dx
        deltay = (y - ymin) - y0 * dy
        deltaz = (z - zmin) - z0 * dz

        old_deltax = (old_x - xmin) - old_x0 * dx
        old_deltay = (old_y - ymin) - old_y0 * dy
        old_deltaz = (old_z - zmin) - old_z0 * dz

        # --- shape weights at old and new positions ---
        xw, yw, zw = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(deltax, deltay, deltaz, dx, dy, dz),
            operand=None,
        )
        oxw, oyw, ozw = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            lambda _: get_second_order_weights(old_deltax, old_deltay, old_deltaz, dx, dy, dz),
            operand=None,
        )

        if len(xw) == 3:
            xw, yw, zw = pad_to_5(xw), pad_to_5(yw), pad_to_5(zw)
            oxw, oyw, ozw = pad_to_5(oxw), pad_to_5(oyw), pad_to_5(ozw)


        # align old weights into new-cell frame (roll by shift = old_i0 - new_i0)
        shift_x = ((old_x0 - x0 + Nx//2) % Nx) - Nx//2
        shift_y = ((old_y0 - y0 + Ny//2) % Ny) - Ny//2
        shift_z = ((old_z0 - z0 + Nz//2) % Nz) - Nz//2
        oxw = _roll_old_weights_to_new_frame(oxw, shift_x) if x_active else oxw
        oyw = _roll_old_weights_to_new_frame(oyw, shift_y) if y_active else oyw
        ozw = _roll_old_weights_to_new_frame(ozw, shift_z) if z_active else ozw

        # --- build Esirkepov W on compact stencil ---
        if x_active and y_active and z_active:
            Wx_, Wy_, Wz_ = get_3D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, Np)
        elif (x_active and y_active and (not z_active)) or (x_active and z_active and (not y_active)) or (y_active and z_active and (not x_active)):
            # Your helper handles XY / XZ / YZ planes based on which N==1.
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

            Wx_, Wy_, Wz_ = get_2D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, Np, null_dim=null_dim)
        elif x_active and (not y_active) and (not z_active):
            # 1D in x: Esirkepov reduces to 1D continuity;
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, Np, dim=0)
        elif y_active and (not x_active) and (not z_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, Np, dim=1)
        elif z_active and (not x_active) and (not y_active):
            Wx_, Wy_, Wz_ = get_1D_esirkepov_weights(xw, yw, zw, oxw, oyw, ozw, Np, dim=2)

        dJx =  q / (dy * dz) / dt * jnp.ones(Np)
        dJy =  q / (dx * dz) / dt * jnp.ones(Np)
        dJz =  q / (dx * dy) / dt * jnp.ones(Np)
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

        # --- stencil points (unwrapped for order, wrapped for scatter) ---
        _, xpts = _stencil_points(x0, Nx)
        _, ypts = _stencil_points(y0, Ny)
        _, zpts = _stencil_points(z0, Nz)

        # # --- scatter compact stencil directly into J ---
        # # Jx_loc[i,j,k,p] adds to cell (xpts[i,p], ypts[j,p], zpts[k,p])
        Sx, Sy, Sz = Jx_loc.shape[0], Jx_loc.shape[1], Jx_loc.shape[2]  # typically 5,5,5 (or 5,5,1 etc)

        if x_active:
            for i in range(Sx):
                for j in range(Sy):
                    for k in range(Sz):
                        xi = xpts[i, :]
                        yj = ypts[j, :]
                        zk = zpts[k, :]

                        Jx = Jx.at[xi, yj, zk].add(Jx_loc[i, j, k, :], mode="drop")

        if y_active:
            for i in range(Sx):
                for j in range(Sy):
                    for k in range(Sz):
                        xi = xpts[i, :]
                        yj = ypts[j, :]
                        zk = zpts[k, :]

                        Jy = Jy.at[xi, yj, zk].add(Jy_loc[i, j, k, :], mode="drop")

        if z_active:
            for i in range(Sx):
                for j in range(Sy):
                    for k in range(Sz):
                        xi = xpts[i, :]
                        yj = ypts[j, :]
                        zk = zpts[k, :]

                        Jz = Jz.at[xi, yj, zk].add(Jz_loc[i, j, k, :], mode="drop")

    not_3D = (not x_active) or (not y_active) or (not z_active)
    # check if simulation is not 3D

    if not_3D:
        J_rhov = J_from_rhov(particles, J, constants, world, grid)
        # calculate J from rho*v for inactive dimensions
        Jx = J_rhov[0] if not x_active else Jx
        Jy = J_rhov[1] if not y_active else Jy
        Jz = J_rhov[2] if not z_active else Jz

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
        # weights if x direction is active
        return Wx_, Wy_, Wz_

    def y_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for j in range(len(y_weights)):
            Wy_ = Wy_.at[2, j, 2, :].set( (y_weights[j] - old_y_weights[j]) )
        # weights if y direction is active
        return Wx_, Wy_, Wz_
    
    def z_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for k in range(len(z_weights)):
            Wz_ = Wz_.at[2, 2, k, :].set( (z_weights[k] - old_z_weights[k]) )
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