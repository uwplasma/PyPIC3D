import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from jax import lax
# import external libraries

from PyPIC3D.utils import digital_filter, wrap_around, bilinear_filter
from PyPIC3D.shapes import get_first_order_weights, get_second_order_weights

@partial(jit, static_argnames=("filter",))
def J_from_rhov(particles, J, constants, world, grid, filter='bilinear'):
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

    for species in particles:
        shape_factor = species.get_shape()
        charge = species.get_charge()
        dq = charge / (dx * dy * dz)
        # calculate the charge density contribution per particle
        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
        # get the particles positions and velocities

        x = x - vx * world['dt'] / 2
        y = y - vy * world['dt'] / 2
        z = z - vz * world['dt'] / 2
        # step back to half time step positions for proper time staggering

        x0 = jnp.floor( (x - grid[0][0]) / dx).astype(int)
        y0 = jnp.floor( (y - grid[1][0]) / dy).astype(int)
        z0 = jnp.floor( (z - grid[2][0]) / dz).astype(int)
        # calculate the nearest grid point based on shape factor

        deltax_node = (x - grid[0][0]) - (x0 * dx)
        deltay_node = (y - grid[1][0]) - (y0 * dy)
        deltaz_node = (z - grid[2][0]) - (z0 * dz)
        # Calculate the difference between the particle position and the nearest grid point

        deltax_face = (x - grid[0][0]) - (x0 + 0.5) * dx
        deltay_face = (y - grid[1][0]) - (y0 + 0.5) * dy
        deltaz_face = (z - grid[2][0]) - (z0 + 0.5) * dz
        # Calculate the difference between the particle position and the nearest staggered cell face

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

        x_weights_node, y_weights_node, z_weights_node = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights( deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_node, deltay_node, deltaz_node, dx, dy, dz),
            operand=None
        )

        x_weights_face, y_weights_face, z_weights_face = jax.lax.cond(
            shape_factor == 1,
            lambda _: get_first_order_weights( deltax_face, deltay_face, deltaz_face, dx, dy, dz),
            lambda _: get_second_order_weights(deltax_face, deltay_face, deltaz_face, dx, dy, dz),
            operand=None
        )
        # get the weights for node and face positions

        xpts_ = jnp.stack(xpts, axis=0)  # (Sx, Np)
        ypts_ = jnp.stack(ypts, axis=0)  # (Sy, Np)
        zpts_ = jnp.stack(zpts, axis=0)  # (Sz, Np)
        # stack the point indices for easier indexing
        x_weights_face_ = jnp.stack(x_weights_face, axis=0)  # (Sx, Np)
        y_weights_face_ = jnp.stack(y_weights_face, axis=0)  # (Sy, Np)
        z_weights_face_ = jnp.stack(z_weights_face, axis=0)  # (Sz, Np)
        # stack the face weights for easier indexing
        x_weights_node_ = jnp.stack(x_weights_node, axis=0)  # (Sx, Np)
        y_weights_node_ = jnp.stack(y_weights_node, axis=0)  # (Sy, Np)
        z_weights_node_ = jnp.stack(z_weights_node, axis=0)  # (Sz, Np)
        # stack the node weights for easier indexing

        n_Sx, n_Sy, n_Sz = xpts_.shape[0], ypts_.shape[0], zpts_.shape[0]
        # get the stencil sizes
        ii, jj, kk = jnp.meshgrid(jnp.arange(n_Sx), jnp.arange(n_Sy), jnp.arange(n_Sz), indexing="ij")
        # create a meshgrid of stencil indices
        combos = jnp.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1)  # (M, 3)
        # create all combinations of stencil indices

        def idx_and_dJ_values(idx):
            i, j, k = idx
            # unpack the stencil indices
            ix = xpts_[i]
            iy = ypts_[j]
            iz = zpts_[k]
            # get the grid indices for this stencil point
            valx = (dq * vx) * x_weights_face_[i] * y_weights_node_[j] * z_weights_node_[k]
            valy = (dq * vy) * x_weights_node_[i] * y_weights_face_[j] * z_weights_node_[k]
            valz = (dq * vz) * x_weights_node_[i] * y_weights_node_[j] * z_weights_face_[k]
            # calculate the current contributions for this stencil point
            return ix, iy, iz, valx, valy, valz
        
        ix, iy, iz, valx, valy, valz = jax.vmap(idx_and_dJ_values)(combos)  # each: (M, Np)
        # vectorized computation of indices and current contributions

        Jx = Jx.at[(ix, iy, iz)].add(valx, mode="drop")
        Jy = Jy.at[(ix, iy, iz)].add(valy, mode="drop")
        Jz = Jz.at[(ix, iy, iz)].add(valz, mode="drop")
        # deposit the current contributions into the global J arrays

    def filter_func(J_, filter):
        J_ = jax.lax.cond(
            filter == 'bilinear',
            lambda J_: bilinear_filter(J_),
            lambda J_: J_,
            operand=J_
        )

        alpha = constants['alpha']
        J_ = jax.lax.cond(
            filter == 'digital',
            lambda J_: digital_filter(J_, alpha),
            lambda J_: J_,
            operand=J_
        )
        return J_
    # define a filtering function

    Jx = filter_func(Jx, filter)
    Jy = filter_func(Jy, filter)
    Jz = filter_func(Jz, filter)
    # apply the selected filter to each component of J
    J = (Jx, Jy, Jz)

    return J

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


def Esirkepov_current(particles, J, constants, world, grid, filter=None):
    """
    Local per-particle Esirkepov deposition that works for 1D/2D/3D by setting inactive dims to size 1.
    J is a tuple (Jx,Jy,Jz) arrays shaped (Nx,Ny,Nz).
    """
    Jx, Jy, Jz = J
    Nx, Ny, Nz = Jx.shape
    dx, dy, dz, dt = world["dx"], world["dy"], world["dz"], world["dt"]
    xmin, ymin, zmin = grid[0][0], grid[1][0], grid[2][0]
 
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
        x, y, z = species.get_forward_position()
        vx, vy, vz = species.get_velocity()
        shape_factor = species.get_shape()
        N_particles = species.get_number_of_particles()
        # get the particle properties

        old_x = x - vx * dt
        old_y = y - vy * dt
        old_z = z - vz * dt
        # calculate old positions from new positions and velocities
        
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

        shift_x = x0 - old_x0
        shift_y = y0 - old_y0
        shift_z = z0 - old_z0
        # calculate the shift between old and new grid points

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
        x_minus1 = wrap_around(x0 - 1, Nx)
        y_minus1 = wrap_around(y0 - 1, Ny)
        z_minus1 = wrap_around(z0 - 1, Nz)
        # calculate the left grid point
        x_minus2 = wrap_around(x0 - 2, Nx)
        y_minus2 = wrap_around(y0 - 2, Ny)
        z_minus2 = wrap_around(z0 - 2, Nz)
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
            lambda _: -(q / (dy * dz)) / dt * jnp.ones(N_particles),
            lambda _: q * vx / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        dJy = jax.lax.cond(
            y_active,
            lambda _: -(q / (dx * dz)) / dt * jnp.ones(N_particles),
            lambda _: q * vy / (dx * dy * dz) * jnp.ones(N_particles),
            operand=None,
        )

        dJz = jax.lax.cond(
            z_active,
            lambda _: -(q / (dx * dy)) / dt * jnp.ones(N_particles),
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
        # Jy currents
        Jy_loc = jnp.cumsum(Fy, axis=1)
        # Jz currents
        Jz_loc = jnp.cumsum(Fz, axis=2)
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
            Wy_ = Wy_.at[i, 2, 2, :].set( (x_weights[i] + old_x_weights[i]) / 2 )
            Wz_ = Wz_.at[i, 2, 2, :].set( (x_weights[i] + old_x_weights[i]) / 2 )
            # use a midpoint average for inactive directions
        # weights if x direction is active
        return Wx_, Wy_, Wz_

    def y_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for j in range(len(y_weights)):
            Wy_ = Wy_.at[2, j, 2, :].set( (y_weights[j] - old_y_weights[j]) )
            # weights for y direction
            Wx_ = Wx_.at[2, j, 2, :].set( (y_weights[j] + old_y_weights[j]) / 2 )
            Wz_ = Wz_.at[2, j, 2, :].set( (y_weights[j] + old_y_weights[j]) / 2 )
            # use a midpoint average for inactive directions
        # weights if y direction is active
        return Wx_, Wy_, Wz_
    
    def z_active(Wx_, Wy_, Wz_, x_weights, y_weights, z_weights, old_x_weights, old_y_weights, old_z_weights):
        for k in range(len(z_weights)):
            Wz_ = Wz_.at[2, 2, k, :].set( (z_weights[k] - old_z_weights[k]) )
            # weights for z direction
            Wx_ = Wx_.at[2, 2, k, :].set( (z_weights[k] + old_z_weights[k]) / 2 )
            Wy_ = Wy_.at[2, 2, k, :].set( (z_weights[k] + old_z_weights[k]) / 2 )
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