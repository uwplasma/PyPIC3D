import jax.numpy as jnp
import toml
from jax import jit
from jax import tree_util
from jax.tree_util import register_pytree_node_class


def grab_laser_keys(config):
    """
    Extracts and returns a list of keys from the given configuration dictionary
    that start with the prefix 'laser'.
    Args:
        config (dict): A dictionary containing configuration keys and values.
    Returns:
        list: A list of keys from the config dictionary that start with 'laser'.
    """
    laser_keys = []
    for key in config.keys():
        if key[:5] == 'laser':
            laser_keys.append(key)
    return laser_keys


def load_lasers_from_toml(config, constants, world, grid, staggered_grid):
    """
    Load laser pulses from a TOML file.

    Parameters:
    - config (dict): Dictionary containing the configuration.

    Returns:
    - list: List of LaserPulse objects.
    """
    laser_keys = grab_laser_keys(config)
    lasers = []

    for toml_key in laser_keys:
        try:
            print(f"Loading laser: {toml_key}")
            laser_config = config[toml_key]
            laser = LaserPulse(laser_config['max_electric_field'], laser_config['k0'], laser_config['omega0'], laser_config['pulse_width'], \
                laser_config['xstart'], laser_config['ystart'], laser_config['zstart'], laser_config['width'], constants, world, grid, staggered_grid)
            lasers.append(laser)
        except Exception as e:
            print(f"Error loading laser: {toml_key}")
            print(e)

    return lasers

@register_pytree_node_class
class LaserPulse:
    def __init__(self, max_electric_field, k0, omega0, pulse_width, \
                  xstart, ystart, zstart, width, constants, world, grid, staggered_grid, points=40):
        self.max_electric_field = max_electric_field
        self.k0 = k0
        self.omega0 = omega0
        self.pulse_width = pulse_width
        self.constants = constants
        # parameters for the laser pulse

        self.points = points
        # number of points to inject the laser pulse
        dx = world['dx']
        # spatial resolution in the x-direction
        dy = world['dy']
        # spatial resolution in the y-direction
        dz = world['dz']
        # spatial resolution in the z-direction
        self.xstart = int(xstart / dx)
        self.ystart = int(ystart / dy)
        self.zstart = int(zstart / dz)
        # starting point of the laser pulse
        self.width = width
        # width of the laser pulse
        self.world = world
        self.grid = grid
        self.staggered_grid = staggered_grid
        # world, grid and staggered grid for the laser pulse

    def electric_field(self, t, x, y, z):
        """
        Calculate the electric field of a laser pulse at a given time and position.

        Parameters:
        t (float): Time at which to evaluate the electric field.
        x (float): x-coordinate at which to evaluate the electric field.
        y (float): y-coordinate at which to evaluate the electric field.
        z (float): z-coordinate at which to evaluate the electric field.

        Returns:
        float: The electric field value at the specified time and position.
        """
        weight = 1.0
        time_window = jnp.exp(-((t - self.pulse_width / 2) ** 2) / (2 * (self.pulse_width / 8) ** 2))
        y_gaussian = jnp.exp(-((y[None, :, None] - self.ystart) ** 2) / (2 * self.width ** 2))
        z_gaussian = jnp.exp(-((z[None, None, :] - self.zstart) ** 2) / (2 * self.width ** 2))
        x_sin = jnp.sin(self.k0 * x[:, None, None] - self.omega0 * t)
        return weight * time_window * y_gaussian * z_gaussian * self.max_electric_field * x_sin

    def magnetic_field(self, t, x, y, z):
        """
        Calculate the magnetic field at a given point in space and time.

        Parameters:
        t (float): Time at which to evaluate the magnetic field.
        x (float): X-coordinate of the point in space.
        y (float): Y-coordinate of the point in space.
        z (float): Z-coordinate of the point in space.

        Returns:
        float: The magnetic field at the given point in space and time.
        """
        weight = 1.0
        time_window = jnp.exp(-((t - self.pulse_width / 2) ** 2) / (2 * (self.pulse_width / 8) ** 2))
        y_gaussian = jnp.exp(-((y[None, :, None] - self.ystart) ** 2) / (2 * self.width ** 2))
        z_gaussian = jnp.exp(-((z[None, None, :] - self.zstart) ** 2) / (2 * self.width ** 2))
        x_sin = jnp.sin(self.k0 * x[:, None, None] - self.omega0 * t)
        C = self.constants['C']
        return weight * time_window * y_gaussian * z_gaussian * self.max_electric_field * x_sin / C

    
    def inject_incident_fields(self, Ex, Ey, Ez, Bx, By, Bz, t):
        """
        Injects the laser fields into the simulation grid.

        Parameters:
        -----------
        Ex : ndarray
            The x-component of the electric field.
        Ey : ndarray
            The y-component of the electric field.
        Ez : ndarray
            The z-component of the electric field.
        Bx : ndarray
            The x-component of the magnetic field.
        By : ndarray
            The y-component of the magnetic field.
        Bz : ndarray
            The z-component of the magnetic field.
        t : float
            The current time in the simulation.

        Returns:
        --------
        None
        """
        x_points = self.grid[0]
        y_points = self.grid[1]
        z_points = self.grid[2]
        # get the grid points for the electric field
        Ey = Ey.at[self.xstart:self.xstart+self.points, :, :].add(self.electric_field(t, x_points[self.xstart:self.xstart+self.points], y_points, z_points))
        # inject the electric field
        staggered_x_points = self.staggered_grid[0]
        staggered_y_points = self.staggered_grid[1]
        staggered_z_points = self.staggered_grid[2]
        # get the staggered grid points for the magnetic field
        Bz = Bz.at[self.xstart:self.xstart+self.points, :, :].add(self.magnetic_field(t, staggered_x_points[self.xstart:self.xstart+self.points], staggered_y_points, staggered_z_points))
        # inject the magnetic field
        return Ex, Ey, Ez, Bx, By, Bz
    

    # Register LaserPulse as a pytree
    def tree_flatten(laser_pulse):
        children = (
            laser_pulse.max_electric_field, laser_pulse.k0, laser_pulse.omega0, laser_pulse.pulse_width,
            laser_pulse.xstart, laser_pulse.ystart, laser_pulse.zstart, laser_pulse.width, laser_pulse.constants, laser_pulse.world, 
            laser_pulse.grid, laser_pulse.staggered_grid, laser_pulse.points
        )
        aux_data = None
        return children, aux_data
    
    @classmethod
    def tree_unflatten(aux_data, children):
        max_electric_field, k0, omega0, pulse_width, xstart, ystart, zstart, \
            width, constants, world, grid, staggered_grid, points = children
        
        return LaserPulse(
            max_electric_field=max_electric_field, 
            k0=k0, 
            omega0=omega0, 
            pulse_width=pulse_width, 
            xstart=xstart, 
            ystart=ystart, 
            zstart=zstart, 
            width=width, 
            constants=constants,
            world=world, 
            grid=grid,
            staggered_grid=staggered_grid,
            points=points
        )

# Example usage
if __name__ == "__main__":
    max_electric_field = 1e9
    # maximum electric field
    k0 = 2 * jnp.pi / 800e-9
    # wave number
    omega0 = 2 * jnp.pi * 3e8 / 800e-9
    print(omega0)
    print(k0)
    # angular frequency
    pulse_width = 10e-15
    # pulse width
    C = 3e8
    constants = {'C': C}
    # speed of light
    xstart = 0
    # x start
    ystart = 15
    # y start
    zstart = 15
    # z start
    width = 2
    # width of the laser pulse
    world = {'dx': 1, 'dy': 1, 'dz': 1, 'Nx': 32, 'Ny': 32, 'Nz': 32, 'x_wind': 32, 'y_wind': 32, 'z_wind': 32}
    # world parameters
    grid = (jnp.arange(0, 32, 1), jnp.arange(0, 32, 1), jnp.arange(0, 32, 1))
    # grid points
    staggered_grid = (jnp.arange(0.5, 32.5, 1), jnp.arange(0.5, 32.5, 1), jnp.arange(0.5, 32.5, 1))
    # staggered grid points
    laser = LaserPulse(max_electric_field, k0, omega0, pulse_width, xstart, ystart, zstart, width, constants, world, grid, staggered_grid)
    # create a laser pulse object
    Ex = jnp.zeros((32, 32, 32))
    Ey = jnp.zeros((32, 32, 32))
    Ez = jnp.zeros((32, 32, 32))
    Bx = jnp.zeros((32, 32, 32))
    By = jnp.zeros((32, 32, 32))
    Bz = jnp.zeros((32, 32, 32))
    # initialize the fields
    t = 0
    # time
    Ex, Ey, Ez, Bx, By, Bz = laser.inject_laser(Ex, Ey, Ez, Bx, By, Bz, t)
    # inject the laser pulse into the fields
    
    import matplotlib.pyplot as plt

    # Plotting function
    def plot_field_slices(field, title):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        slices = [field[16, :, :], field[:, 16, :], field[:, :, 16]]
        slice_titles = ['XY Slice', 'XZ Slice', 'YZ Slice']
        
        for ax, slice, slice_title in zip(axes, slices, slice_titles):
            cax = ax.imshow(slice, origin='lower')
            ax.set_title(slice_title)
            fig.colorbar(cax, ax=ax)
        
        fig.suptitle(title)
        plt.show()

    # Plot different slices of the electric and magnetic fields
    plot_field_slices(Ex, 'Ex Field Slices')
    plot_field_slices(Ey, 'Ey Field Slices')
    plot_field_slices(Ez, 'Ez Field Slices')
    plot_field_slices(Bx, 'Bx Field Slices')
    plot_field_slices(By, 'By Field Slices')
    plot_field_slices(Bz, 'Bz Field Slices')