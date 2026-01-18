import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import plotly.graph_objects as go
import jax
from functools import partial

def plot_positions(particles, t, x_wind, y_wind, z_wind, path):
    """
    Makes an interactive 3D plot of the positions of the particles using Plotly.

    Args:
        particles (list): A list of ParticleSpecies objects containing positions.
        t (float): The time value.
        x_wind (float): The x-axis wind limit.
        y_wind (float): The y-axis wind limit.
        z_wind (float): The z-axis wind limit.

    Returns:
        None
    """
    fig = go.Figure()

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan']
    for idx, species in enumerate(particles):
        x, y, z = species.get_position()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=2, color=colors[idx % len(colors)]),
            name=f'Species {idx + 1}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-(2/3)*x_wind, (2/3)*x_wind]),
            yaxis=dict(range=[-(2/3)*y_wind, (2/3)*y_wind]),
            zaxis=dict(range=[-(2/3)*z_wind, (2/3)*z_wind]),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        title="Particle Positions"
    )

    if not os.path.exists(f"{path}/data/positions"):
        os.makedirs(f"{path}/data/positions")

    fig.write_html(f"{path}/data/positions/particles.{t:09}.html")

def write_particles_phase_space(particles, t, path):
    """
    Write the phase space of the particles to a file.

    Args:
        particles (Particles): The particles to be written.
        t (ndarray): The time values.
        name (str): The name of the plot.

    Returns:
        None
    """
    if not os.path.exists(f"{path}/data/phase_space/x"):
        os.makedirs(f"{path}/data/phase_space/x")
    if not os.path.exists(f"{path}/data/phase_space/y"):
        os.makedirs(f"{path}/data/phase_space/y")
    if not os.path.exists(f"{path}/data/phase_space/z"):
        os.makedirs(f"{path}/data/phase_space/z")
    # Create directory if it doesn't exist

    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        name = species.get_name().replace(" ", "")

        x_phase_space = jnp.stack((x, vx), axis=-1)
        y_phase_space = jnp.stack((y, vy), axis=-1)
        z_phase_space = jnp.stack((z, vz), axis=-1)

        jnp.save(f"{path}/data/phase_space/x/{name}_phase_space.{t:09}.npy", x_phase_space)
        jnp.save(f"{path}/data/phase_space/y/{name}_phase_space.{t:09}.npy", y_phase_space)
        jnp.save(f"{path}/data/phase_space/z/{name}_phase_space.{t:09}.npy", z_phase_space)
    # write the phase space of the particles to a file

def particles_phase_space(particles, world, t, name, path):
    """
    Plot the phase space of the particles.

    Args:
        particles (Particles): The particles to be plotted.
        t (ndarray): The time values.
        name (str): The name of the plot.

    Returns:
        None
    """

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    idx = 0
    order = 10
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(x, vx, c=colors[idx], zorder=order, s=1)
        idx += 1
        order -= 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    #plt.ylim(-1e10, 1e10)
    plt.xlim(-(2/3)*x_wind, (2/3)*x_wind)
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/x/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(y, vy, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.xlim(-(2/3)*y_wind, (2/3)*y_wind)
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/y/{name}_phase_space.{t:09}.png", dpi=150)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.get_position()
        vx, vy, vz = species.get_velocity()
        plt.scatter(z, vz, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.xlim(-(2/3)*z_wind, (2/3)*z_wind)
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/z/{name}_phase_space.{t:09}.png", dpi=150)
    plt.close()


def plot_initial_histograms(particle_species, world, name, path):
    """
    Generates and saves histograms for the initial positions and velocities 
    of particles in a simulation.

    Parameters:
        particle_species (object): An object representing the particle species, 
                                   which provides methods `get_position()` and 
                                   `get_velocity()` to retrieve particle positions 
                                   (x, y, z) and velocities (vx, vy, vz).
        world (dict): A dictionary containing the simulation world parameters, 
                      specifically the wind dimensions with keys 'x_wind', 
                      'y_wind', and 'z_wind'.
        name (str): A string representing the name of the particle species or 
                    simulation, used in the titles of the histograms and filenames.
        path (str): The directory path where the histogram images will be saved.

    Saves:
        Six histogram images:
            - Initial X, Y, and Z position histograms.
            - Initial X, Y, and Z velocity histograms.
        The images are saved in the specified `path` directory with filenames 
        formatted as "{name}_initial_<property>_histogram.png".
    """

    x, y, z = particle_species.get_position()
    vx, vy, vz = particle_species.get_velocity()

    x_wind = world['x_wind']
    y_wind = world['y_wind']
    z_wind = world['z_wind']


    plt.hist(x, bins=50)
    plt.xlabel("X")
    plt.ylabel("Number of Particles")
    plt.xlim(-(2/3)*x_wind, (2/3)*x_wind)
    plt.title(f"{name} Initial X Position Histogram")
    plt.savefig(f"{path}/{name}_initial_x_histogram.png", dpi=150)
    plt.close()

    plt.hist(y, bins=50)
    plt.xlabel("Y")
    plt.ylabel("Number of Particles")
    plt.xlim(-(2/3)*y_wind, (2/3)*y_wind)
    plt.title(f"{name} Initial Y Position Histogram")
    plt.savefig(f"{path}/{name}_initial_y_histogram.png", dpi=150)
    plt.close()

    plt.hist(z, bins=50)
    plt.xlabel("Z")
    plt.ylabel("Number of Particles")
    plt.xlim(-(2/3)*z_wind, (2/3)*z_wind)
    plt.title(f"{name} Initial Z Position Histogram")
    plt.savefig(f"{path}/{name}_initial_z_histogram.png", dpi=150)
    plt.close()

    plt.hist(vx, bins=50)
    plt.xlabel("X Velocity")
    plt.ylabel("Number of Particles")
    plt.title(f"{name} Initial X Velocity Histogram")
    plt.savefig(f"{path}/{name}_initial_x_velocity_histogram.png", dpi=150)
    plt.close()

    plt.hist(vy, bins=50)
    plt.xlabel("Y Velocity")
    plt.ylabel("Number of Particles")
    plt.title(f"{name} Initial Y Velocity Histogram")
    plt.savefig(f"{path}/{name}_initial_y_velocity_histogram.png", dpi=150)
    plt.close()

    plt.hist(vz, bins=50)
    plt.xlabel("Z Velocity")
    plt.ylabel("Number of Particles")
    plt.title(f"{name} Initial Z Velocity Histogram")
    plt.savefig(f"{path}/{name}_initial_z_velocity_histogram.png", dpi=150)
    plt.close()


@partial(jax.jit, static_argnums=(0))
def write_data(filename, time, data):
    """
    Write the given time and data to a file using JAX's callback mechanism.
    This function is designed to be used with JAX's just-in-time compilation (jit) to optimize performance.

    Args:
        filename (str): The name of the file to write to.
        time (float): The time value to write.
        data (any): The data to write.

    Returns:
        None
    """

    def write_to_file(filename, time, data):
        with open(filename, "a") as f:
            f.write(f"{time}, {data}\n")

    return jax.debug.callback(write_to_file, filename, time, data, ordered=True)

