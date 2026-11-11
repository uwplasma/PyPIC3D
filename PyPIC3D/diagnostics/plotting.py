import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
import jax
import numpy as np

from PyPIC3D.diagnostics.output_adapters import particles_for_output
from PyPIC3D.utilities.grids import grid_axis_width


def _expanded_axis_range(center_axis):
    lower, upper = center_axis[1], center_axis[-1]
    width = grid_axis_width(center_axis)
    center = 0.5 * (lower + upper)
    return [float(center - (2 / 3) * width), float(center + (2 / 3) * width)]

def plot_positions(particles, t, static_parameters, dynamic_parameters, path, species_config=None, species_names=None):
    """
    Makes an interactive 3D plot of the positions of the particles using Plotly.

    Args:
        particles (list): A list of ParticleSpecies objects containing positions.
        t (float): The time value.
        static_parameters (dict): Compile-time/run parameters.
        dynamic_parameters (dict): Scalar/grid parameters.

    Returns:
        None
    """
    fig = go.Figure()

    center_grid = dynamic_parameters.grids.center

    particles = particles_for_output(
        particles,
        species_config=species_config,
        species_names=species_names,
        static_parameters=static_parameters,
        dynamic_parameters=dynamic_parameters,
    )
    # Keep tiled fixed-capacity storage out of the plotting loop.  From here on
    # the code sees the same species-like diagnostic objects as ordinary output.

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan']
    for idx, species in enumerate(particles):
        x, y, z = species.x_diagnostic[:, 0], species.x_diagnostic[:, 1], species.x_diagnostic[:, 2]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=2, color=colors[idx % len(colors)]),
            name=species.name
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=_expanded_axis_range(center_grid[0])),
            yaxis=dict(range=_expanded_axis_range(center_grid[1])),
            zaxis=dict(range=_expanded_axis_range(center_grid[2])),
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        title="Particle Positions"
    )

    if not os.path.exists(f"{path}/data/positions"):
        os.makedirs(f"{path}/data/positions")

    fig.write_html(f"{path}/data/positions/particles.{t:09}.html")

    # write the phase space of the particles to a file

def particles_phase_space(particles, static_parameters, dynamic_parameters, t, name, path, species_config=None, species_names=None):
    """
    Plot the phase space of the particles.

    Args:
        particles (Particles): The particles to be plotted.
        t (ndarray): The time values.
        name (str): The name of the plot.

    Returns:
        None
    """

    center_grid = dynamic_parameters.grids.center

    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    idx = 0
    order = 10
    particles = particles_for_output(
        particles,
        species_config=species_config,
        species_names=species_names,
        static_parameters=static_parameters,
        dynamic_parameters=dynamic_parameters,
    )
    # Phase-space plots are an output boundary; tile-major storage is flattened
    # once here rather than leaking into the plotting loops.

    for species in particles:
        x, y, z = species.x_diagnostic[:, 0], species.x_diagnostic[:, 1], species.x_diagnostic[:, 2]
        vx, vy, vz = species.u[:, 0], species.u[:, 1], species.u[:, 2]
        plt.scatter(x, vx, c=colors[idx], zorder=order, s=1)
        idx += 1
        order -= 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    #plt.ylim(-1e10, 1e10)
    plt.xlim(*_expanded_axis_range(center_grid[0]))
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/x/{name}_phase_space.{t:09}.png", dpi=300)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.x_diagnostic[:, 0], species.x_diagnostic[:, 1], species.x_diagnostic[:, 2]
        vx, vy, vz = species.u[:, 0], species.u[:, 1], species.u[:, 2]
        plt.scatter(y, vy, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.xlim(*_expanded_axis_range(center_grid[1]))
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/y/{name}_phase_space.{t:09}.png", dpi=150)
    plt.close()

    idx = 0
    for species in particles:
        x, y, z = species.x_diagnostic[:, 0], species.x_diagnostic[:, 1], species.x_diagnostic[:, 2]
        vx, vy, vz = species.u[:, 0], species.u[:, 1], species.u[:, 2]
        plt.scatter(z, vz, c=colors[idx])
        idx += 1
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.xlim(*_expanded_axis_range(center_grid[2]))
    plt.title(f"{name} Phase Space")
    plt.savefig(f"{path}/data/phase_space/z/{name}_phase_space.{t:09}.png", dpi=150)
    plt.close()


def plot_initial_histograms(particle_record, dynamic_parameters, name, path):
    """
    Generates and saves histograms for the initial positions and velocities 
    of particles in a simulation.

    Parameters:
        particle_record (ParticleOutputRecord): Flattened tiled particle output
                                   with diagnostic positions and velocities.
        dynamic_parameters (dict): Scalar/grid parameters with wind dimensions.
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

    x, y, z = particle_record.x_diagnostic[:, 0], particle_record.x_diagnostic[:, 1], particle_record.x_diagnostic[:, 2]
    vx, vy, vz = particle_record.u[:, 0], particle_record.u[:, 1], particle_record.u[:, 2]

    center_grid = dynamic_parameters.grids.center


    plt.hist(x, bins=50)
    plt.xlabel("X")
    plt.ylabel("Number of Particles")
    plt.xlim(*_expanded_axis_range(center_grid[0]))
    plt.title(f"{name} Initial X Position Histogram")
    plt.savefig(f"{path}/{name}_initial_x_histogram.png", dpi=150)
    plt.close()

    plt.hist(y, bins=50)
    plt.xlabel("Y")
    plt.ylabel("Number of Particles")
    plt.xlim(*_expanded_axis_range(center_grid[1]))
    plt.title(f"{name} Initial Y Position Histogram")
    plt.savefig(f"{path}/{name}_initial_y_histogram.png", dpi=150)
    plt.close()

    plt.hist(z, bins=50)
    plt.xlabel("Z")
    plt.ylabel("Number of Particles")
    plt.xlim(*_expanded_axis_range(center_grid[2]))
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


def initial_tiled_histogram_data(particles):
    """Reduce active, addressable particle shards to the six initial histograms."""
    n_species = particles.active.shape[3]
    lower = np.full((n_species, 6), np.inf)
    upper = np.full((n_species, 6), -np.inf)
    counts = np.zeros((n_species, 6, 50), dtype=np.int64)
    edges = np.empty((n_species, 6, 51))

    # Find common bin ranges before counting each shard. Inactive capacity does
    # not contribute, including uninitialized values in inactive particle slots.
    for x_shard, u_shard, active_shard in zip(
            particles.x.addressable_shards, particles.u.addressable_shards,
            particles.active.addressable_shards):
        x, u, active = (np.asarray(jax.device_get(shard.data))
                        for shard in (x_shard, u_shard, active_shard))
        first_species = active_shard.index[3].start or 0
        for local_species in range(active.shape[3]):
            species = first_species + local_species
            mask = active[..., local_species, :]
            if not np.any(mask):
                continue
            for offset, values in ((0, x), (3, u)):
                selected = values[..., local_species, :, :][mask]
                lower[species, offset:offset+3] = np.minimum(
                    lower[species, offset:offset+3], selected.min(axis=0))
                upper[species, offset:offset+3] = np.maximum(
                    upper[species, offset:offset+3], selected.max(axis=0))

    # NumPy's ordinary 50-bin convention uses [0, 1] for empty input and
    # expands constant data by half a unit on either side.
    for species in range(n_species):
        for component in range(6):
            if lower[species, component] == np.inf:
                lower[species, component], upper[species, component] = 0.0, 1.0
            dtype = particles.x.dtype if component < 3 else particles.u.dtype
            limits = np.asarray([lower[species, component], upper[species, component]], dtype=dtype)
            edges[species, component] = np.histogram_bin_edges(limits, bins=50)

    for x_shard, u_shard, active_shard in zip(
            particles.x.addressable_shards, particles.u.addressable_shards,
            particles.active.addressable_shards):
        x, u, active = (np.asarray(jax.device_get(shard.data))
                        for shard in (x_shard, u_shard, active_shard))
        first_species = active_shard.index[3].start or 0
        for local_species in range(active.shape[3]):
            species = first_species + local_species
            mask = active[..., local_species, :]
            for offset, values in ((0, x), (3, u)):
                selected = values[..., local_species, :, :][mask]
                for axis in range(3):
                    component = offset + axis
                    limits = (lower[species, component], upper[species, component])
                    counts[species, component] += np.histogram(selected[:, axis], bins=50, range=limits)[0]

    return counts, edges


def plot_initial_tiled_histograms(particles, species_names, dynamic_parameters, path):
    """Save initial histograms without gathering the tiled particle arrays.

    Call before the leapfrog seed, while the stored positions and velocities
    are both at the configured initial time. This is a single-process output;
    its addressable shards can span any number of local devices.
    """
    counts, edges = initial_tiled_histogram_data(particles)
    center_grid = dynamic_parameters.grids.center
    for species in range(particles.active.shape[3]):
        name = f"species_{species}" if species_names is None else species_names[species].replace(" ", "_")
        for component in range(6):
            axis = component % 3
            label = "XYZ"[axis]
            velocity = component >= 3
            # Weighted bin edges reproduce plt.hist's bar style without
            # retaining any particle data after the shard reductions.
            plt.hist(edges[species, component, :-1], bins=edges[species, component],
                     weights=counts[species, component])
            plt.xlabel(label + (" Velocity" if velocity else ""))
            plt.ylabel("Number of Particles")
            if not velocity:
                plt.xlim(*_expanded_axis_range(center_grid[axis]))
            quantity = "Velocity" if velocity else "Position"
            plt.title(f"{name} Initial {label} {quantity} Histogram")
            suffix = "xyz"[axis] + ("_velocity" if velocity else "")
            plt.savefig(f"{path}/{name}_initial_{suffix}_histogram.png", dpi=150)
            plt.close()


def write_data(filename, time, data):
    """
    Write one scalar diagnostic sample.

    Args:
        filename (str): The name of the file to write to.
        time (float): The time value to write.
        data (any): The data to write.

    Returns:
        None
    """

    time_value = np.asarray(jax.device_get(time)).item()
    data_value = np.asarray(jax.device_get(data))
    if data_value.ndim == 0:
        data_value = data_value.item()
    else:
        data_value = data_value.tolist()

    with open(filename, "a") as f:
        f.write(f"{time_value}, {data_value}\n")
