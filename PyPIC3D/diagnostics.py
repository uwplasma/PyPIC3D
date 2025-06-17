import jax
import jax.numpy as jnp
import toml


def load_probes_from_toml(toml_file, grid, output_dir):
    """
    Load probes from a TOML configuration file.

    Args:
        config_file (str): Path to the TOML configuration file.
        grid (tuple): A tuple containing the grid coordinates (x, y, z).
        output_dir (str): Directory where probe data will be saved.

    Returns:
        list: A list of Probe objects initialized with the parameters from the TOML file.
    """

    probes = []

    for probe in toml_file.get('probes', []):
        name = probe['name']
        x = probe['x']
        y = probe['y']
        z = probe['z']
        fields = probe.get('fields', [0, 1, 2])  # Default to all fields if not specified
        probes.append(Probe(name, x, y, z, fields, grid, output_dir))

    return probes


class Probe:
    def __init__(self, name, x, y, z, fields, grid, output_dir):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.fields = fields
        self.grid = grid
        self.output_dir = output_dir
        self.write_path = f"{output_dir}/data/probes/{name}.txt"
        # Initialize the probe with position and field names


        self.x_idx = jnp.searchsorted(grid[0], x)
        self.y_idx = jnp.searchsorted(grid[1], y)
        self.z_idx = jnp.searchsorted(grid[2], z)
        # get index points

    def write_values(self, t, E, B, J):
        values = []
        # create a empty list to store the values

        if 0 in self.fields:
            value = jnp.sqrt( E[0]**2 + E[1]**2 + E[2]**2 )[self.x_idx, self.y_idx, self.z_idx]
            # Calculate the magnitude of the electric field
            values.append(value)
        if 1 in self.fields:
            value = jnp.sqrt( B[0]**2 + B[1]**2 + B[2]**2 )[self.x_idx, self.y_idx, self.z_idx]
            # Calculate the magnitude of the magnetic field
            values.append(value)
        if 2 in self.fields:
            value = jnp.sqrt( J[0]**2 + J[1]**2 + J[2]**2 )[self.x_idx, self.y_idx, self.z_idx]
            # Calculate the magnitude of the current density
            values.append(value)
        # calculate the values for the fields


        with open(self.write_path, 'a') as f:
            f.write(f"{t}, ")
            for value in values:
                f.write(f"{value}, ")
            f.write("\n")
        # Write the values to the file