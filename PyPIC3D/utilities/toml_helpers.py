import argparse
from datetime import datetime
import importlib.metadata
import os
from numbers import Integral

import jax
import jax.numpy as jnp
import plotly
import toml
import tqdm

from PyPIC3D.utilities.parameters import dynamic_parameters_for_output, static_parameters_for_output


def load_config_file():
    """
    Parse the command-line configuration path and load its TOML contents.
    """
    parser = argparse.ArgumentParser(description="3D PIC code using Jax")
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    config_file = args.config
    print(f"Using Configuration File: {config_file}")
    toml_file = toml.load(config_file)
    return toml_file


def grab_field_keys(config):
    """
    Extract configuration keys with the ``field`` prefix.
    """
    field_keys = []
    for key in config.keys():
        if key[:5] == 'field':
            field_keys.append(key)
    return field_keys


def grab_previous_field_keys(config):
    """
    Extract previous-field blocks used by time-centered field solvers.
    """
    field_keys = []
    for key in config.keys():
        if key[:14] == "previous_field":
            field_keys.append(key)
    return field_keys


def _add_external_field_to_tiled_component(component, external_field, static_parameters, dynamic_parameters, field_name):
    """
    Add one physical field array into the active interiors of a tiled component.
    """

    tile_nx, tile_ny, tile_nz = [int(width) for width in static_parameters.tile_shape]
    g = int(static_parameters.guard_cells)
    ntx, nty, ntz = component.shape[:3]
    interior_shape = (int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz))
    if external_field.shape != interior_shape:
        raise ValueError(
            f"Shape mismatch for field '{field_name}': external field shape {external_field.shape} "
            f"does not match expected interior shape {interior_shape}"
        )

    for tx in range(ntx):
        for ty in range(nty):
            for tz in range(ntz):
                ix = tx * tile_nx
                iy = ty * tile_ny
                iz = tz * tile_nz
                block = external_field[ix:ix + tile_nx, iy:iy + tile_ny, iz:iz + tile_nz]
                component = component.at[
                    tx, ty, tz,
                    g:g + tile_nx,
                    g:g + tile_ny,
                    g:g + tile_nz,
                ].add(block)

    return component


def load_previous_fields_from_toml(previous_fields, config, static_parameters, dynamic_parameters):
    """
    Load previous D/B field components from TOML blocks named previous_fieldN.

    The static-metric solver stores D at integer time and B at half-integer
    time. Optional previous_fieldN blocks let a run initialize the older
    time levels from npy arrays instead of duplicating the current fields.
    """

    field_keys = grab_previous_field_keys(config)
    field_components = [component for field in previous_fields for component in field]

    for toml_key in field_keys:
        field_name = config[toml_key]['name']
        field_type = config[toml_key]['type']
        field_path = config[toml_key]['path']
        print(f"Loading previous field: {field_name} from {field_path}")

        if field_type < 0 or field_type > 5:
            raise ValueError("Previous static-metric fields must be D or B components with type 0 through 5")

        external_field = jnp.load(field_path)
        field_components[field_type] = _add_external_field_to_tiled_component(
            jnp.zeros_like(field_components[field_type]),
            external_field,
            static_parameters,
            dynamic_parameters,
            field_name,
        )
        print(f"Previous field loaded successfully: {field_name}")

    return tuple(field_components[:3]), tuple(field_components[3:6])


def load_external_fields_from_toml(fields, external_fields, config, static_parameters, dynamic_parameters):
    """
    Load evolved and prescribed external fields from TOML field blocks.
    """

    field_keys = grab_field_keys(config)
    external_E, external_B = external_fields

    for toml_key in field_keys:
        field_name = config[toml_key]['name']
        field_type = config[toml_key]['type']
        field_path = config[toml_key]['path']
        evolve = config[toml_key].get('evolve', True)
        print(f"Loading field: {field_name} from {field_path}")

        external_field = jnp.load(field_path)

        if not evolve and (field_type < 0 or field_type > 5):
            raise ValueError("External-only fields must be electric or magnetic field components with type 0 through 5")

        if evolve:
            # Evolved fields are part of the self-consistent Maxwell solve.
            # This is the original behavior and remains the default.
            fields[field_type] = _add_external_field_to_tiled_component(
                fields[field_type],
                external_field,
                static_parameters,
                dynamic_parameters,
                field_name,
            )
        else:
            # External-only E/B fields are invisible to Maxwell's equations.
            # They are added back only for particle pushes and diagnostics.
            if field_type < 3:
                external_E = list(external_E)
                external_E[field_type] = _add_external_field_to_tiled_component(
                    external_E[field_type],
                    external_field,
                    static_parameters,
                    dynamic_parameters,
                    field_name,
                )
                external_E = tuple(external_E)
            else:
                external_B = list(external_B)
                b_index = field_type - 3
                external_B[b_index] = _add_external_field_to_tiled_component(
                    external_B[b_index],
                    external_field,
                    static_parameters,
                    dynamic_parameters,
                    field_name,
                )
                external_B = tuple(external_B)

        print(f"Field loaded successfully: {field_name}")

    return fields, (external_E, external_B)


def load_dark_fields_from_toml(config, static_parameters, dynamic_parameters):
    """Load dark_fieldN physical t=0 data, then seed the Proca leapfrog.

    Types 0..2 are E', 3..5 are A', and 6 is phi'. All are evolved fields.
    """
    from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import initialize_dark_photon_fields

    E, A, phi = initialize_dark_photon_fields(static_parameters, dynamic_parameters)
    components = [*E, *A, phi]
    for key, block in config.items():
        if not key.startswith("dark_field"):
            continue
        field_type = block["type"]
        if isinstance(field_type, bool) or not isinstance(field_type, Integral) or not 0 <= field_type <= 6:
            raise ValueError("Dark field type must be an integer from 0 through 6")
        if not block.get("evolve", True):
            raise ValueError("Dark initial fields must be evolved (evolve=true)")
        values = jnp.load(block["path"])
        if not jnp.issubdtype(values.dtype, jnp.number) or jnp.iscomplexobj(values) or not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("Dark initial fields must contain finite real numeric values")
        components[field_type] = _add_external_field_to_tiled_component(
            components[field_type], values, static_parameters, dynamic_parameters, block.get("name", key),
        )
    return initialize_dark_photon_fields(
        static_parameters, dynamic_parameters, E=components[:3], A=components[3:6], phi=components[6],
    )


def update_parameters_from_toml(config, static_parameters, dynamic_parameters, plotting_parameters):
    """
    Update run parameters with values from a TOML configuration.
    """

    for key, value in config.get("simulation_parameters", {}).items():
        if key in static_parameters:
            static_parameters[key] = value
        if key in dynamic_parameters:
            dynamic_parameters[key] = value

    for key, value in config.get("static_parameters", {}).items():
        if key in static_parameters:
            static_parameters[key] = value

    for key, value in config.get("dynamic_parameters", {}).items():
        if key in dynamic_parameters:
            dynamic_parameters[key] = value

    for key, value in config.get("plotting", {}).items():
        if key in plotting_parameters:
            plotting_parameters[key] = value

    return static_parameters, dynamic_parameters, plotting_parameters


def dump_parameters_to_toml(simulation_stats, static_parameters, dynamic_parameters, plasma_parameters, plotting_parameters, particles):
    """
    Dump run, plotting, and tiled particle species data into output TOML.
    """

    output_path = static_parameters.output_dir
    output_file = os.path.join(output_path, "data/output.toml")
    plotting_parameters_for_output = {
        key: value
        for key, value in plotting_parameters.items()
        if key not in ("field_map", "particle_species_names", "particle_species_metadata")
    }

    config = {
        "simulation_stats": simulation_stats,
        "static_parameters": static_parameters_for_output(static_parameters),
        "dynamic_parameters": dynamic_parameters_for_output(dynamic_parameters),
        'plasma_parameters': jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, plasma_parameters),
        "plotting": jax.tree_util.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, plotting_parameters_for_output),
        "particles": []
    }

    n_species = particles.active.shape[3]
    species_names = plotting_parameters.get("particle_species_names")
    species_metadata = plotting_parameters.get("particle_species_metadata")
    tile_shape = static_parameters.tile_shape
    tile_shape = [int(width) for width in tile_shape]

    for species_index in range(n_species):
        if species_names is None:
            name = f"species_{species_index}"
        else:
            name = species_names[species_index]

        active_particles = int(jnp.sum(particles.active[:, :, :, species_index, :]))
        if species_metadata is None:
            particle_dict = {"name": name}
        else:
            particle_dict = dict(
                jax.tree_util.tree_map(
                    lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x,
                    species_metadata[species_index],
                )
            )

        particle_dict["storage"] = "tiled"
        particle_dict["active_particles"] = active_particles
        particle_dict["tile_shape"] = tile_shape
        config["particles"].append(particle_dict)

    config["version"] = {
        "PyPIC3D_version": importlib.metadata.version('PyPIC3D'),
        "date": datetime.now().strftime("%Y-%m-%d")
    }

    package_versions = {
        "jax": jax.__version__,
        "toml": toml.__version__,
        "plotly": plotly.__version__,
        "tqdm": tqdm.__version__,
    }

    config["package_versions"] = package_versions

    with open(output_file, 'w') as f:
        toml.dump(config, f)
