"""Generate neutral warm two-stream particle files for dark_two_stream.toml."""

import argparse
from pathlib import Path

import numpy as np
import toml

def initial_positions(n, length):
    """Uniformly spaced positions in a periodic box."""
    x = np.zeros((n, 3))
    x[:, 0] = length * ((np.arange(n) + 0.5) / n - 0.5)
    return x

def initial_velocities(n, beam_speed, vth, seed=42):
    """Counterstreaming beams with mirrored thermal velocities."""
    rng = np.random.default_rng(seed)
    thermal = rng.normal(0.0, vth, (n, 3))
    thermal -= thermal.mean(axis=0)
    forward, backward = thermal.copy(), -thermal.copy()
    forward[:, 0] += beam_speed
    backward[:, 0] -= beam_speed
    return forward, backward

def initial_data(n, length, beam_speed, electron_vth, ion_vth):

    x = initial_positions(n, length)
    forward, backward = initial_velocities(n, beam_speed, electron_vth)
    # Two ions at each electron position cancel charge locally. Opposite
    # ion velocities cancel their initial current without making ions cold.
    rng = np.random.default_rng(42)
    ion_thermal = rng.normal(0.0, ion_vth, (n, 3))

    electron1_x = x.copy()
    electron1_v = forward.copy()
    electron2_x = x.copy()
    electron2_v = backward.copy()
    ion1_x = np.concatenate((x, x))
    ion1_v = np.concatenate((ion_thermal, -ion_thermal))
    # define the initial populations

    return electron1_x, electron1_v, electron2_x, electron2_v, ion1_x, ion1_v


if __name__ == "__main__":

    seed = 42
    beam_speed = 5.0e7
    # initial parameters

    toml_path = Path("dark_two_stream.toml")
    config = toml.load(toml_path)
    # define the toml config file

    electron_vth = config["particle1"]["vth"]
    ion_vth = config["particle3"]["vth"]
    # define the thermal speeds for electrons and ions

    x_wind = config["simulation_parameters"]["x_wind"]
    # define the initial data for the two-stream instability

    n = config["particle1"]["N_particles"]
    # define the number of particles for each electron beam

    electron1_x, electron1_v, electron2_x, electron2_v, ion1_x, ion1_v = initial_data(
        n, x_wind, beam_speed, electron_vth, ion_vth)
    # define the initial positions and velocities for the two electron beams and the ions

    for species, positions, velocities in (
        ("particle1", electron1_x, electron1_v),
        ("particle2", electron2_x, electron2_v),
        ("particle3", ion1_x, ion1_v),
    ):
        for axis, component in enumerate("xyz"):
            for name, values in ((f"initial_{component}", positions[:, axis]),
                                 (f"initial_v{component}", velocities[:, axis])):
                path = Path(config[species][name])
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, values)
    # The native loader reads one one-dimensional array per component.

    electron_plasma_frequency = np.sqrt(
        4 * np.pi * config["particle1"]["charge"] ** 2 * n / config["simulation_parameters"]["x_wind"] ** 3 / config["particle1"]["mass"])

    print(f"Electron plasma frequency: {electron_plasma_frequency:.3e} rad/s")
