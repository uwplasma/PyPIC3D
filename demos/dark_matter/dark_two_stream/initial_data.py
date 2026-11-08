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
    """Mirrored thermal velocities with a common current seed."""
    rng = np.random.default_rng(seed)
    thermal = rng.normal(0.0, vth, (n, 3))
    thermal -= thermal.mean(axis=0)
    forward, backward = thermal.copy(), -thermal.copy()
    forward[:, 0] += beam_speed
    backward[:, 0] += beam_speed
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

    np.save(Path(config["particle1"]["initial_x"]), electron1_x)
    np.save(Path(config["particle1"]["initial_vx"]), electron1_v[:, 0])
    np.save(Path(config["particle1"]["initial_vy"]), electron1_v[:, 1])
    np.save(Path(config["particle1"]["initial_vz"]), electron1_v[:, 2])
    np.save(Path(config["particle2"]["initial_x"]), electron2_x)
    np.save(Path(config["particle2"]["initial_vx"]), electron2_v[:, 0])
    np.save(Path(config["particle2"]["initial_vy"]), electron2_v[:, 1])
    np.save(Path(config["particle2"]["initial_vz"]), electron2_v[:, 2])
    np.save(Path(config["particle3"]["initial_x"]), ion1_x)
    np.save(Path(config["particle3"]["initial_vx"]), ion1_v[:, 0])
    np.save(Path(config["particle3"]["initial_vy"]), ion1_v[:, 1])
    np.save(Path(config["particle3"]["initial_vz"]), ion1_v[:, 2])
    # save the initial data

    electron_plasma_frequency = np.sqrt(
        4 * np.pi * config["particle1"]["charge"] ** 2 * n / config["simulation_parameters"]["x_wind"] ** 3 / config["particle1"]["mass"])

    print(f"Electron plasma frequency: {electron_plasma_frequency:.3e} rad/s")