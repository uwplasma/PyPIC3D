"""HDF5 round trips that retain addressable field and particle shards."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import openpmd_api as io

from PyPIC3D.diagnostics import async_writer, openPMD, output_adapters
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from tests.kernel_fixtures import kernel_parameters


def parameters(num_tiles=2, shape=(8, 2, 1), boundaries=(0, 0, 0)):
    return kernel_parameters(
        Nx=shape[0], Ny=shape[1], Nz=shape[2],
        tile_shape=(shape[0] // num_tiles, shape[1], shape[2]), guard_cells=2,
        x_wind=4., y_wind=2., z_wind=1., dt=.2,
        boundary_conditions=boundaries, particle_boundary_conditions=(0, 1, 1),
        C=2.)


def shard(array, static):
    return jax.device_put(array, NamedSharding(static.field_mesh, P("tile_x", "tile_y", "tile_z")))


def field_with_guard_sentinels(static, shape):
    """Distinct cell values reveal gaps, duplicated tiles, or leaked guards."""
    reference = jnp.arange(shape[0] * shape[1] * shape[2], dtype=jnp.float64).reshape(shape)
    tx, ty, tz = static.tile_shape
    tiles = reference.reshape(shape[0] // tx, tx, shape[1] // ty, ty, shape[2] // tz, tz)
    tiles = tiles.transpose(0, 2, 4, 1, 3, 5)
    g = static.guard_cells
    tiles = jnp.pad(tiles, ((0, 0),) * 3 + ((g, g),) * 3, constant_values=-9999.)
    return shard(tiles, static), reference


def particle_fixture(static):
    # Inactive NaNs must never enter a record. Tile 0 has two electrons,
    # tile 1 has one; ions occupy the opposite pattern and species 2 is empty.
    x = jnp.full((2, 1, 1, 3, 3, 3), jnp.nan)
    u = jnp.full_like(x, jnp.nan)
    active = jnp.zeros(x.shape[:-1], dtype=bool)
    live = ((0, 0, 0, -1.95, .8), (0, 0, 2, -.7, .2),
            (1, 0, 1, .8, -.4), (0, 1, 1, -1.2, .3),
            (1, 1, 0, .3, -.2), (1, 1, 2, 1.5, .5))
    for tile, species, slot, position, velocity in live:
        x = x.at[tile, 0, 0, species, slot].set(jnp.array([position, .1, -.1]))
        u = u.at[tile, 0, 0, species, slot].set(jnp.array([velocity, .05, -.02]))
        active = active.at[tile, 0, 0, species, slot].set(True)
    particles = TiledParticles(shard(x, static), shard(u, static), shard(active, static))
    species = SpeciesConfig(jnp.array([-1., 1., 0.]), jnp.array([2., 3., 4.]),
                            jnp.array([5., 6., 7.]), jnp.ones((3, 3), dtype=bool))
    return particles, species, ("beam electrons", "background ions", "empty species")


def read_particle_records(path, step=0):
    series = io.Series(str(path), io.Access.read_only)
    try:
        iteration = series.iterations[step]
        pending = {}
        for name, species in iteration.particles.items():
            records = {}
            for record_name in ("position", "positionOffset", "momentum"):
                for component in "xyz":
                    records[record_name + "/" + component] = species[record_name][component].load_chunk()
            for record_name in ("gamma", "mass", "charge", "weighting"):
                records[record_name] = species[record_name][io.Record_Component.SCALAR].load_chunk()
            pending[name] = records
        series.flush()
        result = jax.tree.map(lambda array: jnp.array(array, copy=True), pending)
        jax.block_until_ready(result)
        return result, iteration.time
    finally:
        series.close()


def simulation_config(folder, position_file, *, output, initial=False):
    return {
        "simulation_parameters": dict(
            output_dir=str(folder), solver="electrodynamic_yee", Nt=0 if initial else 3,
            Nx=8, Ny=1, Nz=1, x_wind=4., y_wind=1., z_wind=1.,
            particle_tile_nx=4, particle_tile_ny=1, particle_tile_nz=1,
            particle_tile_capacity_factor=2., particle_batch_size=8,
            dt=.02, C=1., eps=1., mu=1., kb=1., current_calculation="j_from_rhov", filter_j="none"),
        "plotting": dict(
            plot_openpmd_fields=output and not initial, plot_openpmd_particles=output and not initial,
            dump_fields=initial, dump_particles=initial, plotting_interval=1,
            openpmd_field_queue_size=1, openpmd_particle_queue_size=1),
        "particle1": dict(
            name="test beam", N_particles=2, charge=.1, mass=1., weight=.05, temperature=0.,
            initial_x=str(position_file), initial_y=0., initial_z=0.,
            initial_vx=.2, initial_vy=.03, initial_vz=0.),
    }


