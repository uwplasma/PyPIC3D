import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import jax
import numpy as np
import jax.numpy as jnp

from PyPIC3D.diagnostics import async_writer
from PyPIC3D.diagnostics import openPMD
from PyPIC3D.diagnostics.openPMD import _ensure_openpmd_array
from PyPIC3D.particles.particle_class import SpeciesConfig, TiledParticles
from PyPIC3D.relativity.core import build_yee_metric
from PyPIC3D.relativity.interpolate_metric import particle_lorentz_factor
from tests.kernel_fixtures import (
    build_tiled_particles,
    field_tiles_from_global,
    kernel_parameters,
    kernel_parameters_from_values,
    particle_parameters_from_values,
    particle_species,
    species_names as names_for_species,
    vector_tiles_from_global,
)


class FakeRecord:
    def __init__(self):
        self.shape = None
        self.dataset_shape = None
        self.unit_SI = None
        self.data = None
        self.chunks = []

    def reset_dataset(self, dataset):
        extent = getattr(dataset, "extent", None)
        if extent is not None:
            self.dataset_shape = tuple(extent)

    def store_chunk(self, array, offset, extent):
        self.shape = tuple(extent)
        self.data = np.array(array, copy=True)
        self.chunks.append((tuple(offset), tuple(extent), np.array(array, copy=True)))


class FakeMesh:
    def __init__(self):
        self.records = {}
        self.axis_labels = None
        self.grid_spacing = None
        self.grid_global_offset = None
        self.unit_dimension = None

    def __getitem__(self, component):
        if component not in self.records:
            self.records[component] = FakeRecord()
        return self.records[component]


class FakeMeshes(dict):
    def __getitem__(self, name):
        if name not in self:
            self[name] = FakeMesh()
        return dict.__getitem__(self, name)


class FakeParticleRecord(dict):
    def __getitem__(self, component):
        if component not in self:
            self[component] = FakeRecord()
        return dict.__getitem__(self, component)


class FakeParticleSpecies(dict):
    def __getitem__(self, record_name):
        if record_name not in self:
            if record_name in ("position", "positionOffset", "momentum"):
                self[record_name] = FakeParticleRecord()
            else:
                self[record_name] = FakeRecord()
        return dict.__getitem__(self, record_name)


class FakeParticles(dict):
    def __getitem__(self, species_name):
        if species_name not in self:
            self[species_name] = FakeParticleSpecies()
        return dict.__getitem__(self, species_name)


class FakeIteration:
    def __init__(self):
        self.meshes = FakeMeshes()
        self.particles = FakeParticles()
        self.time = None
        self.dt = None
        self.time_unit_SI = None


class FakeIterations(dict):
    def __getitem__(self, iteration):
        if iteration not in self:
            self[iteration] = FakeIteration()
        return dict.__getitem__(self, iteration)


class FakeSeries:
    def __init__(self):
        self.iterations = FakeIterations()

    def set_attribute(self, name, value):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def _zero_field(shape):
    return tuple(jnp.zeros(shape) for _ in range(3))


def _parameter_values(tile_shape=(4, 2, 1)):
    return {
        "dt": 0.2,
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "x_wind": 4.0,
        "y_wind": 2.0,
        "z_wind": 1.0,
        "Nx": 4,
        "Ny": 2,
        "Nz": 1,
        "tile_shape": tile_shape,
        "guard_cells": 2,
        "boundary_conditions": {"x": 0, "y": 0, "z": 0},
        "particle_boundary_conditions": {"x": 0, "y": 0, "z": 0},
    }


def _record_data(record):
    if len(record.chunks) <= 1:
        return record.data

    n = sum(int(extent[0]) for _offset, extent, _data in record.chunks)
    out = jnp.zeros((n,), dtype=record.chunks[0][2].dtype)
    for offset, extent, data in record.chunks:
        start = int(offset[0])
        stop = start + int(extent[0])
        out = out.at[start:stop].set(data)
    return out


def _species(name, charge, mass, weight, x1):
    return particle_species(
        name=name,
        charge=charge,
        mass=mass,
        weight=weight,
        x1=jnp.asarray(x1),
        u1=jnp.ones_like(x1) * 0.1,
    )


class OpenPMDDiagnosticsTestsFixtures:
    pass
