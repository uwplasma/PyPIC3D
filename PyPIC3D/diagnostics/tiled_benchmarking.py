from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.grid_and_stencil import BC_PERIODIC
from PyPIC3D.deposition.current_methods import CURRENT_J_FROM_RHOV
from PyPIC3D.deposition.direct_deposition_tiled import direct_J_from_tiled_particles
from PyPIC3D.diagnostics.output_adapters import particles_for_output, vector_field_for_output
from PyPIC3D.electrodynamic_tiled import time_loop_electrodynamic_tiled
from PyPIC3D.particles.species_class import particle_species
from PyPIC3D.particles.tiled_particle_initialization import to_tiled_particles
from PyPIC3D.particles.tiled_particle_refresh import (
    refresh_tiled_particle_tiles,
    update_tiled_particle_positions,
)
from PyPIC3D.particles.tiled_particles import TiledParticles
from PyPIC3D.pusher.tiled_pusher import tiled_particle_push
from PyPIC3D.solvers.yee_tiled import (
    empty_tiled_scalar_field,
    empty_tiled_vector_field,
    tile_grid_axes,
    tile_vector_field,
    update_tiled_B,
    update_tiled_E,
)
from PyPIC3D.utils import add_external_fields, build_yee_grid, compute_energy


@dataclass
class StageSpec:
    name: str
    function: object
    args: tuple
    kwargs: dict
    static_argnames: tuple = ()


@dataclass
class SyntheticTiledYeeState:
    particles: TiledParticles
    species_config: object
    fields: tuple
    world: dict
    constants: dict
    tile_shape: tuple
    g: int


def unused_curl(Ex, Ey, Ez):
    return None


def _pad_tiled_particle_capacity(tiled_particles, slots_per_tile):
    current_slots = tiled_particles.active.shape[-1]
    if current_slots >= slots_per_tile:
        return tiled_particles

    pad_slots = int(slots_per_tile) - int(current_slots)
    vector_pad = [(0, 0)] * tiled_particles.x.ndim
    vector_pad[-2] = (0, pad_slots)
    scalar_pad = [(0, 0)] * tiled_particles.active.ndim
    scalar_pad[-1] = (0, pad_slots)

    return TiledParticles(
        x=jnp.pad(tiled_particles.x, vector_pad),
        u=jnp.pad(tiled_particles.u, vector_pad),
        active=jnp.pad(tiled_particles.active, scalar_pad),
    )


def build_synthetic_tiled_yee_state(
    nx=16,
    ny=16,
    nz=16,
    particles_per_species=2048,
    species_count=2,
    shape=1,
    seed=0,
    tile_shape=(8, 8, 8),
    slots_per_tile=None,
    g=2,
    dt=None,
):
    """
    Build a deterministic tiled electrodynamic Yee benchmark state.
    """

    x_wind = 1.0
    y_wind = 1.0
    z_wind = 1.0
    dx = x_wind / nx
    dy = y_wind / ny
    dz = z_wind / nz
    if dt is None:
        dt = 0.05 * min(dx, dy, dz)

    world = {
        "dt": dt,
        "Nt": 1,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "Nx": nx,
        "Ny": ny,
        "Nz": nz,
        "x_wind": x_wind,
        "y_wind": y_wind,
        "z_wind": z_wind,
        "shape_factor": shape,
        "current_calculation": CURRENT_J_FROM_RHOV,
        "tile_shape": tile_shape,
        "guard_cells": g,
        "boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
        "particle_boundary_conditions": {"x": BC_PERIODIC, "y": BC_PERIODIC, "z": BC_PERIODIC},
    }
    vertex_grid, center_grid = build_yee_grid(world)
    world["grids"] = {"center": center_grid, "vertex": vertex_grid}
    world["grids"]["tiled_center_grid"] = tile_grid_axes(center_grid, world, tile_shape, num_guard_cells=g)
    world["grids"]["tiled_vertex_grid"] = tile_grid_axes(vertex_grid, world, tile_shape, num_guard_cells=g)

    constants = {
        "eps": 1.0,
        "mu": 1.0,
        "C": 1.0,
        "kb": 1.0,
        "alpha": 1.0,
    }

    key = jax.random.PRNGKey(seed)
    particles = []
    for species_index in range(species_count):
        key, x_key, y_key, z_key, vx_key, vy_key, vz_key = jax.random.split(key, 7)
        x1 = jax.random.uniform(
            x_key, (particles_per_species,), minval=-0.45 * x_wind, maxval=0.45 * x_wind
        )
        x2 = jax.random.uniform(
            y_key, (particles_per_species,), minval=-0.45 * y_wind, maxval=0.45 * y_wind
        )
        x3 = jax.random.uniform(
            z_key, (particles_per_species,), minval=-0.45 * z_wind, maxval=0.45 * z_wind
        )
        v1 = 0.01 * jax.random.normal(vx_key, (particles_per_species,))
        v2 = 0.01 * jax.random.normal(vy_key, (particles_per_species,))
        v3 = 0.01 * jax.random.normal(vz_key, (particles_per_species,))

        charge_sign = 1.0 if species_index % 2 == 0 else -1.0
        particles.append(
            particle_species(
                name=f"benchmark_species_{species_index}",
                N_particles=particles_per_species,
                charge=charge_sign,
                mass=1.0,
                T=0.0,
                v1=v1,
                v2=v2,
                v3=v3,
                x1=x1,
                x2=x2,
                x3=x3,
                xwind=x_wind,
                ywind=y_wind,
                zwind=z_wind,
                dx=dx,
                dy=dy,
                dz=dz,
                dt=dt,
            )
        )

    simulation_parameters = {
        "particle_tile_nx": tile_shape[0],
        "particle_tile_ny": tile_shape[1],
        "particle_tile_nz": tile_shape[2],
        "particle_tile_capacity_factor": 1.0,
    }
    tiled_particles, species_config = to_tiled_particles(particles, world, simulation_parameters)
    if slots_per_tile is not None:
        tiled_particles = _pad_tiled_particle_capacity(tiled_particles, slots_per_tile)

    global_shape = (nx + 2, ny + 2, nz + 2)
    E = tuple(jnp.zeros(global_shape) for _ in range(3))
    B = tuple(jnp.zeros(global_shape) for _ in range(3))
    E_tiles = tile_vector_field(E, world, tile_shape, num_guard_cells=g)
    B_tiles = tile_vector_field(B, world, tile_shape, num_guard_cells=g)
    J_tiles = empty_tiled_vector_field(world, tile_shape, num_guard_cells=g)
    rho_tiles = empty_tiled_scalar_field(world, tile_shape, num_guard_cells=g)
    phi_tiles = empty_tiled_scalar_field(world, tile_shape, num_guard_cells=g)
    external_fields = (
        tuple(jnp.zeros_like(component) for component in E_tiles),
        tuple(jnp.zeros_like(component) for component in B_tiles),
    )
    fields = (E_tiles, B_tiles, J_tiles, rho_tiles, phi_tiles, external_fields, None)

    return SyntheticTiledYeeState(
        particles=tiled_particles,
        species_config=species_config,
        fields=fields,
        world=world,
        constants=constants,
        tile_shape=tile_shape,
        g=g,
    )


def _tiled_particle_push_stage(case, relativistic=True, particle_pusher="boris"):
    E_tiles, B_tiles, *_ = case.fields
    push_E_tiles, push_B_tiles = add_external_fields(E_tiles, B_tiles, case.fields[5])
    with jax.named_scope("tiled_particle_push"):
        return tiled_particle_push(
            case.particles,
            case.species_config,
            push_E_tiles,
            push_B_tiles,
            case.world,
            case.constants,
            case.tile_shape,
            case.g,
            relativistic=relativistic,
            particle_pusher=particle_pusher,
        )


def _tiled_particle_retile_stage(case):
    with jax.named_scope("tiled_particle_retile"):
        moved = update_tiled_particle_positions(case.particles, case.species_config, case.world["dt"])
        return refresh_tiled_particle_tiles(moved, case.world, case.tile_shape)


def _tiled_current_deposition_stage(case, particles, J_func):
    _, _, J_tiles, *_ = case.fields
    with jax.named_scope("tiled_current_deposition"):
        return J_func(
            particles,
            case.species_config,
            J_tiles,
            case.constants,
            case.world,
            tile_shape=case.tile_shape,
            g=case.g,
        )


def _tiled_field_update_stage(case, J_tiles):
    E_tiles, B_tiles, *_ = case.fields
    with jax.named_scope("tiled_field_update"):
        E_tiles = update_tiled_E(E_tiles, B_tiles, J_tiles, case.world, case.constants, unused_curl, case.tile_shape, case.g)
        B_tiles = update_tiled_B(E_tiles, B_tiles, case.world, case.constants, unused_curl, case.tile_shape, case.g)
    return E_tiles, B_tiles


def _tiled_diagnostics_stage(case):
    E_tiles, B_tiles, *_ = case.fields
    with jax.named_scope("tiled_diagnostics"):
        return compute_energy(
            case.particles,
            E_tiles,
            B_tiles,
            case.world,
            case.constants,
            species_config=case.species_config,
        )


def _tiled_output_bridge_stage(case):
    E_tiles, B_tiles, *_ = case.fields
    with jax.named_scope("tiled_output_bridge"):
        E = vector_field_for_output(E_tiles, case.world)
        B = vector_field_for_output(B_tiles, case.world)
        particles = particles_for_output(case.particles, case.species_config)
    return E, B, particles


def benchmark_tiled_pic_step(case, J_func, relativistic=True, particle_pusher="boris"):
    with jax.named_scope("tiled_pic_step"):
        return time_loop_electrodynamic_tiled(
            case.particles,
            case.species_config,
            case.fields,
            case.world,
            case.constants,
            unused_curl,
            J_func,
            "tiled_yee",
            tile_shape=case.tile_shape,
            g=case.g,
            relativistic=relativistic,
            particle_pusher=particle_pusher,
        )


def build_tiled_stage_specs(case, J_func=None, relativistic=True, particle_pusher="boris"):
    """
    Return independently timed tiled Yee stage specs.

    These specs identify the expensive tile-management stages before larger
    kernel rewrites: push, retile, deposition, field update, diagnostics, and
    the current global output bridge.
    """

    if J_func is None:
        J_func = partial(direct_J_from_tiled_particles, filter="none")

    pushed_particles = _tiled_particle_push_stage(case, relativistic=relativistic, particle_pusher=particle_pusher)
    deposited_J = _tiled_current_deposition_stage(case, pushed_particles, J_func)

    return {
        "tiled_pic_step": StageSpec(
            "tiled_pic_step",
            benchmark_tiled_pic_step,
            (case, J_func),
            {"relativistic": relativistic, "particle_pusher": particle_pusher},
            ("J_func", "relativistic", "particle_pusher"),
        ),
        "tiled_particle_push": StageSpec(
            "tiled_particle_push",
            _tiled_particle_push_stage,
            (case,),
            {"relativistic": relativistic, "particle_pusher": particle_pusher},
            ("relativistic", "particle_pusher"),
        ),
        "tiled_particle_retile": StageSpec(
            "tiled_particle_retile",
            _tiled_particle_retile_stage,
            (case,),
            {},
            (),
        ),
        "tiled_current_deposition": StageSpec(
            "tiled_current_deposition",
            _tiled_current_deposition_stage,
            (case, pushed_particles, J_func),
            {},
            ("J_func",),
        ),
        "tiled_field_update": StageSpec(
            "tiled_field_update",
            _tiled_field_update_stage,
            (case, deposited_J),
            {},
            (),
        ),
        "tiled_diagnostics": StageSpec(
            "tiled_diagnostics",
            _tiled_diagnostics_stage,
            (case,),
            {},
            (),
        ),
        "tiled_output_bridge": StageSpec(
            "tiled_output_bridge",
            _tiled_output_bridge_stage,
            (case,),
            {},
            (),
        ),
    }
