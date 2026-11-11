"""MPI output acceptance: two ranks, one or several addressable tiles per rank."""

import argparse
from dataclasses import replace
import os
from unittest.mock import patch

from mpi4py import MPI
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import openpmd_api as io

from tests.kernel_fixtures import kernel_parameters
from PyPIC3D.distributed import tiled_field_from_initializer
from PyPIC3D.particles.particle_class import TiledParticles, SpeciesConfig
from PyPIC3D.diagnostics.async_writer import make_tiled_field_snapshot, make_tiled_particle_snapshot
from PyPIC3D.diagnostics import openPMD
from PyPIC3D.diagnostics.openPMD import (
    TiledMeshLayout, write_tiled_field_snapshot_openpmd, write_tiled_particle_snapshot_openpmd,
    write_openpmd_initial_fields, write_openpmd_initial_particles,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument("--local-devices", type=int, default=int(os.environ.get("JAX_NUM_CPU_DEVICES", "1")))
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    assert comm.size == jax.process_count() == 2
    assert jax.local_device_count() == args.local_devices
    ntiles = jax.device_count()
    static, dynamic = kernel_parameters(
        Nx=4*ntiles, Ny=1, Nz=4, x_wind=2*ntiles, y_wind=1., z_wind=2.,
        tile_shape=(4, 1, 4), guard_cells=2, boundary_conditions=(1, 0, 1), C=10.,
    )
    layout = TiledMeshLayout((4*ntiles, 1, 4), (4, 1, 4), guard_cells=2)
    field = tiled_field_from_initializer(lambda x, y, z: x+2*y+3*z, static, dynamic)
    assert len(field.addressable_shards) == args.local_devices
    names = ("one tile", "uneven", "empty")
    species = SpeciesConfig(jnp.array([-1., 1., 0.]), jnp.array([2., 3., 1.]),
                            jnp.array([4., 5., 1.]), jnp.ones((3, 3), dtype=bool))

    def positions(index):
        tile = index[0].start
        value = jnp.zeros((1, 1, 1, 3, 3, 3), dtype=jnp.float64)
        value = value.at[..., 0].set(-ntiles+2*tile+.25+.25*jnp.arange(3))
        return jax.device_get(value)

    def velocities(index):
        value = jnp.zeros((1, 1, 1, 3, 3, 3), dtype=jnp.float64)
        return jax.device_get(value.at[..., 0].set(.2))

    def active(index):
        tile = index[0].start
        value = jnp.zeros((1, 1, 1, 3, 3), dtype=bool)
        value = value.at[0, 0, 0, 0, 0].set(tile == 0)
        value = value.at[0, 0, 0, 1, :].set(jnp.arange(3) < 1+tile % 3)
        return jax.device_get(value)

    sharding = NamedSharding(static.field_mesh, P("tile_x", "tile_y", "tile_z"))
    shape = (ntiles, 1, 1, 3, 3, 3)
    particles = TiledParticles(
        jax.make_array_from_callback(shape, sharding, positions),
        jax.make_array_from_callback(shape, sharding, velocities),
        jax.make_array_from_callback(shape[:-1], sharding, active),
    )
    assert not particles.x.is_fully_addressable
    assert len(particles.x.addressable_shards) == args.local_devices
    # Reject full simulation-array transfers; tiny replicated species metadata is allowed.
    device_get = jax.device_get
    def local_device_get(value):
        if isinstance(value, jax.Array) and not value.is_fully_addressable:
            raise AssertionError("output transferred a global distributed array")
        return device_get(value)

    with patch("jax.device_get", side_effect=local_device_get):
        initial = make_tiled_particle_snapshot(particles, step=0, time=0., species_names=names,
                                               species_config=species, initial=True)
        local_bytes = sum(data.nbytes for _, data in initial.x_shards)
        assert local_bytes == sum(shard.data.nbytes for shard in particles.x.addressable_shards)
        assert local_bytes < particles.x.nbytes
        # Each component has an independently permuted host-shard list.
        initial = replace(initial, x_shards=list(reversed(initial.x_shards)),
                          active_shards=list(reversed(initial.active_shards)))
        if args.fail_rank >= 0:
            abort = openPMD._abort_parallel_write
            def record_writer_abort(communicator):
                print(f"WRITER_ABORT rank={comm.rank}", flush=True)
                abort(communicator)
            with patch.object(openPMD, "_abort_parallel_write", side_effect=record_writer_abort):
                if comm.rank == args.fail_rank:
                    with patch.object(openPMD, "_iter_snapshot_particle_chunks",
                                      side_effect=RuntimeError("injected rank-local output failure")):
                        write_tiled_particle_snapshot_openpmd(
                            initial, output_dir=args.directory, filename="failure", static_parameters=static,
                            dynamic_parameters=dynamic, file_extension=".h5", comm=comm)
                else:
                    write_tiled_particle_snapshot_openpmd(
                        initial, output_dir=args.directory, filename="failure", static_parameters=static,
                        dynamic_parameters=dynamic, file_extension=".h5", comm=comm)
            raise AssertionError("injected failure did not abort MPI")

        write_openpmd_initial_fields({"rho": field}, static, dynamic, args.directory, comm=comm)
        write_openpmd_initial_particles(particles, static, dynamic, args.directory,
                                       species_config=species, species_names=names, comm=comm)
        for mode in ("fields", "particles", "both"):
            folder = os.path.join(args.directory, mode)
            for step in range(2):
                if mode != "particles":
                    snapshot = make_tiled_field_snapshot({"rho": field+step}, step=step, time=step*dynamic.dt)
                    write_tiled_field_snapshot_openpmd(snapshot, output_dir=folder, filename="fields",
                        dynamic_parameters=dynamic, layout=layout, file_extension=".h5", comm=comm)
                if mode != "fields":
                    snapshot = replace(initial, step=step, time=step*float(dynamic.dt), initial=False)
                    write_tiled_particle_snapshot_openpmd(snapshot, output_dir=folder, filename="particles",
                        static_parameters=static, dynamic_parameters=dynamic, file_extension=".h5", comm=comm)

    comm.Barrier()
    if comm.rank == 0:
        x = -ntiles + .5*jnp.arange(4*ntiles)
        y = -0.5+jnp.arange(1)
        z = -1.+.5*jnp.arange(4)
        field_reference = x[:, None, None]+2*y[None, :, None]+3*z[None, None, :]
        particle_reference = (
            jnp.array([-ntiles+.25]),
            jnp.array([-ntiles+2*tile+.25+.25*slot for tile in range(ntiles)
                       for slot in range(1+tile % 3)]),
            jnp.empty(0),
        )
        series = io.Series(os.path.join(args.directory, "data", "initial_fields", "initial_fields.h5"),
                           io.Access.read_only)
        record = series.iterations[0].meshes["rho"][io.Mesh_Record_Component.SCALAR]
        assert tuple(record.shape) == layout.global_shape
        data = record.load_chunk()
        series.flush()
        assert bool(jnp.array_equal(jnp.asarray(data), field_reference))
        series.close()
        for mode in ("fields", "particles", "both"):
            folder = os.path.join(args.directory, mode)
            for kind in ("fields", "particles"):
                path = os.path.join(folder, kind+".h5")
                if kind != mode and mode != "both":
                    assert not os.path.exists(path)
                    continue
                series = io.Series(path, io.Access.read_only)
                assert series.get_attribute("software") == "PyPIC3D"
                assert list(series.iterations) == [0, 1]
                for step in series.iterations:
                    iteration = series.iterations[step]
                    assert iteration.time == step*float(dynamic.dt)
                    if kind == "fields":
                        record = iteration.meshes["rho"][io.Mesh_Record_Component.SCALAR]
                        assert tuple(record.shape) == layout.global_shape
                        data = record.load_chunk()
                        series.flush()
                        assert bool(jnp.array_equal(jnp.asarray(data), field_reference+step))
                    else:
                        for name, expected in zip(names, particle_reference):
                            group = iteration.particles[name.replace(" ", "_")]
                            record = group["position"]["x"]
                            assert tuple(record.shape) == expected.shape
                            if expected.size:
                                data = record.load_chunk()
                                series.flush()
                                assert bool(jnp.allclose(jnp.asarray(data), expected-.2*dynamic.dt/2,
                                                         rtol=0, atol=1.e-14))
                series.close()
        for name, expected in zip(names, particle_reference):
            path = os.path.join(args.directory, "data", "initial_particles",
                                name.replace(" ", "_")+"_initial_particles.h5")
            series = io.Series(path, io.Access.read_only)
            record = series.iterations[0].particles[name.replace(" ", "_")]["position"]["x"]
            assert tuple(record.shape) == expected.shape
            if expected.size:
                data = record.load_chunk()
                series.flush()
                assert bool(jnp.array_equal(jnp.asarray(data), expected))
            series.close()
        print(f"MPI shard output passed: {comm.size} ranks, {ntiles} tiles, fields/particles/initial readback", flush=True)
    comm.Barrier()
    jax.distributed.shutdown()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
