"""Two-rank native-input/shared-output integration test, launched with MPI.

PYPIC_DISTRIBUTED=1 JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=1 mpiexec -n 2 \
  python -m tests.distributed_tests.multihost_io_worker --directory /tmp/test-io \
  --require-mpi-output
"""

import argparse
import os
import socket
from mpi4py import MPI
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tests.kernel_fixtures import kernel_parameters
from PyPIC3D.distributed import tiled_field_from_initializer
from PyPIC3D.relativity.core import D_FIELD_LOCATIONS
from PyPIC3D.particles.distributed_initialization import load_distributed_particles
from PyPIC3D.diagnostics.async_writer import make_tiled_field_snapshot, make_tiled_particle_snapshot
from PyPIC3D.diagnostics.parallel_openpmd import open_parallel_series, write_parallel_snapshot
import openpmd_api as io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--require-mpi-output", action="store_true")
    parser.add_argument("--evolve", action="store_true")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    assert comm.size == 2 and jax.process_count() == 2
    print(f"rank={comm.rank} host={socket.gethostname()} local_devices={jax.local_devices()} "
          f"global_devices={jax.devices()}", flush=True)
    static, dynamic = kernel_parameters(Nx=8, Ny=4, Nz=4, tile_shape=(4, 4, 4), guard_cells=2,
                                         boundary_conditions=(1, 1, 1))
    block = dict(name="ions", N_particles=7, charge=1.0, mass=1.0, temperature=0.0)
    for name in ("x", "y", "z", "vx", "vy", "vz"):
        block["initial_"+name] = os.path.join(args.directory, name+".npy")
    if comm.rank == 0:
        os.makedirs(args.directory, exist_ok=True)
        for name in ("x", "y", "z", "vx", "vy", "vz"):
            values = jnp.asarray([-0.5, -1.5, -1.0, -0.2, -0.7, -1.8, -0.3]) if name == "x" else jnp.zeros(7)
            jnp.save(block["initial_"+name], values)
    comm.Barrier()
    particles, species, names, metadata = load_distributed_particles(
        {"particle1": block}, static, dynamic, comm=comm, chunk_size=2)
    assert len(particles.x.addressable_shards) == 1
    shard = particles.active.addressable_shards[0]
    tile = shard.index[0].start
    assert int(jnp.sum(shard.data)) == (7 if tile == 0 else 0)
    expected = jnp.asarray([-0.5, -1.5, -1.0, -0.2, -0.7, -1.8, -0.3])
    if tile == 0:
        assert bool(jnp.array_equal(particles.x.addressable_shards[0].data[0, 0, 0, 0, :7, 0], expected))
    field = tiled_field_from_initializer(lambda x, y, z: x+2*y+3*z, static, dynamic)
    assert len(field.addressable_shards) == 1
    assert field.addressable_shards[0].device.process_index == comm.rank
    from PyPIC3D.utilities.toml_helpers import dump_parameters_to_toml
    if comm.rank == 0:
        os.makedirs(os.path.join(args.directory, "data"), exist_ok=True)
    comm.Barrier()
    dump_parameters_to_toml({}, static._replace(output_dir=args.directory), dynamic, {},
                            {"particle_species_names": names, "particle_species_metadata": metadata}, particles)
    comm.Barrier()
    if comm.rank == 0:
        import toml
        output = toml.load(os.path.join(args.directory, "data", "output.toml"))
        assert output["particles"][0]["active_particles"] == 7
    if args.require_mpi_output or io.variants.get("mpi", False):
        electric = tuple(tiled_field_from_initializer(lambda x, y, z: x+2*y+3*z, static, dynamic,
                                                       location=location) for location in D_FIELD_LOCATIONS)
        field_series = open_parallel_series(args.directory, comm, filename="fields.h5")
        particle_series = open_parallel_series(args.directory, comm, filename="particles.h5")
        for step in range(2):
            fields = make_tiled_field_snapshot({"lapse": field+step, "E": electric},
                                               step=step, time=step*float(dynamic.dt))
            snapshot = make_tiled_particle_snapshot(particles, step=step, time=fields.time,
                                                     species_names=names, species_config=species)
            write_parallel_snapshot(field_series, comm, static, dynamic, fields=fields)
            write_parallel_snapshot(particle_series, comm, static, dynamic, particles=snapshot)
        field_series.close()
        particle_series.close()
        comm.Barrier()
        if comm.rank == 0:
            series = io.Series(os.path.join(args.directory, "fields.h5"), io.Access.read_only)
            lapse = series.iterations[1].meshes["lapse"][io.Mesh_Record_Component.SCALAR]
            assert tuple(lapse.shape) == (8, 4, 4)
            values = lapse.load_chunk()
            electric_values = []
            for component in "xyz":
                record = series.iterations[1].meshes["E"][component]
                assert tuple(record.shape) == (8, 4, 4)
                electric_values.append(record.load_chunk())
            particle_series = io.Series(os.path.join(args.directory, "particles.h5"), io.Access.read_only)
            record = particle_series.iterations[1].particles["ions"]["position"]["x"]
            assert tuple(record.shape) == (7,)
            positions = record.load_chunk()
            series.flush()
            particle_series.flush()
            coordinates = [axis[1] + jnp.arange(n)*spacing for axis, n, spacing in
                           zip(dynamic.grids.center, (8, 4, 4), (dynamic.dx, dynamic.dy, dynamic.dz))]
            analytical = (coordinates[0][:, None, None] + 2*coordinates[1][None, :, None]
                          + 3*coordinates[2][None, None, :] + 1)
            assert bool(jnp.array_equal(jnp.asarray(values), analytical))
            for component, data in enumerate(electric_values):
                coordinates = [axis[1] + (jnp.arange(n)+0.5*(a == component))*spacing
                               for a, (axis, n, spacing) in enumerate(zip(
                                   dynamic.grids.center, data.shape, (dynamic.dx, dynamic.dy, dynamic.dz)))]
                analytical = (coordinates[0][:, None, None] + 2*coordinates[1][None, :, None]
                              + 3*coordinates[2][None, None, :])
                assert bool(jnp.array_equal(jnp.asarray(data), analytical))
            assert bool(jnp.array_equal(jnp.asarray(positions), expected))
            series.close()
            particle_series.close()
            print("MPI output roundtrip passed: uniform mesh chunks and empty-rank particles", flush=True)
        comm.Barrier()
    elif comm.rank == 0:
        print("Native MPI input passed; shared MPI output not run because openPMD has mpi=False", flush=True)
    if args.evolve:
        from PyPIC3D.__main__ import run_PyPIC3D
        write = args.require_mpi_output or io.variants.get("mpi", False)
        config = {
            "simulation_parameters": dict(solver="dynamic_metric", output_dir=os.path.join(args.directory, "evolution"),
                                          Nx=8, Ny=4, Nz=4, x_wind=4.0, y_wind=2.0, z_wind=2.0,
                                          particle_tile_nx=4, particle_tile_ny=4, particle_tile_nz=4,
                                          Nt=2, dt=1.e-4, metric_backreaction=True, benchmark=True),
            "plotting": dict(plot_openpmd_fields=write, plot_openpmd_particles=write, plotting_interval=1,
                             dump_fields=write, dump_particles=write),
            "particle1": dict(block, charge=0.0),
        }
        result = run_PyPIC3D(config, field_initializers={"Ex": lambda x, y, z: 1.e-3*jnp.sin(jnp.pi*x/2)})
        state = result[1].bssn_parameters.state
        assert len(state.conformal_factor.addressable_shards) == 1
        assert not state.conformal_factor.is_fully_addressable
        assert float(result[1].bssn_parameters.time) == 2.e-4
        assert all(bool(jnp.all(jnp.isfinite(leaf.addressable_shards[0].data))) for leaf in state)
        assert float(jnp.max(jnp.abs(state.trace_K))) > 0
        # Writing diagnostics must not change any evolved field or particle state.
        quiet_config = {**config, "simulation_parameters": {**config["simulation_parameters"],
                        "output_dir": os.path.join(args.directory, "evolution_without_output")},
                        "plotting": dict(plotting_interval=1)}
        quiet = run_PyPIC3D(quiet_config, field_initializers={"Ex": lambda x, y, z: 1.e-3*jnp.sin(jnp.pi*x/2)})
        for left, right in zip(jax.tree.leaves((result[4], result[5], state)),
                               jax.tree.leaves((quiet[4], quiet[5], quiet[1].bssn_parameters.state))):
            for a, b in zip(left.addressable_shards, right.addressable_shards):
                assert bool(jnp.array_equal(a.data, b.data))
        local_bytes = sum(shard.data.nbytes for leaf in state for shard in leaf.addressable_shards)
        print(f"rank={comm.rank} BSSN local state bytes={local_bytes}", flush=True)
        from PyPIC3D.utilities.toml_helpers import dump_parameters_to_toml
        dump_parameters_to_toml({"total_iterations": 2}, result[0], result[1], result[3], result[2], result[4])
        comm.Barrier()
        if comm.rank == 0:
            import toml
            output = toml.load(os.path.join(args.directory, "evolution", "data", "output.toml"))
            assert output["particles"][0]["active_particles"] == 7
            if write:
                output_dir = os.path.join(args.directory, "evolution", "data")
                assert os.path.isfile(os.path.join(output_dir, "initial_fields", "initial_fields.h5"))
                assert os.path.isfile(os.path.join(output_dir, "initial_particles", "ions_initial_particles.h5"))
                assert os.path.isfile(os.path.join(output_dir, "particles.h5"))
                assert not os.path.exists(os.path.join(output_dir, "simulation.h5"))
                series = io.Series(os.path.join(output_dir, "fields.h5"), io.Access.read_only)
                expected_records = {"E", "B", "J", "conformal_factor", "trace_K", "lapse",
                                    "conformal_connection", "shift"}
                expected_records.update(prefix+"_"+suffix for prefix in ("conformal_metric", "traceless_K")
                                        for suffix in ("xx", "xy", "xz", "yy", "yz", "zz"))
                assert list(series.iterations) == [0, 1]
                for step in series.iterations:
                    iteration = series.iterations[step]
                    assert set(iteration.meshes) == expected_records
                    assert iteration.time == step*1.e-4
                    values = []
                    for name in expected_records:
                        components = ("x", "y", "z") if name in ("E", "B", "J", "conformal_connection", "shift") else (io.Mesh_Record_Component.SCALAR,)
                        assert set(iteration.meshes[name]) == set(components)
                        for component in components:
                            record = iteration.meshes[name][component]
                            assert tuple(record.shape) == (8, 4, 4)
                            values.append(record.load_chunk())
                    series.flush()
                    assert all(bool(jnp.all(jnp.isfinite(jnp.asarray(value)))) for value in values)
                series.close()
            print("Actual two-process initialize_simulation/run_PyPIC3D steps passed", flush=True)
    comm.Barrier()
    if comm.rank == 0:
        print("Two-process native input/local-shard checks passed", flush=True)
    jax.distributed.shutdown()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
