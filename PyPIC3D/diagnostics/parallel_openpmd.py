"""Main-thread MPI entry points using the common tiled openPMD writer."""

import openpmd_api as io

from PyPIC3D.diagnostics.openPMD import (
    TiledMeshLayout, _abort_parallel_write, _open_openpmd_series,
    write_tiled_field_snapshot_to_iteration,
    write_tiled_particle_snapshot_to_iteration,
)


def open_parallel_series(output_dir, comm=None, filename="simulation.h5"):
    """Open a common series; only a supplied communicator enables MPI."""
    return _open_openpmd_series(output_dir, filename, file_extension=None, comm=comm,
                               access_mode=io.Access.create)


def write_parallel_snapshot(series, comm, static, dynamic, *, fields=None, particles=None):
    """Write owned chunks collectively with the serial mesh and particle schema."""
    try:
        snapshot = fields if fields is not None else particles
        if fields is not None and particles is not None:
            if fields.step != particles.step or fields.time != particles.time:
                raise ValueError("Field and particle snapshots must describe the same output time.")
        iteration = series.iterations[int(snapshot.step)]
        iteration.open()
        iteration.time = float(snapshot.time)
        iteration.dt = float(dynamic.dt)
        iteration.time_unit_SI = 1.0
        buffers = []

        if fields is not None:
            layout = TiledMeshLayout(
                global_shape=tuple(int(getattr(dynamic, "N" + axis)) for axis in "xyz"),
                tile_shape=tuple(int(width) for width in static.tile_shape),
                guard_cells=int(static.guard_cells),
            )
            buffers.extend(write_tiled_field_snapshot_to_iteration(
                iteration, fields, dynamic_parameters=dynamic, layout=layout))
        if particles is not None:
            buffers.extend(write_tiled_particle_snapshot_to_iteration(
                iteration, particles, static, dynamic, comm=comm))

        # Empty ranks must enter the same flush and close operations.
        iteration.series_flush()
        iteration.close()
    except BaseException:
        _abort_parallel_write(comm)
        raise
