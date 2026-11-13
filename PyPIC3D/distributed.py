"""Distributed startup and addressable-only Cartesian field initialization."""

import os


def initialize_distributed():
    """Opt in with PYPIC_DISTRIBUTED=1 before importing PyPIC3D.

    MPI and JAX share process numbering. Slurm may expose one GPU per task;
    PYPIC_LOCAL_DEVICE_IDS overrides JAX's MPI local-rank selection in that case.
    """
    if os.environ.get("PYPIC_DISTRIBUTED", "0") != "1":
        return
    from mpi4py import MPI
    import jax

    if not jax.distributed.is_initialized():
        ids = os.environ.get("PYPIC_LOCAL_DEVICE_IDS")
        jax.distributed.initialize(
            cluster_detection_method="mpi4py",
            local_device_ids=None if ids is None else [int(v) for v in ids.split(",")],
            initialization_timeout=int(os.environ.get("PYPIC_INITIALIZATION_TIMEOUT", "120")),
        )
    if (jax.process_index(), jax.process_count()) != (MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size):
        raise ValueError("MPI and JAX must use the same process numbering")


def mpi_communicator():
    """Return the simulation communicator only for a multiprocess JAX run."""
    import jax
    if jax.process_count() == 1:
        return None
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if (jax.process_index(), jax.process_count()) != (comm.rank, comm.size):
        raise ValueError("MPI and JAX must use the same process numbering")
    return comm


def tiled_field_from_initializer(initializer, static, dynamic, *, location=("C", "C", "C")):
    """Read native .npy blocks or evaluate a callback on addressable tiles only.

    A callback receives broadcast coordinate arrays (x,y,z), including guards,
    at the requested Yee location. A native file keeps the existing Nx,Ny,Nz
    contract; physical walls are subsequently prepared by the solver.
    """
    import numpy as np
    import jax
    from jax.sharding import NamedSharding
    from PyPIC3D.boundary_conditions.ghost_cells import SCALAR_TILE_SPEC

    widths = tuple(int(n) for n in static.tile_shape)
    counts = tuple(int(n) for n in static.field_mesh.devices.shape)
    g = int(static.guard_cells)
    local_shape = tuple(n + 2*g for n in widths)
    shape = counts + local_shape
    spacing = tuple(float(getattr(dynamic, "d" + axis)) for axis in "xyz")
    origin = tuple(float(axis[1]) for axis in dynamic.grids.center)
    data = np.load(initializer, mmap_mode="r", allow_pickle=False) if isinstance(initializer, (str, os.PathLike)) else None
    if data is not None and data.shape != tuple(n*w for n, w in zip(counts, widths)):
        raise ValueError("Native field file must have shape (Nx, Ny, Nz) without guards")

    def tile_data(index):
        tile = tuple(part.start or 0 for part in index[:3])
        if data is not None:
            value = np.zeros(local_shape, dtype=np.float64)
            source = tuple(slice(t*w, (t+1)*w) for t, w in zip(tile, widths))
            value[tuple(slice(g, g+w) for w in widths)] = data[source]
        elif callable(initializer):
            coordinates = tuple(origin[a] + (tile[a]*widths[a] + np.arange(-g, widths[a]+g)
                               + (0.5 if location[a] == "V" else 0.0))*spacing[a] for a in range(3))
            value = np.broadcast_to(np.asarray(initializer(coordinates[0][:, None, None],
                                     coordinates[1][None, :, None], coordinates[2][None, None, :]),
                                     dtype=np.float64), local_shape)
        else:
            value = np.full(local_shape, initializer, dtype=np.float64)
        return value[None, None, None]

    return jax.make_array_from_callback(shape, NamedSharding(static.field_mesh, SCALAR_TILE_SPEC),
                                        tile_data, dtype=np.float64)
