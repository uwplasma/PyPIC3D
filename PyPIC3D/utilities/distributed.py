"""Distributed startup and addressable-only Cartesian field initialization."""

import os


def initialize_distributed():
    """Opt in with PYPIC_DISTRIBUTED=1 before importing PyPIC3D.

    JAX detects Slurm and Open MPI launches on its own. PYPIC_COORDINATOR_ADDRESS,
    PYPIC_PROCESS_ID and PYPIC_NUM_PROCESSES describe a launch it cannot detect.
    Slurm may expose one GPU per task; PYPIC_LOCAL_DEVICE_IDS overrides JAX's
    local-device selection in that case.
    """
    if os.environ.get("PYPIC_DISTRIBUTED", "0") != "1":
        return
    import jax

    if jax.distributed.is_initialized():
        return
    ids = os.environ.get("PYPIC_LOCAL_DEVICE_IDS")
    processes = os.environ.get("PYPIC_NUM_PROCESSES")
    process = os.environ.get("PYPIC_PROCESS_ID")
    jax.distributed.initialize(
        coordinator_address=os.environ.get("PYPIC_COORDINATOR_ADDRESS"),
        num_processes=None if processes is None else int(processes),
        process_id=None if process is None else int(process),
        local_device_ids=None if ids is None else [int(v) for v in ids.split(",")],
        initialization_timeout=int(os.environ.get("PYPIC_INITIALIZATION_TIMEOUT", "120")),
    )


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
