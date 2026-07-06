from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple, Union
import queue
import threading
import traceback

import jax
import numpy as np

from PyPIC3D.diagnostics.openPMD import (
    TiledMeshLayout,
    write_tiled_field_snapshot_openpmd,
)

ArrayLike = Any
ShardIndex = Tuple[Union[slice, int], ...]
HostShard = Tuple[ShardIndex, np.ndarray]
HostShardList = List[HostShard]
FieldValue = Union[ArrayLike, Sequence[ArrayLike]]
SnapshotFieldValue = Union[HostShardList, Tuple[HostShardList, HostShardList, HostShardList]]


@dataclass(frozen=True)
class TiledFieldSnapshot:
    """
    Host-owned snapshot of tiled field data.

    Scalar fields store a list of ``(shard_index, host_array)`` chunks. Vector
    fields store one such list for each component. The first three array axes
    are tile indices; the last three are tile-local grid indices including
    guard cells.
    """

    step: int
    time: float
    fields: Mapping[str, SnapshotFieldValue]


def _fields_to_tiled_output_map(fields):
    """Return the tiled field components currently written by field diagnostics."""
    E, B, J, rho, phi, external_fields, *rest = fields
    external_E, external_B = external_fields
    return {
        "E": E,
        "B": B,
        "J": J,
        "rho": rho,
        "phi": phi,
        "external_E": external_E,
        "external_B": external_B,
    }


def _copy_array_to_host_shards(arr):
    """
    Copy a possibly-sharded JAX array to host-owned NumPy chunks.

    Only addressable shards are copied. In ordinary single-process runs this is
    the complete tiled field. Multi-process distributed output can build on this
    same shard contract later.
    """
    if hasattr(arr, "addressable_shards"):
        out = []
        for shard in arr.addressable_shards:
            host = np.array(jax.device_get(shard.data), copy=True, order="C")
            out.append((tuple(shard.index), host))
        return out

    try:
        arr = jax.device_get(arr)
    except Exception:
        pass

    host = np.array(arr, copy=True, order="C")
    full_index = tuple(slice(0, n) for n in host.shape)
    return [(full_index, host)]


def prefetch_field_map_to_host(field_map):
    """
    Start asynchronous copies for JAX arrays that support it.

    ``make_tiled_field_snapshot`` still performs the real host copy before the
    snapshot enters the queue.
    """
    def _prefetch(value):
        if hasattr(value, "copy_to_host_async"):
            value.copy_to_host_async()

    for value in field_map.values():
        if isinstance(value, (tuple, list)):
            for component in value:
                _prefetch(component)
        else:
            _prefetch(value)


def make_tiled_field_snapshot(field_map, *, step, time):
    copied = {}

    for name, value in field_map.items():
        is_vector = isinstance(value, (tuple, list)) and len(value) == 3
        if is_vector:
            copied[name] = tuple(_copy_array_to_host_shards(component) for component in value)
        else:
            copied[name] = _copy_array_to_host_shards(value)

    return TiledFieldSnapshot(
        step=int(step),
        time=float(time),
        fields=copied,
    )


class AsyncTiledOpenPMDFieldWriter:
    """
    Bounded background queue writer for tiled openPMD field output.

    The simulation thread copies tile-major fields to a host snapshot and puts
    that snapshot in a bounded queue. The writer thread strips guard cells and
    writes tile interiors as chunks in the global openPMD mesh.
    """

    def __init__(
        self,
        *,
        output_dir,
        filename,
        world,
        global_shape,
        tile_shape,
        guard_cells=1,
        active_dims=(1, 1, 1),
        file_extension=".bp",
        dtype=np.float64,
        queue_size=2,
        raise_writer_errors_on_close=True,
    ):
        queue_size = max(1, int(queue_size))

        self.output_dir = output_dir
        self.filename = filename
        self.world = world
        self.layout = TiledMeshLayout(
            global_shape=tuple(int(width) for width in global_shape),
            tile_shape=tuple(int(width) for width in tile_shape),
            guard_cells=guard_cells,
            active_dims=active_dims,
            dtype=dtype,
        )
        self.file_extension = file_extension
        self.raise_writer_errors_on_close = bool(raise_writer_errors_on_close)

        self._queue = queue.Queue(maxsize=queue_size)
        self._thread = None
        self._closed = False
        self._error = None
        self._error_traceback = None

    def start(self):
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def enqueue(self, snapshot, *, block=True):
        if self._closed:
            raise RuntimeError("Cannot enqueue after writer.close().")
        if self._error is not None:
            raise RuntimeError(
                "Async openPMD writer failed. Original traceback:\n"
                f"{self._error_traceback}"
            ) from self._error

        try:
            self._queue.put(snapshot, block=block)
            return True
        except queue.Full:
            return False

    def enqueue_fields(self, field_map, *, step, time, block=True):
        snapshot = make_tiled_field_snapshot(field_map, step=step, time=time)
        return self.enqueue(snapshot, block=block)

    def close(self, raise_errors=None):
        if self._closed:
            return

        self._closed = True
        if self._thread is not None:
            self._queue.put(None)
            self._queue.join()
            self._thread.join()
            self._thread = None

        if raise_errors is None:
            raise_errors = self.raise_writer_errors_on_close
        if self._error is not None and raise_errors:
            raise RuntimeError(
                "Async openPMD writer failed. Original traceback:\n"
                f"{self._error_traceback}"
            ) from self._error

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return

                if self._error is None:
                    write_tiled_field_snapshot_openpmd(
                        item,
                        output_dir=self.output_dir,
                        filename=self.filename,
                        world=self.world,
                        layout=self.layout,
                        file_extension=self.file_extension,
                    )

            except BaseException as exc:
                self._error = exc
                self._error_traceback = traceback.format_exc()
            finally:
                self._queue.task_done()


def create_async_tiled_openpmd_field_writer(
    world,
    output_dir,
    *,
    filename="fields",
    file_extension=".h5",
    queue_size=2,
):
    writer = AsyncTiledOpenPMDFieldWriter(
        output_dir=output_dir,
        filename=filename,
        world=world,
        global_shape=(int(world["Nx"]), int(world["Ny"]), int(world["Nz"])),
        tile_shape=tuple(int(width) for width in world["tile_shape"]),
        guard_cells=int(world["guard_cells"]),
        active_dims=(1, 1, 1),
        file_extension=file_extension,
        queue_size=queue_size,
    )
    writer.start()
    return writer


def enqueue_openpmd_field_output(field_writer, fields, world, plot_t, t, *, block=True):
    field_map = _fields_to_tiled_output_map(fields)
    prefetch_field_map_to_host(field_map)
    return field_writer.enqueue_fields(
        field_map,
        step=int(plot_t),
        time=float(t * world["dt"]),
        block=block,
    )
