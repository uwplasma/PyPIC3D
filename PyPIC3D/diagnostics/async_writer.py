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
    write_tiled_particle_snapshot_openpmd,
)

ArrayLike = Any
ShardIndex = Tuple[Union[slice, int], ...]
HostShard = Tuple[ShardIndex, np.ndarray]
HostShardList = List[HostShard]
FieldValue = Union[ArrayLike, Sequence[ArrayLike]]
SnapshotFieldValue = Union[HostShardList, Tuple[HostShardList, HostShardList, HostShardList]]


_OPENPMD_WRITE_LOCK = threading.Lock()


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


@dataclass(frozen=True)
class TiledParticleSnapshot:
    """
    Host-owned snapshot of tiled particle data.

    The snapshot keeps the fixed-capacity tile/species/slot layout. The writer
    thread compacts active slots by species before writing openPMD records.
    """

    step: int
    time: float
    species_names: Tuple[str, ...]
    x_shards: HostShardList
    u_shards: HostShardList
    active_shards: HostShardList
    species_charge: np.ndarray
    species_mass: np.ndarray
    species_weight: np.ndarray


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


def prefetch_tiled_particles_to_host(particles):
    for value in (particles.x, particles.u, particles.active):
        if hasattr(value, "copy_to_host_async"):
            value.copy_to_host_async()


def _species_names_for_output(species_names, n_species):
    if species_names is None:
        return tuple(f"species_{species_index}" for species_index in range(n_species))
    return tuple(str(name) for name in species_names)


def _species_array_to_host(value):
    value = jax.device_get(value)
    return np.asarray(value, dtype=np.float64)


def make_tiled_particle_snapshot(
    particles,
    *,
    step,
    time,
    species_names,
    species_config,
):
    names = _species_names_for_output(species_names, int(particles.active.shape[3]))

    return TiledParticleSnapshot(
        step=int(step),
        time=float(time),
        species_names=names,
        x_shards=_copy_array_to_host_shards(particles.x),
        u_shards=_copy_array_to_host_shards(particles.u),
        active_shards=_copy_array_to_host_shards(particles.active),
        species_charge=_species_array_to_host(species_config.charge),
        species_mass=_species_array_to_host(species_config.mass),
        species_weight=_species_array_to_host(species_config.weight),
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
        static_parameters,
        dynamic_parameters,
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
        self.static_parameters = static_parameters
        self.dynamic_parameters = dynamic_parameters
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
                    # openPMD/HDF5 native writes are serialized across field and
                    # particle diagnostics; host snapshot construction remains
                    # outside this lock.
                    with _OPENPMD_WRITE_LOCK:
                        write_tiled_field_snapshot_openpmd(
                            item,
                            output_dir=self.output_dir,
                            filename=self.filename,
                            dynamic_parameters=self.dynamic_parameters,
                            layout=self.layout,
                            file_extension=self.file_extension,
                        )

            except BaseException as exc:
                self._error = exc
                self._error_traceback = traceback.format_exc()
            finally:
                self._queue.task_done()


class AsyncTiledOpenPMDParticleWriter:
    """
    Bounded background queue writer for tiled openPMD particle output.
    """

    def __init__(
        self,
        *,
        output_dir,
        filename,
        static_parameters,
        dynamic_parameters,
        file_extension=".bp",
        dtype=np.float64,
        queue_size=2,
        raise_writer_errors_on_close=True,
    ):
        queue_size = max(1, int(queue_size))

        self.output_dir = output_dir
        self.filename = filename
        self.static_parameters = static_parameters
        self.dynamic_parameters = dynamic_parameters
        self.file_extension = file_extension
        self.dtype = dtype
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
                "Async openPMD particle writer failed. Original traceback:\n"
                f"{self._error_traceback}"
            ) from self._error

        try:
            self._queue.put(snapshot, block=block)
            return True
        except queue.Full:
            return False

    def enqueue_particles(self, particles, *, step, time, species_config, species_names=None, block=True):
        snapshot = make_tiled_particle_snapshot(
            particles,
            step=step,
            time=time,
            species_config=species_config,
            species_names=species_names,
        )
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
                "Async openPMD particle writer failed. Original traceback:\n"
                f"{self._error_traceback}"
            ) from self._error

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return

                if self._error is None:
                    # Share the same native-write lock as field diagnostics so
                    # independent output streams never enter openPMD concurrently.
                    with _OPENPMD_WRITE_LOCK:
                        write_tiled_particle_snapshot_openpmd(
                            item,
                            output_dir=self.output_dir,
                            filename=self.filename,
                            static_parameters=self.static_parameters,
                            dynamic_parameters=self.dynamic_parameters,
                            file_extension=self.file_extension,
                            dtype=self.dtype,
                        )

            except BaseException as exc:
                self._error = exc
                self._error_traceback = traceback.format_exc()
            finally:
                self._queue.task_done()


def create_async_tiled_openpmd_field_writer(
    static_parameters,
    dynamic_parameters,
    output_dir,
    *,
    filename="fields",
    file_extension=".h5",
    queue_size=2,
):
    writer = AsyncTiledOpenPMDFieldWriter(
        output_dir=output_dir,
        filename=filename,
        static_parameters=static_parameters,
        dynamic_parameters=dynamic_parameters,
        global_shape=(int(dynamic_parameters.Nx), int(dynamic_parameters.Ny), int(dynamic_parameters.Nz)),
        tile_shape=tuple(int(width) for width in static_parameters.tile_shape),
        guard_cells=int(static_parameters.guard_cells),
        active_dims=(1, 1, 1),
        file_extension=file_extension,
        queue_size=queue_size,
    )
    writer.start()
    return writer


def create_async_tiled_openpmd_particle_writer(
    static_parameters,
    dynamic_parameters,
    output_dir,
    *,
    filename="particles",
    file_extension=".h5",
    queue_size=2,
):
    writer = AsyncTiledOpenPMDParticleWriter(
        output_dir=output_dir,
        filename=filename,
        static_parameters=static_parameters,
        dynamic_parameters=dynamic_parameters,
        file_extension=file_extension,
        queue_size=queue_size,
    )
    writer.start()
    return writer


def enqueue_openpmd_field_output(field_writer, fields, dynamic_parameters, plot_t, t, *, block=True):
    field_map = _fields_to_tiled_output_map(fields)
    prefetch_field_map_to_host(field_map)
    return field_writer.enqueue_fields(
        field_map,
        step=int(plot_t),
        time=float(t * dynamic_parameters.dt),
        block=block,
    )


def enqueue_openpmd_particle_output(
    particle_writer,
    particles,
    dynamic_parameters,
    plot_t,
    t,
    *,
    species_config,
    species_names=None,
    block=True,
):
    prefetch_tiled_particles_to_host(particles)
    return particle_writer.enqueue_particles(
        particles,
        step=int(plot_t),
        time=float(t * dynamic_parameters.dt),
        species_config=species_config,
        species_names=species_names,
        block=block,
    )
