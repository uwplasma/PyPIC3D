"""Assemble the per-process openPMD series a multiprocess run writes.

Each JAX process writes its own ``<stem>_pNNN.<ext>`` series without talking to
any other process. This tool joins a group back into the single
``<stem>.<ext>`` file that a one-process run produces, which every analysis
script already reads.

The HDF5 backend reports every dataset as one whole chunk, so a reader cannot
discover which regions a file actually wrote. ``PyPIC3D.diagnostics.openPMD``
therefore records ownership as attributes, and this tool reads them back.
"""

from dataclasses import dataclass
from typing import Tuple

import argparse
import os
import re

import numpy as np
import openpmd_api as io

from PyPIC3D.utilities.simulation_helpers import setup_pmd_files


PROCESS_SUFFIX_PATTERN = re.compile(r"^(?P<stem>.+)_p(?P<index>\d+)(?P<extension>\.h5|\.bp)$")

PARTICLE_VECTOR_RECORDS = ("position", "positionOffset", "momentum")
PARTICLE_SCALAR_RECORDS = ("weighting", "charge", "mass", "gamma")

# openPMD orders the seven base dimensions L, M, T, I, theta, N, J.
UNIT_DIMENSION_ORDER = ("L", "M", "T", "I", "theta", "N", "J")

OWNERSHIP_ATTRIBUTES = frozenset({
    "pypic3dOwnedTiles", "pypic3dOwnedTileCounts", "pypic3dOwnedTileCount",
    "pypic3dTileCounts", "pypic3dTileShape", "pypic3dGlobalShape",
    "pypic3dProcessIndex", "pypic3dProcessCount",
})

STANDARD_MESH_ATTRIBUTES = frozenset({
    "geometry", "geometryParameters", "dataOrder", "axisLabels", "gridSpacing",
    "gridGlobalOffset", "gridUnitSI", "unitDimension", "timeOffset", "unitSI",
    "position",
})


DEFAULT_MAX_PENDING_BYTES = 1 << 30


@dataclass(frozen=True)
class TileGeometry:
    """Global tile decomposition plus which tiles each source file owns."""

    global_shape: Tuple[int, ...]
    tile_shape: Tuple[int, ...]
    tile_counts: Tuple[int, ...]
    owned: Tuple[np.ndarray, ...]

    def ordered_tiles(self):
        """Yield (source, tile id) in ascending global tile order."""
        pairs = [(int(tile), source)
                 for source, ids in enumerate(self.owned) for tile in ids]
        pairs.sort()
        for tile, source in pairs:
            yield source, tile

    def tile_offset(self, tile):
        """Global cell offset of a flattened tile id."""
        return [int(index)*int(width) for index, width
                in zip(np.unravel_index(int(tile), self.tile_counts), self.tile_shape)]


@dataclass
class MergeReport:
    output: str
    sources: Tuple[str, ...]
    iterations: Tuple[int, ...]


def _resolve_pmd(path):
    """Resolve PyPIC3D's one-line .pmd pointer to the series it names."""
    with open(path) as pointer:
        lines = [line.strip() for line in pointer if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"{path} must name exactly one series, found {len(lines)} lines")
    return os.path.join(os.path.dirname(path), lines[0])


def discover_series_groups(inputs):
    """Group ``<stem>_pNNN.<ext>`` siblings named by paths, files or directories."""
    groups = {}

    def add(path):
        directory, name = os.path.split(os.path.abspath(path))
        match = PROCESS_SUFFIX_PATTERN.match(name)
        if match is None:
            return False
        key = (directory, match.group("stem"), match.group("extension"))
        groups.setdefault(key, {})[int(match.group("index"))] = os.path.join(directory, name)
        return True

    for entry in inputs:
        entry = str(entry)
        if entry.endswith(".pmd") and os.path.isfile(entry):
            entry = _resolve_pmd(entry)
        if not os.path.exists(entry):
            raise FileNotFoundError(entry)
        # BP series are directories too; recognize members before scanning folders.
        if add(entry):
            # One member names its whole group; pick up its siblings.
            directory, name = os.path.split(os.path.abspath(entry))
            match = PROCESS_SUFFIX_PATTERN.match(name)
            prefix = match.group("stem") + "_p"
            for sibling in sorted(os.listdir(directory)):
                if sibling.startswith(prefix):
                    add(os.path.join(directory, sibling))
        elif os.path.isdir(entry):
            for name in sorted(os.listdir(entry)):
                add(os.path.join(entry, name))

    ordered = []
    for (directory, stem, extension), members in sorted(groups.items()):
        paths = tuple(members[index] for index in sorted(members))
        ordered.append((directory, stem, extension, paths))
    return ordered


def _attribute(attributable, name):
    return attributable.get_attribute(name) if attributable.contains_attribute(name) else None


def _int_vector(attributable, name):
    """openPMD returns a length-one numeric vector attribute as a scalar."""
    return np.atleast_1d(np.asarray(attributable.get_attribute(name), dtype=np.int64))


def read_tile_geometry(source_iterations):
    """Read ownership metadata and check the sources describe one decomposition."""
    global_shape = tile_shape = tile_counts = None
    owned = []
    for index, iteration in enumerate(source_iterations):
        shape = tuple(int(n) for n in iteration.get_attribute("pypic3dGlobalShape"))
        widths = tuple(int(n) for n in iteration.get_attribute("pypic3dTileShape"))
        counts = tuple(int(n) for n in iteration.get_attribute("pypic3dTileCounts"))
        if global_shape is None:
            global_shape, tile_shape, tile_counts = shape, widths, counts
        elif (shape, widths, counts) != (global_shape, tile_shape, tile_counts):
            raise ValueError(
                f"Source {index} describes tiles {widths} of {counts} over {shape}, "
                f"but source 0 describes {tile_shape} of {tile_counts} over {global_shape}.")
        count = int(iteration.get_attribute("pypic3dOwnedTileCount"))
        ids = (_int_vector(iteration, "pypic3dOwnedTiles")
               if count else np.empty(0, dtype=np.int64))
        if ids.size != count:
            raise ValueError(f"Source {index} lists {ids.size} tiles but claims {count}.")
        owned.append(np.sort(ids))
    return TileGeometry(global_shape, tile_shape, tile_counts, tuple(owned))


def validate_tile_coverage(geometry):
    """Owned tiles must partition the global tile grid exactly once."""
    coverage = np.zeros(int(np.prod(geometry.tile_counts)), dtype=np.int64)
    for ids in geometry.owned:
        np.add.at(coverage, ids, 1)
    missing = np.flatnonzero(coverage == 0)
    duplicated = np.flatnonzero(coverage > 1)
    if missing.size or duplicated.size:
        raise ValueError(
            f"Owned tiles do not partition the {geometry.tile_counts} tile grid: "
            f"{missing.size} unwritten (first {missing[:5].tolist()}), "
            f"{duplicated.size} duplicated (first {duplicated[:5].tolist()}). "
            "The group is incomplete or mixes runs with different process counts.")


def _copy_extra_attributes(source, destination, skip=frozenset()):
    for name in source.attributes:
        if name in skip or name in OWNERSHIP_ATTRIBUTES:
            continue
        destination.set_attribute(name, source.get_attribute(name))


def _copy_unit_dimension(source, destination):
    values = np.asarray(source.unit_dimension, dtype=float)
    exponents = {getattr(io.Unit_Dimension, symbol): float(value)
                 for symbol, value in zip(UNIT_DIMENSION_ORDER, values) if value}
    if exponents:
        destination.unit_dimension = exponents


def _drain(pending, destination_record, extent, source_iterations, destination_iteration):
    """Complete the deferred loads, then the stores, then release the buffers."""
    if not pending:
        return
    for iteration in source_iterations:
        iteration.series_flush()
    for offset, data in pending:
        destination_record.store_chunk(data, offset, extent)
    destination_iteration.series_flush()
    pending.clear()


def _merge_meshes(source_iterations, destination_iteration, geometry,
                  max_pending_bytes=DEFAULT_MAX_PENDING_BYTES):
    """Overlay every source's owned tiles into one global-extent mesh record."""
    for name, source_mesh in source_iterations[0].meshes.items():
        destination_mesh = destination_iteration.meshes[name]
        destination_mesh.geometry = source_mesh.geometry
        destination_mesh.data_order = source_mesh.data_order
        destination_mesh.axis_labels = source_mesh.axis_labels
        destination_mesh.grid_spacing = source_mesh.grid_spacing
        destination_mesh.grid_global_offset = source_mesh.grid_global_offset
        destination_mesh.set_grid_unit_SI(source_mesh.grid_unit_SI)
        destination_mesh.unit_SI = source_mesh.unit_SI
        _copy_unit_dimension(source_mesh, destination_mesh)
        if source_mesh.contains_attribute("timeOffset"):
            destination_mesh.time_offset = source_mesh.time_offset
        _copy_extra_attributes(source_mesh, destination_mesh, STANDARD_MESH_ATTRIBUTES)

        for component, source_record in source_mesh.items():
            destination_record = destination_mesh[component]
            destination_record.reset_dataset(
                io.Dataset(source_record.dtype, list(geometry.global_shape)))
            destination_record.unit_SI = source_record.unit_SI
            destination_record.position = source_record.position
            _copy_extra_attributes(source_record, destination_record, STANDARD_MESH_ATTRIBUTES)

            # load_chunk is deferred until the source flushes, and a stored
            # buffer must stay alive until the destination flushes. Tiles are
            # drained in batches so one component need not fit in memory.
            extent = list(geometry.tile_shape)
            tile_bytes = int(np.prod(extent))*np.dtype(source_record.dtype).itemsize
            per_batch = max(1, max_pending_bytes // max(1, tile_bytes))
            pending = []
            for source_index, tile in geometry.ordered_tiles():
                offset = geometry.tile_offset(tile)
                record = source_iterations[source_index].meshes[name][component]
                pending.append((offset, record.load_chunk(offset, extent)))
                if len(pending) >= per_batch:
                    _drain(pending, destination_record, extent,
                           source_iterations, destination_iteration)
            _drain(pending, destination_record, extent,
                   source_iterations, destination_iteration)


def _particle_blocks(source_iterations, species_name, geometry):
    """Locate each owned tile's slice of every source's local records."""
    blocks = []
    for source_index, iteration in enumerate(source_iterations):
        species = iteration.particles[species_name]
        tiles = _int_vector(species, "pypic3dOwnedTiles")
        counts = _int_vector(species, "pypic3dOwnedTileCounts")
        if tiles.size != counts.size:
            raise ValueError(f"Source {source_index} lists {tiles.size} tiles for "
                             f"{species_name} but {counts.size} counts.")
        local = np.cumsum(counts) - counts
        for tile, start, count in zip(tiles, local, counts):
            if count:
                blocks.append((int(tile), source_index, int(start), int(count)))
    blocks.sort()
    return blocks


def _merge_particles(source_iterations, destination_iteration, geometry):
    """Concatenate locally sized records back into global tile order."""
    for species_name, source_species in source_iterations[0].particles.items():
        destination_species = destination_iteration.particles[species_name]
        blocks = _particle_blocks(source_iterations, species_name, geometry)
        total = sum(block[3] for block in blocks)

        present = set(dict(source_species.items()))
        components = []
        for record_name in PARTICLE_VECTOR_RECORDS:
            if record_name in present:
                components.extend((record_name, axis) for axis in ("x", "y", "z"))
        for record_name in PARTICLE_SCALAR_RECORDS:
            if record_name in present:
                components.append((record_name, io.Mesh_Record_Component.SCALAR))

        for record_name, component in components:
            source_record = source_species[record_name][component]
            destination_record = destination_species[record_name][component]
            destination_record.reset_dataset(io.Dataset(source_record.dtype, [int(total)]))
            destination_record.unit_SI = source_record.unit_SI

            pending, running = [], 0
            for tile, source_index, start, count in blocks:
                record = source_iterations[source_index].particles[species_name][record_name][component]
                pending.append((running, count, record.load_chunk([start], [count])))
                running += count
            for iteration in source_iterations:
                iteration.series_flush()
            for offset, count, data in pending:
                destination_record.store_chunk(data, [offset], [count])
            destination_iteration.series_flush()
            pending.clear()

        for record_name in PARTICLE_VECTOR_RECORDS + PARTICLE_SCALAR_RECORDS:
            if record_name in present:
                _copy_unit_dimension(source_species[record_name], destination_species[record_name])
        _copy_extra_attributes(source_species, destination_species)


def merge_series(sources, destination, *, names=(), iterations=None, log=None,
                 max_pending_bytes=DEFAULT_MAX_PENDING_BYTES):
    """Copy owned regions from already-open sources into an open destination."""
    steps = sorted(set(sources[0].iterations))
    for index, series in enumerate(sources[1:], start=1):
        if sorted(set(series.iterations)) != steps:
            raise ValueError(f"Source {index} holds iterations "
                             f"{sorted(set(series.iterations))}, source 0 holds {steps}.")
    if iterations is not None:
        steps = [step for step in steps if step in set(iterations)]

    _copy_extra_attributes(sources[0], destination)
    if names:
        destination.set_attribute("pypic3dMergedFrom", [os.path.basename(name) for name in names])

    for step in steps:
        source_iterations = [series.iterations[step] for series in sources]
        geometry = read_tile_geometry(source_iterations)
        validate_tile_coverage(geometry)

        out = destination.iterations[step]
        out.time = source_iterations[0].time
        out.dt = source_iterations[0].dt
        out.time_unit_SI = source_iterations[0].time_unit_SI
        _copy_extra_attributes(source_iterations[0], out, frozenset({"time", "dt", "timeUnitSI"}))

        _merge_meshes(source_iterations, out, geometry, max_pending_bytes)
        _merge_particles(source_iterations, out, geometry)
        if log is not None:
            log(f"iteration {step}: {int(np.prod(geometry.tile_counts))} tiles "
                f"from {len(sources)} files")
    return steps


def merge_paths(inputs, output=None, *, overwrite=False, write_pmd=True, iterations=None,
                log=None, max_pending_bytes=DEFAULT_MAX_PENDING_BYTES):
    """Merge every ``_pNNN`` group named by ``inputs``; return one report each."""
    groups = discover_series_groups(inputs)
    if not groups:
        raise FileNotFoundError(
            "No _pNNN openPMD series found. A single-process run writes one unsuffixed "
            "file and needs no merge.")

    if output is not None and len(groups) > 1:
        raise ValueError(
            f"--output names one series but {len(groups)} groups were found "
            f"({', '.join(stem for _, stem, _, _ in groups)}). Name one group member, "
            "or drop --output to merge each in place.")

    reports = []
    for directory, stem, extension, paths in groups:
        target = output if output is not None else os.path.join(directory, stem + extension)
        if os.path.exists(target) and not overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite to replace it")

        sources = [io.Series(path, io.Access.read_only) for path in paths]
        _validate_group(sources, paths)
        destination = io.Series(target, io.Access.create)
        try:
            steps = merge_series(sources, destination, names=paths, iterations=iterations,
                                 log=log, max_pending_bytes=max_pending_bytes)
        finally:
            destination.close()
            for series in sources:
                series.close()
        if write_pmd:
            target_directory, target_name = os.path.split(os.path.abspath(target))
            target_stem, target_extension = os.path.splitext(target_name)
            setup_pmd_files(target_directory, target_stem, target_extension)
        reports.append(MergeReport(target, paths, tuple(steps)))
    return reports


def _validate_group(sources, paths):
    """The group must be exactly one run's processes, each present once."""
    expected = _attribute(sources[0], "pypic3dProcessCount")
    indices = []
    for series, path in zip(sources, paths):
        count = _attribute(series, "pypic3dProcessCount")
        if count != expected:
            raise ValueError(f"{path} was written by a run of {count} processes, "
                             f"but {paths[0]} by a run of {expected}.")
        indices.append(_attribute(series, "pypic3dProcessIndex"))
    if expected is None:
        raise ValueError(f"{paths[0]} carries no PyPIC3D ownership metadata; it was not "
                         "written by this version of PyPIC3D.")
    if sorted(indices) != list(range(int(expected))):
        raise ValueError(f"Expected processes 0..{int(expected)-1}, found {sorted(indices)}. "
                         "Stale files from a run with a different process count?")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="PyPIC3D-merge-openpmd",
        description="Join the per-process openPMD series of a multiprocess PyPIC3D run.")
    parser.add_argument("inputs", nargs="*", default=["."],
                        help="Data directories, one group member, or a .pmd pointer")
    parser.add_argument("-o", "--output", help="Merged series path (default <stem><ext>)")
    parser.add_argument("--iterations", help="Comma-separated subset of steps to merge")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing merged file")
    parser.add_argument("--no-pmd", action="store_true", help="Do not rewrite the .pmd pointer")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    steps = None
    if args.iterations:
        steps = [int(value) for value in args.iterations.split(",") if value.strip()]
    log = print if args.verbose else None
    try:
        reports = merge_paths(args.inputs or ["."], args.output, overwrite=args.overwrite,
                              write_pmd=not args.no_pmd, iterations=steps, log=log)
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    except FileNotFoundError as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    for report in reports:
        print(f"merged {len(report.sources)} files into {report.output} "
              f"({len(report.iterations)} iterations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
