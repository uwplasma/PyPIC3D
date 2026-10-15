# Christopher Woolford 2024
# Public API for the boundary_conditions subpackage.

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    BC_CONDUCTING,
)

from PyPIC3D.boundary_conditions.ghost_cells import (
    MESH_AXES,
    SCALAR_TILE_SPEC,
    VECTOR_TILE_SPEC,
    BC_TYPE_FIELD,
    BC_TYPE_PARTICLE,
    boundary_plane,
    face_mask,
    owned_nodes,
    staggered_mirror,
    make_field_mesh,
    axis_permutations,
    make_tiled_halo_op,
    make_tiled_wall_op,
    particle_vector_reflecting_parity,
    source_boundaries,
    update_tiled_ghost_cells,
    fold_tiled_ghost_cells,
    apply_tiled_zero_boundary,
    apply_tiled_constant_boundary,
    apply_tiled_pec_boundary,
)

from PyPIC3D.boundary_conditions.supergaussian import (
    SUPERGAUSSIAN_WALLS,
    load_supergaussian_from_toml,
    build_supergaussian_envelope,
    apply_tiled_supergaussian_absorber,
)
