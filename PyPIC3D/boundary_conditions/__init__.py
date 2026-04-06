# Christopher Woolford 2024
# Public API for the boundary_conditions subpackage.

from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_PERIODIC,
    BC_CONDUCTING,
)

from PyPIC3D.boundary_conditions.boundaryconditions import (
    update_ghost_cells,
    fold_ghost_cells,
    apply_conducting_bc,
    apply_scalar_conducting_bc,
    apply_supergaussian_boundary_condition,
    apply_zero_boundary_condition,
)
