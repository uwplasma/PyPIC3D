from tests.support.yee_fixtures import (
    YeeTiledFixtures as SharedFixtures,
    assemble_tiled_vector_field,
    ghost_cells,
    tile_vector_field,
    update_B,
    update_E,
)


class YeeTiledFixtures(SharedFixtures):
    def _reference_update_E(self, E, B, J, parameter_set, dynamic_values):
        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, tile_shape, int(parameter_set["guard_cells"]))
        static_parameters, dynamic_parameters = self._split_parameters(reference_parameters, dynamic_values)
        E_reference, pml_state = update_E(
            tile_vector_field(E, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            tile_vector_field(B, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            tile_vector_field(J, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            static_parameters,
            dynamic_parameters,
        )
        return assemble_tiled_vector_field(
            E_reference,
            reference_parameters,
            tile_shape,
            num_guard_cells=int(reference_parameters["guard_cells"]),
        ), pml_state

    def _reference_update_B(self, E, B, parameter_set, dynamic_values):
        tile_shape = (parameter_set["Nx"], parameter_set["Ny"], parameter_set["Nz"])
        reference_parameters = self._copy_parameters_for_tile_shape(parameter_set, tile_shape, int(parameter_set["guard_cells"]))
        static_parameters, dynamic_parameters = self._split_parameters(reference_parameters, dynamic_values)
        B_reference, pml_state = update_B(
            tile_vector_field(E, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            tile_vector_field(B, reference_parameters, tile_shape, num_guard_cells=int(reference_parameters["guard_cells"])),
            static_parameters,
            dynamic_parameters,
        )
        return assemble_tiled_vector_field(
            B_reference,
            reference_parameters,
            tile_shape,
            num_guard_cells=int(reference_parameters["guard_cells"]),
        ), pml_state

    def _reference_yee_step(self, E, B, J, parameter_set, dynamic_values):
        E_reference, pml_state = self._reference_update_E(E, B, J, parameter_set, dynamic_values)
        B_reference, pml_state = self._reference_update_B(E_reference, B, parameter_set, dynamic_values)
        return E_reference, B_reference, pml_state

    def _copy_parameters_for_tile_shape(self, parameter_set, tile_shape, g=2):
        reference_parameters = dict(parameter_set)
        reference_parameters["boundary_conditions"] = dict(parameter_set["boundary_conditions"])
        reference_parameters["grids"] = dict(parameter_set["grids"])
        reference_parameters["tile_shape"] = tuple(int(width) for width in tile_shape)
        reference_parameters["guard_cells"] = int(g)
        reference_parameters["field_mesh"] = ghost_cells.make_field_mesh((
            int(reference_parameters["Nx"]) // int(tile_shape[0]),
            int(reference_parameters["Ny"]) // int(tile_shape[1]),
            int(reference_parameters["Nz"]) // int(tile_shape[2]),
        ))
        return reference_parameters
