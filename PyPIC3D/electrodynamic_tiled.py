from PyPIC3D.solvers.yee_tiled import update_tiled_B, update_tiled_E


def time_loop_electrodynamic_tiled(
    particles,
    fields,
    world,
    constants,
    curl_func,
    J_func,
    solver,
    tile_shape,
    relativistic=True,
    particle_pusher="boris",
):
    """
    Advance tiled electrodynamic fields by one Maxwell step without pushing particles.
    """

    del J_func, solver, relativistic, particle_pusher

    E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state = fields
    # unpack the tiled field state.  This first tiled-field pass updates
    # Maxwell fields only; particles and particle boundary conditions are
    # intentionally not touched.

    E_tiles = update_tiled_E(E_tiles, B_tiles, J_tiles, world, constants, curl_func, tile_shape)
    # update electric field from B and the supplied tiled current

    B_tiles = update_tiled_B(E_tiles, B_tiles, world, constants, curl_func, tile_shape)
    # update magnetic field from the newly updated electric field

    fields = (E_tiles, B_tiles, J_tiles, rho, phi, external_fields, pml_state)
    # pack the tiled field state

    return particles, fields
