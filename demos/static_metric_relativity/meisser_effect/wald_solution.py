"""
Analytical Entity/FPIC Wald fields on PyPIC3D's tiled Yee grid.

The initial field is the Schwarzschild Wald solution while the analytical
comparison field is the stationary solution for the rotating Kerr metric.
The evolved variables are the native densities ``sqrt(gamma) D^i`` and
``sqrt(gamma) B^i`` of the evolution metric.
"""

from pathlib import Path
from types import SimpleNamespace
import math
import os

import toml


os.environ.setdefault("MPLCONFIGDIR", "/tmp/fpic-wald-demo-matplotlib")

import jax
import jax.numpy as jnp

from PyPIC3D.boundary_conditions.ghost_cells import make_field_mesh
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.diagnostics.static_metric import divergence
from PyPIC3D.boundary_conditions.grid_and_stencil import (
    BC_CONSTANT,
    BC_PERIODIC,
)
from PyPIC3D.boundary_conditions.supergaussian import (
    load_supergaussian_from_toml,
)
from PyPIC3D.utilities.parameters import DynamicParameters, GridParameters, StaticParameters
from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.relativity.field_interpolation import reconstruct_vector
from PyPIC3D.relativity.field_state import densitize_vector, physical_vector
from PyPIC3D.relativity.metrics.kerr_schild import initialize_kerr_schild_spherical_metric
from PyPIC3D.utilities.grids import build_tiled_yee_grids, build_yee_grid


jax.config.update("jax_enable_x64", True)


def load_configuration(config_path):
    """Load the demo TOML and resolve paths relative to that file."""

    config_path = Path(config_path).resolve()
    config = toml.load(config_path)
    output_dir = Path(config["output"]["directory"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    config["output"]["directory"] = str(output_dir.resolve())
    config["config_path"] = str(config_path)
    return config


def event_horizon_radius(mass, spin):
    """Return the outer Kerr event-horizon radius in geometrized units."""

    return float(mass) + math.sqrt(float(mass) ** 2 - float(spin) ** 2)


def grid_bounds(config):
    """Return the radial, polar, and azimuthal domain bounds."""

    physics = config["physics"]
    grid = config["grid"]
    r_inner = event_horizon_radius(physics["mass"], physics["spin"])
    dtheta = 2.0 * math.pi / int(grid["ntheta"])
    theta_inner = float(grid["theta_offset_fraction"]) * dtheta

    return (
        (r_inner, float(grid["r_outer"])),
        (theta_inner, 2.0 * math.pi + theta_inner),
        (float(grid["phi_inner"]), float(grid["phi_outer"])),
    )


def absorber_width(config):
    """Return the outer matching-layer width in length units and grid cells."""

    (r_inner, r_outer), _, _ = grid_bounds(config)
    nr = int(config["grid"]["nr"])
    width = float(config["absorber"]["fraction"]) * (r_outer - r_inner)
    dr = (r_outer - r_inner) / nr
    return width, max(1, int(math.ceil(width / dr)))


def build_pypic_parameters(config):
    """Build PyPIC3D parameters, coordinate grids, and absorber parameters."""

    physics = config["physics"]
    grid = config["grid"]
    time = config["time"]
    absorber = config["absorber"]

    nr = int(grid["nr"])
    ntheta = int(grid["ntheta"])
    nphi = int(grid["nphi"])
    guard_cells = int(grid["guard_cells"])
    (r_inner, r_outer), (theta_inner, theta_outer), (phi_inner, phi_outer) = (
        grid_bounds(config)
    )

    dr = (r_outer - r_inner) / nr
    dtheta = (theta_outer - theta_inner) / ntheta
    dphi = (phi_outer - phi_inner) / nphi
    total_steps = int(round(float(time["t_end"]) / float(time["dt"])))

    static_parameters = StaticParameters(
        name="FPIC Schwarzschild-to-Kerr Wald demo",
        output_dir=config["output"]["directory"],
        Nt=total_steps,
        verbose=False,
        benchmark=False,
        solver="static_metric",
        electrostatic=False,
        relativistic=True,
        particle_pusher="hybrid_boris_geodesic",
        current_deposition="GR_direct",
        current_filter="none",
        metric="kerr_schild_spherical",
        metric_mass=float(physics["mass"]),
        metric_spin=float(physics["spin"]),
        shape_factor=1,
        guard_cells=guard_cells,
        tile_shape=(nr, ntheta, nphi),
        particle_tile_capacity_factor=1.0,
        pml_active=False,
        supergaussian_active=False,
        supergaussian_layers=(),
        boundary_conditions=(BC_CONSTANT, BC_PERIODIC, BC_PERIODIC),
        particle_boundary_conditions=(BC_CONSTANT, BC_PERIODIC, BC_PERIODIC),
        field_mesh=make_field_mesh((1, 1, 1)),
        # These parameters are not used in this demo,
        # but are required for the PyPIC3D parameter dataclass.
        electrostatic_schwarz_tol=None,
        electrostatic_schwarz_max_iterations=None,
        electrostatic_local_cg_tol=None,
        electrostatic_local_cg_max_iterations=None,
        particle_batch_size=1,
    )

    grid_setup = SimpleNamespace(
        dx=jnp.asarray(dr),
        dy=jnp.asarray(dtheta),
        dz=jnp.asarray(dphi),
        Nx=jnp.asarray(nr),
        Ny=jnp.asarray(ntheta),
        Nz=jnp.asarray(nphi),
        x_wind=jnp.asarray(r_outer - r_inner),
        y_wind=jnp.asarray(theta_outer - theta_inner),
        z_wind=jnp.asarray(phi_outer - phi_inner),
        x_min=jnp.asarray(r_inner),
        y_min=jnp.asarray(theta_inner),
        z_min=jnp.asarray(phi_inner),
    )
    center_grid, vertex_grid = build_yee_grid(grid_setup)
    grids = GridParameters(
        vertex=vertex_grid,
        center=center_grid,
        tiled_vertex_grid=(),
        tiled_center_grid=(),
    )
    dynamic_parameters = DynamicParameters(
        dt=jnp.asarray(time["dt"]),
        dx=jnp.asarray(dr),
        dy=jnp.asarray(dtheta),
        dz=jnp.asarray(dphi),
        Nx=jnp.asarray(nr),
        Ny=jnp.asarray(ntheta),
        Nz=jnp.asarray(nphi),
        x_wind=jnp.asarray(r_outer - r_inner),
        y_wind=jnp.asarray(theta_outer - theta_inner),
        z_wind=jnp.asarray(phi_outer - phi_inner),
        C=jnp.asarray(1.0),
        eps=jnp.asarray(1.0),
        mu=jnp.asarray(1.0),
        kb=jnp.asarray(1.0),
        alpha=jnp.asarray(1.0),
        grids=grids,
    )

    tiled_center_grid, tiled_vertex_grid = build_tiled_yee_grids(
        static_parameters,
        dynamic_parameters,
    )
    dynamic_parameters = dynamic_parameters._replace(
        grids=grids._replace(
            tiled_center_grid=tiled_center_grid,
            tiled_vertex_grid=tiled_vertex_grid,
        )
    )

    _, width_cells = absorber_width(config)
    supergaussian = load_supergaussian_from_toml(
        [
            {
                "wall": "+x",
                "width": width_cells,
                "order": float(absorber["order"]),
                "target_reflection": float(absorber["target_reflection"]),
            }
        ],
        dynamic_parameters,
    )
    absorber_parameters = static_parameters._replace(
        supergaussian_active=True,
        supergaussian_layers=supergaussian[-1],
    )

    return static_parameters, dynamic_parameters, absorber_parameters


def initialize_metric(static_parameters, dynamic_parameters, spin=None):
    """Build the production spherical Kerr-Schild Yee metric."""

    if spin is not None:
        static_parameters = static_parameters._replace(metric_spin=float(spin))
    return initialize_kerr_schild_spherical_metric(
        static_parameters,
        dynamic_parameters,
        mass=static_parameters.metric_mass,
        spin=static_parameters.metric_spin,
    )


def _entity_potential(metric, B0, spin):
    """Evaluate the Entity Eq. (59) four-potential at one metric location."""

    beta_squared = jnp.einsum(
        "...i,...ij,...j->...",
        metric.shift,
        metric.gamma,
        metric.shift,
    )
    g00 = -metric.lapse**2 + beta_squared

    beta_r = metric.shift[..., 0]
    h_rr = metric.gamma[..., 0, 0]
    h_rphi = metric.gamma[..., 0, 2]
    h_phiphi = metric.gamma[..., 2, 2]

    A0 = 0.5 * B0 * (h_rphi * beta_r + 2.0 * spin * g00)
    A_r = 0.5 * B0 * (h_rphi + 2.0 * spin * h_rr * beta_r)
    A_theta = jnp.zeros_like(A0)
    A_phi = 0.5 * B0 * (h_phiphi + 2.0 * spin * h_rphi * beta_r)
    return A0, A_r, A_theta, A_phi


def entity_wald_potential(metric, B0, spin):
    """Return ``A_0`` at centers and ``A_i`` on the native D locations."""

    A0 = _entity_potential(metric.center, B0, spin)[0]
    A_spatial = (
        _entity_potential(metric.D[0], B0, spin)[1],
        _entity_potential(metric.D[1], B0, spin)[2],
        _entity_potential(metric.D[2], B0, spin)[3],
    )
    A_phi_center = _entity_potential(metric.center, B0, spin)[3]
    return A0, A_spatial, A_phi_center


def covariant_E_from_potential(metric_location, dynamic_parameters, B0, spin):
    """Compute stationary ``E_i = partial_i A_0`` on one Yee location."""

    A0 = _entity_potential(metric_location, B0, spin)[0]
    spacings = (
        dynamic_parameters.dx,
        dynamic_parameters.dy,
        dynamic_parameters.dz,
    )
    return tuple(
        (
            jnp.roll(A0, -1, axis=axis + 3)
            - jnp.roll(A0, 1, axis=axis + 3)
        )
        / (2.0 * spacing)
        for axis, spacing in enumerate(spacings)
    )


def B_density_from_potential(A_spatial, dynamic_parameters):
    """Compute the metric-free density ``sqrt(gamma) B^i = epsilon^ijk partial_j A_k``."""

    A_r, A_theta, A_phi = A_spatial
    dr = dynamic_parameters.dx
    dtheta = dynamic_parameters.dy
    dphi = dynamic_parameters.dz

    dAphi_dtheta = (jnp.roll(A_phi, -1, axis=4) - A_phi) / dtheta
    dAtheta_dphi = (jnp.roll(A_theta, -1, axis=5) - A_theta) / dphi
    dAr_dphi = (jnp.roll(A_r, -1, axis=5) - A_r) / dphi
    dAphi_dr = (jnp.roll(A_phi, -1, axis=3) - A_phi) / dr
    dAtheta_dr = (jnp.roll(A_theta, -1, axis=3) - A_theta) / dr
    dAr_dtheta = (jnp.roll(A_r, -1, axis=4) - A_r) / dtheta

    return (
        dAphi_dtheta - dAtheta_dphi,
        dAr_dphi - dAphi_dr,
        dAtheta_dr - dAr_dtheta,
    )


def contravariant_D_from_covariant_E(E_on_D_locations, B, metric):
    """Invert FPIC Eq. (10) locally on each native D location.

    ``B`` and the returned ``D`` are physical contravariant vectors in
    ``metric``; B is transferred through its density in that metric.
    """

    B_density = densitize_vector(B, metric.B)
    D = []
    for target_component, target_location in enumerate(D_FIELD_LOCATIONS):
        target_metric = metric.D[target_component]
        B_on_target = tuple(
            value / target_metric.sqrt_gamma
            for value in reconstruct_vector(B_density, B_FIELD_LOCATIONS, target_location)
        )

        beta = target_metric.shift
        beta_cross_B = (
            beta[..., 1] * B_on_target[2] - beta[..., 2] * B_on_target[1],
            beta[..., 2] * B_on_target[0] - beta[..., 0] * B_on_target[2],
            beta[..., 0] * B_on_target[1] - beta[..., 1] * B_on_target[0],
        )
        D_covariant = jnp.stack(
            tuple(
                (E_component - target_metric.sqrt_gamma * cross_component)
                / target_metric.lapse
                for E_component, cross_component in zip(
                    E_on_D_locations[target_component],
                    beta_cross_B,
                )
            ),
            axis=-1,
        )
        D_contravariant = jnp.einsum(
            "...ij,...j->...i",
            target_metric.gamma_inv,
            D_covariant,
        )
        D.append(D_contravariant[..., target_component])
    return tuple(D)


def initialize_wald_fields(
    config,
    static_parameters,
    dynamic_parameters,
    evolution_metric,
    field_spin,
    constitutive_metric,
):
    """Source densitized Wald D/B for the requested field and metric spins.

    The physical seed fields of the constitutive metric are stored as
    densities of the evolution metric, the metric the solver divides by.
    """

    B0 = float(config["physics"]["B0"])
    _, A_spatial, A_phi_center = entity_wald_potential(
        constitutive_metric,
        B0,
        field_spin,
    )

    # The curl is the density itself, so the evolved magnetic constraint is
    # discrete on the actual Kerr grid.
    B = refresh_fields(
        B_density_from_potential(A_spatial, dynamic_parameters),
        static_parameters, B_FIELD_LOCATIONS, 'B', evolution_metric,
    )

    E_on_D_locations = tuple(
        covariant_E_from_potential(
            constitutive_metric.D[component],
            dynamic_parameters,
            B0,
            field_spin,
        )
        for component in range(3)
    )
    D = contravariant_D_from_covariant_E(
        E_on_D_locations,
        physical_vector(B, evolution_metric.B),
        constitutive_metric,
    )
    D = refresh_fields(
        densitize_vector(D, evolution_metric.D),
        static_parameters, D_FIELD_LOCATIONS, 'D', evolution_metric,
    )
    E_native = tuple(
        E_on_D_locations[component][component] for component in range(3)
    )
    return D, B, E_native, A_phi_center


def initialize_schwarzschild_seed(
    config,
    static_parameters,
    dynamic_parameters,
    evolution_metric,
):
    """Construct the FPIC Schwarzschild seed for later Kerr evolution."""

    seed_spin = float(config["physics"]["seed_spin"])
    seed_metric = initialize_metric(
        static_parameters,
        dynamic_parameters,
        spin=seed_spin,
    )
    return initialize_wald_fields(
        config,
        static_parameters,
        dynamic_parameters,
        evolution_metric,
        seed_spin,
        seed_metric,
    )


def initialize_kerr_target(
    config,
    static_parameters,
    dynamic_parameters,
    evolution_metric,
):
    """Construct the stationary rotating Wald comparison and absorber target."""

    spin = float(config["physics"]["spin"])
    return initialize_wald_fields(
        config,
        static_parameters,
        dynamic_parameters,
        evolution_metric,
        spin,
        evolution_metric,
    )


def center_vector(vector, source_locations, metric):
    """Transfer a native Yee density to cell centers as a physical vector."""

    return tuple(
        value / metric.center.sqrt_gamma
        for value in reconstruct_vector(vector, source_locations, ("C", "C", "C"))
    )


def parallel_electric_field(D, B, metric, B0):
    """Compute the FPIC diagnostic ``gamma_ij D^i B^j / B0^2`` at centers from densities."""

    D_center = jnp.stack(center_vector(D, D_FIELD_LOCATIONS, metric), axis=-1)
    B_center = jnp.stack(center_vector(B, B_FIELD_LOCATIONS, metric), axis=-1)
    return jnp.einsum(
        "...i,...ij,...j->...",
        D_center,
        metric.center.gamma,
        B_center,
    ) / float(B0) ** 2


def weighted_magnetic_divergence(B, metric, dynamic_parameters):
    """Compute ``partial_i(sqrt(gamma) B^i) / sqrt(gamma)`` on vertices from densities."""

    return divergence(B, dynamic_parameters, forward=True) / metric.vertex.sqrt_gamma


def physical_center_axes(dynamic_parameters):
    """Return physical radial and polar center-grid axes."""

    return (
        dynamic_parameters.grids.center[0][1:-1],
        dynamic_parameters.grids.center[1][1:-1],
    )


def physical_component(field, static_parameters):
    """Return the physical single-tile volume of a scalar field."""

    guard_cells = int(static_parameters.guard_cells)
    return field[
        0,
        0,
        0,
        guard_cells:-guard_cells,
        guard_cells:-guard_cells,
        guard_cells:-guard_cells,
    ]
