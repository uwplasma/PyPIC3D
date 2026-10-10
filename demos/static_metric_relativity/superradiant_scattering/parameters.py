"""Demo configuration and finite Cartesian Kerr-Schild interior continuation."""
from dataclasses import dataclass, fields
from functools import partial
import math
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import toml

from PyPIC3D.relativity.core import build_yee_metric, location_grid
from PyPIC3D.relativity.metrics.flat import _flat_cartesian_metric_at_position
from PyPIC3D.utilities.grids import build_yee_grid, build_tiled_yee_grids
from PyPIC3D.utilities.parameters import build_static_parameters, build_dynamic_parameters


@dataclass(frozen=True)
class Parameters:
    mass: float = 1.0
    spin: float = 0.9  # dimensionless a/M
    flat: bool = False
    cells: int = 128
    half_width: float = 8.0
    core_radius: float = 0.35
    absorber_radius: float = 0.70
    absorber_rate: float = 20.0
    packet_inner: float = 3.0  # Euclidean radii, unlike the Kerr interior masks
    packet_outer: float = 6.0
    omega: float = 0.25
    initial_energy: float = 1.e-6
    courant: float = 0.2
    end_time: float = 500.0
    output_interval: float = 0.5
    fit_start: float | None = None
    fit_end: float | None = None
    output_directory: str = "data/superradiant_scattering"

    @property
    def a(self):
        return self.mass * self.spin

    @property
    def horizon(self):
        return self.mass * (1 + math.sqrt(1 - self.spin**2))

    @property
    def omega_h(self):
        return self.a / (2 * self.mass * self.horizon)

    @property
    def spacing(self):
        return 2 * self.half_width / self.cells

    def validate(self):
        positive = ("mass", "half_width", "core_radius", "absorber_radius",
                    "absorber_rate", "packet_inner", "packet_outer", "omega",
                    "initial_energy", "courant", "end_time", "output_interval")
        for key in positive:
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not isinstance(self.flat, bool):
            raise ValueError("flat must be a boolean")
        if not math.isfinite(self.spin) or not 0 <= self.spin < 1:
            raise ValueError("spin must satisfy 0 <= a/M < 1")
        if isinstance(self.cells, bool) or not isinstance(self.cells, int) or self.cells < 16:
            raise ValueError("cells must be an integer >= 16")
        if self.courant > 0.5:
            raise ValueError("courant must be <= 0.5")
        if not 0 < self.core_radius < self.absorber_radius < self.horizon:
            raise ValueError("Require 0 < core_radius < absorber_radius < horizon")
        # Nested Kerr-r surfaces are confocal oblate ellipsoids. Their minimum
        # separation occurs at the equator. A constitutive transfer plus curl
        # reaches no farther than two cells along each coordinate direction.
        separation = (math.hypot(self.horizon, self.a)
                      - math.hypot(self.absorber_radius, self.a))
        if not self.flat and separation <= 2 * math.sqrt(3) * self.spacing:
            raise ValueError("Resolve the horizon: interior-to-horizon separation must "
                             "exceed 2*sqrt(3)*dx; increase cells or reduce box size")
        if not self.packet_inner < self.packet_outer < self.half_width - 2*self.spacing:
            raise ValueError("Packet must fit inside the box with a two-cell wall margin")
        if self.packet_outer - self.packet_inner < 4*self.spacing:
            raise ValueError("Packet support must span at least four cells")
        if not self.flat and self.packet_inner <= math.hypot(self.horizon, self.a) + 2*self.spacing:
            raise ValueError("Packet must start outside the horizon with a two-cell margin")
        for name in ("fit_start", "fit_end"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.fit_start is not None and self.fit_end is not None and self.fit_start >= self.fit_end:
            raise ValueError("fit_start must precede fit_end")


def load_parameters(path):
    """TOML sections group fields for readability; unspecified fields use defaults."""
    config = toml.load(Path(path))
    values = {}
    allowed = {field.name for field in fields(Parameters)}
    for section in config.values():
        if not isinstance(section, dict):
            raise ValueError("Put configuration fields inside TOML sections")
        for key, value in section.items():
            if key not in allowed or key in values:
                raise ValueError(f"Unknown or repeated configuration field: {key}")
            values[key] = value
    result = Parameters(**values)
    result.validate()
    return result


def kerr_radius(position, a):
    rho2 = jnp.sum(position**2, axis=-1)
    return jnp.sqrt(jnp.maximum(0., 0.5 * (rho2-a*a +
                    jnp.sqrt((rho2-a*a)**2 + 4*a*a*position[..., 2]**2))))


def continued_metric(position, mass, a, core_radius):
    """Exact Kerr for r>=core_radius; finite positive ADM data within the core.

    Inside the core the capped ell need not be unit length, so determinant
    and inverse use its actual norm. This continuation is not a BH solution;
    all fields there are excised and it is excluded from energy integrals.
    """
    x, y, z = position
    r = jnp.maximum(kerr_radius(position, a), core_radius)
    ell = jnp.array(((r*x+a*y)/(r*r+a*a), (r*y-a*x)/(r*r+a*a), z/r))
    two_h = 2*mass*r**3/(r**4 + a*a*z*z)
    determinant = 1 + two_h*jnp.dot(ell, ell)
    outer = jnp.outer(ell, ell)
    eye = jnp.eye(3, dtype=position.dtype)
    return (determinant**-0.5, two_h/determinant*ell,
            eye+two_h*outer, eye-two_h/determinant*outer, jnp.sqrt(determinant))


def positions(dynamic, location):
    axes = location_grid(dynamic.grids.tiled_center_grid,
                         dynamic.grids.tiled_vertex_grid, location)
    return jnp.stack(jnp.broadcast_arrays(axes[0][..., :, None, None],
                     axes[1][..., None, :, None], axes[2][..., None, None, :]), -1)


def build_runtime(p):
    """Single-device tiled arrays, in float64; no particle runtime is constructed."""
    p.validate()
    jax.config.update("jax_enable_x64", True)
    config = dict(Nx=p.cells, Ny=p.cells, Nz=p.cells, dx=p.spacing, dy=p.spacing,
                  dz=p.spacing, dt=1., x_wind=2*p.half_width, y_wind=2*p.half_width,
                  z_wind=2*p.half_width, x_min=-p.half_width, y_min=-p.half_width,
                  z_min=-p.half_width)
    static = build_static_parameters(dict(
        **config, name="superradiant_scattering", solver="static_metric",
        metric="flat_cartesian" if p.flat else "kerr_schild_cartesian",
        metric_mass=p.mass, metric_spin=p.a, shape_factor=1, guard_cells=3,
        tile_shape=(p.cells,)*3, boundary_conditions=(1, 1, 1),
        particle_boundary_conditions=(1, 1, 1),
        pml_active=False, supergaussian_active=False))
    center, vertex = build_yee_grid(SimpleNamespace(**config))
    tc, tv = build_tiled_yee_grids(static, SimpleNamespace(
        **config, grids=SimpleNamespace(center=center, vertex=vertex)))
    config["grids"] = dict(center=center, vertex=vertex,
                           tiled_center_grid=tc, tiled_vertex_grid=tv)
    dynamic = build_dynamic_parameters(config)
    point_metric = (_flat_cartesian_metric_at_position if p.flat else
                    partial(continued_metric, mass=p.mass, a=p.a, core_radius=p.core_radius))
    metric = jax.jit(lambda d: build_yee_metric(d, point_metric))(dynamic)
    max_speed = 0.
    # Include every native location and halos, including continued core data.
    for sample in (*metric.D, *metric.B, metric.center, metric.vertex):
        if not all(bool(jnp.all(jnp.isfinite(v))) for v in sample):
            raise ValueError("Nonfinite metric sample")
        if not bool(jnp.all(sample.sqrt_gamma > 0)):
            raise ValueError("Metric volume must be positive")
        speeds = jnp.abs(sample.shift) + sample.lapse[..., None]*jnp.sqrt(
            jnp.diagonal(sample.gamma_inv, axis1=-2, axis2=-1))
        max_speed = max(max_speed, float(jnp.max(jnp.sum(speeds, axis=-1)))/p.spacing)
    cfl_dt = p.courant/max_speed
    # Integer output chunks with an exactly attained final time.
    steps = math.ceil(p.end_time/min(cfl_dt, p.output_interval))
    dynamic = dynamic._replace(dt=jnp.asarray(p.end_time/steps))
    return static, dynamic, metric, steps, cfl_dt
