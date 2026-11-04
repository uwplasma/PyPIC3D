"""Numerical-relativity state carried by the dynamical PIC parameters."""

from typing import NamedTuple
import math
from numbers import Integral, Real

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters as BSSNEvolutionParameters
from JAX_BSSN.bssn.variables import BSSNVariables


class BSSNParameters(NamedTuple):
    """Evolved geometry, NR controls, source strength, and coordinate time."""

    state: object
    evolution_parameters: object
    gravity_G: object
    time: object


def load_bssn_parameters_from_toml(config, static_parameters, dynamic_parameters,
                                  *, state=None, evolution_parameters=None):
    """Read native ``[metric]`` initial arrays and gauge/evolution controls.

    Every omitted or literal-false state entry creates a zero array. Files
    contain global C-node arrays without guards, with tensor components first.
    The conformal factor is W, so physical gamma_ij = gamma_tilde_ij / W**2.
    No state projection, field densitization, or leapfrog seeding occurs here.

    Without ``[metric]``, retain the legacy flat default, ``[bssn_parameters]``
    table, and Python override behavior.
    """
    if "metric" not in config:
        legacy = dict(config.get("bssn_parameters", {}))
        gravity_G = legacy.pop("gravity_G", 1.0)
        time = legacy.pop("time", 0.0)
        return build_bssn_parameters(
            static_parameters, dynamic_parameters, state=state,
            evolution_parameters=(legacy if evolution_parameters is None else evolution_parameters),
            gravity_G=gravity_G, time=time,
        )

    if "bssn_parameters" in config or state is not None or evolution_parameters is not None:
        raise ValueError("[metric] cannot be combined with [bssn_parameters], bssn_state, "
                         "or bssn_evolution_parameters; use one initialization source")
    metric_config = config["metric"]
    if not isinstance(metric_config, dict):
        raise ValueError("[metric] must be a TOML table")

    state_names = BSSNVariables._fields
    boundary_names = ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc")
    numeric_names = ("gravity_G", "time", "eta", "kappa", "nu", "g", "mad_q")
    derived_names = {"dx", "dt", "x_min", "y_min", "z_min"}
    forbidden = set(metric_config) & derived_names
    if forbidden:
        raise ValueError(f"[metric] {', '.join(sorted(forbidden))} are derived from the PIC grid and timestep")
    allowed = set(state_names + boundary_names + numeric_names + ("gauge", "zero_shift"))
    unknown = set(metric_config) - allowed
    if unknown:
        raise ValueError(f"Unknown [metric] keys: {', '.join(sorted(unknown))}")

    controls = {}
    for name in numeric_names:
        if name in metric_config:
            value = metric_config[name]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"[metric] {name} must be a finite real number")
            controls[name] = value
    if "gauge" in metric_config:
        gauge = metric_config["gauge"]
        if gauge not in ("harmonic", "1+log"):
            raise ValueError("[metric] gauge must be 'harmonic' or '1+log'")
        controls["gauge"] = {"harmonic": 0, "1+log": 1}[gauge]
    if "zero_shift" in metric_config:
        if not isinstance(metric_config["zero_shift"], bool):
            raise ValueError("[metric] zero_shift must be a boolean")
        # JAX_BSSN freezes the supplied shift; it does not replace it by zero.
        controls["zero_shift"] = int(metric_config["zero_shift"])
    for name in boundary_names:
        if name not in metric_config:
            continue
        value = metric_config[name]
        if isinstance(value, str) and value in ("periodic", "sommerfeld"):
            controls[name] = {"periodic": 0, "sommerfeld": 1}[value]
        elif isinstance(value, Integral) and not isinstance(value, bool) and value in (0, 1):
            controls[name] = int(value)
        else:
            raise ValueError(f"[metric] {name} must be 'periodic', 'sommerfeld', 0, or 1")

    # Reuse the native allocation and grid checks, including the x64 and
    # single-device requirements, before reading potentially large files.
    gravity_G = controls.pop("gravity_G", 1.0)
    time = controls.pop("time", 0.0)
    parameters = build_bssn_parameters(
        static_parameters, dynamic_parameters,
        evolution_parameters=controls,
        gravity_G=gravity_G, time=time,
    )
    values = []
    for name, template in zip(state_names, parameters.state):
        path = metric_config.get(name, False)
        if path is False:
            value = jnp.zeros(template.shape, dtype=jnp.float64)
        elif isinstance(path, str):
            value = jnp.load(path, allow_pickle=False)
            if not hasattr(value, "dtype") or not jnp.issubdtype(value.dtype, jnp.number) or jnp.iscomplexobj(value):
                raise ValueError(f"[metric] {name} must contain a real numeric array")
            if value.shape != template.shape:
                raise ValueError(f"[metric] {name} has shape {value.shape}; expected {template.shape} "
                                 "on global C nodes without guards")
            value = jnp.asarray(value, dtype=jnp.float64)
        else:
            raise ValueError(f"[metric] {name} must be an initialization filename or literal false")
        if not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError(f"[metric] {name} must contain finite values")
        values.append(value)
    state = BSSNVariables(*values)

    # Zero defaults are literal data, not requests for identity/one fallbacks.
    # Check the geometry before determinant normalization and inversion can
    # obscure an omitted geometry file with floors or NaNs.
    if not bool(jnp.all(state.conformal_factor > 0)):
        raise ValueError("[metric] conformal_factor must contain positive W; omitted/false creates zeros")
    gamma = jnp.moveaxis(state.conformal_metric, (0, 1), (-2, -1))
    if not bool(jnp.allclose(gamma, jnp.swapaxes(gamma, -2, -1), rtol=1.e-12, atol=1.e-12)):
        raise ValueError("[metric] conformal_metric must be symmetric")
    if not bool(jnp.all(jnp.linalg.eigvalsh(gamma) > 0)):
        raise ValueError("[metric] conformal_metric must be positive definite; omitted/false creates zeros")
    return parameters._replace(state=state)


def build_bssn_parameters(static_parameters, dynamic_parameters, *, state=None,
                          evolution_parameters=None, gravity_G=1.0, time=0.0):
    """Build flat initial geometry unless the caller supplies BSSN variables.

    The first coupling uses Cartesian geometrized units and one device. BSSN
    has one spatial spacing; unresolved periodic axes need not share it.
    The default flat state is not a matter constraint solve.
    """
    tile_counts = dynamic_parameters.grids.tiled_center_grid[0].shape[:3]
    shape = tuple(count * width + (bc == 1) for count, width, bc in zip(
        tile_counts, static_parameters.tile_shape, static_parameters.boundary_conditions))
    spacings = (dynamic_parameters.dx, dynamic_parameters.dy, dynamic_parameters.dz)
    active_spacings = [float(spacing) for spacing, size in zip(spacings, shape) if size > 1]
    dx = active_spacings[0] if active_spacings else float(dynamic_parameters.dx)
    if any(abs(spacing - dx) > 1.e-12 * abs(dx) for spacing in active_spacings):
        raise ValueError("BSSN requires equal spacing on all resolved Cartesian axes")
    if any(float(value) != 1.0 for value in (dynamic_parameters.C, dynamic_parameters.eps, dynamic_parameters.mu)):
        raise ValueError("Dynamical relativity requires normalized C=1, eps=1, mu=1")
    if static_parameters.field_mesh is not None and static_parameters.field_mesh.size != 1:
        raise ValueError("Dynamical relativity currently requires one device")
    if not jax.config.x64_enabled:
        raise ValueError("Dynamical relativity requires jax_enable_x64=True")

    if state is None:
        zero = jnp.zeros(shape, dtype=jnp.float64)
        identity = jnp.broadcast_to(jnp.eye(3)[:, :, None, None, None], (3, 3) + shape)
        state = BSSNVariables(identity, jnp.ones_like(zero), jnp.zeros_like(identity),
                              zero, jnp.zeros((3,) + shape), jnp.ones_like(zero),
                              jnp.zeros((3,) + shape))
    elif state.conformal_factor.shape != shape:
        raise ValueError("BSSN state must match the physical PIC center-node allocation")

    if evolution_parameters is None or isinstance(evolution_parameters, dict):
        overrides = {} if evolution_parameters is None else evolution_parameters
        origins = tuple(axis[1] for axis in dynamic_parameters.grids.center)
        evolution_parameters = BSSNEvolutionParameters(
            dx=dx, dt=dynamic_parameters.dt, zero_shift=1, gauge=0,
            x_min=origins[0], y_min=origins[1], z_min=origins[2])
        evolution_parameters = evolution_parameters._replace(**overrides)
    if abs(float(evolution_parameters.dx) - dx) > 1.e-12 * abs(dx):
        raise ValueError("BSSN and PIC spatial spacings must agree")
    if float(evolution_parameters.dt) != float(dynamic_parameters.dt):
        raise ValueError("BSSN and PIC macro timesteps must agree")
    if static_parameters.current_filter != "none":
        from PyPIC3D.relativity.filtering import validate_gr_filter

        validate_gr_filter(static_parameters)
        if any(int(getattr(evolution_parameters, name)) != 0
               for name in ("xl_bc", "xr_bc", "yl_bc", "yr_bc", "zl_bc", "zr_bc")):
            raise ValueError("Matched GR filtering requires all six BSSN boundaries to be periodic")

    evolution_parameters = jax.tree.map(jnp.asarray, evolution_parameters)
    return BSSNParameters(state, evolution_parameters, jnp.asarray(gravity_G), jnp.asarray(time))
