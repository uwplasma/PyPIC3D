"""Host-side BZ output and cumulative radial absorption budgets."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class RadialBoundaryBudget:
    """Cumulative two-species radial absorption budget, packed into eleven slots by as_array."""
    absorbed_count: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    absorbed_charge: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    removed_grid_charge: float = 0.0
    radial_current_outflow: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def accumulate(self, report, dt):
        self.absorbed_count += np.asarray(report.absorbed_count)
        self.absorbed_charge += np.asarray(report.absorbed_charge)
        self.removed_grid_charge += float(jnp.sum(report.removed_grid_charge))
        self.radial_current_outflow += np.asarray(report.radial_current_outflow) * dt

    def as_array(self):
        """Counts, charges, removed cloud charge, then time-integrated face flux."""
        return np.concatenate((self.absorbed_count.reshape(-1),
                               self.absorbed_charge.reshape(-1),
                               [self.removed_grid_charge], self.radial_current_outflow))


def plot_diagnostics(snapshot, output):
    if __package__:
        from .plot_entity_bz import Normalization, make_figure
    else:
        from plot_entity_bz import Normalization, make_figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    radii = tuple(r for r in (2., 3., 4., 5.) if snapshot['r'][0] <= r <= snapshot['r'][-1])
    figure, _ = make_figure(snapshot, Normalization.from_snapshot(snapshot), radii=radii)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def output_metadata(p, static, dynamic):
    """Flat scalar arrays shared by snapshots and the final analysis dump.

    NaN for maximum_timestep means there is no explicit cap on the CFL step.
    Stored particle positions are spherical and momenta are covariant and
    leapfrog-staggered.
    Field arrays retain their tiled Yee layout and guards.
    """
    data = {name: np.asarray(np.nan if value is None else value)
            for name, value in asdict(p).items()}
    data.update(dt=np.asarray(dynamic.dt), B0=np.asarray(p.B0),
                omega_h=np.asarray(p.omega_h), horizon=np.asarray(p.horizon),
                n0_total=np.asarray(p.n0),
                particle_batch_size=np.asarray(static.particle_batch_size),
                horizon_field_cells=np.asarray(static.horizon_field_cells),
                field_discretization=np.asarray('metric_finite_difference'),
                constraint_form=np.asarray('conformal_density'))
    return data


def save_final_state(path, particles, species, fields, step, metadata, dynamic, budget):
    """Write analysis arrays without recovery history or executable objects."""
    from PyPIC3D.relativity.field_state import physical_fields
    fields = physical_fields(fields)
    arrays = dict(metadata, x=particles.x, u=particles.u, active=particles.active,
                  step=np.asarray(step), time=np.asarray(step*float(dynamic.dt)),
                  rho=fields[3], phi=fields[4], boundary_budget=budget)
    for label, vector in (("D", fields[0]), ("B", fields[1]), ("J", fields[2])):
        arrays.update({f"{label}_{i}": value for i, value in enumerate(vector)})
    for name in ('charge', 'mass', 'weight', 'update_x'):
        arrays['species_'+name] = getattr(species, name)
    for label, grids in (('center', dynamic.grids.tiled_center_grid),
                         ('vertex', dynamic.grids.tiled_vertex_grid)):
        arrays.update({f'grid_{label}_{axis}': value for axis, value in zip('xyz', grids)})
    np.savez(path, **{name: np.asarray(jax.device_get(value)) for name, value in arrays.items()})


def prepare_output_directory(output):
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Output directory must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output


