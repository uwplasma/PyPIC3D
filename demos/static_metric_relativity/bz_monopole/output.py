"""Host-side BZ output and cumulative radial absorption budgets."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.relativity.field_state import physical_fields


@dataclass
class RadialBoundaryBudget:
    """Cumulative two-species radial absorption budget, packed into eleven slots by as_array.

    The totals stay on device so the step loop never blocks; they reach the
    host only when a snapshot calls as_array.
    """
    absorbed_count: jax.Array = field(default_factory=lambda: jnp.zeros((2, 2)))
    absorbed_charge: jax.Array = field(default_factory=lambda: jnp.zeros((2, 2)))
    removed_grid_charge: jax.Array = field(default_factory=lambda: jnp.zeros(()))
    radial_current_outflow: jax.Array = field(default_factory=lambda: jnp.zeros(2))

    def accumulate(self, report, dt):
        self.absorbed_count = self.absorbed_count + report.absorbed_count
        self.absorbed_charge = self.absorbed_charge + report.absorbed_charge
        self.removed_grid_charge = self.removed_grid_charge + jnp.sum(report.removed_grid_charge)
        self.radial_current_outflow = self.radial_current_outflow + report.radial_current_outflow*dt

    def as_array(self):
        """Counts, charges, removed cloud charge, then time-integrated face flux."""
        count, charge, removed, outflow = jax.device_get(
            (self.absorbed_count, self.absorbed_charge,
             self.removed_grid_charge, self.radial_current_outflow))
        return np.concatenate((np.asarray(count).reshape(-1), np.asarray(charge).reshape(-1),
                               [float(removed)], np.asarray(outflow)))


def plot_diagnostics(snapshot, output):
    """Draw the in-run figure with the same settings as a standalone plot_bz.py run."""
    from plot_bz import DPI, Normalization, available_radii, make_figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure = make_figure(snapshot, Normalization.from_snapshot(snapshot),
                         radii=available_radii(snapshot))
    figure.savefig(output, dpi=DPI)
    plt.close(figure)


def output_metadata(p, dynamic):
    """Flat scalar arrays shared by snapshots and the final analysis dump.

    NaN for maximum_timestep means there is no explicit cap on the CFL step.
    Stored particle positions are spherical and momenta are covariant and
    leapfrog-staggered.
    Field arrays retain their tiled Yee layout and guards.
    """
    data = {name: np.asarray(np.nan if value is None else value)
            for name, value in asdict(p).items()}
    # Only the derived @property values; asdict(p) already holds every field.
    data.update(dt=np.asarray(dynamic.dt), B0=np.asarray(p.B0),
                omega_h=np.asarray(p.omega_h), horizon=np.asarray(p.horizon),
                n0_total=np.asarray(p.n0),
                field_discretization=np.asarray('metric_finite_difference'),
                constraint_form=np.asarray('conformal_density'))
    return data


def save_final_state(path, particles, species, fields, step, metadata, dynamic, budget):
    """Write analysis arrays without recovery history or executable objects."""
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


