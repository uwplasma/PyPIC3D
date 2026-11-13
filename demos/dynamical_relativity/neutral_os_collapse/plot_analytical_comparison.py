"""Compare the final neutral OS radial and time metrics with Schwarzschild.

Use the stationary reconstruction
    g_rr = gamma_rr + beta_r**2 / (alpha**2 - beta_r * beta^r),
then transform the component to areal radius: g_RR = g_rr / (dR/dr)**2.
The stationary time shift leaves g_tt = -alpha**2 + gamma_rr * beta**2
unchanged, with no rescaling of the time coordinate.
This treats the final snapshot as stationary; it is not an exact time
coordinate transformation of a still-evolving spacetime.

Run with no arguments. Paths and plot limits are set below. Only the final
iteration is read. The prescribed initial mass is used without fitting or
boundary rescaling, in geometrized G = c = 1 units. Spherical symmetry and
stationarity are assumed, not measured by this plotter.
"""

from pathlib import Path

import matplotlib
import numpy as np
import openpmd_api as io

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = DEMO_DIRECTORY / "outputs/data"
OUTPUT_DIRECTORY = DEMO_DIRECTORY / "outputs/analytical_comparison"
# Keep the prescribed mass consistent with initial_metric.py.
TOTAL_MASS = 1.0
INNER_RADIUS_LIMIT = 6.0


def read_final_state():
    """Read only the final positive-x ray from native Cartesian output."""
    series = io.Series(str(DATA_DIRECTORY / "fields.h5"), io.Access.read_only)
    step = max(series.iterations)
    iteration = series.iterations[step]
    scalar = io.Mesh_Record_Component.SCALAR
    reference = iteration.meshes['conformal_factor']
    spacing = np.asarray(reference.grid_spacing)*reference.grid_unit_SI
    origin = np.asarray(reference.grid_global_offset)*reference.grid_unit_SI + np.asarray(reference[scalar].position)*spacing
    center = np.rint(-origin/spacing).astype(int)
    shape = reference[scalar].shape
    count = [shape[0]-center[0],1,1]
    def read(name, component=scalar):
        return iteration.meshes[name][component].load_chunk(center.tolist(),count)
    pending = {name:read(name) for name in ('lapse','conformal_factor','conformal_metric_xx','conformal_metric_yy','conformal_metric_zz')}
    pending['beta'] = read('shift','x')
    series.flush()
    values = {name:np.asarray(value).reshape(-1) for name,value in pending.items()}
    fields = dict(r=origin[0]+np.arange(center[0],shape[0])*spacing[0],
                  alpha=values['lapse'],beta=values['beta'],chi=values['conformal_factor']**2,
                  conformal_grr=values['conformal_metric_xx'],
                  conformal_gt=.5*(values['conformal_metric_yy']+values['conformal_metric_zz']))
    series.close()
    return fields, float(spacing[0])


def main():
    fields, dr = read_final_state()
    mass = TOTAL_MASS
    alpha = fields["alpha"]
    beta = fields["beta"]  # Saved beta is contravariant beta^r.
    gamma_rr = fields["conformal_grr"] / fields["chi"]
    beta_r = gamma_rr * beta

    R = fields["r"] * np.sqrt(fields["conformal_gt"] / fields["chi"])
    # Fourth-order centered derivative; leave unsupported edge cells as gaps.
    dR_dr = np.full_like(R, np.nan)
    dR_dr[2:-2] = (R[:-4] - 8*R[1:-3] + 8*R[3:-1] - R[4:]) / (12*dr)
    denominator = alpha**2 - beta_r * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        g_rr = gamma_rr + beta_r**2 / denominator
        g_RR = g_rr / dR_dr**2

    # Omit the two unsupported inner stencil cells on the final trumpet. Keep
    # gaps as NaNs so a plot cannot join across a pole or a turning point.
    # R(0)=0, so argmin cannot identify the outer branch of a trumpet.
    outward_branch = np.max(np.where(R[1:] <= R[:-1], np.arange(1, R.size), 0))
    resolved_D = 32*np.finfo(denominator.dtype).eps * (alpha**2 + np.abs(beta_r * beta))
    exterior = ((np.arange(R.size) >= max(2, outward_branch))
                & (dR_dr > 0.0) & (denominator > resolved_D)
                & np.isfinite(R) & np.isfinite(denominator)
                & np.isfinite(g_RR) & (g_RR > 0.0))
    g_RR = np.where(exterior, g_RR, np.nan)
    # The stationary time shift leaves g_tt unchanged; reuse the g_RR mask.
    g_tt = np.where(np.isfinite(g_RR), -denominator, np.nan)
    horizon = 2.0 * mass
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.0 - 2.0 * mass / R

    # Schwarzschild has its own exterior mask, independent of the numerical pole.
    g_RR_exact = np.full_like(R, np.nan)
    g_tt_exact = np.full_like(R, np.nan)
    analytical_exterior = ((np.arange(R.size) >= outward_branch) & (R > horizon))
    g_RR_exact[analytical_exterior] = 1.0 / f[analytical_exterior]
    g_tt_exact[analytical_exterior] = -f[analytical_exterior]
    radius = R / mass

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for component, numerical, analytical, scale, label, formula in (
        ("grr", g_RR, g_RR_exact, "log", r"$g_{RR}$",
         r"$g_{RR}=[\gamma_{rr}+\beta_r^2/(\alpha^2-\beta_r\beta^r)]/(dR/dr)^2$"),
        ("gtt", g_tt, g_tt_exact, "linear", r"$g_{tt}$",
         r"$g_{tt}=-\alpha^2+\gamma_{rr}(\beta^r)^2$"),
    ):
        for extent, upper in (("Inner", INNER_RADIUS_LIMIT), ("Full", radius[-1])):
            figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
            figure.suptitle(
                "Numerical vs. Analytical Comparison of the Final Black Hole Spacetime\n"
                + formula,
                fontsize=12,
            )
            visible = radius <= upper
            axis.plot(radius[visible], numerical[visible], color="tab:blue", linewidth=1.8,
                      label=f"Numerical {component}")
            axis.plot(radius[visible], analytical[visible], "--",
                      color="tab:orange", linewidth=1.8, label="Schwarzschild, initial M")
            axis.set_yscale(scale)
            axis.set_xlim(horizon / mass, min(upper, radius[-1]))
            axis.set_xlabel(r"Areal radius $R/M$")
            axis.set_ylabel(label)
            axis.set_title(f"{extent} range", fontsize=12)
            axis.grid(alpha=0.25, which="both")
            axis.legend(fontsize=10)
            path = OUTPUT_DIRECTORY / f"final_{component}_{extent.lower()}.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            print(f"Saved {path}")

    print("Reconstruction assumes spherical symmetry and a stationary final snapshot.")


if __name__ == "__main__":
    main()
