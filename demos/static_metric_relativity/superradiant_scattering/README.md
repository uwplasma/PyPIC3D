# Vacuum Kerr scattering in a PEC box

This experiment evolves a one-time inward electromagnetic dipole packet in
a fixed Cartesian Kerr-Schild metric. There are no particles, deposited
currents, external fields, or continuing sources. All six outer faces use
PyPIC3D's **existing physical FIDO D/B PEC projection**. E and H are auxiliary
covectors used by the production Maxwell update and the Killing-energy budget.

Run from the repository root, with PyPIC3D and its dependencies installed:

```bash
python -m demos.static_metric_relativity.superradiant_scattering.run_demo \
  --config demos/static_metric_relativity/superradiant_scattering/smoke.toml

python -m demos.static_metric_relativity.superradiant_scattering.run_demo \
  --config demos/static_metric_relativity/superradiant_scattering/kerr.toml

python -m demos.static_metric_relativity.superradiant_scattering.run_demo \
  --config demos/static_metric_relativity/superradiant_scattering/schwarzschild.toml
```

Use `JAX_PLATFORMS=cpu` for a CPU run, or select a GPU with JAX's usual
environment settings. The demo currently uses one device and one tile.
The full 128^3, float64 run stores eight metric samples per lattice index
plus fields and workspace; allow several GB of memory and substantial run
time. The 48^3 smoke configuration retains the same spatial spacing in a
smaller box and evolves for just 0.1 M. It tests startup, not amplification.

`--output-dir PATH` overrides the configured directory. Existing demo files
are protected unless `--overwrite` is supplied; that flag replaces only the
four files owned by this demo. TOML sections group dataclass fields; omitted
fields use the defaults in `parameters.py`. Unknown or repeated fields are
errors. The nonrotating preset inherits the same grid and packet as the main
Kerr preset. To construct another short control, use the Python interface:

```python
from dataclasses import replace
from demos.static_metric_relativity.superradiant_scattering.parameters import load_parameters
from demos.static_metric_relativity.superradiant_scattering.run_demo import run

p = load_parameters("demos/static_metric_relativity/superradiant_scattering/smoke.toml")
run(replace(p, spin=0., packet_inner=2.3, packet_outer=2.85,
            half_width=3.25, cells=64, output_directory="data/nonrotating_smoke"))
```

## Initial packet and units

The default is G=c=M=1 in Gaussian electromagnetic units, a/M=0.9,
omega M=0.25, a [-8 M,8 M]^3 box, and 128^3 cells. `mass` and all lengths,
times and frequencies are in code units; lengths are not automatically
rescaled when mass changes. `spin` is dimensionless a/M. The carrier lies
below Omega_H=a/(2 M r_+), but the compact packet has a broad spectrum.
The box also selects its own resonances; these inputs do not guarantee growth.

The seed uses the electric-parity dipole Hertz potential

    Pi = (ex + i ey) f(r+t)/r,
    f(s) = bump(s) exp(-i omega s),
    densitized D = Re(curl_backward curl_forward Pi),
    densitized B = Re(curl_forward d_t Pi).

Here r is the Euclidean radius and `bump` is a smooth compact envelope with
support between `packet_inner` and `packet_outer`. The scalar angular factor
(ex+i ey).rhat is proportional to Y_11, with the convention
exp[i(m phi-omega t)]. Taking the real field includes its conjugate
negative-frequency partner; it is not a second positive-frequency m=-1 seed.
Compatible Yee curls give zero discrete vacuum divergence at initialization.
This is an exactly inward Hertz solution in flat space before discretization;
in Kerr it is an approximately ingoing, constraint-satisfying seed rather
than a separated Kerr eigenfunction. The runner checks inward integrated
FIDO flux before evolution. Reflections from the Cartesian box mix modes.

The synchronized initial exterior FIDO energy is normalized to 1e-6.
Midpoint RK2 constructs the negative-time levels needed by the production
vacuum leapfrog stages. All persistent fields are densitized; diagnostics
reconstruct physical fields at the integer D time before measuring energy.

## Interior and boundaries

The Kerr radius, used for horizon/core masks, is different from the seed's
Euclidean radius. Inside `core_radius=0.35 M`, the metric uses a finite radial
cap with a consistent positive spatial determinant and inverse. Fields in
that core are zeroed. A quintic smooth damping rate rises from zero at
`absorber_radius=0.70 M` to `absorber_rate=20/M` at the core. Each field stage
integrates that damping with a second-order exponential centered-curl update.
The exact Kerr metric and unmodified source-free equations apply outside
the buffer; there is no outer sponge. `flat=true` disables both the core
and damping and selects Minkowski space for cavity controls.

Configuration validation requires more than two cell diagonals between the
absorber and horizon, using the minimum separation of their oblate surfaces.
The CFL estimate includes all directions and every metric location, including
halos and continued core data. The timestep is reduced to hit the requested
final time exactly. Output intervals are rounded to whole timesteps and the
actual interval is stored in metadata.

Interior continuation and damping are numerical treatments, not an exact
excision boundary. Their leakage and constraint errors must be assessed by
refinement and buffer variations. The code preserves the existing FIDO PEC
boundary contract, including its behavior in a nonzero shift. No condition
on the auxiliary E/H fields is imposed in addition to that contract.

## Outputs and growth

- `energy.csv`: flushed after every diagnostic, with total FIDO energy
  outside the excised core, exterior FIDO and Killing energies, amplification,
  equivalent amplitude, horizon/wall Killing fluxes, accumulated flux and
  budget residual, divergence residuals, native FIDO PEC residuals, and
  a field-finiteness flag.
- `metadata.json`: complete parameters, timestep, horizon, backend, progress,
  and completion/failure status.
- `growth.json`: fit status, energy and amplitude growth rates, descriptive
  OLS slope error, R squared, and e-folding times for resolved positive slopes.
- `energy_growth.png`: energies, normalized energy/amplitude, and boundary fluxes.

FIDO energy is the proper-volume integral of (D_i D^i+B_i B^i)/(8 pi).
Killing energy is the integral of (E_i D^i+H_i B^i)/(8 pi) with the same
proper-volume measure. All volume terms use VVV cell centers, excluding
halos and duplicate endpoints. The Kerr exterior is r>=r_+; flat controls
use the whole box. Exterior Killing density can be negative in the ergoregion.

The Killing flux density is the coordinate cross product E x H/(4 pi).
Positive wall flux exits the box; positive horizon flux enters the hole
and leaves the exterior. The horizon flux uses the coordinate faces of the
staircase separating inside/outside cells, not an exact oblate surface.
The budget residual is U_K(t)-U_K(0)+integral(F_wall+F_horizon)dt.
These collocated diagnostics and staircase fluxes are continuum estimates,
not an exactly conserved discrete Hamiltonian. Their mismatch must converge.
The radial FIDO-flux moment is a volume integral used to check the initial
packet direction, not a luminosity on a sphere.

The fit uses log exterior **FIDO** energy; amplitude growth is half the
energy slope, and sqrt(U_ext/U_ext(0)) is an energy-equivalent amplitude,
not a projection onto an isolated l=m=1 mode. The default fit starts at the
later of half the recorded duration and one box crossing time (2 half_width).
At least eight samples, R squared >=0.8, and a slope exceeding three OLS
standard errors are needed for a descriptive growing/decaying classification.
Correlated oscillations mean that this OLS error is not a physical uncertainty.
Short runs return `insufficient_data`; constant or poorly fitted traces
return `unresolved`. The fitter reports decay and never clips it into growth.

Reanalyze a saved trace with an explicit window:

```bash
python -m demos.static_metric_relativity.superradiant_scattering.analysis \
  data/superradiant_scattering --fit-start 250 --fit-end 500
```

## Validation and interpreting amplification

Run the focused tests with:

```bash
JAX_PLATFORMS=cpu python -m unittest tests.code_tests.superradiant_demo_test -v
```

Short tests check seed constraints and direction, metric continuation,
quadrature and halo exclusion, PEC residuals, second-order startup/timestep
convergence, absorber sensitivity before the packet reaches the buffer,
file output, and synthetic growth/decay fits. The flat packet control checks
the native staggered leapfrog energy invariant separately from interpolated
cell-center energy; the latter need not be constant on a coarse grid.

The 48^3 CPU smoke run to 0.1 M gave U_ext/U_ext(0)=0.910827 and exterior
relative divergence residuals below 2e-15. Its five-cell-wide radial packet
is deliberately inexpensive and poorly resolved for energy measurements:
the Killing-energy budget mismatch is about 5.4% of its initial energy.
These are startup diagnostics, not a physical scattering amplification or
decay measurement. The default fit correctly reports `insufficient_data`.

A positive energy slope by itself does not
establish black-hole superradiance. Before making that claim, run matching
Kerr/nonrotating and flat controls, refine the grid and timestep separately,
vary the core and absorber, and extend the duration through many reflections.
Inspect negative horizon Killing flux, measured wall exchange, budget error,
and constraint convergence alongside growth. Change the fit window and
check cavity oscillations. No converged long-time growth is claimed by this demo.

For the physical background see [Brito, Cardoso and Pani, Superradiance](https://arxiv.org/abs/1501.06570).
