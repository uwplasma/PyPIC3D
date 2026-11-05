#!/usr/bin/env python3
"""Plot the four BZ-monopole panels of Entity Paper II, Figure 6.

Reference: Galishnikova et al., https://arxiv.org/abs/2511.17701,
Section 4.5 and Figure 6. These plots use PyPIC3D output, not Entity data.

Run from this directory, with no arguments, after run_bz_monopole.py finishes::

    python plot_bz.py

It reads the final diagnostics written to DATA_DIR and writes one figure
beside them. Edit the constants below to retarget it. No JAX, GPU, simulation
initialization, or LaTeX installation is required.

Panels: (a) -H_phi/B0 with contours of measured poloidal magnetic flux;
(b) H_phi/B0 at r=2,3,4,5, with the leading-order -a*sin(theta)^2/8 curve;
(c) Omega/omega_h at the same radii; (d) L_EM/(B0^2*omega_h^2/6).
Hphi, omega, and luminosity are the runner's collocated diagnostics. Profiles
are linearly interpolated in radius without angular smoothing. Flux contours
integrate the saved radial_flux=sqrt(gamma)*B^r with the trapezoidal rule.

Normalization uses B0 and omega_h embedded in each NumPy snapshot.
In particular, it does not replace omega_h by the a/r_H shorthand printed in
the preprint. The figure always displays the actual saved time, not the
paper's t=200.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Must match SimulationParameters.output_directory in simulation_parameters.py.
# Spelled out here so plotting never imports JAX.
DATA_DIR = Path('data')
FINAL_DIAGNOSTICS = 'diagnostics.npz'
FIGURE_NAME = 'figure6_diagnostics.png'
RADII = (2., 3., 4., 5.)
DPI = 150

REQUIRED = ('r', 'theta', 'Hphi', 'omega', 'radial_flux', 'luminosity', 'time')
# Normalization scalars; output_metadata writes every one of these.
SCALARS = ('spin', 'horizon', 'B0', 'omega_h', 'r_min', 'r_max', 'sponge_start')


@dataclass(frozen=True)
class Normalization:
    spin: float
    horizon: float
    B0: float
    omega_h: float
    r_min: float
    r_max: float
    sponge_start: float

    @property
    def luminosity_bz(self):
        return self.B0**2 * self.omega_h**2 / 6

    @classmethod
    def from_snapshot(cls, p):
        spin = float(p['spin'])
        if not np.isfinite(spin) or not 0 < spin < 1:
            raise ValueError('Figure 6 normalization requires 0 < spin < 1 (omega_h and L_BZ must be nonzero).')
        return cls(spin, float(p['horizon']), float(p['B0']), float(p['omega_h']),
                   float(p['r_min']), float(p['r_max']), float(p['sponge_start']))



def theta_bounds(data):
    """The saved diagnostic grid defines its angular extent."""
    theta = np.asarray(data['theta'], dtype=float)
    return float(theta[0]), float(theta[-1])


def load_snapshot(path):
    """Load only the plot diagnostics; never deserialize pickle objects."""
    with np.load(path, allow_pickle=False) as saved:
        missing = set(REQUIRED + SCALARS) - set(saved.files)
        if missing:
            raise ValueError(f'{path}: missing {", ".join(sorted(missing))}. '
                             'Use a self-contained snapshot_*.npz or diagnostics.npz file.')
        return {key: np.asarray(saved[key], dtype=float) for key in REQUIRED + SCALARS}


def radial_profile(r, values, radius):
    """Interpolate to the requested physical radius, rather than rounding it."""
    if not r[0] <= radius <= r[-1]:
        raise ValueError(f'Profile radius {radius:g} is outside sampled radii [{r[0]:g}, {r[-1]:g}].')
    high = int(np.searchsorted(r, radius))
    if high == 0 or r[high] == radius:
        return values[high].copy()
    low = high-1
    fraction = (radius-r[low])/(r[high]-r[low])
    return (1-fraction)*values[low] + fraction*values[high]


def poloidal_flux(theta, radial_flux):
    """Psi(r,theta)-Psi(r,theta_min) from measured sqrt(gamma)*B^r (no 2*pi factor)."""
    increments = .5*(radial_flux[:, 1:]+radial_flux[:, :-1])*np.diff(theta)
    return np.concatenate((np.zeros((len(radial_flux), 1)), np.cumsum(increments, axis=1)), axis=1)


def cell_edges(nodes, lower, upper):
    if nodes[0] < lower-1e-12 or nodes[-1] > upper+1e-12:
        raise ValueError('Snapshot coordinates lie outside the snapshot domain.')
    return np.r_[lower, .5*(nodes[:-1]+nodes[1:]), upper]


def available_radii(data, radii=RADII):
    """Keep the profile radii the snapshot actually samples."""
    return tuple(r for r in radii if data['r'][0] <= r <= data['r'][-1])


def make_figure(data, norm, *, radii=RADII):
    """Return the Figure 6 Matplotlib Figure for one snapshot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle

    r, theta = data['r'], data['theta']
    if any(not np.isfinite(rad) or rad < norm.horizon for rad in radii):
        raise ValueError('Profile radii must be finite and on or outside the event horizon.')
    h_profiles = [radial_profile(r, data['Hphi']/norm.B0, rad) for rad in radii]
    o_profiles = [radial_profile(r, data['omega']/norm.omega_h, rad) for rad in radii]
    exterior = r >= norm.horizon
    if np.count_nonzero(exterior) < 2:
        raise ValueError('Need at least two exterior radial samples for the figure.')
    with plt.rc_context({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif',
                         'font.size': 10, 'axes.titlesize': 12, 'axes.linewidth': .8,
                         'xtick.direction': 'in', 'ytick.direction': 'in'}):
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.8), layout='constrained',
                                 gridspec_kw={'width_ratios': [1, 1.12, 1.12, 1.22]})
        signal = -data['Hphi']/norm.B0
        positive = np.isfinite(signal) & (signal > 0) & exterior[:, None]
        maximum = float(np.max(signal[positive])) if positive.any() else .03
        upper = max(.03, maximum)
        cmap = plt.get_cmap('inferno').copy()
        cmap.set_bad('#d9d9d9')
        re, te = np.meshgrid(cell_edges(r, norm.r_min, norm.r_max),
                              cell_edges(theta, *theta_bounds(data)), indexing='ij')
        mesh = axes[0].pcolormesh(re*np.sin(te), re*np.cos(te),
                                 np.ma.array(signal, mask=~positive), shading='flat',
                                 cmap=cmap, norm=LogNorm(1e-4, upper), rasterized=True)
        colorbar = fig.colorbar(mesh, ax=axes[0], orientation='horizontal',
                               location='top', shrink=.96, pad=.04, extend='both')
        colorbar.ax.tick_params(labelsize=8)
        rr, tt = np.meshgrid(r, theta, indexing='ij')
        psi = poloidal_flux(theta, data['radial_flux'])/norm.B0
        psi = np.ma.masked_where(~np.isfinite(psi) | ~np.broadcast_to(exterior[:, None], psi.shape), psi)
        # Fixed flux levels label measured field lines; contours are not imposed radial rays.
        levels = np.linspace(0, 2, 19)[1:-1]
        if psi.count() and float(psi.max()) > float(psi.min()):
            levels = levels[(levels > psi.min()) & (levels < psi.max())]
            if levels.size:
                axes[0].contour(rr*np.sin(tt), rr*np.cos(tt), psi, levels=levels,
                                colors='white', linewidths=.65, alpha=.85)
        axes[0].add_patch(Circle((0, 0), norm.horizon, color='black', zorder=5))
        axes[0].set(xlim=(0, norm.r_max), ylim=(-norm.r_max, norm.r_max), aspect='equal',
                    xlabel=r'$x/r_g$', ylabel=r'$z/r_g$', title=r'(a) $-H_\phi/B_0$')
        has_masked = bool(np.any((~np.isfinite(signal) | (signal <= 0)) & exterior[:, None]))

        colors = plt.get_cmap('viridis')(np.linspace(.12, .85, len(radii)))
        for radius, hp, op, color in zip(radii, h_profiles, o_profiles, colors):
            axes[1].plot(theta, np.ma.masked_invalid(hp), color=color, lw=1.35)
            axes[2].plot(theta, np.ma.masked_invalid(op), color=color, lw=1.35,
                         label=rf'$r={radius:g}\,r_g$')
        analytic = -norm.spin*np.sin(theta)**2/8
        axes[1].plot(theta, analytic, 'k--', lw=1.35, label='BZ leading order')
        axes[1].legend(loc='upper right', fontsize=8, frameon=True, facecolor='white', framealpha=.9)
        axes[2].axhline(.5, color='black', ls='--', lw=1, alpha=.7)
        axes[2].legend(loc='best', fontsize=8, frameon=False)
        axes[1].set_title(r'(b) $H_\phi/B_0$')
        axes[2].set_title(r'(c) $\Omega/\omega_H$')
        for ax in axes[1:3]:
            ax.set(xlim=(0, np.pi), xlabel=r'$\theta$')
            ax.set_xticks([0, np.pi/2, np.pi], ['0', r'$\pi/2$', r'$\pi$'])
        luminosity = data['luminosity']/norm.luminosity_bz
        axes[3].plot(r[exterior], np.ma.masked_invalid(luminosity[exterior]), color='#2a718e', lw=1.6)
        axes[3].axhline(1., color='black', ls='--', lw=1)
        axes[3].axvspan(norm.sponge_start, norm.r_max, color='#d9d9d9', alpha=.45, zorder=0)
        axes[3].text((norm.sponge_start+norm.r_max)/2, .97, 'sponge', rotation=90,
                     va='top', ha='center', transform=axes[3].get_xaxis_transform(), fontsize=8, color='.4')
        axes[3].set(xlim=(norm.horizon, norm.r_max), xlabel=r'$r/r_g$',
                    title=r'(d) $L_{\rm EM}/L_{\rm BZ}$')
        for ax in axes[1:]:
            ax.grid(alpha=.15, lw=.6)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)
        title = (rf'BZ monopole - Entity Fig. 6 diagnostics    $t={float(data["time"]):g}\,r_g/c$'
                 rf'    $a={norm.spin:g}$')
        fig.suptitle(title, fontsize=13)
        if has_masked:
            fig.supxlabel(r'Gray map cells: $-H_\phi/B_0\leq0$ or undefined; black disk: event horizon.', fontsize=8)
    return fig


def main():
    """Plot the final saved diagnostics; edit the constants above to retarget."""
    import matplotlib.pyplot as plt
    directory = DATA_DIR.expanduser().resolve()
    path = directory/FINAL_DIAGNOSTICS
    data = load_snapshot(path)
    figure = make_figure(data, Normalization.from_snapshot(data),
                         radii=available_radii(data))
    destination = directory/FIGURE_NAME
    figure.savefig(destination, dpi=DPI)
    plt.close(figure)
    print(f'{path.name}: t={float(data["time"]):g} -> {destination}')


if __name__ == '__main__':
    main()
