"""Energy amplification plots and descriptive exponential fits (not proof of SR)."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def fit_growth(time, energy, start, end):
    time, energy = np.asarray(time, float), np.asarray(energy, float)
    selected = (time >= start) & (time <= end)
    result = dict(status='insufficient_data', fit_start=float(start), fit_end=float(end),
                  samples=int(selected.sum()), energy_growth_rate=None,
                  amplitude_growth_rate=None, energy_efolding_time=None,
                  amplitude_efolding_time=None, slope_standard_error=None, r_squared=None)
    if not np.all(np.isfinite(time)) or np.any(~np.isfinite(energy[selected])) or np.any(energy[selected] <= 0):
        return dict(result, status='invalid_data')
    if selected.sum() < 8:
        return result
    x, y = time[selected], np.log(energy[selected])
    x0, y0 = x.mean(), y.mean()
    xx = np.sum((x-x0)**2)
    if xx <= 0 or np.any(np.diff(x) <= 0):
        return dict(result, status='invalid_data')
    slope = np.sum((x-x0)*(y-y0))/xx
    residual = y-(y0+slope*(x-x0))
    rss, tss = np.sum(residual**2), np.sum((y-y0)**2)
    stderr = np.sqrt(rss/(len(x)-2)/xx)
    r_squared = 1-rss/tss if tss > 1.e-28 else 0.
    resolved = r_squared >= 0.8 and abs(slope) > max(3*stderr, 1.e-12/(x[-1]-x[0]))
    status = ('growing' if slope > 0 else 'decaying') if resolved else 'unresolved'
    return dict(result, status=status, energy_growth_rate=float(slope),
                amplitude_growth_rate=float(slope/2), slope_standard_error=float(stderr),
                r_squared=float(r_squared), log_energy_at_mean_time=float(y0),
                mean_fit_time=float(x0),
                energy_efolding_time=float(1/slope) if slope > 0 and resolved else None,
                amplitude_efolding_time=float(2/slope) if slope > 0 and resolved else None)


def analyze(directory, *, fit_start=None, fit_end=None):
    for name, value in (('fit_start', fit_start), ('fit_end', fit_end)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    if fit_start is not None and fit_end is not None and fit_start >= fit_end:
        raise ValueError('fit_start must precede fit_end')
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    trace = np.atleast_1d(np.genfromtxt(directory/'energy.csv', delimiter=',', names=True))
    time, energy = trace['time'], trace['fido_energy_exterior']
    p = metadata['parameters']
    start = fit_start if fit_start is not None else p.get('fit_start')
    end = fit_end if fit_end is not None else p.get('fit_end')
    if start is None:
        start = max(float(time[-1])/2, 2*p['half_width'])
    if end is None:
        end = float(time[-1])
    fit = fit_growth(time, energy, start, end)
    summary = dict(fit, energy_amplification=float(energy[-1]/energy[0]),
                   equivalent_amplitude_amplification=float(np.sqrt(energy[-1]/energy[0])),
                   interpretation='Descriptive FIDO-energy fit; physical superradiance is unverified.',
                   uncertainty_note='OLS standard error ignores correlated cavity oscillations; not a physical error bar.')
    (directory/'growth.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    axes[0].plot(time, trace['fido_energy_total'], label='FIDO: domain minus core')
    axes[0].plot(time, energy, label='FIDO: exterior')
    axes[0].plot(time, trace['killing_energy_exterior'], label='Killing: exterior')
    axes[0].set(ylabel='EM energy')
    axes[0].legend()
    axes[1].plot(time, energy/energy[0], label='Exterior energy / initial')
    axes[1].plot(time, np.sqrt(energy/energy[0]), label='Equivalent amplitude / initial')
    if fit['energy_growth_rate'] is not None:
        selected = (time >= start) & (time <= end)
        prediction = np.exp(fit['log_energy_at_mean_time'] +
                            fit['energy_growth_rate']*(time[selected]-fit['mean_fit_time']))
        axes[1].plot(time[selected], prediction/energy[0], '--', label=f"Energy fit: {fit['status']}")
    axes[1].set(ylabel='Amplification', yscale='log')
    axes[1].legend()
    axes[2].plot(time, trace['killing_flux_horizon_in'], label='Into horizon (staircase)')
    axes[2].plot(time, trace['killing_flux_wall_out'], label='Out through PEC walls')
    axes[2].set(xlabel='Coordinate time (G=c=1)', ylabel='Killing-energy flux')
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle('Vacuum Kerr PEC cavity — physical growth not yet established')
    fig.savefig(directory/'energy_growth.png', dpi=160)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--fit-start', type=float)
    parser.add_argument('--fit-end', type=float)
    args = parser.parse_args()
    print(json.dumps(analyze(args.directory, fit_start=args.fit_start, fit_end=args.fit_end), indent=2))


if __name__ == '__main__':
    main()
