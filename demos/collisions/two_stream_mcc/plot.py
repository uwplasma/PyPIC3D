"""Compare the collisionless and electron-neutral MCC two-stream runs.

Run from the repository root after
    PyPIC3D --config demos/collisions/two_stream_mcc/two_stream.toml
    PyPIC3D --config demos/collisions/two_stream_mcc/two_stream_mcc.toml
"""
import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs/two_stream_mcc"
RUNS = {"collisionless": ("collisionless", ("tab:red", "tab:blue", "tab:green")),
        "mcc": ("with electron-neutral MCC", ("tab:orange", "tab:cyan", "tab:olive"))}

fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
for run, (label, colors) in RUNS.items():
    load = lambda f: np.loadtxt(os.path.join(OUTPUT_DIR, run, "data", f), delimiter=",", ndmin=2)
    field, kinetic, momentum = load("electric_field_energy.txt"), load("kinetic_energy.txt"), load("total_momentum.txt")
    for k, y in enumerate((field[:, 1], kinetic[:, 1] / kinetic[0, 1], momentum[:, 1] / momentum[0, 1])):
        ax[k].plot(field[:, 0], y, color=colors[k], lw=2, label=label)
    print(f"{run:14s} peak E-field energy = {field[:, 1].max():.3e} J, final kinetic / initial = {kinetic[-1, 1] / kinetic[0, 1]:.4f}")
ax[0].set(ylabel="Electric field energy [J]", title="Weakly ionized two-stream comparison")
ax[1].set(ylabel="Kinetic energy / initial")
ax[2].set(ylabel=r"$\sum m|\mathbf{v}|$ / initial", xlabel="Time [s]")
for a in ax:
    a.legend()
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "two_stream_mcc.png"), dpi=300)
print(f"Saved {OUTPUT_DIR}/two_stream_mcc.png")
