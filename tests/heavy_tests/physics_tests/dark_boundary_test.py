"""Reflecting Proca cavities and outgoing transverse/longitudinal PML packets."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from PyPIC3D.boundary_conditions.PML import load_pml_from_toml
from PyPIC3D.solvers.dark_matter_yee.dark_photon_fields import (
    initialize_dark_photon_fields, synchronized_dark_fields, dark_divergence,
)
from PyPIC3D.solvers.dark_matter_yee.pml import initialize_dark_pml, advance_dark_pml
from tests.kernel_fixtures import kernel_parameters
from tests.support.dark_photon_fixtures import evolve, interior, wave


def cavity(s, d, longitudinal, time=0., discrete=False):
    E, A, phi = initialize_dark_photon_fields(s, d)
    k = np.pi/float(d.x_wind)
    kh = 2*np.sin(k*float(d.dx)/2)/float(d.dx) if discrete else k
    omega = float(d.C)*np.sqrt(kh**2+s.dark_mu**2)
    xc = d.grids.tiled_center_grid[0][..., None, None] + d.x_wind/2
    xv = d.grids.tiled_vertex_grid[0][..., None, None] + d.x_wind/2
    component = 0 if longitudinal else 1
    shape = jnp.cos(k*xv) if longitudinal else jnp.sin(k*xc)
    A, E = list(A), list(E)
    A[component] = jnp.broadcast_to(shape*np.cos(omega*time), phi.shape)
    amplitude = float(d.C)**2*s.dark_mu**2/omega if longitudinal else omega
    E[component] = jnp.broadcast_to(amplitude*shape*np.sin(omega*time), phi.shape)
    if longitudinal:
        phi = jnp.broadcast_to(float(d.C)**2*kh/omega*jnp.sin(k*xc)*np.sin(omega*time), phi.shape)
    return tuple(E), tuple(A), phi


def packet(s, d, longitudinal=False, oblique=False):
    """Right-going spectral packet satisfying the discrete free-wave constraint."""
    E, A, phi = initialize_dark_photon_fields(s, d)
    nx, ny = int(d.Nx), int(d.Ny)
    dx, dy = float(d.dx), float(d.dy)
    x = np.arange(nx)*dx-float(d.x_wind)/2
    y = np.arange(ny)*dy-float(d.y_wind)/2
    kx = 2*np.pi*np.fft.fftfreq(nx, dx)[:, None]
    ky = 2*np.pi*np.fft.fftfreq(ny, dy)[None, :]
    if oblique:
        envelope = np.exp(-((x[:, None]+3)**2+(y[None, :]+3)**2)/(2*1.5**2))
        amplitude = envelope*np.cos(2*(x[:, None]+y[None, :]+6))
    else:
        amplitude = np.exp(-(x[:, None]+6)**2/(2*1.5**2))*np.cos(3*(x[:, None]+6))
    ahat = np.fft.fft2(amplitude)
    ahat[0, 0] = 0.
    kxh, kyh = 2*np.sin(kx*dx/2)/dx, 2*np.sin(ky*dy/2)/dy
    omega = np.sign(kx+ky)*float(d.C)*np.sqrt(kxh**2+kyh**2+s.dark_mu**2)
    # Zero-frequency directions carry no packet; avoid a sign ambiguity there.
    ahat[omega == 0.] = 0.
    omega = np.where(omega == 0., 1., omega)
    component = 0 if longitudinal else 2
    offset = np.exp(1j*kx*dx/2) if longitudinal else 1.
    avec = np.fft.ifft2(ahat*offset).real
    evec = np.fft.ifft2(1j*(float(d.C)**2*s.dark_mu**2/omega if longitudinal else omega)*ahat*offset).real
    A, E = list(A), list(E)
    A[component] = A[component].at[interior(s)].set(avec[None, None, None, ..., None])
    E[component] = E[component].at[interior(s)].set(evec[None, None, None, ..., None])
    if longitudinal:
        values = np.fft.ifft2(float(d.C)**2*kxh/omega*ahat).real
        phi = phi.at[interior(s)].set(values[None, None, None, ..., None])
    return initialize_dark_photon_fields(s, d, tuple(E), tuple(A), phi)


def interior_energy(fields, state, s, d, mask):
    E, A, phi, B = synchronized_dark_fields(fields, s, d, state)
    idx = interior(s)
    density = sum(e[idx]**2 + d.C**2*b[idx]**2 + d.C**2*s.dark_mu**2*a[idx]**2 for e, a, b in zip(E, A, B))
    density = density+s.dark_mu**2*phi[idx]**2
    return jnp.sum(density*mask)


class TestDarkBoundaryPhysics(unittest.TestCase):
    def test_pml_temporal_convergence(self):
        for longitudinal in (False, True):
            results = []
            for steps in (20, 40, 80):
                s, d = kernel_parameters(Nx=32, Ny=1, Nz=1, x_wind=2*np.pi,
                                         y_wind=1., z_wind=1., dt=.4/steps, dark_mu=.7,
                                         solver="dark_matter_yee", pml_active=True)
                fields = initialize_dark_photon_fields(s, d, *wave(s, d, longitudinal, discrete=True))
                profiles = (jnp.full((34, 3, 3), .5), jnp.zeros((34, 3, 3)), jnp.zeros((34, 3, 3)))
                fields, state = initialize_dark_pml(fields, s, d, profiles)
                J = tuple(jnp.zeros_like(v) for v in fields[0])
                fields, state = jax.jit(lambda f, p: jax.lax.fori_loop(
                    0, steps, lambda _, pair: advance_dark_pml(pair[0], J, pair[1], s, d), (f, p),
                ))(fields, state)
                results.append(synchronized_dark_fields(fields, s, d, state))
            differences = [np.sqrt(sum(float(jnp.mean((x-y)**2)) for x, y in
                                      zip(jax.tree.leaves(a), jax.tree.leaves(b))))
                           for a, b in zip(results[:-1], results[1:])]
            order = np.log2(differences[0]/differences[1])
            self.assertTrue(1.8 < order < 2.2, (longitudinal, order))

    def test_cavity_spatial_and_temporal_convergence(self):
        for longitudinal in (False, True):
            for temporal in (False, True):
                errors = []
                for refinement in range(3):
                    n = 32 if temporal else 32*2**refinement
                    steps = 20*2**refinement if temporal else 1024
                    s, d = kernel_parameters(Nx=n, Ny=1, Nz=1, x_wind=2., y_wind=1., z_wind=1.,
                                             dt=.4/steps, C=1.3, dark_mu=.7, boundary_conditions=(1, 0, 0),
                                             solver="dark_matter_yee")
                    initial = initialize_dark_photon_fields(s, d, *cavity(s, d, longitudinal, discrete=temporal))
                    result = evolve(initial, s, d, steps)
                    exact_E, _, exact_phi = cavity(s, d, longitudinal, .4, discrete=temporal)
                    _, exact_A, _ = cavity(s, d, longitudinal, .4-float(d.dt)/2, discrete=temporal)
                    errors.append(np.sqrt(sum(float(jnp.mean((a[interior(s)]-b[interior(s)])**2))
                                              for a, b in zip(jax.tree.leaves(result), jax.tree.leaves((exact_E, exact_A, exact_phi))))))
                    residual = dark_divergence(result[0], s, d)+s.dark_mu**2*result[2]
                    np.testing.assert_allclose(residual[interior(s)][..., 1:, :, :], 0., atol=3e-12)
                orders = np.log2(np.array(errors[:-1])/errors[1:])
                with self.subTest(longitudinal=longitudinal, temporal=temporal, orders=orders):
                    self.assertTrue(np.all((orders > 1.8) & (orders < 2.2)), orders)

    def test_pml_packet_reflection_and_long_time_stability(self):
        for mass, longitudinal, oblique in ((0., False, False), (.7, False, False),
                                            (.7, True, False), (2., True, False), (.7, False, True)):
            n, length, width = (96, 24., 16) if oblique else (384, 48., 48)
            ny = n if oblique else 1
            dt = .04 if oblique else .03
            steps = round((48. if oblique else 72.)/dt)
            s, d = kernel_parameters(Nx=n, Ny=ny, Nz=1, x_wind=length, y_wind=length if oblique else 1.,
                                     z_wind=1., dt=dt, dark_mu=mass, boundary_conditions=(1, 1 if oblique else 0, 0),
                                     solver="dark_matter_yee", pml_active=True)
            fields = packet(s, d, longitudinal, oblique)
            walls = ("-x", "+x", "-y", "+y") if oblique else ("+x",)
            profiles = load_pml_from_toml([dict(wall=wall, thickness=width) for wall in walls], s, d)[4]
            seeded, state = initialize_dark_pml(fields, s, d, profiles)
            J = tuple(jnp.zeros_like(v) for v in fields[0])
            mask = jnp.ones((n, ny, 1)).at[-width:].set(0.)
            if oblique:
                mask = mask.at[:width].set(0.).at[:, :width].set(0.).at[:, -width:].set(0.)

            # Sample the non-PML energy throughout the run; the late maximum
            # measures returning packets without relying on a single instant.
            def run(f, p):
                def step(pair, _):
                    f, p = jax.lax.fori_loop(0, 20, lambda _, pair: advance_dark_pml(pair[0], J, pair[1], s, d), pair)
                    return (f, p), interior_energy(f, p, s, d, mask)
                return jax.lax.scan(step, (f, p), None, length=steps//20)
            (final, state), energies = jax.jit(run)(seeded, state)
            reference = evolve(fields, s, d, steps//20*20)
            reflected = float(interior_energy(reference, None, s, d, mask))
            returned = float(jnp.max(energies[len(energies)*3//4:]))
            initial = float(interior_energy(fields, None, s, d, mask))
            with self.subTest(mass=mass, longitudinal=longitudinal, oblique=oblique):
                self.assertGreater(reflected, .1*initial)
                self.assertLess(returned/reflected, .01)
                self.assertLess(float(jnp.max(energies)), 1.05*initial)
                for value in jax.tree.leaves((final, state)):
                    self.assertTrue(np.all(np.isfinite(value)))


if __name__ == "__main__":
    unittest.main()
