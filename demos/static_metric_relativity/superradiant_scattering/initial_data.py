"""An inward electric-parity l=m=1 Hertz packet and compatible Yee curls."""
import jax.numpy as jnp

from PyPIC3D.relativity.core import D_FIELD_LOCATIONS
from .parameters import positions


def curl(vector, spacing, *, forward):
    """D-location potential -> B with forward differences; reverse with backward."""
    def difference(value, axis):
        return ((jnp.roll(value, -1, axis=axis+3)-value) if forward else
                (value-jnp.roll(value, 1, axis=axis+3)))/spacing
    return tuple(difference(vector[(i+2)%3], (i+1)%3) -
                 difference(vector[(i+1)%3], (i+2)%3) for i in range(3))


def hertz_potential(position, p, time=0.):
    """Pi=(ex+i ey) f(r+t)/r with inward phase exp[-i omega(r+t)].

    The angular scalar p.n is sin(theta) exp(i phi), proportional to Y_11.
    In flat space D=curl curl Pi and B=curl d_t Pi solve vacuum Maxwell.
    In Kerr these curls initialize *densities*: constraint-satisfying,
    approximately ingoing data, not a separated Kerr eigenfunction.
    """
    r = jnp.sqrt(jnp.sum(position**2, axis=-1))
    width = (p.packet_outer-p.packet_inner)/2
    center = (p.packet_outer+p.packet_inner)/2
    s = r+time
    u = (s-center)/width
    inside = jnp.abs(u) < 1
    denominator = jnp.maximum(1-u*u, 1.e-12)
    envelope = jnp.where(inside, jnp.exp(1-1/denominator), 0.)
    derivative = envelope*(-2*u/(width*denominator**2))
    phase = jnp.exp(-1j*p.omega*s)
    safe_r = jnp.maximum(r, 1.e-12)
    f = envelope*phase/safe_r
    ft = (derivative-1j*p.omega*envelope)*phase/safe_r
    polarization = jnp.array([1., 1j, 0.])
    return f[..., None]*polarization, ft[..., None]*polarization


def seed_fields(p, dynamic):
    potential, derivative = [], []
    for i, location in enumerate(D_FIELD_LOCATIONS):
        pi, pi_t = hertz_potential(positions(dynamic, location), p)
        potential.append(jnp.real(pi[..., i]))
        derivative.append(jnp.real(pi_t[..., i]))
    D = curl(curl(tuple(potential), p.spacing, forward=True), p.spacing, forward=False)
    B = curl(tuple(derivative), p.spacing, forward=True)
    return D, B
