"""Vacuum specialization of production GR leapfrog, with an interior buffer."""
import jax
import jax.numpy as jnp
from typing import NamedTuple

from PyPIC3D.relativity.core import B_FIELD_LOCATIONS, D_FIELD_LOCATIONS
from PyPIC3D.boundary_conditions.staggered import refresh_fields
from PyPIC3D.solvers.GR_yee import static_metric
from PyPIC3D.solvers.GR_yee.static_metric import (
    compute_covariant_E as electric_aux,
    compute_covariant_H as magnetic_aux,
)
from .initial_data import curl
from .parameters import kerr_radius, positions


class Geometry(NamedTuple):
    flat: bool
    cells: int
    half_width: float
    a: float
    horizon: float
    spacing: float
    core_radius: float


@jax.tree_util.register_pytree_node_class
class VacuumEvolution:
    """All state entries are native densities, not FIDO physical components."""

    def __init__(self, p, static, dynamic, metric):
        # Only geometry participates in JIT cache keys. Output paths, fit
        # windows, duration, and buffer rates cannot force recompilation.
        self.p = Geometry(p.flat, p.cells, p.half_width, p.a, p.horizon, p.spacing, p.core_radius)
        self.static, self.dynamic, self.metric = static, dynamic, metric
        self.buffers = {}
        for kind, locations in (("D", D_FIELD_LOCATIONS), ("B", B_FIELD_LOCATIONS)):
            masks, rates = [], []
            for loc in locations:
                r = kerr_radius(positions(dynamic, loc), p.a)
                u = jnp.clip((p.absorber_radius-r)/(p.absorber_radius-p.core_radius), 0., 1.)
                rate = p.absorber_rate*u**3*(10-15*u+6*u*u)
                rates.append(jnp.zeros_like(rate) if p.flat else rate)
                masks.append(jnp.ones_like(r, dtype=bool) if p.flat else r > p.core_radius)
            self.buffers[kind] = (tuple(masks), tuple(rates))

    def tree_flatten(self):
        # Metric arrays are runtime arguments, never enormous XLA constants.
        return (self.dynamic, self.metric, self.buffers), (self.p, self.static)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        result = object.__new__(cls)
        result.p, result.static = auxiliary
        result.dynamic, result.metric, result.buffers = children
        return result

    def refresh(self, vector, kind):
        mask, _ = self.buffers[kind]
        vector = tuple(jnp.where(m, v, 0.) for m, v in zip(mask, vector))
        return refresh_fields(vector, self.static,
            D_FIELD_LOCATIONS if kind == "D" else B_FIELD_LOCATIONS, kind, self.metric)

    def absorb(self, old, updated, kind, dt):
        """Second-order exponential integration of -sigma V with centered curl.

        Buffer coefficients equal one/zero at all outer walls and halos, so
        these linear combinations preserve the production PEC projections.
        """
        masks, rates = self.buffers[kind]
        return tuple(jnp.where(mask, jnp.exp(-rate*dt)*before +
                     jnp.exp(-rate*dt/2)*(after-before), 0.)
                     for before, after, mask, rate in zip(old, updated, masks, rates))

    def update_B(self, E, B, dt):
        new = static_metric.update_B(E, B, self.metric, self.static, self.dynamic, dt)
        return self.absorb(B, new, "B", dt)

    def update_D(self, D, H, dt):
        zero = tuple(jnp.zeros_like(v) for v in D)
        new = static_metric.update_D(D, H, zero, self.metric, self.static, self.dynamic, dt)
        return self.absorb(D, new, "D", dt)

    def synchronized(self, state):
        D, Bhalf, Dprev, Bprev = state
        Dhalf = tuple((a+b)/2 for a, b in zip(D, Dprev))
        Bminus = tuple((a+b)/2 for a, b in zip(Bhalf, Bprev))
        B = self.update_B(electric_aux(Dhalf, Bhalf, self.metric), Bminus, self.dynamic.dt)
        return D, B

    def step(self, state):
        D, Bhalf, Dprev, _ = state
        _, B = self.synchronized(state)
        Dhalf = tuple((a+b)/2 for a, b in zip(D, Dprev))
        E, H = electric_aux(D, B, self.metric), magnetic_aux(D, B, self.metric)
        Bnext = self.update_B(E, Bhalf, self.dynamic.dt)
        Dmid = self.update_D(Dhalf, H, self.dynamic.dt)
        Dnext = self.update_D(D, magnetic_aux(Dmid, Bnext, self.metric), self.dynamic.dt)
        return Dnext, Bnext, D, Bhalf

    def rhs(self, pair):
        D, B = pair
        dD = curl(magnetic_aux(D, B, self.metric), self.p.spacing, forward=False)
        dB = tuple(-v for v in curl(electric_aux(D, B, self.metric), self.p.spacing, forward=True))
        result = []
        for kind, field, derivative in (("D", D, dD), ("B", B, dB)):
            masks, rates = self.buffers[kind]
            result.append(tuple(jnp.where(mask, d-rate*v, 0.)
                for mask, rate, v, d in zip(masks, rates, field, derivative)))
        return tuple(result)

    def midpoint_start(self, pair, dt):
        first = self.rhs(pair)
        mid = tuple(self.refresh(tuple(v+dt*k/2 for v, k in zip(field, slope)), kind)
                    for field, slope, kind in zip(pair, first, ("D", "B")))
        second = self.rhs(mid)
        return tuple(self.refresh(tuple(v+dt*k for v, k in zip(field, slope)), kind)
                     for field, slope, kind in zip(pair, second, ("D", "B")))

    def initialize(self, D, B):
        pair = self.refresh(D, "D"), self.refresh(B, "B")
        half = self.midpoint_start(pair, -self.dynamic.dt/2)
        previous = self.midpoint_start(pair, -self.dynamic.dt)
        threehalf = self.midpoint_start(pair, -1.5*self.dynamic.dt)
        return pair[0], half[1], previous[0], threehalf[1]

    def advance(self, state, steps):
        return jax.lax.fori_loop(0, steps, lambda _, value: self.step(value), state)
