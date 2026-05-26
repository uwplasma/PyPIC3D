import unittest

import jax.numpy as jnp

from PyPIC3D.diagnostics.openPMD import _fields_to_interior_map


def _zero_field(shape):
    return tuple(jnp.zeros(shape) for _ in range(3))


class OpenPMDDiagnosticsTests(unittest.TestCase):

    def test_fields_to_interior_map_skips_empty_pml_state(self):
        shape = (5, 5, 5)
        E = _zero_field(shape)
        B = _zero_field(shape)
        J = _zero_field(shape)
        rho = jnp.zeros(shape)
        phi = jnp.zeros(shape)
        external_fields = (_zero_field(shape), _zero_field(shape))

        field_map = _fields_to_interior_map((E, B, J, rho, phi, external_fields, None))

        self.assertNotIn("field_1", field_map)
        self.assertEqual(field_map["rho"].shape, (3, 3, 3))

    def test_fields_to_interior_map_skips_active_pml_memory(self):
        shape = (5, 5, 5)
        E = _zero_field(shape)
        B = _zero_field(shape)
        J = _zero_field(shape)
        rho = jnp.zeros(shape)
        phi = jnp.zeros(shape)
        external_fields = (_zero_field(shape), _zero_field(shape))
        pml_state = (_zero_field((3, 3, 3)), _zero_field((3, 3, 3)))

        field_map = _fields_to_interior_map((E, B, J, rho, phi, external_fields, pml_state))

        self.assertNotIn("field_1", field_map)

    def test_fields_to_interior_map_keeps_array_history_fields(self):
        shape = (5, 5, 5)
        E = _zero_field(shape)
        B = _zero_field(shape)
        J = _zero_field(shape)
        rho = jnp.zeros(shape)
        phi = jnp.zeros(shape)
        external_fields = (_zero_field(shape), _zero_field(shape))
        A2 = jnp.ones(shape)

        field_map = _fields_to_interior_map((E, B, J, rho, phi, external_fields, A2))

        self.assertIn("field_1", field_map)
        self.assertEqual(field_map["field_1"].shape, (3, 3, 3))


if __name__ == "__main__":
    unittest.main()
