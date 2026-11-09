"""Check gauge independence of the pressure displayed in the video."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from render_naca_pressure import relative_pressure


class PressureDisplayTests(unittest.TestCase):
    def test_display_is_gauge_invariant_and_masks_solid(self):
        p = np.arange(240, dtype=float).reshape(20, 12)/10
        solid = np.zeros_like(p, dtype=bool)
        solid[8:12, 5:7] = True
        original = p.copy()
        a, ref_a = relative_pressure(p, solid)
        b, ref_b = relative_pressure(p + 1000, solid)
        assert_allclose(a, b, atol=2e-13)
        self.assertAlmostEqual(ref_b-ref_a, 1000)
        self.assertTrue(np.isnan(a[solid]).all())
        self.assertAlmostEqual(np.nanmean(a[:2]), 0.)
        assert_allclose(p, original)


if __name__ == '__main__':
    unittest.main()
