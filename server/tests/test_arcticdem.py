"""arcticdem.py: ArcticDEM's 2 m mosaic made into ground for the near terrain north of 60 N (BeamNG.Earth-RU): the
polar stereographic projection, the files' names, the index, the forest floor, the datum, the blunder check, and
dem.py's blend."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import arcticdem  # noqa: E402
import dem as demmod  # noqa: E402


class ProjectionTest(unittest.TestCase):
    def test_pole_and_true_scale(self):
        x, y = arcticdem.polar_xy(90.0, 10.0)
        self.assertAlmostEqual(float(x), 0.0, places=6)
        self.assertAlmostEqual(float(y), 0.0, places=6)
        # on the central meridian (45 W) the point lies on -y; at 70 N its distance is a * m_c (true scale there)
        x, y = arcticdem.polar_xy(70.0, -45.0)
        mc = math.cos(math.radians(70)) / math.sqrt(1 - arcticdem._E ** 2 * math.sin(math.radians(70)) ** 2)
        self.assertAlmostEqual(float(x), 0.0, places=6)
        self.assertAlmostEqual(float(-y), arcticdem._A * mc, places=3)
        # 45 E is +x, 135 E is +y
        x, y = arcticdem.polar_xy(80.0, 45.0)
        self.assertGreater(float(x), 0)
        self.assertAlmostEqual(float(y), 0.0, places=6)
        x, y = arcticdem.polar_xy(80.0, 135.0)
        self.assertAlmostEqual(float(x), 0.0, places=6)
        self.assertGreater(float(y), 0)

    def test_scale_factor(self):
        # the projection's scale at 70 N is 1: a metre along the parallel is a metre in the plane
        lat = 70.0
        d = 1e-4
        x0, y0 = arcticdem.polar_xy(lat, 10.0)
        x1, y1 = arcticdem.polar_xy(lat, 10.0 + d)
        a, e2 = arcticdem._A, arcticdem._E ** 2
        phi = math.radians(lat)
        ground = a * math.cos(phi) / math.sqrt(1 - e2 * math.sin(phi) ** 2) * math.radians(d)
        self.assertAlmostEqual(math.hypot(float(x1 - x0), float(y1 - y0)) / ground, 1.0, places=6)


class FilesTest(unittest.TestCase):
    def test_file_of_matches_pgc_layout(self):
        # 2m_dem_tiles.vrt: 07_41_1_1 starts at x = 0, y = -3400000; 07_40_2_2 at x = -50000, y = -3350000
        self.assertEqual(arcticdem.file_of(25000, -3375000), "07_41_1_1")
        self.assertEqual(arcticdem.file_of(-25000, -3325000), "07_40_2_2")
        self.assertEqual(arcticdem.file_of(0.0, -3400000.0), "07_41_1_1")
        # the Karelian Isthmus (60.70 N, 29.90 E): file 32_72_2_1 (read from PGC's bucket, 2026-11-11)
        x, y = arcticdem.polar_xy(60.70, 29.90)
        self.assertEqual(arcticdem.file_of(float(x), float(y)), "32_72_2_1")
        self.assertEqual(arcticdem.file_url("32_72_2_1"),
                         arcticdem.BASE + "2m/32_72/32_72_2_1_2m_v4.1_dem.tif")

    def test_index(self):
        known = arcticdem.index()
        self.assertIsNotNone(known)
        self.assertGreater(len(known), 9000)
        self.assertIn("32_72_2_1", known)
        for name in list(known)[:200]:      # every name round-trips through its own centre
            r, c, sr, sc = (int(v) for v in name.split("_"))
            cx = (c - arcticdem.ORIGIN) * arcticdem.SUPER_M + (sc - 0.5) * arcticdem.SUB_M
            cy = (r - arcticdem.ORIGIN) * arcticdem.SUPER_M + (sr - 0.5) * arcticdem.SUB_M
            self.assertEqual(arcticdem.file_of(cx, cy), name)

    def test_covers(self):
        z = 15
        n = 2 ** z
        y60 = int((1 - math.asinh(math.tan(math.radians(60.7))) / math.pi) / 2 * n)
        y55 = int((1 - math.asinh(math.tan(math.radians(55.0))) / math.pi) / 2 * n)
        self.assertTrue(arcticdem.covers(z, 0, y60))
        self.assertFalse(arcticdem.covers(z, 0, y55))


class HelpersTest(unittest.TestCase):
    def test_resample_block(self):
        a = np.arange(16, dtype=np.float32).reshape(4, 4)
        self.assertEqual(arcticdem.resample_block(a, 2).tolist(), [[2.5, 4.5], [10.5, 12.5]])
        self.assertEqual(arcticdem.resample_block(a, 8).shape, (8, 8))
        self.assertEqual(arcticdem.resample_block(a, 4).tolist(), a.tolist())

    def test_grow(self):
        m = np.zeros((7, 7), bool)
        m[3, 3] = True
        self.assertEqual(int(arcticdem.grow(m, 1).sum()), 5)
        self.assertEqual(int(arcticdem.grow(m, 2).sum()), 13)

    def test_min_filter(self):
        a = np.arange(25, dtype=np.float32).reshape(5, 5)
        m = arcticdem.min_filter(a, 3)
        self.assertEqual(float(m[2, 2]), 6.0)
        self.assertEqual(float(m[0, 0]), 0.0)
        self.assertEqual(float(m[4, 4]), 18.0)
        self.assertTrue(np.array_equal(arcticdem.min_filter(a, 1), a))

    def test_ground_fills_under_trees_without_pits(self):
        """The stereo surface over a stand sits only part of the canopy's height up (here 6 m of 15): subtracting the
        canopy dug a 9 m pit; the ground under the stand is now the open ground around it."""
        px = 2.0
        rng = np.random.default_rng(1)
        h = np.full((128, 128), 100.0, np.float32) + rng.normal(0, 0.3, (128, 128)).astype(np.float32)
        canopy = np.zeros((128, 128), np.float32)
        canopy[40:90, 40:90] = 15.0
        h[40:90, 40:90] += 6.0
        ok = np.ones((128, 128), bool)
        base = np.full((128, 128), 104.0, np.float32)
        g = arcticdem.ground(h, canopy, None, ok, base, px)
        self.assertLess(float(np.abs(g[50:80, 50:80] - 100.0).max()), 0.6)
        self.assertLess(float(np.abs(g[:20] - 100.0).max()), 0.6)
        # never above the surface (the speckle smoothing aside: compared with the surface smoothed as much)
        self.assertTrue(np.all(g <= arcticdem.gauss(h, arcticdem.SMOOTH_M / px) + 0.05))

    def test_ground_in_villages_from_the_streets(self):
        px = 2.0
        h = np.full((128, 128), 50.0, np.float32)
        built = np.zeros((128, 128), bool)
        built[30:100, 30:100] = True
        for i in range(34, 96, 12):                          # houses 8 m tall, 8 m wide, streets between
            for j in range(34, 96, 12):
                h[i:i + 4, j:j + 4] = 58.0
        ok = np.ones((128, 128), bool)
        g = arcticdem.ground(h, np.zeros_like(h), built, ok, np.full_like(h, 50.0), px)
        self.assertLess(float(np.abs(g[30:100, 30:100] - 50.0).max()), 0.6)    # no pit, no houses


class FakeRasters:
    pass


class FakeGeoid:
    def __init__(self, n):
        self.n = n

    def tile(self, z, x, y, size=arcticdem.TILE):
        return None if self.n is None else np.full((size, size), self.n, np.float32)


class Fake(arcticdem.ArcticDem):
    """The mosaic as a function: surface heights (above the ellipsoid), canopy and land cover given."""

    def __init__(self, surface, canopy=None, built=None, n=18.0):
        super().__init__("/nonexistent/dem", None, rasters=FakeRasters(), geoid=FakeGeoid(n))
        self._s, self._c, self._b = surface, canopy, built

    def surface(self, z, x, y, size=arcticdem.TILE, min_px=arcticdem.LATTICE_M, margin=0):
        if self._s is None:
            return None
        s = np.pad(self._s, margin, mode="edge") if margin else self._s.copy()
        return s, np.isfinite(s)

    def canopy(self, z, x, y, size=arcticdem.TILE):
        return self._c

    def built(self, z, x, y, size=arcticdem.TILE):
        return self._b

    def sparse_ground(self, z, x, y, size=arcticdem.BIAS_PX):
        return getattr(self, "_sparse", None)


KEY = (15, 18911, 9126)     # a z15 tile on the Karelian Isthmus


class GroundTest(unittest.TestCase):
    def test_datum_from_egm2008(self):
        s = np.full((512, 512), 118.0, np.float32)          # 100 m above the geoid, N = 18 m
        base = np.full((512, 512), 103.0, np.float32)
        h, ok = Fake(s).tile(*KEY, base)
        self.assertTrue(ok.all())
        self.assertTrue(np.allclose(h, 100.0))

    def test_datum_from_offset_without_a_model(self):
        s = np.full((512, 512), 118.0, np.float32)
        base = np.full((512, 512), 100.5, np.float32)
        h, ok = Fake(s, n=None).tile(*KEY, base)
        self.assertTrue(np.allclose(h, 100.5))              # the median offset to Mapterhorn's open ground

    def test_trees_taken_off(self):
        s = np.full((512, 512), 118.0, np.float32)
        canopy = np.zeros((512, 512), np.float32)
        canopy[100:300, 100:300] = 15.0
        s[100:300, 100:300] += 7.0                           # the DSM stands part of the way up the crowns there
        base = np.full((512, 512), 100.0, np.float32)
        h, ok = Fake(s, canopy=canopy).tile(*KEY, base)
        self.assertLess(float(np.abs(h[150:250, 150:250] - 100.0).max()), 0.5)

    def test_sea_and_blunders_dropped(self):
        s = np.full((512, 512), 118.0, np.float32)
        s[:, :10] = 400.0                                    # a blunder: 300 m above the ground
        base = np.full((512, 512), 100.0, np.float32)
        base[-20:] = 0.0                                     # Mapterhorn's sea
        h, ok = Fake(s).tile(*KEY, base)
        self.assertFalse(ok[:, :10].any())
        self.assertFalse(ok[-20:].any())
        self.assertTrue(ok[100:200, 100:200].all())

    def test_nothing_there(self):
        base = np.full((512, 512), 100.0, np.float32)
        self.assertIsNone(Fake(None).tile(*KEY, base))
        with self.assertRaises(ValueError):
            Fake(None).tile(13, 0, 0, base)


class DemBlendTest(unittest.TestCase):
    def _dem(self, arctic):
        d = demmod.Dem.__new__(demmod.Dem)
        d.arctic = arctic
        d.arctic_served = set()
        d.stats = demmod.Counter()
        d._lock = demmod.threading.Lock()
        d._local = demmod.threading.local()
        return d

    def test_blend_feathers_the_edge(self):
        s = np.full((512, 512), 118.0, np.float32)
        s[:, 256:] = np.nan                                  # the mosaic has the western half only
        d = self._dem(Fake(s))
        a = np.full((512, 512), 105.0, np.float32)
        out = d._arctic(KEY, a)
        self.assertAlmostEqual(float(out[256, 10]), 100.0, places=3)     # deep inside: the mosaic's ground
        self.assertAlmostEqual(float(out[256, 500]), 105.0, places=3)    # outside: Mapterhorn's
        mid = out[256, 230:256]
        self.assertTrue(np.all(np.diff(mid) >= -1e-4))       # a slope between them, not a step
        self.assertIn(KEY, d.arctic_served)

    def test_far_tiles_moved_to_its_level(self):
        f = Fake(np.full((512, 512), 118.0, np.float32))
        d = self._dem(f)
        a = np.full((512, 512), 105.0, np.float32)
        far = (14, KEY[1] >> 1, KEY[2] >> 1)
        self.assertIs(d._arctic(far, a), a)                  # no open ground points: Mapterhorn's as it is
        n = 2 ** 14
        lat, lon = arcticdem.tile_lat_lon(14, far[1], far[2], 64)
        f._sparse = (lat.ravel(), lon.ravel(), np.full(64 * 64, 100.0))
        out = d._arctic(far, a)
        self.assertLess(float(np.abs(out[100:400, 100:400] - 100.0).max()), 0.5)     # the near terrain's level
        self.assertIn(far, d.arctic_served)
        del n

    def test_the_south_untouched(self):
        d = self._dem(Fake(np.full((512, 512), 118.0, np.float32)))
        a = np.full((512, 512), 105.0, np.float32)
        self.assertIs(d._arctic((15, 18911, 10500), a), a)               # ~55 N: no mosaic
        self.assertIs(self._dem(None)._arctic(KEY, a), a)

    def test_failure_keeps_mapterhorn(self):
        class Broken(Fake):
            def surface(self, *a, **k):
                raise OSError("offline")
        d = self._dem(Broken(None))
        a = np.full((512, 512), 105.0, np.float32)
        self.assertIs(d._arctic(KEY, a), a)
        self.assertTrue(d._local.partial)

    def test_coarse_at(self):
        d = self._dem(Fake(None))
        self.assertTrue(d.arctic_at(60.70, 29.90))
        self.assertFalse(d.arctic_at(55.75, 37.62))          # Moscow
        self.assertFalse(self._dem(None).arctic_at(60.70, 29.90))


if __name__ == "__main__":
    unittest.main()
