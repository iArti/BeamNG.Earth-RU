"""builtground.py: the ground photo round buildings painted from their footprints (BeamNG.Earth-RU): which footprints
count, the apron, the courtyards (small ones paved, a block garden not), the photo's greenery kept, the shadow."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import builtground  # noqa: E402


def feats_of(rings, classes=None, under=None):
    """A minimal buildings Features stand-in: one polygon per ring of (lon, lat)."""
    rings = [np.asarray(r, np.float64) for r in rings]
    attrs = {}
    if classes is not None:
        attrs["class"] = np.array(classes, dtype=object)
    if under is not None:
        attrs["is_underground"] = np.array(under)
    return SimpleNamespace(kind=np.full(len(rings), 3, np.uint8), attrs=attrs, parts=lambda i: [[rings[i]]],
                           __len__=lambda: len(rings))


class Feats(SimpleNamespace):
    def __len__(self):
        return len(self.kind)


def block(n, walls, w=6):
    """A footprint mask n x n with a closed ring of buildings walls px from the edge, w px deep."""
    m = np.zeros((n, n), bool)
    a, b = walls, n - walls
    m[a:a + w, a:b] = m[b - w:b, a:b] = m[a:b, a:a + w] = m[a:b, b - w:b] = True
    return m


class PadsTest(unittest.TestCase):
    def test_standing_buildings_only(self):
        sq = [(30.0, 60.0), (30.001, 60.0), (30.001, 60.001), (30.0, 60.001)]
        f = feats_of([sq, sq, sq], classes=["apartments", "carport", "roof"], under=[False, False, False])
        f = Feats(**vars(f))
        pads = builtground.building_pads([f])
        self.assertEqual(len(pads), 1)
        lat, lon = pads[0]
        self.assertAlmostEqual(float(lat[0]), 60.0)
        self.assertAlmostEqual(float(lon[1]), 30.001)
        f2 = Feats(**vars(feats_of([sq], under=[True])))
        self.assertEqual(builtground.building_pads([f2]), [])
        self.assertEqual(builtground.building_pads([None]), [])


class MaskTest(unittest.TestCase):
    def test_dilate_and_enclosed(self):
        m = np.zeros((9, 9), bool)
        m[4, 4] = True
        self.assertEqual(int(builtground.dilate(m, 1).sum()), 5)
        ring = block(20, 2, 2)
        inside = builtground.enclosed(ring)
        self.assertTrue(inside[10, 10])
        self.assertFalse(inside[0, 0])
        self.assertFalse(inside[3, 10])             # (the ring itself is not enclosed ground)

    def test_small_parts(self):
        m = np.zeros((10, 10), bool)
        m[0:2, 0:2] = True                          # 4 px
        m[5:10, 5:10] = True                        # 25 px
        out = builtground.small_parts(m, 10)
        self.assertTrue(out[0:2, 0:2].all())
        self.assertFalse(out[5:10, 5:10].any())

    def test_small_courtyard_paved_large_garden_not(self):
        px = 0.5
        small = block(120, 20, 12)                  # a 40 m block, its well courtyard 28 m across (~780 m2)
        paved, _ = builtground.masks(small, px)
        self.assertGreater(float(paved[60, 60]), 0.9)
        self.assertLess(float(paved[5, 5]), 0.1)    # outside the block: the photo's
        big = block(400, 20, 12)                    # 180 m: its garden is far larger than YARD_MAX_M2
        paved, _ = builtground.masks(big, px)
        self.assertLess(float(paved[200, 200]), 0.1)
        self.assertGreater(float(paved[20 + 12 + 2, 200]), 0.9)     # its apron is paved

    def test_greenery_kept(self):
        px = 0.5
        m = block(120, 20, 12)
        green = np.zeros_like(m)
        green[50:70, 50:70] = True                  # a lawn in the yard
        paved, _ = builtground.masks(m, px, green)
        self.assertLess(float(paved[60, 60]), 0.1)
        self.assertGreater(float(paved[38, 38]), 0.9)

    def test_shadow_at_the_walls(self):
        m = np.zeros((100, 100), bool)
        m[40:60, 40:60] = True
        _, shadow = builtground.masks(m, 0.5)
        self.assertLess(float(shadow[50, 61]), 0.95)                # right outside a wall: darker
        self.assertAlmostEqual(float(shadow[50, 95]), 1.0, places=3)     # far off: as it was
        self.assertAlmostEqual(float(shadow[50, 50]), 1.0, places=3)     # under the building: untouched


if __name__ == "__main__":
    unittest.main()
