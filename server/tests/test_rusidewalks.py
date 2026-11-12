"""rusidewalks.py: streets whose sidewalks are mapped as their own lines get no guessed ones (BeamNG.Earth-RU)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rusidewalks  # noqa: E402


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.street = np.array([[0.0, 0.0], [200.0, 0.0]])

    def seg(self, y, x0=0.0, x1=200.0):
        return np.array([[x0, y, x1, y]])

    def test_both_sides(self):
        segs = np.concatenate([self.seg(8.0), self.seg(-8.0)])
        left, right = rusidewalks.coverage(self.street, segs)
        self.assertEqual((left, right), (1.0, 1.0))

    def test_one_side_and_part_of_the_length(self):
        left, right = rusidewalks.coverage(self.street, self.seg(6.0, 0.0, 100.0))
        self.assertAlmostEqual(left, 0.5, places=1)
        self.assertEqual(right, 0.0)

    def test_a_very_short_piece(self):
        street = np.array([[0.0, 0.0], [3.0, 0.0]])        # (shorter than half a step: no sample before)
        self.assertEqual(rusidewalks.coverage(street, self.seg(6.0)), (1.0, 0.0))

    def test_too_far_too_close_or_across(self):
        self.assertEqual(rusidewalks.coverage(self.street, self.seg(40.0)), (0.0, 0.0))     # another street's
        self.assertEqual(rusidewalks.coverage(self.street, self.seg(0.5)), (0.0, 0.0))      # (on the centreline)
        across = np.array([[100.0, -20.0, 100.0, 20.0]])                                    # a crossing
        self.assertEqual(rusidewalks.coverage(self.street, across), (0.0, 0.0))


class Feats:
    """A minimal roads layer: lines of (lon, lat) near 60 N with their classes and subclasses."""

    def __init__(self, lines, classes, subclasses, tags=None):
        coords, coord_off = [], [0]
        for line in lines:
            coords.extend(line)
            coord_off.append(len(coords))
        n = len(lines)
        self.lonlat = np.asarray(coords, np.float64)
        self.kind = np.full(n, 2, np.uint8)
        self.part_off = np.arange(n + 1)
        self.ring_off = np.arange(n + 1)
        self.coord_off = np.asarray(coord_off)
        self.attrs = {"class": np.array(classes, dtype=object), "subclass": np.array(subclasses, dtype=object)}
        if tags is not None:
            self.attrs["osm_sidewalk"] = np.array(tags, dtype=object)
        self.rules = {}

    def __len__(self):
        return len(self.kind)


def line(dy_m, x0=30.0, x1=30.004, lat=60.0):
    d = dy_m / 110574.0
    return [(x0, lat + d), (x1, lat + d)]


class ApplyTest(unittest.TestCase):
    def test_street_with_mapped_sidewalks_is_separate(self):
        f = Feats([line(0), line(9), line(-9), line(300)],
                  ["residential", "footway", "footway", "residential"],
                  ["", "sidewalk", "sidewalk", ""])
        rusidewalks.apply(f)
        tags = f.attrs["osm_sidewalk"]
        self.assertEqual(tags[0], "separate")
        self.assertEqual(tags[3], "")               # (no sidewalk lines beside it: the class rule as before)

    def test_own_tag_and_plain_footways_kept(self):
        f = Feats([line(0), line(9), line(0, lat=60.01), line(9, lat=60.01)],
                  ["residential", "footway", "residential", "footway"],
                  ["", "sidewalk", "", ""], tags=["both", "", "", ""])
        rusidewalks.apply(f)
        tags = f.attrs["osm_sidewalk"]
        self.assertEqual(tags[0], "both")           # the street's own tag says
        self.assertEqual(tags[2], "")               # a park path beside it is not its sidewalk


if __name__ == "__main__":
    unittest.main()
