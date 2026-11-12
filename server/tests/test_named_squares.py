"""surface._plazas: a square mapped only as a named place (place=square, no pedestrian or surface tags) is not paved over
its carriageways (BeamNG.Earth-RU); a pedestrian area or a square with its own surface still is."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import surface  # noqa: E402


class Layer(SimpleNamespace):
    def __len__(self):
        return len(self.attrs["class"])


def layer(rows):
    return Layer(attrs={"class": np.array([r[0] for r in rows], dtype=object),
                        "subtype": np.array([r[1] for r in rows], dtype=object),
                        "tags": np.array([json.dumps(r[2]) for r in rows], dtype=object)})


class NamedSquareTest(unittest.TestCase):
    def test_place_only_square_left_out(self):
        f = layer([("plaza", "pedestrian", {"place": "square", "wikipedia": "ru:x"}),
                   ("pedestrian", "pedestrian", {"highway": "pedestrian", "area": "yes"}),
                   ("plaza", "pedestrian", {"place": "square", "surface": "sett"}),
                   ("plaza", "pedestrian", {"place": "square", "area:highway": "pedestrian"}),
                   ("park", "park", {"leisure": "park"})])
        self.assertEqual(surface._plazas(f).tolist(), [False, True, True, True, False])

    def test_no_tags_column(self):
        f = Layer(attrs={"class": np.array(["plaza"], dtype=object), "subtype": np.array(["pedestrian"], dtype=object)})
        self.assertEqual(surface._plazas(f).tolist(), [True])


if __name__ == "__main__":
    unittest.main()
