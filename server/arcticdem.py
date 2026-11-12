"""The Arctic's 2 m elevation: PGC's ArcticDEM v4.1 mosaic, made into ground for the near terrain (BeamNG.Earth-RU).

Why: north of 60 N outside the national lidar areas (Russia's north-west, Karelia, the Kola peninsula, Siberia's
north, northern Scandinavia) Mapterhorn is Copernicus GLO-30: 30 m pixels of a surface model, forests and all. The
ArcticDEM mosaic covers all land north of 60 N at 2 m, from stereo satellite imagery (Maxar), so the shape of the
ground (ditches, embankments, river banks, rock outcrops, hills under 30 m) is there - but it is a surface model too.

Data (CC BY 4.0, cite: Porter et al. 2023, ArcticDEM Version 4.1 Mosaics, doi:10.7910/DVN/3VDC4W; acknowledge PGC and
NSF): Cloud-Optimised GeoTIFFs on AWS (s3://pgc-opendata-dems, us-west-2), 50 x 50 km each (25100 px of 2 m: 100 m of
overlap all round), EPSG:3413 (WGS 84 / NSIDC Sea Ice Polar Stereographic North: true scale at 70 N, central meridian
45 W), float32 metres above the WGS84 *ellipsoid*, nodata -9999, 512 px tiles, overviews down to 128 m. A file is named
<row>_<col>_<subrow>_<subcol> (file_of: 100 km supertiles from the grid's origin, two by two 50 km tiles each); the
ones that exist are listed in arcticdem_index.json.gz (tools/build_arcticdem_index.py, from PGC's 2m_dem_tiles.vrt).

From surface to ground (tile()):

1. the mosaic read on dem.py's tile grid (bilinear, the overview as fine as the tile's pixels; like hrdem.py);
2. ellipsoidal heights to EGM2008 (Mapterhorn's datum): the geoid height N from gravity.py's EGM2008 model where its
   coefficients are on disk (earthd's real gravity downloads them), on the corners of a z10 grid (N changes by
   centimetres over its ~10-20 km); without them, the median offset to the tile's Mapterhorn heights over open ground;
3. the ground only where the mosaic sees it (ground()): open land (Meta / WRI's canopy height, CHMv2, smoothed to the
   stereo model's scale, under OPEN_CANOPY_M) and, where ESA WorldCover says built-up, only its lowest points (within
   LOW_M of the minimum over LOW_WINDOW_M: streets, yards, not roofs). Under trees and buildings the ground is
   interpolated from the open ground around (normalised convolution, finest of FILL_SIGMAS_M with enough support),
   and past their reach it is Mapterhorn's shape moved to the open ground's level. Never above the surface; the
   stereo model's speckle smoothed over SMOOTH_M.
   (The first try subtracted the canopy model from the surface and opened the built-up areas: the stereo surface does
   not reach the crowns' tops, so every stand became a pit 4-6 m deep, and the villages pits down to their ditches.
   "From fairly flat to a front line of trenches", the user on the Karelian Isthmus, 2026-11-11.)
4. checked against Mapterhorn's tile: pixels more than MAX_DIFF_M off it are dropped (the stereo models' blunders over
   water and in shadow), and the sea (Mapterhorn's exact 0) stays the sea.

dem.py blends the result into its tiles of z15 and deeper with a feathered edge (hrdem.feather), as for HRDEM.

    a = ArcticDem(dem_cache_dir, fetcher)
    got = a.tile(z, x, y, base)   # None, or (heights float32 [512, 512] above EGM2008, valid bool [512, 512])
"""
import gzip
import json
import logging
import math
import os
import threading
from collections import OrderedDict

import numpy as np

import cog as cogmod
import rangeread

log = logging.getLogger("earthd.arcticdem")

SOURCE = "arcticdem"                    # the data credits' id (credits.py) and the block cache's folder
# What ground() makes: raise it with every change to its output. dem.py's decoded mirror and the road plans kept
# across runs (revisit.switches_key) are keyed by it, so nothing built on an older ground is read back.
GROUND_VERSION = 5            # 3: distant tiles moved to its level; 2: the seen ground filled (1: canopy subtracted)
RELEASE = "v4.1-2m"
BASE = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/mosaics/v4.1/"
FILE_URL = BASE + "2m/{folder}/{name}_2m_v4.1_dem.tif"
HOST = "pgc-opendata-dems.s3.us-west-2.amazonaws.com"
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arcticdem_index.json.gz")
ATTRIBUTION = ("ArcticDEM v4.1 mosaic, Polar Geospatial Center (University of Minnesota), NSF awards 1043681, "
               "1559691, 1542736, 1810976, 2129685; CC BY 4.0")
MIN_LAT = 59.8                  # tiles wholly south of this are not asked (the mosaic stops at 60 N, with a margin)
SUPER_M = 100000.0              # supertile size (m)
SUB_M = 50000.0                 # file size without its overlap (m)
ORIGIN = 41                     # the supertile index of x (or y) = 0
NODATA_LIMIT = -1000.0
TILE = 512                      # dem.py's tile size
LATTICE_M = 2.0                 # the mosaic's pixel: finer tile pixels are not read finer
MAX_LEVEL = 6                   # 2 m * 2^6 = 128 m: the coarsest overview
TILE_CACHE_BYTES = 192 * 1024 * 1024
GEOID_Z = 10                    # the geoid height's grid (its corners evaluated, bilinear between)
CANOPY_Z = 14                   # canopy and land cover are read per z14 cell (maprasters' unit)
# The stereo DSM is a smooth sheet over the crowns (its matching windows: ~5-10 m), the canopy model a sharp 1.2 m
# raster with the gaps between the trees in it: smoothed to the DSM's scale before it says what is open.
CANOPY_SMOOTH_M = 6.0           # Gaussian sigma (m)
OPEN_CANOPY_M = 1.0             # smoothed canopy under this: open ground, the mosaic's heights are the ground's
OPEN_EDGE_M = 4.0               # the open ground's edge (m) blended softly into the filled ground
BUILT_CLASS = 50                # ESA WorldCover "built-up"
BUILT_GROW_M = 20.0             # the built-up mask grown by this (WorldCover's 10 m pixels miss a building's edge)
LOW_WINDOW_M = 30.0             # in built-up areas only the ground within LOW_M of the lowest over this window ...
LOW_M = 1.0                     # ... counts as seen (streets and yards between houses, not their roofs)
FILL_SIGMAS_M = (8.0, 24.0, 72.0, 216.0)    # the fill's scales (m), finest first
FILL_SUPPORT = 0.08             # a scale serves a pixel when this share of its neighbourhood is open ground
SMOOTH_M = 2.0                  # the stereo model's speckle (m, Gaussian sigma) smoothed out of the result
MAX_DIFF_M = 50.0               # a pixel this far from Mapterhorn's height is a blunder
MARGIN = 64                     # px of the neighbouring tiles each tile is made with (cut after)
CELL_RASTERS = 16               # z14 canopy / land-cover rasters kept (the nine tiles round one read the same few)
# The distant tiles (z12-z14) keep Mapterhorn's shape but are moved to the mosaic's ground level (sparse_ground; dem.py
# _shift_to): the near terrain stood ~5-12 m below them, a trench with a wall round it a kilometre out (2026-11-11)
BIAS_PX = 64                    # the grid the overview is read on ...
BIAS_M = 32.0                   # ... from the overview with pixels this size
OPEN_CLASSES = (20, 30, 40, 60, 90, 100)    # WorldCover: shrubs, grass, crops, bare, wetland, moss: the surface is ground
MIN_OPEN_PX = 2000              # the median geoid offset needs this many open pixels


# --- EPSG:3413 (polar stereographic, north pole, true scale at 70 N, WGS84) ------------------------------------
_A = 6378137.0
_F = 1 / 298.257223563
_E = math.sqrt(_F * (2 - _F))
_LAT_TS = math.radians(70.0)
_LON0 = math.radians(-45.0)


def _t(phi):
    s = np.sin(phi)
    return np.tan(np.pi / 4 - phi / 2) / ((1 - _E * s) / (1 + _E * s)) ** (_E / 2)


_MC = math.cos(_LAT_TS) / math.sqrt(1 - (_E * math.sin(_LAT_TS)) ** 2)
_TC = float(_t(_LAT_TS))


def polar_xy(lat, lon):
    """EPSG:3413 easting / northing (m) of WGS84 lat / lon (degrees; arrays). Snyder (1987) eq. 21-33, 21-34, 21-35
    with the true-scale latitude's m_c / t_c (the pole is (0, 0); x grows towards 45 E, y towards 135 E)."""
    phi = np.radians(np.asarray(lat, np.float64))
    lam = np.radians(np.asarray(lon, np.float64))
    rho = _A * _MC * _t(phi) / _TC
    return rho * np.sin(lam - _LON0), -rho * np.cos(lam - _LON0)


def file_of(x, y):
    """The mosaic file ("rr_cc_s_s") whose 50 km square holds EPSG:3413 point (x, y)."""
    cx, cy = math.floor(x / SUPER_M), math.floor(y / SUPER_M)
    sx = int((x - cx * SUPER_M) // SUB_M) + 1
    sy = int((y - cy * SUPER_M) // SUB_M) + 1
    return f"{cy + ORIGIN:02d}_{cx + ORIGIN:02d}_{sy}_{sx}"


def file_url(name):
    return FILE_URL.format(folder=name[:5], name=name)


_index = {"files": None, "loaded": False}
_index_lock = threading.Lock()


def index():
    """The set of mosaic files that exist, or None when the list is missing (then every file is asked)."""
    with _index_lock:
        if not _index["loaded"]:
            _index["loaded"] = True
            try:
                with gzip.open(INDEX_PATH, "rt", encoding="ascii") as f:
                    _index["files"] = frozenset(json.load(f)["files"])
            except (OSError, ValueError, KeyError) as e:
                log.warning("ArcticDEM index %s: %s (every file is asked)", INDEX_PATH, e)
        return _index["files"]


def tile_lat_lon(z, x, y, size=TILE, margin=0):
    """Pixel-centre lat / lon grids of Web-Mercator tile (z, x, y) (row 0 north), [size, size], or with margin pixels
    of its neighbours all round [size + 2 margin, size + 2 margin]."""
    n = 2 ** z
    i = (np.arange(-margin, size + margin) + 0.5) / size
    e = size + 2 * margin
    lon = (x + i) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + i) / n))))
    return np.broadcast_to(lat[:, None], (e, e)), np.broadcast_to(lon[None, :], (e, e))


def covers(z, x, y):
    """Whether tile (z, x, y) reaches north of MIN_LAT (the only place the mosaic is asked)."""
    n = 2 ** z
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return north > MIN_LAT


def pixel_m(z, y, size=TILE):
    """Ground size (m) of a pixel of tile row y at zoom z (at its centre latitude)."""
    n = 2 ** z
    lat = math.atan(math.sinh(math.pi * (1 - 2 * (y + 0.5) / n)))
    return 40075016.686 * math.cos(lat) / n / size


def gauss(a, sigma):
    """A separable Gaussian blur (edges held). Pure numpy."""
    if sigma < 0.3:
        return a.astype(np.float32)
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    a = np.asarray(a, np.float32)
    p = np.pad(a, ((r, r), (0, 0)), mode="edge")
    b = np.zeros_like(a)
    for i in range(2 * r + 1):
        b += k[i] * p[i:i + a.shape[0]]
    p = np.pad(b, ((0, 0), (r, r)), mode="edge")
    out = np.zeros_like(a)
    for i in range(2 * r + 1):
        out += k[i] * p[:, i:i + a.shape[1]]
    return out


def grow(mask, px):
    """mask dilated by px pixels (4-neighbour steps). Pure."""
    m = mask.copy()
    for _ in range(max(0, int(px))):
        p = np.pad(m, 1, mode="constant")
        m = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return m


def resample_block(a, size):
    """A square array to size x size: block means when larger by a whole factor, nearest pixels when smaller."""
    n = a.shape[0]
    if n == size:
        return a.astype(np.float32)
    if n > size and n % size == 0:
        f = n // size
        return a.astype(np.float32).reshape(size, f, size, f).mean(axis=(1, 3))
    idx = (np.arange(size) * n // size).astype(np.int64)
    return a[np.ix_(idx, idx)].astype(np.float32)


def min_filter(a, w):
    """The minimum over a w x w window (w odd; edges held). Separable, pure numpy."""
    r = max(0, int(w) // 2)
    if r == 0:
        return a.astype(np.float32)
    out = np.asarray(a, np.float32)
    for axis in (0, 1):
        p = np.pad(out, [(r, r) if i == axis else (0, 0) for i in (0, 1)], mode="edge")
        m = p[tuple(slice(0, out.shape[0]) if i == axis else slice(None) for i in (0, 1))].copy()
        for k in range(1, 2 * r + 1):
            sl = tuple(slice(k, k + out.shape[0]) if i == axis else slice(None) for i in (0, 1)) if axis == 0 else \
                tuple(slice(None) if i == 0 else slice(k, k + out.shape[1]) for i in (0, 1))
            np.minimum(m, p[sl], out=m)
        out = m
    return out


def seen(h, canopy, built, ok, px_m):
    """Where the mosaic's surface is the ground: open land, and in built-up areas only its low points. Pure."""
    s = ok & (canopy < OPEN_CANOPY_M)
    if built is not None and built.any():
        near = grow(built, BUILT_GROW_M / px_m)
        w = max(3, int(round(LOW_WINDOW_M / px_m)) | 1)
        low = (h - min_filter(np.where(ok, h, np.inf).astype(np.float32), w)) < LOW_M
        s &= ~near | low
    return s


def ground(h, canopy, built, ok, base, px_m):
    """The ground under surface h (above EGM2008): h itself where it is seen (seen()), elsewhere filled from the seen
    ground around (normalised convolution, the finest of FILL_SIGMAS_M with FILL_SUPPORT), past its reach base moved
    to the seen ground's level; never above h; the speckle smoothed. Pure."""
    s = seen(h, canopy, built, ok, px_m)
    if not s.any():
        return None
    sv = np.where(s, h, 0.0).astype(np.float32)
    sw = s.astype(np.float32)
    off = float(np.median((h - base)[s]))
    fill = (base + np.float32(off)).astype(np.float32)          # past every scale's reach: Mapterhorn's shape
    # coarse to fine, each scale blended over the coarser ones by its support (a switch between scales was a step: a
    # seam line through the forest, 2026-11-11)
    out = fill.copy()
    for sm in reversed(FILL_SIGMAS_M):
        sig = sm / max(px_m, 0.1)
        den = gauss(sw, sig)
        num = gauss(sv, sig)
        w = np.clip(den / (2.0 * FILL_SUPPORT), 0.0, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            est = np.where(den > 1e-6, num / np.maximum(den, 1e-6), out)
        out = (out * (1.0 - w) + est * w).astype(np.float32)
    # the seen ground as it is, its edge soft
    k = np.clip(gauss(sw, OPEN_EDGE_M / max(px_m, 0.1) / 2.0) * 2.0 - 1.0, 0.0, 1.0) * s
    g = h * k + out * (1.0 - k)
    g = np.minimum(np.where(ok, g, out), np.where(ok, h, np.inf))       # the ground is never above the surface
    return gauss(g, SMOOTH_M / max(px_m, 0.1)).astype(np.float32)


class Geoid:
    """EGM2008 geoid heights from gravity.py's model (its coefficients on disk under gravity_dir; nothing is downloaded
    here). n(lat, lon) is None while there is no model."""

    def __init__(self, gravity_dir):
        self.dir = gravity_dir
        self._model = None
        self._tried = False
        self._corners = {}
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None and not self._tried:
                self._tried = True
                try:
                    import gravity
                    if gravity.egm2008_cached(self.dir):
                        self._model = gravity.load_egm2008(cache_dir=self.dir, log=log.debug)
                    else:
                        log.info("ArcticDEM: no EGM2008 coefficients in %s: geoid from Mapterhorn offsets", self.dir)
                except Exception as e:          # (a damaged file, no native module): the fallback
                    log.warning("ArcticDEM: EGM2008: %s: %s", type(e).__name__, e)
            return self._model

    def _corner(self, cx, cy):
        key = (cx, cy)
        v = self._corners.get(key)
        if v is None:
            n = 2 ** GEOID_Z
            lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * cy / n))))
            lon = cx / n * 360.0 - 180.0
            v = float(self._model.field(lat, lon, 0.0).geoid)
            self._corners[key] = v
        return v

    def tile(self, z, x, y, size=TILE):
        """N (m) on tile (z, x, y)'s pixels, or None without a model."""
        if self._get_model() is None:
            return None
        f = 2.0 ** (GEOID_Z - z)
        i = (np.arange(size) + 0.5) / size
        gx, gy = (x + i) * f, (y + i) * f           # the pixels in z10 tile units
        x0, y0 = int(math.floor(gx.min())), int(math.floor(gy.min()))
        x1, y1 = int(math.floor(gx.max())) + 1, int(math.floor(gy.max())) + 1
        with self._lock:
            c = np.array([[self._corner(cx % (2 ** GEOID_Z), cy) for cx in range(x0, x1 + 1)]
                          for cy in range(y0, y1 + 1)], np.float64)
        u, v = gx - x0, gy - y0
        iu, iv = np.clip(np.floor(u).astype(np.int64), 0, c.shape[1] - 2), np.clip(np.floor(v).astype(np.int64), 0,
                                                                                 c.shape[0] - 2)
        fu, fv = u - iu, v - iv
        top = c[iv][:, iu] * (1 - fu)[None, :] + c[iv][:, iu + 1] * fu[None, :]
        bot = c[iv + 1][:, iu] * (1 - fu)[None, :] + c[iv + 1][:, iu + 1] * fu[None, :]
        return (top * (1 - fv)[:, None] + bot * fv[:, None]).astype(np.float32)


class ArcticDem:
    """dem_cache_dir: dem.py's cache folder (<cache>/dem): the mosaic's blocks go under it, the canopy and land cover
    under <cache>/mapdata/raw (shared with the map data), EGM2008 is read from <cache>/gravity."""

    def __init__(self, dem_cache_dir, fetcher, rasters=None, geoid=None):
        self.cache_dir, self.fetcher = dem_cache_dir, fetcher
        root = os.path.dirname(os.path.abspath(dem_cache_dir))
        self._rasters = rasters
        self._raw_dir = os.path.join(root, "mapdata", "raw")
        self.geoid = geoid if geoid is not None else Geoid(os.path.join(root, "gravity"))
        self._cogs = OrderedDict()          # (name, level) -> Cog, or None
        self._lock = threading.Lock()
        self._tiles = cogmod.TileCache(TILE_CACHE_BYTES)
        self._cell_rasters = OrderedDict()
        self.stats = {"tiles": 0, "empty": 0, "geoid:egm2008": 0, "geoid:offset": 0, "dropped_px": 0}

    # --- the mosaic -------------------------------------------------------------------------------
    def rasters(self):
        if self._rasters is None:
            import maprasters
            with self._lock:
                if self._rasters is None:
                    self._rasters = maprasters.Rasters(self._raw_dir, self.fetcher)
        return self._rasters

    def _cog(self, name, level):
        key = (name, level)
        with self._lock:
            if key in self._cogs:
                self._cogs.move_to_end(key)
                return self._cogs[key]
        remote = rangeread.RemoteFile(file_url(name), None, os.path.join(self.cache_dir, SOURCE, RELEASE, name),
                                      self.fetcher)
        try:
            c = cogmod.Cog(remote, "demand", level=level, tiles=self._tiles)
            if c.epsg not in (None, 3413) or c.sx is None:
                raise cogmod.CogError(f"ArcticDEM {name}: unexpected georeferencing (EPSG {c.epsg})")
        except rangeread.Missing:
            c = None
        except cogmod.CogError as e:
            if level > 0 and "no overview" in str(e):
                c = None
            else:
                raise
        with self._lock:
            self._cogs[key] = c
            while len(self._cogs) > 64:
                self._cogs.popitem(last=False)
        return c

    def surface(self, z, x, y, size=TILE, min_px=LATTICE_M, margin=0):
        """The mosaic's heights (above the ellipsoid) and their validity on tile (z, x, y), or None where it has
        nothing. Network failures raise netfetch.FetchError."""
        if not covers(z, x, y):
            return None
        lat, lon = tile_lat_lon(z, x, y, size, margin)
        px, py = polar_xy(lat, lon)
        want = max(0, min(MAX_LEVEL, int(math.floor(math.log2(max(pixel_m(z, y, size), min_px) / LATTICE_M)))))
        out = np.zeros(lat.shape, np.float32)
        ok = np.zeros(lat.shape, bool)
        cx = np.floor(px / SUB_M).astype(np.int64)
        cy = np.floor(py / SUB_M).astype(np.int64)
        known = index()
        for code in np.unique((cx + 4096) * 8192 + (cy + 4096)):
            qx, qy = int(code) // 8192 - 4096, int(code) % 8192 - 4096
            m = (cx == qx) & (cy == qy)
            name = file_of((qx + 0.5) * SUB_M, (qy + 0.5) * SUB_M)
            if known is not None and name not in known:
                continue
            c = None
            level = want
            while level >= 0:
                c = self._cog(name, level)
                if c is not None or level == 0:
                    break
                level -= 1
            if c is None:
                continue
            col, row = c.pixel_of(px[m], py[m])
            col, row = col - 0.5, row - 0.5
            c0, r0 = int(math.floor(col.min())) - 1, int(math.floor(row.min())) - 1
            c1, r1 = int(math.floor(col.max())) + 3, int(math.floor(row.max())) + 3
            win = c.read_window(c0, r0, c1 - c0, r1 - r0).astype(np.float32)
            bad = ~np.isfinite(win) | (win < NODATA_LIMIT)
            if c.nodata is not None:
                bad |= win == np.float32(c.nodata)
            if bad.all():
                continue
            u, v = col - c0, row - r0
            u0, v0 = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
            fu, fv = (u - u0).astype(np.float32), (v - v0).astype(np.float32)
            a00, a01 = win[v0, u0], win[v0, u0 + 1]
            a10, a11 = win[v0 + 1, u0], win[v0 + 1, u0 + 1]
            good = ~(bad[v0, u0] | bad[v0, u0 + 1] | bad[v0 + 1, u0] | bad[v0 + 1, u0 + 1])
            h = (a00 * (1 - fu) + a01 * fu) * (1 - fv) + (a10 * (1 - fu) + a11 * fu) * fv
            out[m] = np.where(good, h, 0.0)
            ok[m] = good
        if not ok.any():
            return None
        return out, ok

    # --- the ground -------------------------------------------------------------------------------
    def _cells(self, z, x, y):
        """The z14 cell over tile (z >= 14) and the tile's place in it: (cell, sub-tiles per side, col, row)."""
        import mapcells
        s = z - CANOPY_Z
        return mapcells.CellKey(CANOPY_Z, x >> s, y >> s), 1 << s, x - ((x >> s) << s), y - ((y >> s) << s)

    def _cell_raster(self, kind, cell):
        """The cell's canopy / land-cover Raster (maprasters), kept for the neighbouring tiles that read it too."""
        key = (kind, cell)
        with self._lock:
            if key in self._cell_rasters:
                self._cell_rasters.move_to_end(key)
                return self._cell_rasters[key]
        r = getattr(self.rasters(), kind)(cell)
        with self._lock:
            self._cell_rasters[key] = r
            while len(self._cell_rasters) > CELL_RASTERS:
                self._cell_rasters.popitem(last=False)
        return r

    def _around(self, fn, z, x, y, margin):
        """fn's tile (z, x, y) with margin pixels of its neighbours' all round (theirs, from fn), or None when none of
        the nine has any. fn(z, x, y) -> [TILE, TILE] array or None (zeros)."""
        if margin == 0:
            return fn(z, x, y)
        n = 2 ** z
        rows, any_ = [], False
        for dy in (-1, 0, 1):
            row = []
            for dx in (-1, 0, 1):
                yy = y + dy
                a = fn(z, (x + dx) % n, yy) if 0 <= yy < n else None
                any_ |= a is not None
                row.append(np.zeros((TILE, TILE), np.float32) if a is None else a.astype(np.float32))
            rows.append(np.concatenate(row, axis=1))
        if not any_:
            return None
        big = np.concatenate(rows, axis=0)
        return big[TILE - margin:2 * TILE + margin, TILE - margin:2 * TILE + margin]

    def canopy(self, z, x, y, size=TILE):
        """Canopy height (m, float32) on the tile's pixels (0 where none or no data), or None where no CHM file
        exists. z >= CANOPY_Z."""
        cell, k, i, j = self._cells(z, x, y)
        r = self._cell_raster("canopy", cell)
        if r is None:
            return None
        n = r.data.shape[0] // k
        block = r.data[j * n:(j + 1) * n, i * n:(i + 1) * n]
        block = np.where(block == r.nodata, 0, block)
        return resample_block(block, size)

    def built(self, z, x, y, size=TILE):
        """ESA WorldCover's built-up class on the tile's pixels (bool), or None where it has no data."""
        cell = self._cells(z, x, y)[0]
        r = self._cell_raster("worldcover", cell)
        lat, lon = tile_lat_lon(z, x, y, size)
        cls = r.sample(lon, lat)
        if not cls.any():
            return None
        return cls == BUILT_CLASS

    def _built_float(self, z, x, y):
        b = self.built(z, x, y)
        return None if b is None else b.astype(np.float32)

    def tile(self, z, x, y, base):
        """Ground heights above EGM2008 on tile (z >= CANOPY_Z) and their validity, or None where the mosaic has
        nothing. base: the tile's Mapterhorn heights (the datum fallback, the blunder check, the sea). Network
        failures raise netfetch.FetchError: the caller keeps its own heights (never a surface model with its trees)."""
        if z < CANOPY_Z:
            raise ValueError("ArcticDEM ground is made for tiles of z14 and deeper")
        # everything on the tile with MARGIN pixels of its neighbours round it (the fill and the smoothing see across
        # its edges, so neighbouring tiles agree there: each tile alone left a line along every tile edge), cut after
        m = MARGIN
        got = self.surface(z, x, y, margin=m)
        if got is None:
            return None
        h, ok = got
        base = np.pad(base, m, mode="edge")
        ok &= base != 0.0                           # the sea stays the sea
        if not ok[m:-m, m:-m].any():
            return None
        px_m = pixel_m(z, y)
        canopy = self._around(self.canopy, z, x, y, m)
        if canopy is None:
            canopy = np.zeros(h.shape, np.float32)
        else:
            canopy = gauss(canopy, CANOPY_SMOOTH_M / px_m)
        built = self._around(self._built_float, z, x, y, m)
        built = None if built is None else built > 0.5
        # the datum
        n = self.geoid.tile(z, x, y)
        if n is not None:
            n = np.pad(n, m, mode="edge")
            h = h - n
            self._count("geoid:egm2008")
        else:
            open_ = ok & (canopy < 1.0)
            if built is not None:
                open_ &= ~built
            d = (h - base)[open_] if open_.sum() >= MIN_OPEN_PX else (h - base)[ok]
            h = h - np.float32(np.median(d))
            self._count("geoid:offset")
        # the blunders (before anything is taken from the surface)
        far = ok & (np.abs(h - base) > MAX_DIFF_M)
        if far.any():
            ok &= ~far
            self._count("dropped_px", int(far.sum()))
        g = ground(h, canopy, built, ok, base, px_m) if ok.any() else None
        self._count("tiles")
        if g is not None:
            g, ok = g[m:-m, m:-m], ok[m:-m, m:-m]
        if g is None or not ok.any():
            self._count("empty")
            return None
        return np.where(ok, g, 0.0).astype(np.float32), ok

    def sparse_ground(self, z, x, y, size=BIAS_PX):
        """Ground points for lining a distant tile up with the near terrain (dem.py's bias, as for HRDEM): the
        mosaic's coarse overview (BIAS_M pixels) on a size x size grid, where ESA WorldCover says open land (not trees,
        buildings or water), brought to EGM2008. (lat, lon, h) arrays, or None where there are none. Network failures
        raise netfetch.FetchError."""
        got = self.surface(z, x, y, size=size, min_px=BIAS_M)
        if got is None:
            return None
        h, ok = got
        n = self.geoid.tile(z, x, y, size)
        if n is None:
            return None             # (no EGM2008: no absolute level to line up with; the near tiles use offsets)
        import mapcells
        r = self.rasters().worldcover(mapcells.CellKey(z, x, y))
        lat, lon = tile_lat_lon(z, x, y, size)
        cls = r.sample(lon, lat)
        ok &= np.isin(cls, OPEN_CLASSES)
        if ok.sum() < 16:
            return None
        return np.asarray(lat[ok]), np.asarray(lon[ok]), (h - n)[ok].astype(np.float64)

    def _count(self, what, k=1):
        with self._lock:
            self.stats[what] = self.stats.get(what, 0) + k
