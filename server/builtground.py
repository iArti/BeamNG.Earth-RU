"""The ground photo round buildings painted from their footprints (BeamNG.Earth-RU).

Where the only imagery is Sentinel-2 (10 m: Russia, much of the world), the ground photo under a town is a blur of
roofs, courtyards and streets: the buildings, drawn from their exact OpenStreetMap outlines, stand on light roof-coloured
smears that do not match them, and a courtyard reads as roof. The footprints are what the photo should agree with, so
the ground round them is painted from them, in the photo's own pavement tone (surface.photo_pads' painter, as the
parking lots and the roads):

- the apron: APRON_M round every wall (a pavement, the blind area) paved;
- the courtyards: ground enclosed by buildings (closed across passages up to PASSAGE_M wide, as the arches of a city
  block) paved when the yard is small (YARD_MAX_M2: a well courtyard; a block's garden keeps its photo);
- the photo's greenery (lawns, trees in a yard) never paved;
- a contact shadow: the ground AO_M out from the walls darkened, softly, so the buildings sit on it.

surface.paint_square calls photo_built under the roads (they paint over it). BNE_BUILT_GROUND=0 turns it off.

    rgb = photo_built(rgb, frame, x0, y0, size, buildings_feats_list)
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ON = os.environ.get("BNE_BUILT_GROUND", "1") != "0"
APRON_M = 2.5
PASSAGE_M = 4.0                 # gaps between buildings up to this wide closed when looking for enclosed yards
YARD_MAX_M2 = 1500.0            # enclosed yards up to this area are paved (a well courtyard; a block garden is not)
GREEN = 12                      # 2 G - R - B at least: the photo's greenery (lawns, trees), left as it is
YARD_GRID_M = 2.0               # the yards are found on a grid this coarse (m per cell)
AO_M = 3.0
AO_DARK = 0.72                  # the ground right at a wall: this much of its brightness
SKIP_CLASSES = {"roof", "carport", "canopy"}    # open structures: the ground under them shows


def building_pads(feats_list):
    """The outer rings of the buildings' footprints as (lat (k,), lon (k,)) pads (surface._pads_px's form): standing
    buildings only (not underground ones, nor open roofs and carports)."""
    pads = []
    for feats in feats_list:
        if feats is None or len(feats) == 0:
            continue
        kind = np.asarray(feats.kind)
        under = feats.attrs.get("is_underground")
        cls = feats.attrs.get("class")
        for i in np.flatnonzero(kind == 3):
            if under is not None and bool(under[i]):
                continue
            if cls is not None and str(cls[i] or "").strip().lower() in SKIP_CLASSES:
                continue
            for part in feats.parts(int(i)):
                if not part or len(part[0]) < 3:
                    continue
                ring = np.asarray(part[0], np.float64)
                pads.append((ring[:, 1], ring[:, 0]))
    return pads


def dilate(mask, r):
    """mask grown by r pixels (a disc, approximated by 4- and 8-neighbour steps alternately). Pure."""
    m = np.asarray(mask, bool)
    for k in range(max(0, int(round(r)))):
        p = np.pad(m, 1, mode="constant")
        g = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if k % 2:
            g |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        m = g
    return m


def enclosed(mask):
    """The pixels of ~mask that cannot be reached from the image's border without crossing mask. Pure."""
    h, w = mask.shape
    # (.copy(): an image made from an array shares its buffer, and Pillow 13's flood fill left it unchanged)
    im = Image.fromarray(np.pad(np.where(mask, 255, 0).astype(np.uint8), 1, mode="constant"), "L").copy()
    ImageDraw.floodfill(im, (0, 0), 128)
    a = np.asarray(im)[1:-1, 1:-1]
    return a == 0


def soften(a, r):
    """A mask (0..1) blurred over radius r pixels (PIL's box blur twice: a near-Gaussian; 8-bit, plenty for a soft
    edge). Pure."""
    a = np.asarray(a, np.float32)
    r = int(round(r))
    if r < 1:
        return a
    im = Image.fromarray(np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8), "L")
    im = im.filter(ImageFilter.BoxBlur(r)).filter(ImageFilter.BoxBlur(r))
    return np.asarray(im, np.float32) / 255.0


def small_parts(mask, max_px):
    """The connected parts (4-neighbour) of mask with at most max_px pixels. Pure (a breadth-first walk over the
    mask's own pixels: the yards' grid is coarse, YARD_GRID_M)."""
    h, w = mask.shape
    flat = mask.ravel()
    seen = np.zeros(flat.shape, bool)
    out = np.zeros(flat.shape, bool)
    for start in np.flatnonzero(flat).tolist():
        if seen[start]:
            continue
        seen[start] = True
        part, i = [start], 0
        while i < len(part):
            p = part[i]
            i += 1
            r, c = divmod(p, w)
            for q, okq in ((p - w, r > 0), (p + w, r < h - 1), (p - 1, c > 0), (p + 1, c < w - 1)):
                if okq and flat[q] and not seen[q]:
                    seen[q] = True
                    part.append(q)
        if len(part) <= max_px:
            out[part] = True
    return out.reshape(h, w)


def masks(foot, px_m, green=None):
    """(paved coverage float 0..1, shadow factor float AO_DARK..1) for a footprint mask (bool, px_m metres a pixel);
    green: the photo's greenery (bool, the same shape), never paved. Pure."""
    h, w = foot.shape
    apron = dilate(foot, APRON_M / px_m)
    # the yards, on a coarser grid (a block photo is 2048-4096 px)
    f = max(1, int(round(YARD_GRID_M / px_m)))
    gh, gw = -(-h // f), -(-w // f)
    small = np.zeros((gh * f, gw * f), bool)
    small[:h, :w] = foot
    small = small.reshape(gh, f, gw, f).any(axis=(1, 3))
    gpx = px_m * f
    closed = dilate(small, PASSAGE_M / 2 / gpx)
    yard = small_parts(enclosed(closed), int(YARD_MAX_M2 / (gpx * gpx)))     # courtyards; a block's garden stays
    yard = np.repeat(np.repeat(yard, f, axis=0), f, axis=1)[:h, :w]
    paved = apron | yard
    if green is not None:
        paved &= ~(green & ~dilate(foot, 1.0 / px_m))       # the photo's lawns and trees stay (not at the very wall)
    paved = soften(paved.astype(np.float32), max(1.0, 0.5 / px_m))
    near = dilate(foot, AO_M / px_m) & ~foot
    shade = soften(near.astype(np.float32), AO_M / px_m / 2)
    shadow = 1.0 - (1.0 - AO_DARK) * shade * (~foot)
    return np.clip(paved, 0.0, 1.0), shadow.astype(np.float32)


def photo_built(rgb, frame, x0, y0, size, feats_list):
    """The photo rgb (H x W x 3, row 0 north, over [x0, x0 + size]² of frame) with the ground round the buildings of
    feats_list painted from their footprints (see the module's text). Pure (a new array, or rgb itself)."""
    if not ON or rgb is None or not feats_list:
        return rgb
    import surface
    import fast
    h, w = rgb.shape[:2]
    pads = building_pads(feats_list)
    cov = surface._pads_px(frame, x0, y0, size, w, h, pads) if pads else None
    if cov is None:
        return rgb
    px_m = size / float(w)
    g = np.asarray(rgb, np.int16)
    green = soften(((2 * g[..., 1] - g[..., 0] - g[..., 2]) >= GREEN).astype(np.float32), 1.0 / px_m) > 0.5
    foot = cov >= 0.5
    paved, shadow = masks(foot, px_m, green)
    # the roofs out of the tone's samples (a pink or red roof smeared round its walls tinted the pavement it was
    # sampled for): the pavement's own colour under the footprints first
    base = np.array(np.asarray(rgb, np.uint8))
    base[foot] = np.array(surface.PHOTO_ROAD_RGBA["paved"][:3], np.uint8)
    fast.require(231, "the ground round buildings in the photo's own pavement tone")
    radius = max(1, int(round(surface.PHOTO_TONE_RADIUS_M * w / size)))
    out = fast._ef.paint_cover_toned(np.ascontiguousarray(base, np.uint8), np.ascontiguousarray(paved, np.float32),
                                     np.array(surface.PHOTO_ROAD_RGBA["paved"], np.float32),
                                     np.array(surface.PHOTO_ROAD_TONE["paved"], np.float32), radius,
                                     surface.PHOTO_TONE_SAMPLE_LUMA, surface.PHOTO_TONE_SAMPLE_SAT,
                                     surface.PHOTO_TONE_GREEN)
    return np.clip(np.asarray(out, np.float32) * shadow[..., None], 0, 255).astype(np.uint8)
