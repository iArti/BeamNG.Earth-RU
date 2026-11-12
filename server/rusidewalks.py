"""Streets whose sidewalks are mapped as their own lines get no guessed ones (BeamNG.Earth-RU).

Russia's mappers (and many elsewhere) draw a street's sidewalks as footway=sidewalk lines beside it and leave the street
itself without a sidewalk tag: in Saint Petersburg and its surroundings most map cells have dozens to over a hundred
such lines and almost no tagged street. The road rules then guessed a sidewalk on both sides of every
town street (roads.has_sidewalks' class and land-cover rule) and the mapped line was drawn beside it as well: two
sidewalks, the mapped one where it really is, the guessed one at the curb.

apply(feats) reads a roads layer's own sidewalk lines (europewalks.mapped_walk's test: class footway with subclass,
footway tag or rule "sidewalk") and, for each street (SIDEWALK_CLASSES and trunks) without a sidewalk tag, measures
along it (every STEP_M) which share of its length has such a line SIDE_MIN_M..SIDE_MAX_M off its centreline, running
within MAX_ANGLE_DEG of it. A street with COVER_MIN of its length seen on either side is tagged "separate" (the map's
own word for it: no guessed sidewalk, roads.SIDEWALK_TAG), so its mapped lines are its sidewalks. Done once per layer
(kept on it), before the road specs read the tags.
"""
import math

import numpy as np

VERSION = "ru-sidewalks-v1"         # the road specs' cache key carries it (roads._road_specs_kept)
STREETS = {"residential", "tertiary", "secondary", "primary", "unclassified", "living_street", "trunk"}
STEP_M = 10.0
SIDE_MIN_M = 1.5                    # a sidewalk line's distance from the street's centreline: at least ...
SIDE_MAX_M = 28.0                   # ... and at most (a wide avenue's sidewalk behind its lawns)
MAX_ANGLE_DEG = 30.0
ALONG_MAX_M = 3.0                   # the nearest point of a line this far at most ahead of or behind the sample
COVER_MIN = 0.3                     # the share of a street's length with a sidewalk line beside it, either side


def _is_sidewalk(feats, cls, sub, footway, i):
    import roads
    if cls[i] != "footway" or int(feats.kind[i]) != 2:
        return False
    return ((sub is not None and str(sub[i]) == "sidewalk") or (footway is not None and str(footway[i]) == "sidewalk")
            or roads._rule_value(feats.rules.get("subclass"), i) == "sidewalk")


def _lines(feats, i):
    out = []
    for p in range(feats.part_off[i], feats.part_off[i + 1]):
        for r in range(feats.ring_off[p], feats.ring_off[p + 1]):
            a, b = feats.coord_off[r], feats.coord_off[r + 1]
            if b - a >= 2:
                out.append(np.asarray(feats.lonlat[a:b], np.float64))
    return out


def coverage(street, segs, step=STEP_M):
    """(left share, right share) of a street polyline (k, 2 metres) with a sidewalk segment (m, 4: x0 y0 x1 y1) beside
    it at each STEP_M sample: SIDE_MIN_M..SIDE_MAX_M off, within MAX_ANGLE_DEG. Pure."""
    if len(street) < 2 or not len(segs):
        return 0.0, 0.0
    d = np.diff(street, axis=0)
    seg_len = np.hypot(d[:, 0], d[:, 1])
    total = float(seg_len.sum())
    if total < 1.0:
        return 0.0, 0.0
    at = np.arange(step / 2, total, step)
    if not len(at):                     # (a piece shorter than half a step: its middle)
        at = np.array([total / 2])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    k = np.clip(np.searchsorted(cum, at, side="right") - 1, 0, len(d) - 1)
    t = (at - cum[k]) / np.maximum(seg_len[k], 1e-9)
    pts = street[k] + d[k] * t[:, None]
    dirs = d[k] / np.maximum(seg_len[k], 1e-9)[:, None]
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    ab_len = np.hypot(ab[:, 0], ab[:, 1])
    ab_dir = ab / np.maximum(ab_len, 1e-9)[:, None]
    cos_min = math.cos(math.radians(MAX_ANGLE_DEG))
    left = right = 0
    for p, u in zip(pts, dirs):
        ap = p - a
        s = np.clip((ap * ab).sum(1) / np.maximum(ab_len ** 2, 1e-9), 0.0, 1.0)
        near = a + ab * s[:, None]
        v = near - p
        dist = np.hypot(v[:, 0], v[:, 1])
        par = np.abs(ab_dir @ u) >= cos_min
        abreast = np.abs(v @ u) <= ALONG_MAX_M          # (beside the sample, not past a line's end ahead or behind)
        ok = par & abreast & (dist >= SIDE_MIN_M) & (dist <= SIDE_MAX_M)
        if not ok.any():
            continue
        cross = u[0] * v[ok, 1] - u[1] * v[ok, 0]
        left += bool((cross > 0).any())
        right += bool((cross < 0).any())
    n = len(pts)
    return left / n, right / n


def apply(feats):
    """Tag the layer's streets with sidewalks mapped beside them "separate" (osm_sidewalk), once per layer. Never fails:
    whatever goes wrong here leaves the tags as they were (the road specs must be made)."""
    st = feats.__dict__
    if st.get("_ru_sidewalks") is not None or not len(feats):
        return
    st["_ru_sidewalks"] = 0
    try:
        _apply(feats)
    except Exception as e:      # (a layer this does not understand: the class rule, as upstream)
        import logging
        logging.getLogger("earthd.rusidewalks").warning("sidewalks from the map's lines: %s: %s", type(e).__name__, e)


def _apply(feats):
    st = feats.__dict__
    a = feats.attrs
    cls_col = a.get("class")
    if cls_col is None:
        return
    cls = cls_col.astype(str)
    sub, footway = a.get("subclass"), a.get("osm_footway")
    walk_ids = [i for i in np.flatnonzero(cls == "footway").tolist() if _is_sidewalk(feats, cls, sub, footway, i)]
    if not walk_ids:
        return
    lat0 = float(np.mean(feats.lonlat[:, 1]))
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110574.0

    def xy(ll):
        return np.column_stack([ll[:, 0] * kx, ll[:, 1] * ky])
    segs = []
    for i in walk_ids:
        for line in _lines(feats, i):
            q = xy(line)
            segs.append(np.column_stack([q[:-1], q[1:]]))
    segs = np.concatenate(segs) if segs else np.zeros((0, 4))
    lo = np.minimum(segs[:, :2], segs[:, 2:])
    hi = np.maximum(segs[:, :2], segs[:, 2:])
    tags = a.get("osm_sidewalk")
    tags = np.array([""] * len(feats), dtype=object) if tags is None else np.array(tags, dtype=object)
    changed = 0
    for i in np.flatnonzero(np.isin(cls, list(STREETS))).tolist():
        if str(tags[i] or "").strip() not in ("", "None", "nan"):
            continue                        # (the street's own tag says)
        best = 0.0
        for line in _lines(feats, i):
            q = xy(line)
            bx0, by0 = q.min(0) - SIDE_MAX_M
            bx1, by1 = q.max(0) + SIDE_MAX_M
            m = (hi[:, 0] >= bx0) & (lo[:, 0] <= bx1) & (hi[:, 1] >= by0) & (lo[:, 1] <= by1)
            if m.any():
                best = max(best, *coverage(q, segs[m]))
        if best >= COVER_MIN:
            tags[i] = "separate"
            changed += 1
    if changed:
        a["osm_sidewalk"] = tags
    st["_ru_sidewalks"] = changed
