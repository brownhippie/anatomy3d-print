"""What the hair in a photo does, read from the hair itself, for the builder
to sculpt it: which way its strands run at every hair pixel, how much they
swing from side to side and over what length (waves), how fine its texture
is (sleek, strands, curls, coils), and where its hairline runs over the
forehead.

- Strand direction: the structure tensor of the hair's brightness (strands
  are thin lines; across them the brightness changes most), after evening
  out the hair's local contrast (black hair is dark, its strands still
  show). Coherence: how clearly one direction wins -- high on straight or
  wavy hair, low on curls and coils.
- Waves: strands traced through that direction field, both ways from 300
  seeds; how far each one's heading swings either side of its own course
  (degrees), and at what period (its autocorrelation's first trough is half
  a wave).
- Texture: brightness detail at strand scale against detail at the scale
  of locks and shading -- near nothing on sleek hair, most on coils.
- Hairline: across the forehead, the first row of hair going up from the
  brows.
- Parting: on top of the head, the line the strands fall away from to
  either side.
- Ropes (braids, cornrows, locs, twists): the hair's detail lives at the
  width of its ropes as much as at strand scale (loose hair's fades past
  its strands; sleek hair's is all shading); the ropes' width from the
  brightness across them repeating.
"""
from typing import Optional

import numpy as np

KINDS = ["sleek", "straight", "wavy", "curly", "coily", "corded"]
_MIN_HAIR_PX = 400
_MIN_HAIR = 0.03  # less hair than this (x the face's size squared) shows: nothing to read
_DERIV = 0.004  # brightness derivatives: Gaussian sigma, x the face's size
_TENSOR = 0.025  # the tensor averaged over this, x the face's size
_TRACE = 0.8  # strands traced this far each way for waves, x the face's size
_WAVE = (0.08, 1.0)  # a wave's length, x the face's size
_COURSE = 0.3  # a strand's own course: its heading smoothed over this, x the face's size
# measured on our test photos: sleek black updo 0.16, straight 0.38, wavy
# 0.59, short tight coils 1.06 (fineness); straight 0.71 and wavy 0.64
# coherence with 15-18 and 30 degrees of swing, curls and coils 0.24-0.28
_SLEEK = 0.25
_COMBED = 0.5
_WAVY_SWING = 24.0
_COILY = 0.85
# ropes: detail at a hundredth of the face's size over detail at strand
# scale, and shading's over strand's -- measured on 20 photos (MIT,
# huggingface.co/datasets/nrhone/male-black-hairstyles): braids, cornrows,
# locs and twists 0.93-1.40, fades, afros, 360 waves and buzz cuts 0.50-0.87
# (a sleek flat top 0.97, but its shading 4.8 where ropes' is 0.8-1.4)
_ROPE = 0.9
_ROPE_SHADE = 1.6
_ROPE_FINE = 0.5  # and textured: ropes' fineness 0.69-1.1 (a sleek updo's rope ratio is 1.01, its fineness 0.21)
_ROPE_W = (0.03, 0.2)  # a rope's width, x the face's size
_PART_MIN = 1.0  # how clearly the strands fall away either side of a parting (-1..2)


def _structure(gray: np.ndarray, hair: np.ndarray, s1: float, s2: float):
    from scipy.ndimage import gaussian_filter

    w = hair.astype(np.float32)
    # the hair's own contrast, evened out: (g - local mean) / local spread
    sw = gaussian_filter(w, 3 * s2) + 1e-6
    mean = gaussian_filter(gray * w, 3 * s2) / sw
    var = gaussian_filter((gray - mean) ** 2 * w, 3 * s2) / sw
    g = np.where(hair, (gray - mean) / np.sqrt(var + 1e-4), 0.0).astype(np.float32)
    gx = gaussian_filter(g, s1, order=(0, 1))
    gy = gaussian_filter(g, s1, order=(1, 0))
    jxx = gaussian_filter(gx * gx * w, s2) / (gaussian_filter(w, s2) + 1e-6)
    jyy = gaussian_filter(gy * gy * w, s2) / (gaussian_filter(w, s2) + 1e-6)
    jxy = gaussian_filter(gx * gy * w, s2) / (gaussian_filter(w, s2) + 1e-6)
    # the strands run across the brightness's strongest change
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    tr = jxx + jyy
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (tr + 1e-6)
    return np.mod(theta, np.pi).astype(np.float32), np.clip(coh, 0, 1).astype(np.float32), g


def _trace(field_c2, field_s2, hair, y, x, ty, tx, steps):
    """Positions stepped one px at a time along the strands from (y, x),
    starting along (ty, tx); NaN once a trace leaves the hair."""
    from scipy.ndimage import map_coordinates

    h, w = hair.shape
    hairf = hair.astype(np.float32)
    ys, xs, alive = [], [], np.ones(len(y), bool)
    for _ in range(steps):
        cc = map_coordinates(field_c2, [y, x], order=1, mode="nearest")
        ss = map_coordinates(field_s2, [y, x], order=1, mode="nearest")
        ang = 0.5 * np.arctan2(ss, cc)
        ny, nx = np.sin(ang), np.cos(ang)
        flip = ny * ty + nx * tx < 0
        ny, nx = np.where(flip, -ny, ny), np.where(flip, -nx, nx)
        y, x, ty, tx = y + ny, x + nx, ny, nx
        alive &= (y >= 0) & (y < h - 1) & (x >= 0) & (x < w - 1)
        alive &= map_coordinates(hairf, [np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)], order=0) > 0.5
        ys.append(np.where(alive, y, np.nan))
        xs.append(np.where(alive, x, np.nan))
    return np.array(ys).T, np.array(xs).T


def _waves(theta: np.ndarray, coh: np.ndarray, hair: np.ndarray, size: float, seed: int = 0):
    """(heading swing in degrees, wave length in px or None)."""
    from scipy.ndimage import gaussian_filter1d

    ys, xs = np.nonzero(hair & (coh > 0.3))
    if len(ys) < 50:
        return 0.0, None
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ys), size=min(300, len(ys)), replace=False)
    y0, x0 = ys[pick].astype(np.float64), xs[pick].astype(np.float64)
    ty, tx = np.sin(theta[ys[pick], xs[pick]]), np.cos(theta[ys[pick], xs[pick]])
    steps = int(_TRACE * size)
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    fy, fx = _trace(c2, s2, hair, y0, x0, ty, tx, steps)
    by, bx = _trace(c2, s2, hair, y0, x0, -ty, -tx, steps)
    swings, acs = [], []
    lag_n = int(_WAVE[1] * size)
    for i in range(len(pick)):
        py = np.concatenate([by[i][::-1], [y0[i]], fy[i]])
        px = np.concatenate([bx[i][::-1], [x0[i]], fx[i]])
        ok = np.isfinite(py)
        a = b = steps
        while a > 0 and ok[a - 1]:
            a -= 1
        while b < len(ok) - 1 and ok[b + 1]:
            b += 1
        if b - a + 1 < 0.4 * size:
            continue
        yy, xx = gaussian_filter1d(py[a:b + 1], 2.0), gaussian_filter1d(px[a:b + 1], 2.0)
        head = np.unwrap(np.arctan2(np.diff(yy), np.diff(xx)))
        det = head - gaussian_filter1d(head, _COURSE * size)  # off its own course
        cut = int(0.1 * size)
        det = det[cut:-cut] if len(det) > 3 * cut else det
        if len(det) < 20:
            continue
        swings.append(float(np.degrees(det.std())))
        dn = (det - det.mean()) / (det.std() + 1e-6)
        n = len(dn)
        acs.append(np.array([np.mean(dn[:n - L] * dn[L:]) if n - L > 10 else np.nan for L in range(lag_n)]))
    if len(swings) < 10:
        return 0.0, None
    ac = np.nanmean(np.array(acs), axis=0)
    lo = int(_WAVE[0] * size / 2)
    trough = lo + int(np.nanargmin(ac[lo:])) if np.isfinite(ac[lo:]).any() else None
    wave = 2.0 * trough if trough is not None and ac[trough] < -0.15 and trough < len(ac) - 2 else None
    return float(np.median(swings)), (None if wave is None else float(wave))


def _fineness(gray: np.ndarray, hair: np.ndarray, size: float) -> float:
    """Brightness detail at strand scale over detail at the scale of locks
    and shading (difference-of-Gaussians bands, in the hair's middle)."""
    from scipy.ndimage import gaussian_filter

    core = hair & (gaussian_filter(hair.astype(np.float32), 0.02 * size) > 0.95)
    if core.sum() < 50:
        core = hair

    def band(sc):
        sg = max(0.5, sc * size)
        return float(np.std((gaussian_filter(gray, sg) - gaussian_filter(gray, 1.6 * sg))[core]))

    fine = np.mean([band(sc) for sc in (0.003, 0.005, 0.008)])
    coarse = np.mean([band(sc) for sc in (0.02, 0.032, 0.05)])
    return float(fine / (coarse + 1e-6))


def _ropes(gray: np.ndarray, hair: np.ndarray, size: float, seed: int = 0):
    """(detail at rope scale / at strand scale, shading / strand scale, the
    ropes' width in px or None): difference-of-Gaussians bands of the
    hair's evened-out brightness in its middle; the width from brightness
    sampled across the ropes' course repeating (its autocorrelation's
    clearest peak)."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    theta, coh, g = _structure(gray, hair, max(1.0, 0.012 * size), max(2.0, 0.05 * size))
    core = hair & (gaussian_filter(hair.astype(np.float32), 0.02 * size) > 0.9)
    if core.sum() < 50:
        return 0.0, 0.0, None

    def band(sc):
        sg = max(0.5, sc * size)
        return float(np.std((gaussian_filter(g, sg) - gaussian_filter(g, 1.6 * sg))[core]))

    fine = band(0.004) + 1e-6
    rope, shade = band(0.01) / fine, band(0.035) / fine
    width = None
    ys, xs = np.nonzero(core & (coh > 0.25))
    if len(ys) >= 50:
        rng = np.random.default_rng(seed)
        k = rng.choice(len(ys), min(400, len(ys)), replace=False)
        y, x = ys[k].astype(np.float64), xs[k].astype(np.float64)
        th = theta[ys[k], xs[k]]
        L = int(0.35 * size)
        t = np.arange(-L, L + 1)
        v = map_coordinates(gaussian_filter(g, max(0.5, 0.004 * size)),
                            [y[:, None] + np.cos(th)[:, None] * t, x[:, None] - np.sin(th)[:, None] * t], order=1, mode="nearest")
        v = (v - v.mean(1, keepdims=True)) / (v.std(1, keepdims=True) + 1e-6)
        ac = np.array([np.mean(v[:, :v.shape[1] - j] * v[:, j:]) for j in range(L)])
        best = 0.02
        for j in range(max(int(_ROPE_W[0] * size), 1), min(int(_ROPE_W[1] * size), L - 1)):
            if ac[j] >= ac[j - 1] and ac[j] >= ac[j + 1] and ac[j] - ac[:j].min() > best:
                best, width = ac[j] - ac[:j].min(), float(j)
    return float(rope), float(shade), width


def _hairline(labels: np.ndarray, f: np.ndarray, hair_label: int, hat_label: int):
    """Across the forehead, the first row of hair going up from the brows
    (photo px), or None where the forehead runs into a hat or past the head."""
    brow_y = float((f[105, 1] + f[334, 1]) / 2.0)
    ruler = float(f[2, 1] - brow_y)
    x0, x1 = float(f[54, 0]), float(f[284, 0])  # the forehead's sides
    cols = np.arange(int(round(min(x0, x1))), int(round(max(x0, x1))) + 1)
    top = int(max(brow_y - 3.0 * ruler, 0))
    rows_out = []
    for c in cols:
        if not 0 <= c < labels.shape[1]:
            rows_out.append(np.nan)
            continue
        col = labels[top:int(brow_y), c][::-1]  # from the brows up
        hit = np.flatnonzero(np.isin(col, (hair_label, hat_label)))
        if not len(hit) or labels[int(brow_y) - hit[0], c] == hat_label:
            rows_out.append(np.nan)
        else:
            rows_out.append(float(int(brow_y) - hit[0]))
    rows_out = np.array(rows_out)
    good = np.isfinite(rows_out)
    if good.mean() < 0.5:
        return None
    mid = rows_out[len(rows_out) // 2 - 2:len(rows_out) // 2 + 3]
    mid = mid[np.isfinite(mid)]
    return {"cols": cols[good].astype(np.float32), "rows": rows_out[good].astype(np.float32),
            "height": round(float((brow_y - (np.median(mid) if len(mid) else np.nanmedian(rows_out))) / ruler), 2)}


def _parting(theta: np.ndarray, coh: np.ndarray, hair: np.ndarray, f: np.ndarray, box, size: float):
    """A parting: on top of the head, the line the strands fall away from to
    either side (left of it they run down to the left, right of it down to
    the right). Photo px points, or None."""
    from scipy.ndimage import gaussian_filter1d

    bx0, by0 = box[0], box[1]
    brow = float((f[105, 1] + f[334, 1]) / 2) - by0
    xl, xr = float(f[234, 0]) - bx0, float(f[454, 0]) - bx0
    tx = np.cos(theta) * np.sign(np.sin(theta) + 1e-9)  # the strand's sideways run, taken pointing down
    w = coh * hair
    win = max(int(0.06 * size), 3)
    pts = []
    rows = np.flatnonzero(hair[:, int(max(xl, 0)):int(min(xr, hair.shape[1]))].any(axis=1))
    rows = rows[rows < brow - 0.15 * size]
    for r in rows[::2]:
        a = gaussian_filter1d(tx[r] * w[r], 2.0)
        n = gaussian_filter1d(w[r], 2.0) + 1e-6
        best, bc = 0.0, None
        for c in range(int(max(xl + win, win)), int(min(xr - win, hair.shape[1] - win))):
            if not hair[r, c]:
                continue
            left = a[c - win:c].sum() / n[c - win:c].sum()
            right = a[c:c + win].sum() / n[c:c + win].sum()
            sc = right - left  # left side running left (negative), right side running right
            if sc > best:
                best, bc = sc, c
        if bc is not None and best > _PART_MIN:
            pts.append((bc + bx0, r + by0))
    if len(pts) < max(4, 0.3 * len(rows[::2])):
        return None
    pts = np.array(pts, np.float32)
    # one steady line, a fair way down the head, most of the points on it --
    # else no parting (a wrong groove would be an artifact)
    fit = np.polyfit(pts[:, 1], pts[:, 0], 1)
    res = np.polyval(fit, pts[:, 1]) - pts[:, 0]
    keep = np.abs(res) < 0.04 * size
    if keep.sum() < max(6, 0.6 * len(pts)):
        return None
    pts = pts[keep]
    if np.ptp(pts[:, 1]) < 0.2 * size or float(np.std(res[keep])) > 0.02 * size:
        return None
    return pts


def read_hair(rgb: np.ndarray, labels: np.ndarray, box, face_mesh: np.ndarray, hair_label: int, hat_label: int) -> Optional[dict]:
    """See the module's docstring. `labels`, `box`: head_parts' map. None
    when too little hair shows (a cap, a shaved head)."""
    from PIL import Image

    f = np.asarray(face_mesh, dtype=np.float64)[:, :2]
    size = max(float(np.ptp(f[:, 0])), float(np.ptp(f[:, 1])))
    bx0, by0, bx1, by1 = box
    hair = labels[by0:by1, bx0:bx1] == hair_label
    hairline = _hairline(labels, f, hair_label, hat_label)
    if hair.sum() < max(_MIN_HAIR_PX, _MIN_HAIR * size * size):
        return {"kind": None, "hairline": hairline, "size_px": size}
    gray = np.asarray(Image.fromarray(rgb[by0:by1, bx0:bx1, :3]).convert("L"), dtype=np.float32) / 255.0
    theta, coh, _ = _structure(gray, hair, max(0.8, _DERIV * size), max(2.0, _TENSOR * size))
    swing, wave_px = _waves(theta, coh, hair, size)
    fine = _fineness(gray, hair, size)
    c = float(coh[hair].mean())
    rope, shade, rope_px = _ropes(gray, hair, size)
    if fine < _SLEEK:
        kind = "sleek"
    elif rope >= _ROPE and shade <= _ROPE_SHADE and fine >= _ROPE_FINE:
        kind = "corded"
    elif c >= _COMBED:
        kind = "wavy" if swing >= _WAVY_SWING else "straight"
    else:
        kind = "coily" if fine >= _COILY else "curly"
    try:
        part = _parting(theta, coh, hair, f, (bx0, by0), size)
    except Exception:  # noqa: BLE001 - no parting then
        part = None
    return {"box": (bx0, by0, bx1, by1), "theta": theta, "coherence": coh, "size_px": size, "parting": part,
            "kind": kind, "coherence_mean": round(c, 3), "swing_deg": round(swing, 1),
            "wave_px": wave_px if kind == "wavy" else None, "fineness": round(fine, 3), "hairline": hairline,
            "rope": round(rope, 3), "rope_shade": round(shade, 3), "rope_px": rope_px if kind == "corded" else None}
