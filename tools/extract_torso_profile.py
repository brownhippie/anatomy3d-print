"""Torso cross-section profiles from the MakeHuman base mesh (CC0, via the
anny 0.6.1 wheel's mpfb2 data). Levels u: 0 = shoulder joints, 1 = hip
joints. Across s in [-1, 1] (fraction of the torso's half-width at that
level): front and back surface depth relative to the torso's side edges,
divided by the centre-line depth. Output: back_profile.npz"""
import sys
import numpy as np
import trimesh

path = sys.argv[1]
V, groups, cur = [], {}, None
for line in open(path):
    if line.startswith("v "):
        V.append([float(x) for x in line.split()[1:4]])
    elif line.startswith("g "):
        cur = line.split()[1]
        groups.setdefault(cur, [])
    elif line.startswith("f ") and cur is not None:
        groups[cur].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
V = np.array(V)
# Average young adult male: MakeHuman's macro targets for male / young,
# the three ancestry groups weighted equally (average muscle and weight is
# the base itself). Targets: lines "vertex_index dx dy dz".
import gzip, os
tdir = os.path.join(os.path.dirname(os.path.abspath(path)), "targets")
if os.path.isdir(tdir):
    for race in ("african", "asian", "caucasian"):
        for line in gzip.open(os.path.join(tdir, f"{race}-male-young.target.gz"), "rt"):
            parts = line.split()
            if len(parts) == 4 and not line.startswith("#"):
                V[int(parts[0])] += np.array([float(x) for x in parts[1:]]) / 3.0
    print("applied male-young targets (1/3 each ancestry)")
def centroid(g):
    idx = np.unique(np.array([i for f in groups[g] for i in f]))
    return V[idx].mean(0)
tris = []
for f in groups["body"]:
    for k in range(1, len(f) - 1):
        tris.append([f[0], f[k], f[k + 1]])
body = trimesh.Trimesh(V, np.array(tris), process=True)
sh = (centroid("joint-l-shoulder") + centroid("joint-r-shoulder")) / 2
hp = (centroid("joint-l-upper-leg") + centroid("joint-r-upper-leg")) / 2
eye = centroid("joint-l-eye"); head = centroid("joint-head")
front_sign = 1.0 if eye[2] > head[2] else -1.0
print("height", np.ptp(V[:, 1]).round(2), "shoulder y", sh[1].round(2), "hip y", hp[1].round(2), "front +Z" if front_sign > 0 else "front -Z")
U = np.linspace(-0.2, 1.4, 33)
S = np.linspace(-1, 1, 41)
F = np.full((len(U), len(S)), np.nan); B = np.full_like(F, np.nan); W = np.full(len(U), np.nan); D = np.full(len(U), np.nan)
for i, u in enumerate(U):
    y = sh[1] + u * (hp[1] - sh[1])
    sec = body.section(plane_origin=[0, y, 0], plane_normal=[0, 1, 0])
    if sec is None:
        continue
    loop = None
    for lp in sec.discrete:
        if lp[:, 0].min() < sh[0] < lp[:, 0].max() and (loop is None or np.ptp(lp[:, 0]) > np.ptp(loop[:, 0])):
            loop = lp
    if loop is None:
        continue
    x = loop[:, 0] - sh[0]; z = front_sign * loop[:, 2]
    hw = (x.max() - x.min()) / 2; xc = (x.max() + x.min()) / 2
    # side seam height: mean depth of the outer 7% of the width on each side
    # (a single extreme vertex jitters from slice to slice)
    edge = np.abs(x - xc) > 0.93 * hw
    side_z = float(np.mean(z[edge])) if edge.any() else (z[np.argmax(x)] + z[np.argmin(x)]) / 2
    fr = np.full(len(S), np.nan); bk = np.full(len(S), np.nan)
    for j, s in enumerate(S):
        xs = xc + s * hw * 0.999
        a, b = x[:-1], x[1:]
        za, zb = z[:-1], z[1:]
        hit = (a - xs) * (b - xs) <= 0
        hit &= a != b
        if not hit.any():
            continue
        t = (xs - a[hit]) / (b[hit] - a[hit])
        zz = za[hit] + t * (zb[hit] - za[hit])
        fr[j], bk[j] = zz.max(), zz.min()
    d0 = fr[len(S) // 2] - bk[len(S) // 2]
    F[i] = (fr - side_z) / d0; B[i] = (bk - side_z) / d0; W[i] = hw; D[i] = d0
    print(f"u={u:+.2f}  half-width {hw:.2f}  depth {d0:.2f}  centre front {F[i, 20]:+.2f} back {B[i, 20]:+.2f}  back at s=0.5 {B[i, 30]:+.2f}  front at s=0.5 {F[i, 30]:+.2f}")
# smooth across heights (3-tap), leaving gaps alone
for A in (F, B):
    Ak = A.copy()
    for i in range(1, len(U) - 1):
        blk = A[i - 1:i + 2]
        Ak[i] = np.where(np.isfinite(blk).all(0), 0.25 * blk[0] + 0.5 * blk[1] + 0.25 * blk[2], A[i])
    A[:] = Ak
for i, u in enumerate(U):
    if 0.15 <= u <= 1.15:
        print(f"smoothed u={u:+.2f}: centre front {F[i, 20]:+.2f} back {B[i, 20]:+.2f}   at s=0.5 front {F[i, 30]:+.2f} back {B[i, 30]:+.2f}")
sw = abs(centroid("joint-l-shoulder")[0] - centroid("joint-r-shoulder")[0])
ratio = (sh[1] - hp[1]) / sw
print("torso length (shoulder->hip joints) / shoulder-joint distance:", round(ratio, 3))
np.savez(sys.argv[2] if len(sys.argv) > 2 else "back_profile.npz", u=U, s=S, front=F, back=B, half_width=W, depth=D, torso_len_over_shoulders=ratio)
