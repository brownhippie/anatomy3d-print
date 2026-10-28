"""General plugin: the frame at any border. Reads a file that is one row per change (or cuts changes from a raw periodic signal), and reads
every quantity against the system's own first spins, held, with a local level beside, the kick, the gate, the drift and loop, the noise
channel, and the mirror between the two sides when both are given. It only reads. See PLUGIN.md.

usage, per-change file:  python mad_plugin.py data.csv --a current [--b torque] [--noise vib] [--temp temp] [--label phase] [--spread spread]
usage, raw signal:        python mad_plugin.py raw.csv --cut piezo --fs 1000 --a current [--b volts] [--noise piezo] [--label phase]
options: --first 500 --warmup 300 --window 200 --local 50 --follow"""
import argparse
import io
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

P = dict(N_FIRST=500, WARMUP=300, STEADY_PCT=95, CEIL_PCT=99, FLOOR_PCT=1, W_LOCAL=50, MIN_LOCAL=10, WINDOW=200, RHYTHM_MIN=0.5, RHYTHM_MAX=None)


def margin(R_t, R):
    """The owner's line with M_sys = 1/R_t: M = (R_t - R)/R_t. Negative = past the threshold. Every number here comes from this."""
    return (R_t - R) / R_t


# ---------------------------------------------------------------- cutting changes from a raw periodic signal
def crossings(x, h):
    cand = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0)) + 1
    below = np.flatnonzero(x < -h); out = []; last = None
    for c in cand:
        if last is None:
            out.append(int(c)); last = c; continue
        j = np.searchsorted(below, last, side="right")
        if j < below.size and below[j] < c:
            out.append(int(c)); last = c
    return np.array(out, int)


def cut_changes(df, sig, fs, cols, label=None):
    """One change = one cycle of the rhythm found in column `sig`. Returns a per-change frame with the mean of every column in `cols`,
    the change length in samples, the within-change spread of the first column, and the RMS of `sig` over the change."""
    x = df[sig].to_numpy(float); x = x - x.mean()
    sp = np.abs(np.fft.rfft(x * np.hanning(x.size))); fr = np.fft.rfftfreq(x.size, 1 / fs)
    m = (fr >= P["RHYTHM_MIN"]) & (fr <= (P["RHYTHM_MAX"] or 0.45 * fs))
    f = float(fr[m][np.argmax(sp[m])])
    sos = butter(2, [0.6 * f, min(1.6 * f, 0.45 * fs)], btype="band", fs=fs, output="sos")
    y = sosfiltfilt(sos, x); y = y / (y.std() + 1e-12)
    up = crossings(y, 0.3); L = np.diff(up).astype(float)
    med = np.median(L); keep = np.abs(L - med) <= 0.3 * med
    rows = []
    for k in range(L.size):
        if not keep[k] or L[k] < 4:
            continue
        a, b = up[k], up[k + 1]
        r = {c: float(df[c].to_numpy(float)[a:b].mean()) for c in cols}
        r.update(start=int(a), length=int(b - a), spread=float(df[cols[0]].to_numpy(float)[a:b].std() / max(abs(r[cols[0]]), 1e-12)), sig_rms=float(np.sqrt(np.mean(x[a:b] ** 2))))
        if label:
            r["label"] = str(df[label].iloc[a])
        rows.append(r)
    out = pd.DataFrame(rows); out.attrs["rhythm_hz"] = f
    return out


# ---------------------------------------------------------------- reference and readings
def reference(R, spread):
    band = float(np.percentile(spread, P["STEADY_PCT"])); st = spread <= band; Rs = R[st]
    L = float(Rs.mean()); ceil = float(np.percentile(Rs, P["CEIL_PCT"])); floor = float(np.percentile(Rs, P["FLOOR_PCT"]))
    d = pd.Series(R - L).rolling(3, center=True, min_periods=1).median().to_numpy()
    ups = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) + 1
    lens = [b - a for a, b in zip(ups[:-1], ups[1:]) if b - a >= 4]
    return dict(band=band, L=L, ceil=ceil, floor=floor, width_up=ceil - L, width_down=L - floor, loop_len=float(np.median(lens)) if lens else np.nan, n_steady=int(st.sum()))


def local_level(R):
    return pd.Series(R).rolling(P["W_LOCAL"], min_periods=P["MIN_LOCAL"]).mean().shift(1).to_numpy()


def take_first_spins(t, qtys, spread_col, temp_col=None):
    first = t[t["run"] == t["run"].iloc[0]]
    need = P["WARMUP"] + P["N_FIRST"]
    if len(first) < need:
        return None, f"waiting for first spins: {len(first)} changes of the first run, need {need}"
    if temp_col and temp_col in first and first[temp_col].notna().any():
        tt = first[temp_col].to_numpy(float); lev = np.nanmedian(tt[-max(len(tt) // 3, 1):]); w0 = int(np.argmax(tt >= lev - 1.0))
    else:
        w0 = P["WARMUP"]
    fs_ = first.iloc[w0:w0 + P["N_FIRST"]]
    refs = {q: reference(fs_[q].to_numpy(float), fs_[spread_col].to_numpy(float)) for q in qtys}
    for q in qtys:
        kk = fs_[q].diff().abs().to_numpy(float); ok = np.isfinite(kk); refs["kick_" + q] = reference(kk[ok], fs_[spread_col].to_numpy(float)[ok])
    st = fs_[spread_col].to_numpy(float) <= refs[qtys[0]]["band"]
    noise0 = float(np.median(fs_["noise"].to_numpy(float)[st])) if "noise" in fs_ else np.nan
    return dict(refs=refs, noise0=noise0, from_change=int(fs_.index[0])), None


def read_changes(t, qtys, spread_col, state):
    out = t.copy(); refs = state["refs"]
    out["steady"] = t[spread_col].to_numpy(float) <= refs[qtys[0]]["band"]
    if "noise" in t:
        out["noise_ratio"] = t["noise"] / state["noise0"]
    for q in qtys:
        R = t[q].to_numpy(float); r = refs[q]
        out[q + "_M_ceil"] = margin(r["ceil"], R); out[q + "_M_floor"] = (R - r["floor"]) / r["floor"]; out[q + "_M_L"] = margin(r["L"], R)
        Lw = local_level(R); out[q + "_M_ceil_local"] = margin(Lw + r["width_up"], R); out[q + "_M_floor_local"] = (R - (Lw - r["width_down"])) / (Lw - r["width_down"])
        out[q + "_gate"] = np.where(out[q + "_M_ceil"] < 0, "past", np.where(out[q + "_M_floor"] < 0, "short", "in"))
        out[q + "_gate_local"] = np.where(out[q + "_M_ceil_local"] < 0, "past", np.where(out[q + "_M_floor_local"] < 0, "short", "in"))
        out[q + "_nudge_local_pct"] = (R - Lw) / Lw * 100
        out[q + "_kick"] = t[q].diff().abs(); out[q + "_kick_M_ceil"] = margin(refs["kick_" + q]["ceil"], out[q + "_kick"].to_numpy(float))
    if len(qtys) >= 2:
        out["extra_pct"] = out[qtys[0] + "_nudge_local_pct"] + out[qtys[1] + "_nudge_local_pct"]   # zero when up on one side is down on the other
    return out


def windows(tr, qtys, label):
    rows = []; W = P["WINDOW"]
    for a in range(0, len(tr) - W + 1, W):
        w = tr.iloc[a:a + W]; r = dict(label=label, first_change=int(w.index[0]), changes=W, unsteady_pct=float((~w["steady"]).mean() * 100))
        if "noise_ratio" in w:
            r["noise_ratio"] = float(w["noise_ratio"].median())
        if "temp" in w:
            r["temp"] = float(w["temp"].median())
        for q in qtys:
            r[q + "_past_pct"] = float((w[q + "_M_ceil"] < 0).mean() * 100); r[q + "_short_pct"] = float((w[q + "_M_floor"] < 0).mean() * 100)
            r[q + "_past_local_pct"] = float((w[q + "_M_ceil_local"] < 0).mean() * 100); r[q + "_short_local_pct"] = float((w[q + "_M_floor_local"] < 0).mean() * 100)
            r[q + "_drift_pct"] = float(w[q + "_M_L"].median() * 100); r[q + "_depth_pct"] = float(w[q + "_M_L"].abs().mean() * 100)
            r[q + "_kick_past_pct"] = float((w[q + "_kick_M_ceil"] < 0).mean() * 100)
            d = w[q + "_M_L"].to_numpy(float); dm = pd.Series(d).rolling(3, center=True, min_periods=1).median().to_numpy(); ups = np.flatnonzero((dm[:-1] > 0) & (dm[1:] <= 0)) + 1
            lens = [b - a_ for a_, b in zip(ups[:-1], ups[1:]) if b - a_ >= 2]; r[q + "_loop_len"] = float(np.median(lens)) if lens else np.nan
        if len(qtys) >= 2:
            x = w[qtys[0] + "_nudge_local_pct"].to_numpy(float); y = w[qtys[1] + "_nudge_local_pct"].to_numpy(float); ok = np.isfinite(x) & np.isfinite(y)
            r["mirror_slope"] = float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() > 20 and x[ok].std() > 0 else np.nan
            r["extra_sd_pct"] = float(np.nanstd(w["extra_pct"].to_numpy(float)))
        rows.append(r)
    return rows


# ---------------------------------------------------------------- one pass
def run_once(args, state):
    df = pd.read_csv(args.file)
    if args.cut:
        cols = [c for c in (args.a, args.b, args.noise, args.temp) if c]
        t = cut_changes(df, args.cut, args.fs, cols, args.label); rhythm = t.attrs.get("rhythm_hz", np.nan)
        qtys = [c for c in (args.a, args.b) if c]
        if args.noise:
            t["noise"] = t[args.noise] if args.noise != args.cut else t["sig_rms"]
        if args.temp:
            t["temp"] = t[args.temp]
        spread_col = "spread"
        t["run"] = (t["label"] != t["label"].shift()).cumsum() if "label" in t else 0
        t["label"] = t["label"] if "label" in t else "all"
    else:
        t = df.copy(); rhythm = np.nan
        qtys = [c for c in (args.a, args.b) if c]
        if args.noise:
            t["noise"] = t[args.noise]
        if args.temp:
            t["temp"] = t[args.temp]
        spread_col = args.spread or "_spread"
        if spread_col not in t:
            t[spread_col] = t[args.a].diff().abs().fillna(0) / t[args.a].abs().clip(lower=1e-12)   # no inside-change spread given: use the change's own step as the steadiness
        t["label"] = t[args.label].astype(str) if args.label else "all"
        t["run"] = (t["label"] != t["label"].shift()).cumsum()
    if state.get("refs") is None:
        st, msg = take_first_spins(t, qtys, spread_col, "temp" if args.temp else None)
        if st is None:
            print(msg, flush=True); return state, []
        state.update(st); state["rhythm_hz"] = rhythm
        base = args.file.rsplit(".", 1)[0]
        json.dump(dict(rhythm_hz=rhythm, first_spins_from_change=state["from_change"], noise0=state["noise0"], refs=state["refs"]), io.open(base + "_reference.json", "w", encoding="utf-8"), indent=1)
        print(f"first spins taken from change {state['from_change']}" + (f", rhythm {rhythm:.2f} Hz" if np.isfinite(rhythm) else "") + "; " + "; ".join(f"{q}: level {state['refs'][q]['L']:.4g}, floor {state['refs'][q]['floor']:.4g}, ceiling {state['refs'][q]['ceil']:.4g}, loop {state['refs'][q]['loop_len']:.0f}" for q in qtys), flush=True)
    tr = read_changes(t, qtys, spread_col, state)
    base = args.file.rsplit(".", 1)[0]
    tr.to_csv(base + "_changes.csv", index=False)
    rows = []
    for lab in tr["label"].unique():
        rows += windows(tr[tr["label"] == lab], qtys, lab)
    pd.DataFrame(rows).to_csv(base + "_windows.csv", index=False)
    new = rows[state.get("n_windows", 0):]; state["n_windows"] = len(rows)
    for r in new:
        q = qtys[0]
        line = f"[{r['label']}] from change {r['first_change']}: {q} past {r[q + '_past_pct']:.1f} % short {r[q + '_short_pct']:.1f} % (local {r[q + '_past_local_pct']:.1f}/{r[q + '_short_local_pct']:.1f}); drift {r[q + '_drift_pct']:+.2f} % depth {r[q + '_depth_pct']:.2f} % loop {r[q + '_loop_len']:.0f}; kick past {r[q + '_kick_past_pct']:.1f} %; unsteady {r['unsteady_pct']:.0f} %"
        if len(qtys) >= 2:
            line += f"; mirror slope {r['mirror_slope']:+.2f}, extra sd {r['extra_sd_pct']:.3f} %"
        if "noise_ratio" in r:
            line += f"; noise x{r['noise_ratio']:.2f}"
        if "temp" in r:
            line += f"; temp {r['temp']:.1f}"
        print(line, flush=True)
    return state, new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--a", required=True, help="input-side quantity column (R on side A)"); ap.add_argument("--b", help="machine-side quantity column (R on side B), optional")
    ap.add_argument("--noise", help="dilution channel column (vibration, error count...), optional"); ap.add_argument("--temp", help="temperature column, optional"); ap.add_argument("--label", help="phase / run label column, optional")
    ap.add_argument("--spread", help="within-change spread column for the steady flag (per-change files), optional"); ap.add_argument("--cut", help="raw periodic signal column to cut changes from"); ap.add_argument("--fs", type=float, default=1.0, help="sample rate of the raw file (only with --cut)")
    ap.add_argument("--first", type=int, default=500); ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--window", type=int, default=200); ap.add_argument("--local", type=int, default=50); ap.add_argument("--follow", action="store_true")
    args = ap.parse_args()
    P.update(N_FIRST=args.first, WARMUP=args.warmup, WINDOW=args.window, W_LOCAL=args.local)
    state = {}
    while True:
        try:
            state, _ = run_once(args, state)
        except Exception as e:
            print("read error:", e, flush=True)
        if not args.follow:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
