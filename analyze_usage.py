"""
Mobile Data Usage Distribution Analysis  (synthetic-data demo)
==============================================================

Demonstrates the analysis used on a real (confidential) SIM estate, on
SYNTHETIC data so it is fully runnable. The question: what does "typical"
data usage look like when the distribution is heavy-tailed?

Approach
--------
1. Generate a synthetic fleet: idle SIMs + a "light" population + a "heavy"
   population (two log-normals).
2. Show that the arithmetic mean is misleading on the raw scale.
3. Fit a 2-component log-normal mixture with a hand-written EM algorithm
   (no scikit-learn needed) to recover the two populations.
4. Render the story as SVG: raw (linear) vs log-scale with the fitted
   components and the data-pool threshold sitting in the valley between them.

Run:
    pip install -r requirements.txt
    python analyze_usage.py

Author: Sabbir  |  github.com/sab-bir08
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape as _xesc

import numpy as np

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SEED = 8
POOL_LIMIT_MB = 150.0


# ---------------------------------------------------------------------------
# 1. Synthetic SIM fleet
# ---------------------------------------------------------------------------
def make_fleet(n=4000) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n_idle = int(0.17 * n)
    n_active = n - n_idle
    # 79% light, 21% heavy among active SIMs
    is_heavy = rng.random(n_active) < 0.21
    light = rng.lognormal(mean=np.log(1.3), sigma=0.9, size=n_active)   # ~1.3 MB
    heavy = rng.lognormal(mean=np.log(600), sigma=0.55, size=n_active)  # ~600 MB
    active = np.where(is_heavy, heavy, light)
    return np.concatenate([np.zeros(n_idle), active])


# ---------------------------------------------------------------------------
# 2. Hand-written EM for a 2-component Gaussian mixture (on log10 usage)
# ---------------------------------------------------------------------------
def _npdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def fit_lognormal_mixture(usage_mb: np.ndarray, iters=300):
    x = np.log10(usage_mb[usage_mb > 0])
    mu = np.array([np.percentile(x, 20), np.percentile(x, 80)])
    sd = np.array([x.std() / 2, x.std() / 2]) + 0.1
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        r = np.stack([w[k] * _npdf(x, mu[k], sd[k]) for k in range(2)])
        r /= r.sum(0) + 1e-12
        Nk = r.sum(1)
        w = Nk / len(x)
        mu = (r * x).sum(1) / Nk
        sd = np.sqrt((r * (x - mu[:, None]) ** 2).sum(1) / Nk) + 1e-6
    order = np.argsort(mu)            # light component first
    return w[order], mu[order], sd[order], x


def skewness(a: np.ndarray) -> float:
    a = a.astype(float)
    return float(((a - a.mean()) ** 3).mean() / a.std() ** 3)


# ---------------------------------------------------------------------------
# 3. SVG rendering helpers
# ---------------------------------------------------------------------------
INK, GRID, MUTED = "#1f2937", "#e5e7eb", "#6b7280"
LIGHT, HEAVY, BAR, THRESH = "#2563eb", "#dc2626", "#cbd5e1", "#b45309"


def esc(s): return _xesc(str(s))


def _hdr(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="Segoe UI, Helvetica, Arial, sans-serif">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>')


def raw_hist_svg(usage):
    """Linear-scale histogram - looks like a single spike at zero (heavy tail)."""
    w, h, ml, mr, mt, mb = 360, 300, 48, 16, 52, 40
    pw, ph = w - ml - mr, h - mt - mb
    counts, edges = np.histogram(usage, bins=24, range=(0, np.percentile(usage, 99)))
    cmax = counts.max()
    s = [_hdr(w, h)]
    s.append(f'<text x="{ml}" y="24" font-size="14" font-weight="700" fill="{INK}">Raw usage (linear scale)</text>')
    s.append(f'<text x="{ml}" y="42" font-size="11" fill="{MUTED}">One tall spike + a long invisible tail</text>')
    bw = pw / len(counts)
    for i, c in enumerate(counts):
        bh = ph * c / cmax
        x = ml + i * bw
        s.append(f'<rect x="{x:.1f}" y="{mt+ph-bh:.1f}" width="{bw-1:.1f}" height="{bh:.1f}" fill="{BAR}"/>')
    s.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{w-mr}" y2="{mt+ph}" stroke="{INK}"/>')
    s.append(f'<text x="{ml}" y="{h-12}" font-size="10" fill="{MUTED}">0 MB</text>')
    s.append(f'<text x="{w-mr}" y="{h-12}" font-size="10" text-anchor="end" fill="{MUTED}">high</text>')
    s.append(f'<text x="{ml+pw/2}" y="{mt+ph/2}" font-size="11" text-anchor="middle" fill="{MUTED}">long tail hides here</text>')
    s.append('</svg>')
    return "".join(s)


def log_mixture_svg(x_log, w, mu, sd):
    """Log-scale histogram with the two fitted log-normal components."""
    W, H, ml, mr, mt, mb = 760, 340, 56, 24, 56, 52
    pw, ph = W - ml - mr, H - mt - mb
    lo, hi = -1.0, 3.3                      # log10(MB): 0.1 .. ~2000 MB
    nb = 46
    counts, edges = np.histogram(x_log, bins=nb, range=(lo, hi), density=True)
    grid = np.linspace(lo, hi, 240)
    comp = [w[k] * _npdf(grid, mu[k], sd[k]) for k in range(2)]
    dens = comp[0] + comp[1]
    ymax = max(counts.max(), dens.max()) * 1.12

    def X(v): return ml + pw * (v - lo) / (hi - lo)
    def Y(v): return mt + ph * (1 - v / ymax)

    s = [_hdr(W, H)]
    s.append(f'<text x="{ml}" y="26" font-size="16" font-weight="700" fill="{INK}">Same data on a log scale: two populations</text>')
    s.append(f'<text x="{ml}" y="45" font-size="11" fill="{MUTED}">A 2-component log-normal mixture (hand-written EM) separates light and heavy SIMs</text>')
    # histogram bars
    bw = pw / nb
    for i, c in enumerate(counts):
        bh = ph * c / ymax
        s.append(f'<rect x="{X(edges[i]):.1f}" y="{mt+ph-bh:.1f}" width="{bw-1:.1f}" height="{bh:.1f}" fill="{BAR}" opacity="0.8"/>')
    # x ticks at powers of ten
    for p in range(-1, 4):
        gx = X(p)
        if ml <= gx <= W - mr:
            s.append(f'<line x1="{gx:.1f}" y1="{mt}" x2="{gx:.1f}" y2="{mt+ph}" stroke="{GRID}"/>')
            lab = f"{10**p:g} MB" if p < 3 else "1 GB"
            s.append(f'<text x="{gx:.1f}" y="{mt+ph+18:.1f}" font-size="10" text-anchor="middle" fill="{MUTED}">{lab}</text>')
    # component + combined curves
    for k, col in [(0, LIGHT), (1, HEAVY)]:
        pts = " ".join(f"{X(g):.1f},{Y(comp[k][j]):.1f}" for j, g in enumerate(grid))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
    pts = " ".join(f"{X(g):.1f},{Y(dens[j]):.1f}" for j, g in enumerate(grid))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{INK}" stroke-width="1.4" stroke-dasharray="4 3"/>')
    # pool-limit threshold marker (in the valley)
    tx = X(np.log10(POOL_LIMIT_MB))
    s.append(f'<line x1="{tx:.1f}" y1="{mt}" x2="{tx:.1f}" y2="{mt+ph}" stroke="{THRESH}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{tx-6:.1f}" y="{mt+14}" font-size="10" text-anchor="end" fill="{THRESH}" font-weight="600">{POOL_LIMIT_MB:g} MB pool limit</text>')
    # component labels
    s.append(f'<text x="{X(mu[0]):.1f}" y="{Y(w[0]*_npdf(mu[0],mu[0],sd[0]))-8:.1f}" font-size="11" text-anchor="middle" fill="{LIGHT}" font-weight="700">Light ~{10**mu[0]:.0f} MB ({w[0]*100:.0f}%)</text>')
    s.append(f'<text x="{X(mu[1]):.1f}" y="{Y(w[1]*_npdf(mu[1],mu[1],sd[1]))-8:.1f}" font-size="11" text-anchor="middle" fill="{HEAVY}" font-weight="700">Heavy ~{10**mu[1]:.0f} MB ({w[1]*100:.0f}%)</text>')
    s.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{W-mr}" y2="{mt+ph}" stroke="{INK}"/>')
    s.append('</svg>')
    return "".join(s)


def main():
    usage = make_fleet()
    pos = usage[usage > 0]
    w, mu, sd, x_log = fit_lognormal_mixture(usage)

    idle_pct = 100 * (usage == 0).mean()
    print("Synthetic SIM fleet:", len(usage), "SIMs")
    print(f"  idle (0 MB):       {idle_pct:.0f}%")
    print(f"  arithmetic mean:   {usage.mean():.1f} MB   <- misleading")
    print(f"  median:            {np.median(pos):.1f} MB")
    print(f"  geometric mean:    {10**np.log10(pos).mean():.1f} MB")
    print(f"  raw skewness:      {skewness(usage):.1f}   (0 = symmetric)")
    print("\nFitted 2-component log-normal mixture:")
    print(f"  light: {w[0]*100:4.0f}% of active, center ~{10**mu[0]:6.1f} MB")
    print(f"  heavy: {w[1]*100:4.0f}% of active, center ~{10**mu[1]:6.1f} MB")
    print(f"  {POOL_LIMIT_MB:g} MB pool limit sits in the valley between the two.")

    ASSETS.mkdir(exist_ok=True)
    (ASSETS / "usage_raw.svg").write_text(raw_hist_svg(usage), encoding="utf-8")
    (ASSETS / "usage_distribution.svg").write_text(log_mixture_svg(x_log, w, mu, sd), encoding="utf-8")
    print("\nOK - wrote assets/usage_raw.svg and assets/usage_distribution.svg")


if __name__ == "__main__":
    main()
