"""Figures per recording: hypnogram with stage probabilities, spectrogram, spindle / slow-wave density map."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DPI = 150
ORDER = ["W", "R", "N1", "N2", "N3"]           # top to bottom, as clinicians draw hypnograms
COLOURS = {"W": "#e6b84c", "R": "#d1495b", "N1": "#9ecae1", "N2": "#4292c6", "N3": "#08306b"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _clock(start_time: str):
    try:
        return datetime.fromisoformat(start_time) if start_time else None
    except ValueError:
        return None


def _hours_axis(ax, hours: float, start):
    if start is None:
        ax.set_xlabel("Hours from the start of the recording")
        return
    step = 1 if hours <= 14 else 2
    ticks = np.arange(0, hours + 1e-9, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([(start + timedelta(hours=float(t))).strftime("%H:%M") for t in ticks])
    ax.set_xlabel(f"Clock time (recording started {start:%Y-%m-%d %H:%M})")


def hypnogram(path: Path, rid: str, stages: list[str], proba, confidence: np.ndarray, channel: str,
              start_time: str, lights: tuple[float, float] | None, excluded: list[tuple[float, float]]) -> None:
    plt = _plt()
    n = len(stages)
    t = np.arange(n + 1) * 30 / 3600
    hours = n * 30 / 3600
    start = _clock(start_time)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    y = np.array([ORDER.index(s) for s in stages] + [ORDER.index(stages[-1])])
    a1.step(t, y, where="post", color="#333333", lw=1.0)
    rem = np.array([s == "R" for s in stages])
    for i in np.flatnonzero(rem):
        a1.plot([t[i], t[i + 1]], [1, 1], color=COLOURS["R"], lw=3.5, solid_capstyle="butt")
    low = confidence < 0.8
    for i in np.flatnonzero(low):
        a1.axvspan(t[i], t[i + 1], ymin=0.0, ymax=0.06, color="#999999", lw=0)
    for a, b in excluded:
        a1.axvspan(a / 3600, b / 3600, color="#f4a582", alpha=0.35, lw=0)
    if lights:
        for v, lab in zip(lights, ("lights off", "lights on")):
            if v is not None:
                a1.axvline(v / 3600, color="#555555", ls="--", lw=0.8)
                a1.text(v / 3600, -0.45, f" {lab} " , fontsize=7, color="#555555",
                        ha="left" if lab == "lights off" else "right")
    a1.set_yticks(range(len(ORDER)))
    a1.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])
    a1.set_ylim(len(ORDER) - 0.5, -0.6)
    a1.set_title(f"{rid}: automatic sleep stages (YASA, EEG only, {channel})", fontsize=10, pad=16)
    a1.text(0.5, 1.02, "grey ticks: confidence < 80 %   orange: left out of spindle / slow-wave detection",
            transform=a1.transAxes, ha="center", va="bottom", fontsize=7, color="#555555")
    bottom = np.zeros(n)
    for s in ["N3", "N2", "N1", "R", "W"]:
        v = proba[s].to_numpy()
        a2.bar(t[:-1], v, width=30 / 3600, bottom=bottom, align="edge", color=COLOURS[s], lw=0,
               label={"W": "Wake", "R": "REM"}.get(s, s))
        bottom += v
    a2.set_ylim(0, 1)
    a2.set_ylabel("Probability")
    a2.legend(ncol=5, fontsize=7, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.18))
    a2.set_xlim(0, hours)
    _hours_axis(a2, hours, start)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def spectrogram(path: Path, rid: str, x100: np.ndarray, channel: str, start_time: str) -> None:
    """Multitaper-free spectrogram: 30 s Welch windows (4 s segments), 0.5-25 Hz, log power."""
    from scipy.signal import welch

    plt = _plt()
    sf, win = 100.0, 3000
    n = len(x100) // win
    if n < 2:
        raise ValueError("too short for a spectrogram")
    seg = x100[: n * win].reshape(n, win).astype(np.float64)
    f, p = welch(seg, fs=sf, nperseg=400, noverlap=200, axis=1)
    keep = (f >= 0.5) & (f <= 25)
    f, p = f[keep], 10 * np.log10(p[:, keep] + 1e-12)
    lo, hi = np.percentile(p, [2.5, 97.5])
    hours = n * 30 / 3600
    fig, ax = plt.subplots(figsize=(11, 3.4))
    im = ax.pcolormesh(np.arange(n + 1) * 30 / 3600, np.r_[f, f[-1] + (f[1] - f[0])] - (f[1] - f[0]) / 2, p.T,
                       shading="flat", cmap="RdBu_r", vmin=lo, vmax=hi, rasterized=True)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim(0, hours)
    _hours_axis(ax, hours, _clock(start_time))
    ax.set_title(f"{rid}: spectrogram of the staging channel {channel}", fontsize=10)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Power (dB µV²/Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def positions(labels_contacts: list[tuple[str, tuple[str, ...]]]) -> dict[str, np.ndarray]:
    """2-D sensor positions from MNE's standard_1005 template (bipolar pair: the midpoint)."""
    from processor.eeg import template_montage

    pos = template_montage().get_positions()["ch_pos"]
    out = {}
    for label, contacts in labels_contacts:
        pts = [pos[c] for c in contacts if c in pos]
        if contacts and len(pts) == len(contacts):
            out[label] = np.mean(pts, axis=0)
    return out


def density_map(path: Path, rid: str, labels_contacts, spindle: dict[str, float], sw: dict[str, float],
                bipolar: bool) -> bool:
    """Two head maps: spindles and slow waves per minute of N2+N3, per channel. False if fewer than
    4 channels have a position."""
    import mne

    plt = _plt()
    pos3 = positions(labels_contacts)
    names, seen = [], []
    for n, _ in labels_contacts:
        if n in pos3 and n in spindle and not any(np.allclose(pos3[n], q, atol=1e-4) for q in seen):
            names.append(n)
            seen.append(pos3[n])
    if len(names) < 4:
        return False
    info = mne.create_info(names, 100.0, "eeg")
    mont = mne.channels.make_dig_montage(ch_pos={n: pos3[n] for n in names}, coord_frame="head")
    info.set_montage(mont, on_missing="ignore")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.0))
    for ax, vals, title, cmap in ((axes[0], spindle, "Spindles / min of N2+N3", "Purples"),
                                  (axes[1], sw, "Slow waves / min of N2+N3", "Blues")):
        v = np.array([vals.get(n, np.nan) for n in names], dtype=float)
        v = np.nan_to_num(v)
        vmax = max(float(v.max()), 1e-6)
        im, _ = mne.viz.plot_topomap(v, info, axes=ax, show=False, cmap=cmap, vlim=(0, vmax), contours=0,
                                     names=names if len(names) <= 24 else None, sensors=True)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    fig.suptitle(f"{rid}: spindle and slow-wave density ({'bipolar pairs at their midpoints' if bipolar else 'per electrode'})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return True
