"""
eeg-sleep: sleep staging, sleep statistics, spindles and slow waves from clinical scalp EEG
with YASA (Vallat & Walker 2021).

Reads the BIDS-EEG derivative of eeg-prep (or a plain BIDS-EEG dataset, or loose EDF/BDF/
BrainVision/FIF/EEGLAB files) from INPUT_DIR and writes to OUTPUT_DIR/eeg-sleep/[<LABEL>/]:

  hypnogram.csv    recording_id, epoch, start_s, duration_s, stage (W/N1/N2/N3/R), confidence, channel
  sleep_stats.csv  recording_id, metric, value  (YASA's sleep statistics + staging confidence)
  spindles.csv     recording_id, channel, start_s, end_s, duration_s, frequency_hz, amplitude_uv, stage
  slow_waves.csv   recording_id, channel, start_s, end_s, amplitude_uv, frequency_hz, stage
  features.csv     recording_id, channel, feature, value  (spindle_* and sw_* per channel)
  figures/         <recording_id>_hypnogram.png, _spectrogram.png, _density.png
  summary.json, provenance.json

One recording at a time; the staging derivation is streamed at 100 Hz, detection runs in
windows (WINDOW_S) of the detection channels, channels in parallel (THREADS).
"""
from __future__ import annotations

import os
import sys

from processor import resources as R

# Memory per parallel detection worker (one channel of a 2 h window at 100 Hz, YASA's filters and
# Hilbert transform, plus the worker's imports) and for the main process (a 10-min native-rate
# chunk of every detection channel, the staging trace and YASA's features). See README "Runtime".
WORKER_MB = 450
MAIN_MB = 900


def _early_env() -> None:
    """Before numpy is imported: thread pools from THREADS (cgroup-aware), and every cache in
    TMPDIR/HOME (read-only file system on Lambda)."""
    tmp = os.environ.get("TMPDIR") or os.environ.get("HOME") or "/tmp"
    for var in ("HOME", "TMPDIR"):
        d = os.environ.get(var)
        if d:
            try:
                os.makedirs(d, exist_ok=True)
            except OSError:
                pass
    mpl = os.path.join(tmp, "mpl")
    try:
        os.makedirs(mpl, exist_ok=True)
        baked = "/opt/eeg-sleep/matplotlib"
        if os.path.isdir(baked):
            import shutil

            for f in os.listdir(baked):
                if f.endswith(".json") and not os.path.exists(os.path.join(mpl, f)):
                    shutil.copy(os.path.join(baked, f), mpl)
        os.environ.setdefault("MPLCONFIGDIR", mpl)
    except OSError:
        pass
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tmp, "numba"))
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_DATA", os.path.join(tmp, "mne_data"))
    os.environ.setdefault("MNE_USE_CUDA", "false")
    try:
        req = int((os.environ.get("THREADS") or "0").strip())
    except ValueError:
        req = 0
    n, _ = R.resolve_workers(max(0, req), WORKER_MB, MAIN_MB)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(n)


_early_env()

import csv  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from importlib import metadata as importlib_metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from processor import eeg as E  # noqa: E402
from processor import figures as F  # noqa: E402
from processor import sleep as Z  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("eeg-sleep")
log.setLevel(logging.INFO)
for noisy in ("mne", "numba", "joblib", "yasa", "matplotlib", "lightgbm"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
# YASA's classifiers were pickled with scikit-learn 0.24.2; only their LabelEncoder is a
# scikit-learn object, which unpickles unchanged (README "Model").
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

APP = "eeg-sleep"
REPO_ROOT = Path(__file__).resolve().parent.parent
PAD_S = 30.0            # context on each side of a detection window
CITATIONS = ["10.7554/eLife.70092", "10.5665/sleep.5774", "10.1093/jamia/ocy064", "10.3389/fnins.2013.00267",
             "10.1038/s41592-019-0686-2"]
HEADERS = {
    "hypnogram.csv": ["recording_id", "epoch", "start_s", "duration_s", "stage", "confidence", "channel"],
    "sleep_stats.csv": ["recording_id", "metric", "value"],
    "spindles.csv": ["recording_id", "channel", "start_s", "end_s", "duration_s", "frequency_hz", "amplitude_uv",
                     "stage"],
    "slow_waves.csv": ["recording_id", "channel", "start_s", "end_s", "amplitude_uv", "frequency_hz", "stage"],
    "features.csv": ["recording_id", "channel", "feature", "value"],
}
PARAM_NAMES = ("STAGING_CHANNEL", "USE_DEMOGRAPHICS", "MIN_SLEEP_MINUTES", "DETECT_CHANNELS", "DETECT_REFERENCE",
               "SPINDLE_FREQ",
               "SW_AMP_PTP", "EXCLUDE_SEIZURE", "SEIZURE_MARGIN_S", "WINDOW_S", "THREADS", "LABEL")


def version_of(root: Path) -> str:
    try:
        return (root / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("yasa", "lightgbm", "scikit-learn", "mne", "numpy", "scipy", "pandas", "antropy"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def peak_rss_mb() -> float:
    ru = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    return round(ru / (2**20 if sys.platform == "darwin" else 2**10), 1)


def _bool(name: str, value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"{name} must be true or false (got {value!r})")


def _range(name: str, value: str, lo: float, hi: float) -> tuple[float, float]:
    try:
        a, b = (float(v) for v in value.split(","))
    except ValueError:
        raise ValueError(f"{name} must be two numbers separated by a comma (got {value!r})") from None
    if not lo <= a < b <= hi:
        raise ValueError(f"{name} must be low,high within {lo:g}-{hi:g} (got {value!r})")
    return a, b


def env_params() -> dict:
    """Parameters from the environment; raises ValueError on a bad value."""
    e = lambda k, d: (os.environ.get(k) or d).strip()  # noqa: E731
    raw = {
        "STAGING_CHANNEL": e("STAGING_CHANNEL", "auto"),
        "USE_DEMOGRAPHICS": e("USE_DEMOGRAPHICS", "false"),
        "MIN_SLEEP_MINUTES": e("MIN_SLEEP_MINUTES", "60"),
        "DETECT_CHANNELS": e("DETECT_CHANNELS", "all").lower(),
        "DETECT_REFERENCE": e("DETECT_REFERENCE", "auto").lower(),
        "SPINDLE_FREQ": e("SPINDLE_FREQ", "12,15"),
        "SW_AMP_PTP": e("SW_AMP_PTP", "75,350"),
        "EXCLUDE_SEIZURE": e("EXCLUDE_SEIZURE", "true"),
        "SEIZURE_MARGIN_S": e("SEIZURE_MARGIN_S", "60"),
        "WINDOW_S": e("WINDOW_S", "7200"),
        "THREADS": e("THREADS", "0"),
        "LABEL": e("LABEL", ""),
    }
    p = dict(raw)
    p["use_demo"] = _bool("USE_DEMOGRAPHICS", raw["USE_DEMOGRAPHICS"])
    p["min_minutes"] = float(raw["MIN_SLEEP_MINUTES"])
    if p["min_minutes"] < 10:
        raise ValueError("MIN_SLEEP_MINUTES must be >= 10 (YASA smooths its features over 7.5 min)")
    if p["DETECT_CHANNELS"] not in ("all", "frontal-central"):
        raise ValueError(f"DETECT_CHANNELS must be all or frontal-central (got {raw['DETECT_CHANNELS']!r})")
    if p["DETECT_REFERENCE"] not in ("auto", "as-recorded"):
        raise ValueError(f"DETECT_REFERENCE must be auto or as-recorded (got {raw['DETECT_REFERENCE']!r})")
    p["spindle_freq"] = _range("SPINDLE_FREQ", raw["SPINDLE_FREQ"], 9, 18)
    p["sw_ptp"] = _range("SW_AMP_PTP", raw["SW_AMP_PTP"], 10, 2000)
    p["exclude_seizure"] = _bool("EXCLUDE_SEIZURE", raw["EXCLUDE_SEIZURE"])
    p["margin"] = float(raw["SEIZURE_MARGIN_S"])
    if p["margin"] < 0:
        raise ValueError("SEIZURE_MARGIN_S must be >= 0")
    p["window"] = float(raw["WINDOW_S"])
    if p["window"] < 600:
        raise ValueError("WINDOW_S must be >= 600")
    threads = int(raw["THREADS"])
    if threads < 0:
        raise ValueError("THREADS must be >= 0 (0 = every vCPU)")
    p["threads"], p["threads_why"] = R.resolve_workers(threads, WORKER_MB, MAIN_MB)
    if p["LABEL"] and not all(c.isalnum() or c in "-_" for c in p["LABEL"]):
        raise ValueError("LABEL may hold letters, digits, - and _ only")
    return p


class CsvOut:
    def __init__(self, path: Path, header: list[str]):
        self.fh = open(path, "w", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.fh, fieldnames=header, extrasaction="ignore", lineterminator="\n")
        self.w.writeheader()

    def rows(self, rows):
        self.w.writerows(rows)
        self.fh.flush()

    def close(self):
        self.fh.close()


class Skipped(Exception):
    """A designed reason not to analyse a recording (not a failure)."""


class Progress:
    """Logs at most once a minute, so a long step never looks hung."""

    def __init__(self, rid: str, what: str, total_s: float):
        self.rid, self.what, self.total, self.last = rid, what, total_s, time.monotonic()

    def __call__(self, done_s: float):
        if time.monotonic() - self.last >= 60:
            self.last = time.monotonic()
            log.info("%s: %s %.0f%% (%.1f of %.1f h)", self.rid, self.what, 100 * done_s / max(self.total, 1),
                     done_s / 3600, self.total / 3600)


def _num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if not np.isfinite(f) else f"{f:.4f}".rstrip("0").rstrip(".")


def sleep_stats(stages: list[str], conf: np.ndarray, e0: int, e1: int) -> dict[str, float]:
    """YASA's sleep statistics over epochs [e0, e1) (lights off to lights on, else the recording)."""
    import yasa

    vals = [{"W": "WAKE", "R": "REM"}.get(s, s) for s in stages[e0:e1]]
    stats = dict(yasa.Hypnogram(vals, freq="30s", n_stages=5).sleep_statistics())
    c = conf[e0:e1]
    stats["confidence_mean"] = float(100 * c.mean()) if len(c) else float("nan")
    stats["confidence_ge80_pct"] = float(100 * np.mean(c >= 0.8)) if len(c) else float("nan")
    stats["analysis_start_s"] = e0 * Z.EPOCH_S
    stats["analysis_end_s"] = e1 * Z.EPOCH_S
    return stats


def detection_channels(good: list[E.Channel], which: str) -> list[E.Channel]:
    out = [c for c in good if c.contacts and not set(c.contacts) & {"A1", "A2", "M1", "M2"}]
    if which == "frontal-central":
        out = [c for c in out if E.region(c.contacts) in ("frontal", "central", "frontal+central", "central+frontal")]
    return out


def process_recording(rec: E.Recording, params: dict, parallel, fig_dir: Path) -> dict:
    from joblib import delayed

    t0 = time.monotonic()
    raw = E.open_raw(rec.path)
    fs = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    duration = n_times / fs
    if duration < params["min_minutes"] * 60:
        raise Skipped(f"skipped: {duration / 60:.1f} min < MIN_SLEEP_MINUTES {params['min_minutes']:g}")
    if fs < 60:
        raise Skipped(f"skipped: sampling rate {fs:g} Hz < 60 Hz (staging uses 0.4-30 Hz)")
    rec.notes.extend(E.set_units(raw, rec))
    if not rec.start_time and raw.info.get("meas_date") is not None:
        rec.start_time = raw.info["meas_date"].replace(tzinfo=None).isoformat(timespec="seconds")
    hp, hp_where = E.highpass_hz(rec)
    if hp is not None and hp > 0.5:
        rec.notes.append(f"high-pass filter at {hp:g} Hz ({hp_where}): slow waves (0.3-1.5 Hz) and delta are "
                         f"attenuated, so N3 and slow waves are under-counted")
    if rec.age is not None and rec.age < 5:
        rec.notes.append(f"age {rec.age:g} y: YASA's classifiers were trained on recordings from age 5 up "
                         f"(Vallat & Walker 2021); staging in younger children is unreliable")
    chans = E.channels(rec, list(raw.ch_names), list(raw.get_channel_types()))
    good = [c for c in chans if c.status == "good"]
    if not good:
        raise ValueError(f"no good EEG channel among {len(chans)}")
    index = {name: i for i, name in enumerate(raw.ch_names)}
    montage = E.montage_of(rec, good)

    # 1. the staging derivation, streamed at 100 Hz
    if params["STAGING_CHANNEL"].lower() != "auto":
        cands = [Z.parse_requested(params["STAGING_CHANNEL"], good, index)]
    else:
        cands = Z.staging_candidates(good, index, montage)
    if not cands:
        raise Skipped("skipped: no central EEG channel or central bipolar pair to stage on "
                      f"(good channels: {', '.join(c.label for c in good)})")
    cands = cands[:3]
    xs = Z.stream_100hz(raw, cands, progress=Progress(rec.recording_id, "reading the staging channel", duration))
    chosen, why_not = None, []
    for k, d in enumerate(cands):
        x = xs[k]
        per = x[: len(x) // 3000 * 3000].reshape(-1, 3000)
        flat = float(np.mean(per.std(axis=1) < 1.0)) if len(per) else 1.0
        if flat > 0.5:
            why_not.append(f"{d.label}: {100 * flat:.0f}% of epochs flat")
            continue
        chosen, x100 = d, x
        break
    if chosen is None:
        raise ValueError("every staging candidate is flat: " + "; ".join(why_not))
    del xs
    meta = None
    if params["use_demo"]:
        if rec.age is not None and 0 < rec.age < 120 and rec.sex in ("M", "F"):
            meta = {"age": float(rec.age), "male": rec.sex == "M"}
        else:
            rec.notes.append("USE_DEMOGRAPHICS=true but age or sex unknown: staged with the EEG-only classifier")
    ts = time.monotonic()
    st = Z.stage(x100, chosen.label, meta)
    t_stage = time.monotonic() - ts
    n_ep = len(st.stages)

    # 2. time left out of detection, lights
    evs = E.events(rec)
    excluded = E.intervals(evs, E.EXCLUDED_TYPES, duration)
    n_sz = 0
    if params["exclude_seizure"]:
        sz = E.intervals(evs, E.SEIZURE_TYPES, duration)
        n_sz = len(sz)
        excluded += [(max(0.0, a - params["margin"]), min(duration, b + params["margin"])) for a, b in sz]
    excluded.sort()
    lo, lon = E.marker(evs, "lights_off"), E.marker(evs, "lights_on")
    e0 = int(lo // Z.EPOCH_S) if lo is not None and lo < n_ep * Z.EPOCH_S else 0
    e1 = int(np.ceil(lon / Z.EPOCH_S)) if lon is not None and lon > e0 * Z.EPOCH_S else n_ep
    e1 = min(e1, n_ep)
    if lo is None and lon is None:
        rec.notes.append("no lights_off/lights_on events: sleep statistics cover the whole recording")
    stats = sleep_stats(st.stages, st.confidence, e0, e1)

    hyp_rows = [{"recording_id": rec.recording_id, "epoch": i, "start_s": f"{i * Z.EPOCH_S:g}",
                 "duration_s": f"{Z.EPOCH_S:g}", "stage": s, "confidence": f"{st.confidence[i]:.3f}",
                 "channel": chosen.label} for i, s in enumerate(st.stages)]
    stat_rows = [{"recording_id": rec.recording_id, "metric": k, "value": _num(v)} for k, v in stats.items()]

    # 3. spindles and slow waves, per window, channels in parallel
    det = detection_channels(good, params["DETECT_CHANNELS"])
    lab = {c.label: index[c.name] for c in good}
    ears = [next((e for e in side if e in lab), None) for side in (("M1", "A1"), ("M2", "A2"))]
    if params["DETECT_REFERENCE"] == "auto" and montage != "bipolar-longitudinal" and all(ears):
        det_ref = f"linked {'mastoids' if ears[0][0] == 'M' else 'ears'} ({ears[0]}+{ears[1]})/2"
        derivs = [Z.Derivation(c.label, [(index[c.name], 1.0), (lab[ears[0]], -0.5), (lab[ears[1]], -0.5)], det_ref)
                  for c in det]
    else:
        det_ref = "bipolar pairs as recorded" if montage == "bipolar-longitudinal" else \
            f"as recorded ({rec.reference or montage or 'unknown'} reference)"
        derivs = [Z.Derivation(c.label, [(index[c.name], 1.0)], det_ref) for c in det]
    sp_rows, sw_rows, errors = [], [], {}
    minutes = {c.label: 0.0 for c in det}
    t_detect = 0.0
    prog = Progress(rec.recording_id, "spindles and slow waves", duration)
    for c0, c1, r0, r1 in Z.windows(n_ep * Z.EPOCH_S, params["window"], PAD_S):
        if not det:
            break
        tw = time.monotonic()
        xw = Z.stream_100hz(raw, derivs, r0, r1)
        i0 = int(round(r0 * Z.SF))
        n = xw.shape[1]
        full = Z.sample_mask(st.stages, i0 + n, excluded)
        hyp = full[i0:i0 + n]
        core = hyp[int((c0 - r0) * Z.SF):int((c1 - r0) * Z.SF)]
        core_min = float(np.count_nonzero(np.isin(core, (2, 3)))) / Z.SF / 60
        res = parallel(delayed(Z.detect_channel)(xw[k], hyp, i0 / Z.SF, (c0, c1), d.label,
                                                 params["spindle_freq"], params["sw_ptp"])
                       for k, d in enumerate(derivs))
        del xw
        for d, (sp, sw, err) in zip(derivs, res):
            if err:
                errors.setdefault(d.label, f"{c0:.0f}-{c1:.0f} s: {err}")
                continue
            minutes[d.label] += core_min
            sp_rows += [{"recording_id": rec.recording_id, "channel": d.label, "start_s": f"{a:.3f}",
                         "end_s": f"{b:.3f}", "duration_s": f"{du:.3f}", "frequency_hz": f"{fr:.2f}",
                         "amplitude_uv": f"{am:.1f}", "stage": s, "_t": a} for a, b, du, fr, am, s in sp]
            sw_rows += [{"recording_id": rec.recording_id, "channel": d.label, "start_s": f"{a:.3f}",
                         "end_s": f"{b:.3f}", "amplitude_uv": f"{am:.1f}", "frequency_hz": f"{fr:.2f}", "stage": s,
                         "_t": a} for a, b, am, fr, s in sw]
        t_detect += time.monotonic() - tw
        prog(c1)
    sp_rows.sort(key=lambda r: (r["_t"], r["channel"]))
    sw_rows.sort(key=lambda r: (r["_t"], r["channel"]))

    feat_rows, sp_dens, sw_dens = [], {}, {}
    for d in derivs:
        m = minutes[d.label]
        sps = [r for r in sp_rows if r["channel"] == d.label]
        sws = [r for r in sw_rows if r["channel"] == d.label]
        vals = {
            "spindle_density": len(sps) / m if m > 0 else None,
            "spindle_amplitude": np.mean([float(r["amplitude_uv"]) for r in sps]) if sps else None,
            "spindle_frequency": np.mean([float(r["frequency_hz"]) for r in sps]) if sps else None,
            "sw_density": len(sws) / m if m > 0 else None,
            "sw_amplitude": np.mean([float(r["amplitude_uv"]) for r in sws]) if sws else None,
            "nrem_minutes_analysed": m,
        }
        if vals["spindle_density"] is not None:
            sp_dens[d.label], sw_dens[d.label] = vals["spindle_density"], vals["sw_density"]
        feat_rows += [{"recording_id": rec.recording_id, "channel": d.label, "feature": k,
                       "value": "" if v is None else _num(v)} for k, v in vals.items()]

    # 4. figures
    figs = []
    lights = (lo, lon) if (lo is not None or lon is not None) else None
    rid = rec.recording_id
    try:
        p = fig_dir / f"{rid}_hypnogram.png"
        F.hypnogram(p, rid, st.stages, st.proba, st.confidence, chosen.label, rec.start_time, lights, excluded)
        figs.append(("hypnogram", p))
        p = fig_dir / f"{rid}_spectrogram.png"
        F.spectrogram(p, rid, x100, chosen.label, rec.start_time)
        figs.append(("spectrogram", p))
        p = fig_dir / f"{rid}_density.png"
        if F.density_map(p, rid, [(d.label, c.contacts) for d, c in zip(derivs, det)], sp_dens, sw_dens,
                         montage == "bipolar-longitudinal"):
            figs.append(("density", p))
        else:
            rec.notes.append("density map not drawn: fewer than 4 detection channels with a 10-20 position")
    except Exception as exc:  # noqa: BLE001 - a figure must not fail the recording
        rec.notes.append(f"figure failed: {type(exc).__name__}: {exc}")
        log.warning("%s: figure failed: %s\n%s", rid, exc, traceback.format_exc())

    def dens(rows, region_names):
        ch = [d.label for d, c in zip(derivs, det) if E.region(c.contacts) in region_names and minutes[d.label] > 0]
        if not ch:
            return None
        return float(np.mean([sum(1 for r in rows if r["channel"] == x) / minutes[x] for x in ch]))

    head = {
        "duration_h": round(duration / 3600, 3), "epochs": n_ep,
        "TST_min": stats.get("TST"), "SE_pct": stats.get("SE"), "SOL_min": stats.get("SOL"),
        "WASO_min": stats.get("WASO"), "N1_pct": stats.get("%N1"), "N2_pct": stats.get("%N2"),
        "N3_pct": stats.get("%N3"), "REM_pct": stats.get("%REM"),
        "confidence_mean_pct": stats["confidence_mean"], "confidence_ge80_pct": stats["confidence_ge80_pct"],
        "n_spindles": len(sp_rows), "n_slow_waves": len(sw_rows),
        "spindle_density_central": dens(sp_rows, ("central",)),
        "sw_density_frontal": dens(sw_rows, ("frontal",)),
        "nrem_minutes_analysed": round(max(minutes.values()), 2) if minutes else 0.0,
    }
    head = {k: (round(float(v), 3) if v is not None and np.isfinite(float(v)) else None) for k, v in head.items()}
    summary = {
        "status": "ok", "sfreq": fs, "duration_s": round(duration, 3), "montage": montage,
        "staging_channel": chosen.label, "staging_derivation": chosen.how, "classifier": st.classifier,
        "staging_alternatives": [d.label for d in cands], "staging_skipped": why_not,
        "demographics": meta, "n_channels": len(chans), "n_good": len(good),
        "detection_channels": [d.label for d in derivs], "detection_reference": det_ref, "excluded": [[round(a, 1), round(b, 1)] for a, b in excluded],
        "n_seizures_excluded": n_sz, "highpass_hz": hp, "lights": [lo, lon], "stats_epochs": [e0, e1],
        "channels_not_analysed": {c.label: c.reason for c in chans if c.status != "good"},
        "channel_errors": errors, "notes": rec.notes,
        "seconds": {"staging": round(t_stage, 2), "detection": round(t_detect, 2),
                    "total": round(time.monotonic() - t0, 2)},
    }
    return {"hypnogram.csv": hyp_rows, "sleep_stats.csv": stat_rows, "spindles.csv": sp_rows,
            "slow_waves.csv": sw_rows, "features.csv": feat_rows, "summary": summary, "head": head, "figures": figs,
            "start_time": rec.start_time}


def captions(rid: str, kind: str, s: dict, h: dict) -> str:
    ch = s["staging_channel"]
    if kind == "hypnogram":
        return (f"{rid}: automatic sleep stages per 30-s epoch (YASA, EEG only, on {ch}; {s['staging_derivation']}). "
                f"Top: the hypnogram (REM in red; grey ticks mark epochs scored with less than 80 % confidence, "
                f"orange bands are bad segments or seizures left out of spindle and slow-wave detection). "
                f"Bottom: the classifier's probability for each stage. Look for a plausible night: wake at the "
                f"start, N2 and N3 early, REM later and in cycles. Long stretches of low confidence or "
                f"implausible stages (REM straight from wake, N3 in a seizure) need review; mean confidence "
                f"{h['confidence_mean_pct']}%. Research output, not a clinical sleep scoring: review it against "
                f"the EEG.")
    if kind == "spectrogram":
        return (f"{rid}: spectrogram of the staging channel {ch} (30-s epochs, 0.5-25 Hz, log power). Sleep shows "
                f"as broad slow-wave power (red at the bottom) in N3 and sigma bands (12-15 Hz) in N2; wake and "
                f"REM as low slow-wave power with more alpha/beta. Vertical stripes across all frequencies are "
                f"movement or electrode artifacts. Use it to check that the hypnogram follows the EEG.")
    return (f"{rid}: spindles (left) and slow waves (right) per minute of N2+N3 sleep on each channel"
            f"{' (bipolar pairs drawn at their midpoints)' if s['montage'] == 'bipolar-longitudinal' else ''}. "
            f"Spindles are usually highest over the centre and slow waves over the front of the head. A clear "
            f"left-right difference in spindle density can accompany a lesion or a focal epilepsy, but also a bad "
            f"electrode: check the channel's signal. {h['nrem_minutes_analysed']} min of N2+N3 analysed. "
            f"Research output to be reviewed.")


def main() -> int:
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    try:
        params = env_params()
    except ValueError as exc:
        log.error("bad parameter: %s", exc)
        return 2
    out_dir = output_dir / APP / params["LABEL"] if params["LABEL"] else output_dir / APP
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    recs, seen, problems = E.find_recordings(input_dir)
    log.info("input: %d recording(s), %d file(s); staging channel %s, %s, detection on %s channels, "
             "seizures %s, window %g s", len(recs), len(seen), params["STAGING_CHANNEL"],
             "EEG + age/sex" if params["use_demo"] else "EEG only", params["DETECT_CHANNELS"],
             f"excluded (margin {params['margin']:g} s)" if params["exclude_seizure"] else "kept", params["window"])
    log.info("parallel workers: %d (%s)", params["threads"], params["threads_why"])
    for p in problems:
        log.warning("%s: %s", p["input"], p["reason"])

    outs = {name: CsvOut(out_dir / name, header) for name, header in HEADERS.items()}
    prov_recs: dict[str, dict] = {}
    for p in problems:
        prov_recs[p["recording_id"] or p["input"]] = {"input": p["input"], "status": p.get("status", "failed"),
                                                      "error": p["reason"]}
    n_ok = n_skipped = 0
    n_failed = sum(1 for p in problems if p.get("status", "failed") == "failed")
    heads, figures, warnings_ = {}, [], []

    from joblib import Parallel

    with Parallel(n_jobs=params["threads"], backend="loky", max_nbytes=None) as parallel:
        for rec in recs:
            try:
                res = process_recording(rec, params, parallel, fig_dir)
            except Skipped as why:
                log.warning("%s: %s", rec.recording_id, why)
                prov_recs[rec.recording_id] = {"input": rec.rel, "status": str(why), "notes": rec.notes}
                warnings_.append(f"{rec.recording_id}: {why}")
                n_skipped += 1
                continue
            except Exception as exc:  # noqa: BLE001 - one recording's failure must not sink the run
                log.error("%s: FAILED: %s\n%s", rec.recording_id, exc, traceback.format_exc())
                prov_recs[rec.recording_id] = {"input": rec.rel, "status": "failed",
                                               "error": f"{type(exc).__name__}: {exc}"}
                warnings_.append(f"{rec.recording_id}: failed: {type(exc).__name__}: {exc}")
                n_failed += 1
                continue
            for name, out in outs.items():
                out.rows(res[name])
            n_ok += 1
            s, h = res["summary"], res["head"]
            heads[rec.recording_id] = h
            for kind, path in res["figures"]:
                figures.append({"path": f"figures/{path.name}", "caption": captions(rec.recording_id, kind, s, h)})
            for ch, err in s["channel_errors"].items():
                log.warning("%s: channel %s: %s", rec.recording_id, ch, err)
            for note in s["notes"]:
                log.info("%s: %s", rec.recording_id, note)
            for note in s["notes"]:
                if note.startswith("high-pass filter at"):
                    warnings_.append(f"{rec.recording_id}: {note}")
            if rec.age is not None and rec.age < 5:
                warnings_.append(f"{rec.recording_id}: age {rec.age:g} y, below the age range of YASA's training "
                                 f"data (5 y and up): the hypnogram is unreliable")
            if h["confidence_mean_pct"] is not None and h["confidence_mean_pct"] < 70:
                warnings_.append(f"{rec.recording_id}: mean staging confidence {h['confidence_mean_pct']:.0f}% "
                                 f"(< 70%): review the hypnogram")
            if s["montage"] != "bipolar-longitudinal" and not s["staging_channel"].endswith(("M1", "M2", "A1", "A2")):
                warnings_.append(f"{rec.recording_id}: staged on {s['staging_channel']}, not on a central-mastoid "
                                 f"derivation; YASA's accuracy is reported for C4-M1-like channels")
            if s["montage"] == "bipolar-longitudinal":
                warnings_.append(f"{rec.recording_id}: bipolar-only recording, staged on {s['staging_channel']}; "
                                 f"YASA was trained on central-mastoid derivations")
            log.info("%s: %d channels (%d good), %.0f Hz, %s, %.2f h, %d epochs staged on %s (TST %s min, SE %s%%, "
                     "mean confidence %s%%), %d spindles and %d slow waves on %d channels over %s min of N2+N3; "
                     "%.1f s (staging %.1f s, detection %.1f s)", rec.recording_id, s["n_channels"], s["n_good"],
                     s["sfreq"], s["montage"], h["duration_h"], h["epochs"], s["staging_channel"], h["TST_min"],
                     h["SE_pct"], h["confidence_mean_pct"], h["n_spindles"], h["n_slow_waves"],
                     len(s["detection_channels"]), h["nrem_minutes_analysed"], s["seconds"]["total"],
                     s["seconds"]["staging"], s["seconds"]["detection"])
            prov_recs[rec.recording_id] = {"input": rec.rel, **s}
    for out in outs.values():
        out.close()

    status = "ok"
    if not recs and not problems:
        log.error("no recording found under INPUT_DIR (looked for recordings.csv and %s files); it holds %d file(s): %s",
                  "/".join(E.SIGNAL_EXTENSIONS), len(seen), ", ".join(seen[:40]) + (" …" if len(seen) > 40 else ""))
        status = "failed: no input"
    elif n_ok == 0 and n_skipped == 0 and n_failed > 0:
        status = "failed: every recording failed"
    elif n_ok == 0:
        status = "ok: nothing analysed (" + f"{n_skipped} skipped" + (f", {n_failed} failed" if n_failed else "") + ")"

    tables = [
        {"path": "hypnogram.csv", "caption": "Sleep stage (W, N1, N2, N3, R) and the classifier's confidence per 30-s "
                                             "epoch, with the channel it was staged on. Read by eeg-spikes."},
        {"path": "sleep_stats.csv", "caption": "Sleep statistics per recording (YASA, AASM definitions): TIB, SPT, "
                                               "TST, WASO, SOL in minutes; SE, SME and %N1-%REM in percent; stage "
                                               "minutes and latencies (from lights off, or the recording start)."},
        {"path": "spindles.csv", "caption": "Every sleep spindle detected in N2/N3 per channel (YASA spindles_detect)."},
        {"path": "slow_waves.csv", "caption": "Every slow wave detected in N2/N3 per channel (YASA sw_detect)."},
        {"path": "features.csv", "caption": "Per channel: spindle and slow-wave density (per min of N2+N3), mean "
                                            "amplitude and spindle frequency."},
    ]
    finished = datetime.now(timezone.utc)
    summary = {"app": APP, "version": version_of(REPO_ROOT), "recordings": heads, "figures": figures,
               "tables": tables, "warnings": warnings_}
    prov = {
        "app": APP, "version": version_of(REPO_ROOT), "variant": None, "tools": tool_versions(),
        "parameters": {k: params[k] for k in PARAM_NAMES},
        "inputs": seen,
        "started": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "finished": finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 2),
        "peak_rss_mb": peak_rss_mb(),
        "threads": {"requested": int(params["THREADS"]), "used": params["threads"], "why": params["threads_why"]},
        "recordings": prov_recs,
        "citations": CITATIONS,
        "licence_notice": None,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(out_dir / "provenance.json", "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2, default=str)
    log.info("status: %s", status)
    log.info("done: %d recording(s) ok, %d skipped, %d failed, %.1f s, peak RSS %.0f MB -> %s", n_ok, n_skipped,
             n_failed, prov["wall_seconds"], prov["peak_rss_mb"], out_dir)
    return 1 if status.startswith("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
