"""
Sleep staging, spindles and slow waves with YASA (Vallat & Walker 2021), on recordings read a
window at a time.

- `stream_100hz` reads derivations (a channel, or a weighted sum of channels such as C4 - A1)
  in chunks at the native rate and resamples them to 100 Hz with a polyphase filter, so a 24 h
  recording never sits in memory at its native rate.
- `choose_staging` picks the EEG derivation YASA stages on: a central electrode referred to the
  contralateral mastoid or ear (C4-M1, C3-M2, as YASA's classifiers were trained) when both
  electrodes are in the recording, else the fallbacks documented in README "Staging channel".
- `stage` runs `yasa.SleepStaging` (EEG only by default) with a classifier loaded from a
  read-only folder (`YASA_CLASSIFIER_DIR`, `/opt/eeg-sleep/classifiers` in the image).
- `detect_channel` runs `yasa.spindles_detect` and `yasa.sw_detect` on one channel of one
  window, restricted to N2/N3 epochs outside excluded time.
"""
from __future__ import annotations

import glob
import logging
import math
import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from processor import eeg as E

log = logging.getLogger(__name__)

SF = 100.0                      # YASA stages at 100 Hz; detection runs at 100 Hz too
EPOCH_S = 30.0
STAGE_CODES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
YASA_TO_STAGE = {"WAKE": "W", "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "REM": "R", "R": "R"}
CODE_TO_STAGE = {v: k for k, v in STAGE_CODES.items()}
EXCLUDED = -1                   # YASA's "Art" code: not detected in
UNSCORED = -2

MASTOIDS = {"C4": ("M1", "A1"), "C3": ("M2", "A2")}
# Fallbacks (README "Staging channel"), in order, after C4-M1 / C3-M2.
REFERENTIAL_FALLBACK = [("C4", None), ("C3", None), ("Cz", None)]
BIPOLAR_FALLBACK = [("F4", "C4", "P4"), ("F3", "C3", "P3"), ("Fz", "Cz", "Pz"),
                    ("C4", "P4", None), ("C3", "P3", None), ("F4", "C4", None), ("F3", "C3", None),
                    ("Cz", "Pz", None), ("Fz", "Cz", None)]


# --- derivations and resampling ----------------------------------------------------

@dataclass
class Derivation:
    label: str                         # e.g. "C4-A1", "C4", "F4-P4"
    terms: list[tuple[int, float]]     # (channel index in the raw file, weight)
    how: str                           # human-readable, for provenance


def ratio(fs: float) -> tuple[int, int]:
    """(up, down) with up/down = 100/fs."""
    fr = Fraction(SF / fs).limit_denominator(4096)
    return fr.numerator, fr.denominator


def origin_100hz(fs: float, t0: float) -> int:
    """Index, on the 100 Hz grid of the whole recording, of the first sample stream_100hz returns
    for a window starting at t0 s: the window start is snapped down to a multiple of the polyphase
    `down` factor, so event times must be offset by this, not by round(t0 * 100)."""
    up, down = ratio(fs)
    return int(math.floor(t0 * fs / down)) * up


def stream_100hz(raw, derivs: list[Derivation], t0: float = 0.0, t1: float | None = None,
                 chunk_s: float = 600.0, pad_s: float = 10.0, progress=None) -> np.ndarray:
    """
    The derivations over [t0, t1) seconds at 100 Hz, float32 µV, shape (len(derivs), n). Read in
    chunks of ~chunk_s (all channels the derivations need at once) with pad_s of context on each
    side, resampled with scipy's polyphase filter, trimmed back to the chunk.
    """
    from scipy.signal import resample_poly

    fs = float(raw.info["sfreq"])
    up, down = ratio(fs)
    n_total = int(raw.n_times)
    s0 = int(math.floor(t0 * fs / down)) * down   # sample origin_100hz() refers to
    s1 = n_total if t1 is None else min(n_total, int(round(t1 * fs)))
    n_out = int(math.ceil((s1 - s0) * up / down))
    out = np.zeros((len(derivs), n_out), dtype=np.float32)
    picks = sorted({i for d in derivs for i, _ in d.terms})
    col = {p: k for k, p in enumerate(picks)}
    chunk = max(down, int(chunk_s * fs) // down * down)
    pad = max(down, int(pad_s * fs) // down * down)
    pos = s0
    while pos < s1:
        a, b = pos, min(s1, pos + chunk)
        ra, rb = max(0, a - pad), min(n_total, b + pad)
        block = E.read_window(raw, picks, ra, rb)
        x = np.zeros((len(derivs), rb - ra))
        for k, d in enumerate(derivs):
            for i, w in d.terms:
                x[k] += w * block[col[i]]
        del block
        y = resample_poly(x, up, down, axis=1, padtype="line")
        o0 = (a - ra) * up // down
        n = min(int(math.ceil((b - a) * up / down)), y.shape[1] - o0)
        dst = (a - s0) * up // down
        n = min(n, n_out - dst)
        out[:, dst:dst + n] = y[:, o0:o0 + n]
        pos = b
        if progress:
            progress(b / fs)
    return out


def _by_label(good: list[E.Channel], index: dict[str, int]) -> dict[str, int]:
    return {c.label: index[c.name] for c in good}


def parse_requested(req: str, good: list[E.Channel], index: dict[str, int]) -> Derivation:
    """STAGING_CHANNEL given by the user: a channel label (`C4`, `C4-P4`), or `X-Y` built as the
    difference of two referential channels."""
    lab = _by_label(good, index)
    names = {k.lower(): k for k in lab}
    if req.lower() in names:
        k = names[req.lower()]
        return Derivation(k, [(lab[k], 1.0)], f"{k} as recorded (STAGING_CHANNEL)")
    parts = req.split("-")
    if len(parts) == 2:
        a, b = (E.electrode(p) or p for p in parts)
        if a in lab and b in lab:
            return Derivation(f"{a}-{b}", [(lab[a], 1.0), (lab[b], -1.0)], f"{a} minus {b} (STAGING_CHANNEL)")
    raise ValueError(f"STAGING_CHANNEL {req!r} is not a good channel of this recording, nor a difference of two "
                     f"(good channels: {', '.join(sorted(lab))})")


def staging_candidates(good: list[E.Channel], index: dict[str, int], montage: str) -> list[Derivation]:
    """Every derivation YASA could stage on, best first (README "Staging channel")."""
    lab = _by_label(good, index)
    out: list[Derivation] = []
    if montage != "bipolar-longitudinal":
        for c, refs in MASTOIDS.items():
            for r in refs:
                if c in lab and r in lab:
                    out.append(Derivation(f"{c}-{r}", [(lab[c], 1.0), (lab[r], -1.0)],
                                          f"{c} minus {r} (contralateral {'mastoid' if r[0] == 'M' else 'ear'}, "
                                          f"as YASA's training data)"))
                    break
        for c, r in REFERENTIAL_FALLBACK:
            if c in lab:
                out.append(Derivation(c, [(lab[c], 1.0)], f"{c} in the recording's reference "
                                      f"({'average' if montage in ('', 'average', 'referential') else montage})"))
    else:
        for a, b, c in BIPOLAR_FALLBACK:
            if c is not None:
                p1, p2 = f"{a}-{b}", f"{b}-{c}"
                if p1 in lab and p2 in lab:
                    out.append(Derivation(f"{a}-{c}", [(lab[p1], 1.0), (lab[p2], 1.0)],
                                          f"{a}-{c} = ({p1}) + ({p2}): the two central pairs of the bipolar chain "
                                          f"summed into one longer derivation"))
            else:
                p = f"{a}-{b}"
                if p in lab:
                    out.append(Derivation(p, [(lab[p], 1.0)], f"bipolar pair {p} as recorded"))
    return out


# --- staging ----------------------------------------------------------------------

def classifier_path(use_demo: bool) -> Path:
    """The newest bundled LightGBM classifier for EEG only (+demo), from YASA_CLASSIFIER_DIR
    (read-only copy made at image build time) or else from the installed yasa package."""
    d = os.environ.get("YASA_CLASSIFIER_DIR")
    if not d or not os.path.isdir(d):
        import yasa

        d = os.path.join(os.path.dirname(yasa.__file__), "classifiers")
    name = "clf_eeg+demo" if use_demo else "clf_eeg"
    files = glob.glob(os.path.join(glob.escape(d), glob.escape(name) + "_lgb_*.joblib"))
    if not files:
        raise FileNotFoundError(f"no {name}_lgb_*.joblib in {d}")

    def version(f):
        return tuple(int(v) for v in re.findall(r"_lgb_([\d.]+)\.joblib$", f)[0].split("."))
    return Path(max(files, key=version))


@dataclass
class Staging:
    stages: list[str]          # W/N1/N2/N3/R per 30 s epoch
    confidence: np.ndarray     # max probability per epoch
    proba: "object"            # pandas DataFrame, columns W N1 N2 N3 R
    classifier: str


def stage(x100: np.ndarray, label: str, metadata: dict | None) -> Staging:
    """YASA SleepStaging on one 100 Hz trace in µV."""
    import mne
    import yasa

    info = mne.create_info([label], SF, ch_types=["eeg"])
    raw = mne.io.RawArray(x100[None, :].astype(np.float64) * 1e-6, info, verbose="error")
    sls = yasa.SleepStaging(raw, eeg_name=label, metadata=metadata)
    clf = classifier_path(metadata is not None)
    hyp = sls.predict(path_to_model=str(clf))
    stages = [YASA_TO_STAGE[str(v).upper()] for v in hyp.hypno.to_numpy()]
    proba = hyp.proba.copy()
    proba.columns = [YASA_TO_STAGE[str(c).upper()] for c in proba.columns]
    proba = proba.reindex(columns=["W", "N1", "N2", "N3", "R"]).fillna(0.0).reset_index(drop=True)
    return Staging(stages, proba.max(axis=1).to_numpy(), proba, clf.name)


def sample_mask(stages: list[str], n: int, excluded: list[tuple[float, float]]) -> np.ndarray:
    """Per-sample (100 Hz) stage code over n samples: YASA's integer hypnogram, excluded time = -1,
    beyond the last full epoch = -2."""
    m = np.full(n, UNSCORED, dtype=np.int8)
    per = int(EPOCH_S * SF)
    codes = np.array([STAGE_CODES[s] for s in stages], dtype=np.int8)
    k = min(len(codes) * per, n)
    m[:k] = np.repeat(codes, per)[:k]
    for a, b in excluded:
        m[max(0, int(a * SF)):max(0, min(n, int(math.ceil(b * SF))))] = EXCLUDED
    return m


# --- detection --------------------------------------------------------------------

def detect_channel(x: np.ndarray, hyp: np.ndarray, t_offset: float, core: tuple[float, float], name: str,
                   spindle_freq: tuple[float, float], sw_ptp: tuple[float, float]):
    """
    Spindles and slow waves on one channel of one window (100 Hz, µV), in N2/N3 samples only.
    Returns (spindles, slow_waves, error); events are kept when they start inside `core`
    (seconds from the recording start).
    """
    import logging as _l

    import yasa

    _l.getLogger("yasa").setLevel(_l.ERROR)
    sp_rows, sw_rows = [], []
    try:
        if np.count_nonzero(np.isin(hyp, (2, 3))) < 30 * SF:
            return sp_rows, sw_rows, None
        x = x.astype(np.float64)
        sp = yasa.spindles_detect(x, sf=SF, ch_names=[name], hypno=hyp, include=(2, 3), freq_sp=spindle_freq,
                                  verbose="error")
        if sp is not None:
            for r in sp.summary().itertuples(index=False):
                t = float(r.Start) + t_offset
                if core[0] <= t < core[1]:
                    sp_rows.append((t, float(r.End) + t_offset, float(r.Duration), float(r.Frequency),
                                    float(r.Amplitude), CODE_TO_STAGE.get(int(r.Stage), "")))
        sw = yasa.sw_detect(x, sf=SF, ch_names=[name], hypno=hyp, include=(2, 3), amp_ptp=sw_ptp, verbose="error")
        if sw is not None:
            for r in sw.summary().itertuples(index=False):
                t = float(r.Start) + t_offset
                if core[0] <= t < core[1]:
                    sw_rows.append((t, float(r.End) + t_offset, float(r.PTP), float(r.Frequency),
                                    CODE_TO_STAGE.get(int(r.Stage), "")))
    except Exception as exc:  # noqa: BLE001 - one channel-window must not sink the recording
        return sp_rows, sw_rows, f"{type(exc).__name__}: {exc}"
    return sp_rows, sw_rows, None


def windows(duration: float, window_s: float, pad_s: float):
    """(core_start, core_end, read_start, read_end) covering [0, duration) in near-equal windows."""
    n = max(1, math.ceil(duration / window_s - 1e-9))
    step = duration / n
    for i in range(n):
        a, b = i * step, (i + 1) * step
        yield a, b, max(0.0, a - pad_s), min(duration, b + pad_s)
