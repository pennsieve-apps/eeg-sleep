"""
A small synthetic BIDS-EEG derivative that follows workflow-scalp-eeg/docs/conventions.md (the
`eeg` output of eeg-prep), for the tests, `make run` and the release review (`review.generate`):

  <out>/eeg/dataset_description.json, participants.tsv, recordings.csv
  <out>/eeg/sub-01/eeg/sub-01_task-sleep_run-01_desc-preproc_{eeg.edf,eeg.json,channels.tsv,events.tsv}
      65 min, 256 Hz, 19 10-20 channels + A1/A2, average reference (referential)
  <out>/eeg/sub-02/eeg/sub-02_task-sleep_run-01_desc-preproc_{...}
      65 min, 128 Hz, the 18 pairs of the longitudinal bipolar montage only (CHB-MIT style)
  <out>/eeg/sub-03/eeg/sub-03_task-routine_run-01_desc-preproc_{...}
      5 min, 256 Hz, referential: shorter than MIN_SLEEP_MINUTES (60), skipped
  <out>/truth.json   planted spindle and slow-wave times, the sleep-like segments

The "night" is a sequence of sleep-like segments (minutes): 0-8 wake (posterior alpha), 8-12
N1-like (theta), 12-30 N2-like (spindles every ~10 s), 30-48 N3-like (slow waves), 48-56 REM-like
(low-voltage mixed frequency), 56-65 N2-like. Spindles (13 Hz, 1 s, 60 µV peak at C3/C4/Cz) and
slow waves (single 0.8 Hz waves, -100/+60 µV at frontal electrodes) are planted at known times,
with a scalp distribution (central spindles, frontal slow waves) before re-referencing. YASA's
classifier is not expected to stage synthetic data correctly: tests check the output contract, and
check detection against the planted events with a given hypnogram.

sub-01 also has a `bad_segment` (20:00-20:30) and a `seizure` (60:00-61:00) with planted spindles
inside both, which must not be reported, and lights_off / lights_on events at 1 min and 64 min.

Usage: python -m tests.make_fixture <out-dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

SEED = 42
REF10_20 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
            "O1", "O2"]
EARS = ["A1", "A2"]
BIPOLAR = [("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"), ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"),
           ("P3", "O1"), ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"), ("Fp2", "F8"), ("F8", "T8"),
           ("T8", "P8"), ("P8", "O2"), ("Fz", "Cz"), ("Cz", "Pz")]
SEGMENTS = [(0, 8, "W"), (8, 12, "N1"), (12, 30, "N2"), (30, 48, "N3"), (48, 56, "R"), (56, 65, "N2")]
CENTRAL = ("C3", "C4", "Cz")


def gain(kind: str, ch: str) -> float:
    if ch in EARS:
        return 0.0 if kind != "sw" else 0.05
    if kind == "spindle":
        return {"C": 1.0, "F": 0.6, "P": 0.6}.get(ch[0], 0.15) if not ch.startswith("Fp") else 0.3
    if kind == "sw":
        return 1.0 if ch[0] == "F" else {"C": 0.8, "P": 0.5}.get(ch[0], 0.3)
    if kind == "alpha":
        return {"O": 1.0, "P": 0.6}.get(ch[0], 0.15) * (0.8 if ch in ("P7", "P8") else 1.0)
    return 1.0


def pink(n: int, rng: np.random.Generator, std: float) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / x.std() * std


def band_noise(n: int, fs: float, lo: float, hi: float, std: float, rng) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / (x.std() + 1e-12) * std


def spindle(fs: float, amp: float = 60.0, freq: float = 13.0, dur: float = 1.0) -> np.ndarray:
    t = np.arange(int(dur * fs)) / fs
    return amp * np.hanning(len(t)) * np.sin(2 * np.pi * freq * t)


def slow_wave(fs: float, neg: float = 100.0, pos: float = 60.0, freq: float = 0.8) -> np.ndarray:
    """One slow wave: a negative half-wave then a positive one, 1/freq s in all."""
    half = int(fs / freq / 2)
    t = np.arange(half) / half
    return np.concatenate([-neg * np.sin(np.pi * t), pos * np.sin(np.pi * t)])


def add_at(x: np.ndarray, fs: float, t: float, shape: np.ndarray, g: float = 1.0) -> None:
    i = int(round(t * fs))
    j = min(len(x), i + len(shape))
    if 0 <= i < len(x):
        x[i:j] += g * shape[: j - i]


def night(fs: float, minutes: float, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict]:
    """Potentials of the 21 electrodes against an ideal reference, and the planted events."""
    n = int(minutes * 60 * fs)
    chans = REF10_20 + EARS
    x = {c: pink(n, rng, 8.0) for c in chans}
    common = pink(n, rng, 4.0)
    for c in chans:
        x[c] += common
    truth = {"spindles": [], "slow_waves": [], "segments": [[a * 60, b * 60, s] for a, b, s in SEGMENTS if a < minutes]}
    for a, b, st in SEGMENTS:
        if a >= minutes:
            break
        i0, i1 = int(a * 60 * fs), min(n, int(b * 60 * fs))
        m = i1 - i0
        if st == "W":
            alpha = band_noise(m, fs, 9, 11, 20.0, rng)
            beta = band_noise(m, fs, 15, 25, 4.0, rng)
            for c in chans:
                x[c][i0:i1] += gain("alpha", c) * alpha + beta * (c not in EARS)
        elif st == "N1":
            th = band_noise(m, fs, 4, 7, 15.0, rng)
            for c in chans:
                x[c][i0:i1] += th * (c not in EARS)
        elif st in ("N2", "N3"):
            de = band_noise(m, fs, 0.5, 4, 25.0 if st == "N2" else 45.0, rng)
            th = band_noise(m, fs, 4, 7, 10.0, rng)
            for c in chans:
                x[c][i0:i1] += gain("sw", c) * de + th * (c not in EARS)
            if st == "N2":
                t = a * 60 + 5.0
                while t < b * 60 - 5:
                    truth["spindles"].append(round(t, 3))
                    t += rng.uniform(8.0, 12.0)
            else:
                t = a * 60 + 3.0
                while t < b * 60 - 3:
                    truth["slow_waves"].append(round(t, 3))
                    t += rng.uniform(2.0, 3.5)
        elif st == "R":
            th = band_noise(m, fs, 2, 7, 12.0, rng)
            for c in chans:
                x[c][i0:i1] += th * (c not in EARS)
    # Spindles inside the bad segment and the seizure of sub-01 (must be left out)
    extra = [1205.0, 1215.0, 3605.0, 3620.0, 3640.0] if minutes >= 61 else []
    truth["excluded_spindles"] = extra
    sp, sw = spindle(fs), slow_wave(fs)
    for t in truth["spindles"] + extra:
        for c in chans:
            add_at(x[c], fs, t, sp, gain("spindle", c))
    for t in truth["slow_waves"]:
        for c in chans:
            add_at(x[c], fs, t, sw, gain("sw", c))
    return x, truth


def write_edf(path: Path, data: np.ndarray, names: list[str], fs: float) -> None:
    from edfio import Edf, EdfSignal

    sigs = []
    for x, name in zip(data, names):
        r = max(1.0, float(np.max(np.abs(x))) * 1.1)
        sigs.append(EdfSignal(x, sampling_frequency=fs, label=name, physical_dimension="uV", physical_range=(-r, r)))
    Edf(sigs).write(path)


def write_tsv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


CHANNEL_COLS = ["name", "type", "units", "low_cutoff", "high_cutoff", "sampling_frequency", "status",
                "status_description", "contacts", "hemisphere", "region", "x", "y", "z", "original_name"]
EVENT_COLS = ["onset", "duration", "trial_type", "source", "channels"]
REC_COLS = ["recording_id", "subject", "session", "task", "run", "path", "sfreq", "n_channels", "n_good", "duration_s",
            "montage", "reference", "kind", "start_time", "n_seizures", "source_format", "age", "sex"]


def write_recording(root: Path, sub: str, task: str, fs: float, data: np.ndarray, names: list[str], montage: str,
                    contacts: list[str], events=(), bad=(), start_time="", age="", sex="") -> dict:
    stem = f"sub-{sub}_task-{task}_run-01"
    d = root / f"sub-{sub}" / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    write_edf(d / f"{stem}_desc-preproc_eeg.edf", data, names, fs)
    write_tsv(d / f"{stem}_desc-preproc_channels.tsv", CHANNEL_COLS, [
        {"name": n, "type": "EEG", "units": "µV", "low_cutoff": "0.5", "high_cutoff": "n/a",
         "sampling_frequency": f"{fs:g}", "status": "bad" if n in bad else "good",
         "status_description": "flat" if n in bad else "n/a", "contacts": c, "hemisphere": "", "region": "",
         "x": "", "y": "", "z": "", "original_name": n.upper()} for n, c in zip(names, contacts)])
    write_tsv(d / f"{stem}_desc-preproc_events.tsv", EVENT_COLS, list(events))
    (d / f"{stem}_desc-preproc_eeg.json").write_text(json.dumps({
        "SamplingFrequency": fs, "PowerLineFrequency": 50, "EEGReference": montage,
        "RecordingDuration": data.shape[1] / fs}))
    n_sz = sum(1 for e in events if e["trial_type"] == "seizure")
    return {"recording_id": stem, "subject": sub, "session": "", "task": task, "run": "01",
            "path": f"sub-{sub}/eeg/{stem}_desc-preproc_eeg.edf", "sfreq": f"{fs:g}", "n_channels": len(names),
            "n_good": len(names) - len(bad), "duration_s": f"{data.shape[1] / fs:g}",
            "montage": montage, "reference": "average" if montage == "average" else "",
            "kind": "routine", "start_time": start_time, "n_seizures": n_sz, "source_format": "bids-edf",
            "age": age, "sex": sex}


def make(out: Path, minutes: float = 65.0) -> Path:
    rng = np.random.default_rng(SEED)
    root = out / "eeg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "eeg-sleep synthetic fixture", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "eeg-prep", "Description": "tests/make_fixture.py of eeg-sleep"}]}, indent=2))
    write_tsv(root / "participants.tsv", ["participant_id", "age", "sex"],
              [{"participant_id": "sub-01", "age": "34", "sex": "F"}, {"participant_id": "sub-02", "age": "9",
                                                                        "sex": "M"},
               {"participant_id": "sub-03", "age": "50", "sex": "M"}])
    truth: dict = {}
    recs = []

    # sub-01: referential, average reference, 256 Hz, A1/A2 kept
    fs = 256.0
    x, t1 = night(fs, minutes, rng)
    names = REF10_20 + EARS
    data = np.vstack([x[c] for c in names])
    data -= data.mean(axis=0, keepdims=True)
    ev = [{"onset": "60", "duration": "0", "trial_type": "lights_off", "source": "annotation", "channels": ""},
          {"onset": "1200", "duration": "30", "trial_type": "bad_segment", "source": "eeg-prep", "channels": ""},
          {"onset": "3600", "duration": "60", "trial_type": "seizure", "source": "annotation", "channels": ""},
          {"onset": f"{minutes * 60 - 60:g}", "duration": "0", "trial_type": "lights_on", "source": "annotation",
           "channels": ""}]
    recs.append(write_recording(root, "01", "sleep", fs, data, names, "average", names, events=ev,
                                start_time="2026-01-01T22:00:00", age="34", sex="F"))
    truth[recs[-1]["recording_id"]] = {**t1, "excluded": [[1200, 1230], [3600 - 60, 3660 + 60]],
                                       "lights": [60, minutes * 60 - 60]}

    # sub-02: bipolar-only, 128 Hz
    fs = 128.0
    x, t2 = night(fs, minutes, rng)
    names = [f"{a}-{b}" for a, b in BIPOLAR]
    data = np.vstack([x[a] - x[b] for a, b in BIPOLAR])
    recs.append(write_recording(root, "02", "sleep", fs, data, names, "bipolar-longitudinal",
                                [f"{a}|{b}" for a, b in BIPOLAR], age="9", sex="M"))
    truth[recs[-1]["recording_id"]] = {**t2, "excluded": []}

    # sub-03: a 5-minute routine EEG
    fs = 256.0
    x, _ = night(fs, 5.0, rng)
    names = REF10_20
    data = np.vstack([x[c] for c in names])
    data -= data.mean(axis=0, keepdims=True)
    recs.append(write_recording(root, "03", "routine", fs, data, names, "average", names))

    with open(root / "recordings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REC_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
    (out / "truth.json").write_text(json.dumps(truth))
    return out


if __name__ == "__main__":
    print(make(Path(sys.argv[1] if len(sys.argv) > 1 else "data/fixture")))
