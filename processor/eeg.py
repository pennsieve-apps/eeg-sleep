"""
Finding and reading scalp EEG recordings: the BIDS-EEG derivative of eeg-prep (its `eeg`
port), a plain BIDS-EEG dataset, or loose EDF/BDF/BrainVision/FIF/EEGLAB files.

The contract is workflow-scalp-eeg docs/conventions.md:

- `recordings.csv` (anywhere under INPUT_DIR) lists the recordings with `path` relative to the
  folder the CSV is in, plus sfreq, montage, reference, start_time, age and sex;
- `<stem>[_desc-preproc]_channels.tsv` next to the signal file says which channels are good,
  their type and (for bipolar pairs) their `contacts`;
- `<stem>[_desc-preproc]_events.tsv` holds `seizure`, `bad_segment`, `artifact`, `lights_off`,
  `lights_on` rows (HED-SCORE `eventType` `sz...` of SzCORE-style BIDS datasets is read as a seizure).

Signal files that no recordings.csv lists are picked up as well (a plain BIDS dataset or loose
files), with whatever sidecars they have, and `participants.tsv` gives age and sex. Channel names
are canonicalised to the 10-20/10-10 spelling of MNE's `standard_1005` (`EEG Fp1-Ref` and
`Fp1-Avg` -> `Fp1`, `T3` -> `T7`, `FP1-F7` -> `Fp1-F7`). Nothing here loads a whole recording;
callers read one window at a time.

Adapted from processor-ieeg-hfo's `ieeg.py` (same walking, sidecar and window-reading code);
scalp EEG keeps EEG channels only.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

SIGNAL_EXTENSIONS = (".edf", ".bdf", ".vhdr", ".fif", ".set")
# Used only when there is no _channels.tsv: names that are clearly not EEG.
NON_EEG_NAME = re.compile(r"^(ECG|EKG|EMG|EOG|LOC|ROC|CHIN|RESP|PHOTIC|EVENT|STATUS|TRIG|DC\d*|SPO2|SAO2|PULSE|PR$|"
                          r"OSAT|ANNOT|BURSTS|SUPPR|IBI|LEG|ABD|THOR|FLOW|SNORE|POS|VNS|MK|X\d|\-+$|\.$)",
                          re.IGNORECASE)
SEIZURE_TYPES = {"seizure", "ictal", "sz"}
EXCLUDED_TYPES = {"bad_segment", "artifact", "artefact"}
OLD_NAMES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
REF_SUFFIX = re.compile(r"[-_ ](REF|AVG|AV|AR|CAR|LE|AVERAGE|COMMON)$", re.IGNORECASE)


def walk_files(root: Path):
    """Every file under root, following symlinked folders (INPUT_DIR entries are symlinks), once."""
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if not name.startswith("."):
                yield Path(dirpath) / name


def _float(value) -> float | None:
    try:
        v = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(fh, delimiter="\t")]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(fh)]


# --- channel names ----------------------------------------------------------------

def template_montage():
    """MNE's 10-05 template montage (`standard_1005`, renamed `colin27_1005` in MNE 1.13 with the
    same names and positions)."""
    import mne

    try:
        return mne.channels.make_standard_montage("colin27_1005")
    except ValueError:
        return mne.channels.make_standard_montage("standard_1005")


@lru_cache(maxsize=1)
def _standard_names() -> dict[str, str]:
    names = template_montage().ch_names
    out = {n.lower(): n for n in names}
    for old, new in OLD_NAMES.items():
        out[old.lower()] = new
    return out


def electrode(name: str) -> str | None:
    """The canonical 10-20/10-10 name of a single electrode label, or None if it is not one."""
    n = name.strip()
    n = re.sub(r"^EEG[\s_-]+", "", n, flags=re.IGNORECASE)
    n = REF_SUFFIX.sub("", n).strip()
    return _standard_names().get(n.lower())


def canonical(name: str) -> tuple[str, tuple[str, ...]]:
    """
    (canonical name, contacts). A referential channel gives ("C4", ("C4",)); a bipolar pair
    ("Fp1-F7", ("Fp1", "F7")); anything else keeps its name and has no contacts.
    """
    raw = name.strip()
    single = electrode(raw)
    if single:
        return single, (single,)
    s = re.sub(r"^EEG[\s_-]+", "", raw, flags=re.IGNORECASE)
    s = re.sub(r"-\d+$", "", s)  # CHB-MIT repeats a pair as T8-P8-0, T8-P8-1
    parts = [p for p in re.split(r"\s*-\s*", s) if p]
    if len(parts) == 2:
        a, b = electrode(parts[0]), electrode(parts[1])
        if a and b:
            return f"{a}-{b}", (a, b)
        if a and REF_SUFFIX.match("-" + parts[1]):
            return a, (a,)
    return raw, ()


def hemisphere(contacts: tuple[str, ...]) -> str:
    def side(e: str) -> str:
        d = re.findall(r"\d+", e)
        if not d:
            return "Z"
        return "L" if int(d[-1]) % 2 else "R"
    sides = {side(c) for c in contacts}
    if len(sides) == 1:
        return sides.pop()
    sides.discard("Z")
    return sides.pop() if len(sides) == 1 else "Z"


def region(contacts: tuple[str, ...]) -> str:
    """frontal | central | temporal | parietal | occipital | other, from the (first) electrode name."""
    if not contacts:
        return "other"
    regs = []
    for c in contacts:
        u = c.upper()
        if u.startswith(("FP", "AF", "F")) and not u.startswith(("FT", "FC")):
            regs.append("frontal")
        elif u.startswith(("FC", "C")) and not u.startswith("CP"):
            regs.append("central")
        elif u.startswith(("FT", "T", "TP", "A", "M")):
            regs.append("temporal")
        elif u.startswith(("CP", "P")) and not u.startswith("PO"):
            regs.append("parietal")
        elif u.startswith(("PO", "O", "I")):
            regs.append("occipital")
        else:
            regs.append("other")
    return regs[0] if len(set(regs)) == 1 else "+".join(dict.fromkeys(regs))


def in_region(contacts: tuple[str, ...], regions) -> bool:
    """True when the channel touches one of `regions`: a bipolar pair's region is `frontal+central`
    for F4-C4, so F4-C4 counts as frontal and as central, C4-P4 as central and parietal."""
    return bool(set(region(contacts).split("+")) & set(regions))


# --- recordings -------------------------------------------------------------------

@dataclass
class Recording:
    recording_id: str
    path: Path                         # the signal file (.edf, .vhdr, ...)
    rel: str                           # path relative to INPUT_DIR, for logs and provenance
    source: str                        # "recordings.csv" or "scan"
    subject: str = ""
    montage: str = ""                  # from recordings.csv when given
    reference: str = ""
    start_time: str = ""
    age: float | None = None
    sex: str = ""
    channels_tsv: Path | None = None
    events_tsv: Path | None = None
    sidecar_json: Path | None = None
    notes: list[str] = field(default_factory=list)


def signal_prefix(path: Path) -> str:
    """`sub-01_task-x_desc-preproc_eeg.edf` -> `sub-01_task-x_desc-preproc`; `foo.edf` -> `foo`."""
    stem = path.name[: -len(path.suffix)]
    return stem[: -len("_eeg")] if stem.endswith("_eeg") else stem


def recording_id_from_prefix(prefix: str) -> str:
    """The BIDS entity chain of the source recording: drop a `_desc-<label>` entity."""
    return re.sub(r"_desc-[A-Za-z0-9]+", "", prefix)


def _sidecars(rec: Recording) -> None:
    prefix = signal_prefix(rec.path)
    folder = rec.path.parent
    for attr, suffix in (("channels_tsv", "_channels.tsv"), ("events_tsv", "_events.tsv"),
                         ("sidecar_json", "_eeg.json")):
        if getattr(rec, attr) is None:
            p = folder / f"{prefix}{suffix}"
            if p.is_file():
                setattr(rec, attr, p)


def _participants(files: list[Path]) -> dict[str, dict[str, dict]]:
    """{folder of participants.tsv: {subject label: row}}."""
    out: dict[str, dict[str, dict]] = {}
    for p in files:
        if p.name != "participants.tsv":
            continue
        try:
            rows = read_tsv(p)
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        out[os.path.realpath(p.parent)] = {r.get("participant_id", "").removeprefix("sub-"): r for r in rows}
    return out


def _sex(value: str) -> str:
    v = (value or "").strip().lower()
    return {"m": "M", "male": "M", "f": "F", "female": "F"}.get(v, "")


def find_recordings(input_dir: Path) -> tuple[list[Recording], list[str], list[dict]]:
    """
    (recordings, every input file seen (relative), problems). A problem is a listed recording
    whose file is missing, or a duplicate recording id; each is a dict with recording_id,
    input and reason.
    """
    files = list(walk_files(input_dir))
    rel = lambda p: os.path.relpath(p, input_dir)  # noqa: E731
    seen = [rel(p) for p in files]
    problems: list[dict] = []
    recs: list[Recording] = []
    listed: set[str] = set()

    for csv_path in (p for p in files if p.name == "recordings.csv"):
        try:
            rows = read_csv(csv_path)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            problems.append({"recording_id": "", "input": rel(csv_path), "reason": f"unreadable recordings.csv: {exc}"})
            continue
        for row in rows:
            rid, path = row.get("recording_id", ""), row.get("path", "")
            if not rid or not path:
                continue
            sig = csv_path.parent / path
            if not sig.is_file():
                problems.append({"recording_id": rid, "input": rel(csv_path),
                                 "reason": f"recordings.csv lists {path}, which is not in the input"})
                continue
            listed.add(os.path.realpath(sig))
            recs.append(Recording(
                recording_id=rid, path=sig, rel=rel(sig), source="recordings.csv", subject=row.get("subject", ""),
                montage=row.get("montage", ""), reference=row.get("reference", ""),
                start_time=row.get("start_time", ""), age=_float(row.get("age")), sex=_sex(row.get("sex", ""))))

    parts = _participants(files)
    scanned = [p for p in files if p.suffix.lower() in SIGNAL_EXTENSIONS and os.path.realpath(p) not in listed]
    in_deriv = lambda p: "derivatives" in Path(rel(p)).parts[:-1]  # noqa: E731
    raw_too = any(not in_deriv(p) for p in scanned)
    for p in scanned:
        if raw_too and in_deriv(p):
            # a raw BIDS dataset with its derivatives/ folder: the raw recordings are analysed
            problems.append({"recording_id": recording_id_from_prefix(signal_prefix(p)), "input": rel(p),
                             "status": "skipped", "reason": "in the derivatives/ folder of a BIDS dataset"})
            continue
        prefix = signal_prefix(p)
        m = re.match(r"sub-([A-Za-z0-9]+)", p.name)
        rec = Recording(recording_id=recording_id_from_prefix(prefix), path=p, rel=rel(p), source="scan",
                        subject=m.group(1) if m else "")
        if rec.subject:
            d = Path(os.path.realpath(p.parent))
            for folder in [d, *d.parents]:
                row = parts.get(str(folder), {}).get(rec.subject)
                if row:
                    rec.age, rec.sex = _float(row.get("age")), _sex(row.get("sex", ""))
                    break
        recs.append(rec)

    for r in recs:
        _sidecars(r)

    out: list[Recording] = []
    by_id: dict[str, Recording] = {}
    for r in sorted(recs, key=lambda r: (r.source != "recordings.csv", r.rel)):
        if r.recording_id in by_id:
            keep = by_id[r.recording_id]
            problems.append({"recording_id": r.recording_id, "input": r.rel, "status": "skipped",
                             "reason": f"duplicate recording id; {keep.rel} is used"})
            continue
        by_id[r.recording_id] = r
        out.append(r)
    out.sort(key=lambda r: r.recording_id)
    return out, seen, problems


# --- channels -------------------------------------------------------------------

@dataclass
class Channel:
    name: str                    # name in the signal file
    label: str                   # canonical name (`C4`, `Fp1-F7`), used in every table
    contacts: tuple[str, ...]    # electrodes (one for referential, two for a pair, none if unknown)
    status: str                  # good | bad
    reason: str = ""


def channels(rec: Recording, raw_names: list[str], raw_types: list[str]) -> list[Channel]:
    """
    Every channel of the recording with good/bad status and its canonical name. With a
    channels.tsv its type and status columns decide (a channel absent from the file is bad);
    without one, MNE's channel type and a name filter decide. Only EEG channels are good.
    """
    in_file = set(raw_names)
    out: list[Channel] = []
    labels: dict[str, str] = {}   # label (or electrode pair in either order) -> the good channel's label

    def add(name, ctype, status, reason, contacts_col=""):
        label, contacts = canonical(name)
        if contacts_col and "|" in contacts_col:
            cs = tuple(electrode(c) or c for c in contacts_col.split("|"))
            contacts, label = cs, "-".join(cs) if len(cs) == 2 else label
        key = "-".join(sorted(contacts)) if len(contacts) == 2 else label
        if status == "good" and key in labels:
            status, reason = "bad", f"duplicate of {labels[key]}" + (" (reversed)" if labels[key] != label else "")
        if status == "good":
            labels[key] = label
        out.append(Channel(name, label, contacts, status, reason))

    if rec.channels_tsv is not None:
        for row in read_tsv(rec.channels_tsv):
            name = row.get("name", "")
            if not name:
                continue
            ctype = (row.get("type") or "EEG").upper()
            status = (row.get("status") or "good").lower()
            reason = ""
            if name not in in_file:
                reason = "listed in channels.tsv but not in the signal file"
            elif status == "bad":
                reason = row.get("status_description") or "status bad in channels.tsv"
                if reason.lower() == "n/a":
                    reason = "status bad in channels.tsv"
            elif ctype != "EEG":
                reason = f"type {ctype} is not EEG"
            add(name, ctype, "bad" if reason else "good", reason, row.get("contacts", ""))
        return out
    for name, mtype in zip(raw_names, raw_types):
        ctype = mtype.upper()
        if ctype != "EEG":
            add(name, ctype, "bad", f"type {ctype} is not EEG")
        elif NON_EEG_NAME.match(re.sub(r"^EEG[\s_-]+", "", name, flags=re.IGNORECASE)):
            add(name, ctype, "bad", "name looks like a non-EEG channel")
        else:
            add(name, ctype, "good", "")
    return out


def montage_of(rec: Recording, good: list[Channel]) -> str:
    """`bipolar-longitudinal` when most good channels are electrode pairs, else the
    recordings.csv montage, else `referential`."""
    pairs = sum(1 for c in good if len(c.contacts) == 2)
    if good and pairs > len(good) / 2:
        return "bipolar-longitudinal"
    if rec.montage:
        return rec.montage
    if good and sum(1 for c in good if re.search(r"[-_ ](AVG|AV|CAR|AVERAGE)$", c.name, re.IGNORECASE)) > len(good) / 2:
        return "average"   # e.g. Siena's `Fp1-Avg`
    return "referential"


# --- time intervals -------------------------------------------------------------

def events(rec: Recording) -> list[dict]:
    if rec.events_tsv is None:
        return []
    try:
        return read_tsv(rec.events_tsv)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        rec.notes.append(f"events.tsv unreadable: {exc}")
        return []


def _kind(e: dict) -> str:
    t = (e.get("trial_type") or "").strip().lower()
    if t:
        return t
    et = (e.get("eventType") or e.get("event_type") or "").strip().lower()
    if et.startswith("sz"):
        return "seizure"
    if et in ("bckg", "background"):
        return ""
    return et


def intervals(evs: list[dict], kinds: set[str], duration_s: float) -> list[tuple[float, float]]:
    """(start, end) of every event of one of `kinds`; a seizure without a duration runs to the
    end of the recording."""
    out = []
    for e in evs:
        k = _kind(e)
        if k not in kinds:
            continue
        on, dur = _float(e.get("onset")), _float(e.get("duration"))
        if on is None:
            continue
        if dur is None or dur <= 0:
            if k in SEIZURE_TYPES:
                dur = duration_s - on
            else:
                continue
        out.append((max(0.0, on), min(duration_s, on + dur)))
    return sorted(out)


def marker(evs: list[dict], kind: str) -> float | None:
    """Onset of the first `kind` event (lights_off / lights_on)."""
    times = [_float(e.get("onset")) for e in evs if _kind(e) == kind]
    times = [t for t in times if t is not None]
    return min(times) if times else None


def sidecar_line_freq(rec: Recording) -> float | None:
    if rec.sidecar_json is None:
        return None
    try:
        return _float(json.loads(rec.sidecar_json.read_text()).get("PowerLineFrequency"))
    except (OSError, ValueError, AttributeError):
        return None


HIGHPASS_NAME = re.compile(r"^high[-_ ]?pass(?:[-_ ]?filter)?$", re.IGNORECASE)
BANDPASS_NAME = re.compile(r"^band[-_ ]?pass(?:[-_ ]?filter)?$", re.IGNORECASE)
CUTOFF_KEY = re.compile(r"^(?:cutoff[-_ ]?frequency|half[-_ ]amplitude cutoff(?: \(hz\))?|cutoff(?: \(hz\))?|frequency)$",
                        re.IGNORECASE)
BAND_LOW_KEY = re.compile(r"^(?:low[-_ ]?cutoff(?:[-_ ]?frequency)?|lower[-_ ]?cutoff)(?: \(hz\))?$", re.IGNORECASE)


def highpass_hz(rec: Recording) -> tuple[float | None, str]:
    """
    The highest high-pass cutoff the recording's metadata reports, and where it was found:
    in the BIDS sidecar's `SoftwareFilters`, `SourceSoftwareFilters` and `HardwareFilters`, a filter
    named high-pass (`Highpass`, `high-pass filter`, `HighpassFilter`, ...) with a cutoff key
    (`CutoffFrequency`, `half-amplitude cutoff (Hz)`, ...), or a band-pass filter's low cutoff;
    and the `low_cutoff` column of `_channels.tsv`. Low-pass cutoffs (a band-pass `HighCutoff`)
    never count. Slow waves are 0.3-1.5 Hz.
    """
    found: list[tuple[float, str]] = []

    def filters(group, where):
        if not isinstance(group, dict):
            return
        for name, spec in group.items():
            if not isinstance(spec, dict):
                continue
            if HIGHPASS_NAME.match(name.strip()):
                keys = CUTOFF_KEY
            elif BANDPASS_NAME.match(name.strip()):
                keys = BAND_LOW_KEY
            else:
                continue
            for k, v in spec.items():
                f = _float(v)
                if keys.match(k.strip()) and f is not None and f > 0:
                    found.append((f, f"{where}/{name}/{k}"))

    if rec.sidecar_json is not None:
        try:
            j = json.loads(rec.sidecar_json.read_text())
            for key in ("SoftwareFilters", "SourceSoftwareFilters", "HardwareFilters"):
                filters(j.get(key), key)
        except (OSError, ValueError):
            pass
    if rec.channels_tsv is not None:
        try:
            vals = [_float(r.get("low_cutoff")) for r in read_tsv(rec.channels_tsv)]
            vals = [v for v in vals if v is not None and v > 0]
            if vals:
                found.append((float(sorted(vals)[len(vals) // 2]), "channels.tsv low_cutoff (median)"))
        except (OSError, UnicodeDecodeError, csv.Error):
            pass
    if not found:
        return None, ""
    v, where = max(found)
    return v, where


# --- signal ---------------------------------------------------------------------

def open_raw(path: Path):
    """The recording, lazily (nothing loaded). EDF/BDF/BrainVision/FIF/EEGLAB through MNE."""
    import mne

    return mne.io.read_raw(str(path), preload=False, verbose="error")


UNIT_TO_UV = {"uv": 1.0, "µv": 1.0, "μv": 1.0, "microv": 1.0, "mv": 1e3, "v": 1e6, "nv": 1e-3}


def set_units(raw, rec: Recording) -> list[str]:
    """
    Per-channel factors from MNE's values to µV, kept on `raw` for read_window. MNE reads a
    channel whose EDF physical dimension is empty as volts; such files usually hold µV (e.g.
    OpenNeuro ds003555). For those channels the `units` column of `_channels.tsv` decides, else
    the amplitude does (a median |x| above 10 mV read as volts means the numbers are µV).
    Returns notes for provenance.
    """
    import numpy as np

    orig = getattr(raw, "_orig_units", None) or {}
    scale = np.full(len(raw.ch_names), 1e6)
    unknown = [i for i, n in enumerate(raw.ch_names) if str(orig.get(n, "V")).strip().lower() in ("", "n/a", "none")]
    if not unknown:
        raw._uv_scale = scale
        return []
    tsv = {}
    if rec.channels_tsv is not None:
        tsv = {r.get("name", ""): (r.get("units") or "").strip().lower() for r in read_tsv(rec.channels_tsv)}
    by_tsv = [i for i in unknown if tsv.get(raw.ch_names[i]) in UNIT_TO_UV]
    for i in by_tsv:
        scale[i] = UNIT_TO_UV[tsv[raw.ch_names[i]]]  # MNE's value is the number in the file
    notes = []
    if by_tsv:
        notes.append(f"{len(by_tsv)} channel(s) without a unit in the file: unit taken from channels.tsv")
    rest = [i for i in unknown if i not in by_tsv]
    if rest:
        n = min(int(raw.n_times), int(60 * raw.info["sfreq"]))
        start = max(0, int(raw.n_times) // 2 - n // 2)
        x = raw.get_data(picks=rest, start=start, stop=start + n)
        med = np.median(np.abs(x), axis=1)
        for i, m in zip(rest, med):
            if m > 1e-2:
                scale[i] = 1.0
        k = int(np.sum(med > 1e-2))
        if k:
            notes.append(f"{k} channel(s) without a unit in the file and amplitudes only plausible as µV: read as µV")
    raw._uv_scale = scale
    return notes


def read_window(raw, picks: list[int], start: int, stop: int, max_bytes: int = 128 << 20):
    """
    Samples [start, stop) of the picked channels as float64 microvolts, shape (channels,
    samples). Read in groups of channels so the buffer MNE returns stays under max_bytes.
    """
    import numpy as np

    scale = getattr(raw, "_uv_scale", None)
    n = stop - start
    out = np.empty((len(picks), n), dtype=np.float64)
    group = max(1, int(max_bytes // max(1, n * 8)))
    for i in range(0, len(picks), group):
        p = picks[i:i + group]
        f = scale[p][:, None] if scale is not None else 1e6  # MNE returns volts
        out[i:i + group] = raw.get_data(picks=p, start=start, stop=stop) * f
    return out
