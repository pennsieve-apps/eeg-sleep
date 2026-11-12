"""
Tests on the synthetic fixture (tests/make_fixture.py); no Docker, no real data.

YASA's classifier is not meant to stage synthetic data correctly, so staging tests check the
output contract (epochs, stages, confidence, channel, statistics); detection is checked against
the planted spindles and slow waves.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from processor import eeg as E
from processor import main as M
from processor import sleep as Z
from tests import make_fixture

REC1 = "sub-01_task-sleep_run-01"     # referential, average reference, A1/A2, 256 Hz, lights, bad segment, seizure
REC2 = "sub-02_task-sleep_run-01"     # bipolar only, 128 Hz
REC3 = "sub-03_task-routine_run-01"   # 5 min: skipped
STAGES = {"W", "N1", "N2", "N3", "R"}
PROV_KEYS = {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished", "wall_seconds",
             "peak_rss_mb", "threads", "recordings", "citations", "licence_notice"}


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    return make_fixture.make(tmp_path_factory.mktemp("fixture"))


def run(monkeypatch, input_dir: Path, output_dir: Path, **env) -> int:
    for k in M.PARAM_NAMES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("THREADS", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return M.main()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def prov(out: Path) -> dict:
    return json.loads((out / "provenance.json").read_text())


@pytest.fixture(scope="module")
def default_run(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture_dir / "eeg", out) == 0
    finally:
        mp.undo()
    return out / "eeg-sleep"


def test_tables_headers_provenance_and_summary(default_run):
    for name, header in M.HEADERS.items():
        with open(default_run / name) as fh:
            assert next(csv.reader(fh)) == header, name
    p = prov(default_run)
    assert set(p) == PROV_KEYS
    assert p["app"] == "eeg-sleep" and p["version"] == "0.1.0" and p["variant"] is None
    assert "10.7554/eLife.70092" in p["citations"]
    assert p["threads"]["used"] == 1 and p["tools"]["yasa"] == "0.8.1"
    st = {k: v["status"] for k, v in p["recordings"].items()}
    assert st[REC1] == "ok" and st[REC2] == "ok"
    assert st[REC3] == "skipped: 5.0 min < MIN_SLEEP_MINUTES 60"
    s = json.loads((default_run / "summary.json").read_text())
    assert set(s) == {"app", "version", "recordings", "figures", "tables", "warnings"}
    assert set(s["recordings"]) == {REC1, REC2}
    assert isinstance(s["recordings"][REC1]["TST_min"], float)
    kinds = [f["path"].rsplit("_", 1)[1] for f in s["figures"]]
    assert kinds == ["hypnogram.png", "spectrogram.png", "density.png"] * 2
    for f in s["figures"]:
        assert (default_run / f["path"]).is_file() and len(f["caption"]) > 80
    assert any("bipolar-only" in w for w in s["warnings"])


def test_hypnogram_contract(default_run, fixture_dir):
    rows = read_csv(default_run / "hypnogram.csv")
    for rid, channel in ((REC1, "C4-A1"), (REC2, "F4-P4")):
        r = [x for x in rows if x["recording_id"] == rid]
        assert len(r) == 130  # 65 min of 30-s epochs
        assert [int(x["epoch"]) for x in r] == list(range(130))
        assert all(float(x["start_s"]) == 30 * int(x["epoch"]) and x["duration_s"] == "30" for x in r)
        assert {x["stage"] for x in r} <= STAGES
        assert all(0.0 <= float(x["confidence"]) <= 1.0 for x in r)
        assert {x["channel"] for x in r} == {channel}
    assert not [x for x in rows if x["recording_id"] == REC3]
    p = prov(default_run)["recordings"]
    assert p[REC1]["montage"] == "average" and p[REC2]["montage"] == "bipolar-longitudinal"
    assert p[REC1]["classifier"] == "clf_eeg_lgb_0.5.0.joblib"
    assert p[REC1]["detection_reference"].startswith("linked ears")


def test_sleep_stats_use_lights(default_run):
    stats = {(r["recording_id"], r["metric"]): r["value"] for r in read_csv(default_run / "sleep_stats.csv")}
    assert float(stats[(REC1, "analysis_start_s")]) == 60 and float(stats[(REC1, "analysis_end_s")]) == 3840
    assert float(stats[(REC1, "TIB")]) == pytest.approx(63.0)
    assert float(stats[(REC2, "TIB")]) == pytest.approx(65.0)
    for rid in (REC1, REC2):
        tst = float(stats[(rid, "TST")])
        parts = sum(float(stats[(rid, k)]) for k in ("N1", "N2", "N3", "REM"))
        assert tst == pytest.approx(parts, abs=0.01)
        assert 0 <= float(stats[(rid, "SE")]) <= 100


def test_events_only_in_n2_n3_and_never_in_excluded_time(default_run, fixture_dir):
    truth = json.loads((fixture_dir / "truth.json").read_text())
    hyp = {}
    for r in read_csv(default_run / "hypnogram.csv"):
        hyp.setdefault(r["recording_id"], []).append(r["stage"])
    for name in ("spindles.csv", "slow_waves.csv"):
        rows = read_csv(default_run / name)
        assert rows
        for r in rows:
            t = float(r["start_s"])
            assert r["stage"] in ("N2", "N3")
            assert hyp[r["recording_id"]][int(t // 30)] == r["stage"]
            for a, b in truth[r["recording_id"]]["excluded"]:
                assert not a <= t <= b, (name, r)


def _recall(planted, rows, channel, hyp, excluded):
    st = np.array([float(r["start_s"]) for r in rows if r["channel"] == channel])
    en = np.array([float(r["end_s"]) for r in rows if r["channel"] == channel])
    ok = [p for p in planted if hyp[int(p // 30)] in ("N2", "N3") and not any(a <= p <= b + 1 for a, b in excluded)]
    hits = [np.any((st <= p + 1.0) & (en >= p)) for p in ok]
    return float(np.mean(hits)), len(st)


def test_planted_spindles_and_slow_waves_are_found(default_run, fixture_dir):
    truth = json.loads((fixture_dir / "truth.json").read_text())
    hyp = {}
    for r in read_csv(default_run / "hypnogram.csv"):
        hyp.setdefault(r["recording_id"], []).append(r["stage"])
    sp = read_csv(default_run / "spindles.csv")
    for rid, ch in ((REC1, "C4"), (REC1, "Cz"), (REC2, "C4-P4"), (REC2, "F3-C3")):
        rec, n = _recall(truth[rid]["spindles"], [r for r in sp if r["recording_id"] == rid], ch, hyp[rid],
                         truth[rid]["excluded"])
        assert rec >= 0.9, (rid, ch, rec)
    sw = [r for r in read_csv(default_run / "slow_waves.csv") if r["recording_id"] == REC1]
    planted = np.array(truth[REC1]["slow_waves"])
    starts = np.array([float(r["start_s"]) for r in sw if r["channel"] == "Fz"])
    ok = [p for p in planted if hyp[REC1][int(p // 30)] in ("N2", "N3")]
    assert np.mean([np.any(np.abs(starts - p) < 0.6) for p in ok]) >= 0.6
    feats = {(r["recording_id"], r["channel"], r["feature"]): r["value"] for r in read_csv(default_run / "features.csv")}
    assert float(feats[(REC1, "C4", "spindle_density")]) > 3 * float(feats[(REC1, "O1", "spindle_density")] or 0)
    assert float(feats[(REC1, "Fz", "sw_density")]) > float(feats[(REC1, "O1", "sw_density")] or 0)


def test_detect_channel_on_a_known_hypnogram():
    """Detection alone, independent of staging: everything is N2, 13 Hz spindles every 10 s."""
    rng = np.random.default_rng(0)
    fs, n = 100.0, 100 * 600
    x = make_fixture.pink(n, rng, 10.0)
    planted = np.arange(5.0, 595.0, 10.0)
    for t in planted:
        make_fixture.add_at(x, fs, t, make_fixture.spindle(fs))
    hyp = np.full(n, 2, dtype=np.int8)
    hyp[int(300 * fs):int(330 * fs)] = Z.EXCLUDED
    sp, sw, err = Z.detect_channel(x.astype(np.float32), hyp, 100.0, (100.0, 700.0), "C4", (12, 15), (75, 350))
    assert err is None
    starts = np.array([s[0] for s in sp])
    assert starts.min() >= 100.0                       # offset applied
    kept = [p + 100 for p in planted if not 300 <= p <= 330]
    hits = [np.any(np.abs(starts + 0.5 - p - 0.5) < 0.6) for p in kept]
    assert np.mean(hits) >= 0.9 and len(sp) <= len(kept) * 1.1
    assert not [s for s in starts if 400 <= s <= 430]   # the excluded 30 s
    assert all(s[5] == "N2" for s in sp)


def test_staging_channel_parameter_and_demographics(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir / "eeg", tmp_path, STAGING_CHANNEL="C3-A2", USE_DEMOGRAPHICS="true",
               DETECT_CHANNELS="frontal-central", LABEL="c3") == 0
    out = tmp_path / "eeg-sleep" / "c3"
    rows = read_csv(out / "hypnogram.csv")
    assert {r["channel"] for r in rows if r["recording_id"] == REC1} == {"C3-A2"}
    p = prov(out)["recordings"]
    assert p[REC1]["classifier"] == "clf_eeg+demo_lgb_0.5.0.joblib" and p[REC1]["demographics"] == {
        "age": 34.0, "male": False}
    assert p[REC2]["status"] == "failed" and "STAGING_CHANNEL" in p[REC2]["error"]  # no C3/A2 in a bipolar file
    assert set(p[REC1]["detection_channels"]) == {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3", "Cz", "C4"}


def test_loose_chbmit_style_bipolar_edf(fixture_dir, tmp_path, monkeypatch):
    import mne

    src = next((fixture_dir / "eeg" / "sub-02").rglob("*.edf"))
    raw = mne.io.read_raw_edf(src, preload=True, verbose="error")
    (tmp_path / "in").mkdir()
    names = [n.upper() for n in raw.ch_names]
    names[-1] = "CZ-PZ"
    make_fixture.write_edf(tmp_path / "in" / "chb01_03.edf", raw.get_data() * 1e6, names, raw.info["sfreq"])
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out", DETECT_REFERENCE="as-recorded") == 0
    p = prov(tmp_path / "out" / "eeg-sleep")["recordings"]["chb01_03"]
    assert p["status"] == "ok" and p["montage"] == "bipolar-longitudinal" and p["staging_channel"] == "F4-P4"
    assert "Fp1-F7" in p["detection_channels"] and "Cz-Pz" in p["detection_channels"]


def test_no_input_bad_parameters_and_failed_recording(fixture_dir, tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    assert run(monkeypatch, tmp_path / "empty", tmp_path / "o1") == 1
    assert run(monkeypatch, fixture_dir / "eeg", tmp_path / "o2", SPINDLE_FREQ="15,12") == 2
    assert run(monkeypatch, fixture_dir / "eeg", tmp_path / "o2", DETECT_CHANNELS="occipital") == 2
    assert run(monkeypatch, fixture_dir / "eeg", tmp_path / "o2", MIN_SLEEP_MINUTES="5") == 2
    shutil.copytree(fixture_dir / "eeg", tmp_path / "in")
    next((tmp_path / "in" / "sub-01").rglob("*.edf")).write_bytes(b"broken")
    shutil.rmtree(tmp_path / "in" / "sub-02")
    (tmp_path / "in" / "recordings.csv").write_text(
        "\n".join(line for line in (tmp_path / "in" / "recordings.csv").read_text().splitlines()
                  if "sub-02" not in line) + "\n")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o3") == 0   # one failed, one skipped: not all failed
    st = {k: v["status"] for k, v in prov(tmp_path / "o3" / "eeg-sleep")["recordings"].items()}
    assert st[REC1] == "failed" and st[REC3].startswith("skipped")
    assert run(monkeypatch, tmp_path / "in", tmp_path / "o4", MIN_SLEEP_MINUTES="10000") == 0


# --- units ------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("EEG Fp1-Ref", ("Fp1", ("Fp1",))), ("Fp1-Avg", ("Fp1", ("Fp1",))), ("T3", ("T7", ("T7",))),
    ("FP1-F7", ("Fp1-F7", ("Fp1", "F7"))), ("T8-P8-0", ("T8-P8", ("T8", "P8"))), ("CZ", ("Cz", ("Cz",))),
    ("ECG", ("ECG", ())), ("A1", ("A1", ("A1",))), ("EEG T5-REF", ("P7", ("P7",))),
])
def test_canonical_names(name, expected):
    assert E.canonical(name) == expected


def test_duplicate_and_reversed_pairs_are_left_out():
    """CHB-MIT repeats T8-P8 (T8-P8-0, T8-P8-1) and has P7-T7 next to T7-P7."""
    rec = E.Recording("chb", Path("x.edf"), "x.edf", "scan")
    names = ["FP1-F7", "T7-P7", "T8-P8-0", "T8-P8-1", "P7-T7", "ECG", "-"]
    ch = {c.name: c for c in E.channels(rec, names, ["eeg"] * len(names))}
    assert ch["T7-P7"].status == "good" and ch["T8-P8-0"].status == "good"
    assert ch["T8-P8-1"].reason == "duplicate of T8-P8"
    assert ch["P7-T7"].reason == "duplicate of T7-P7 (reversed)"
    assert ch["ECG"].status == "bad" and ch["-"].status == "bad"


def test_units_from_channels_tsv(tmp_path):
    """An EDF without a physical dimension holds µV (ds003555): channels.tsv says so."""
    from edfio import Edf, EdfSignal

    x = np.sin(np.arange(2560) / 10) * 50
    Edf([EdfSignal(x, sampling_frequency=256, label=n, physical_dimension="", physical_range=(-100, 100))
         for n in ("C4", "A1")]).write(tmp_path / "r_eeg.edf")
    (tmp_path / "r_channels.tsv").write_text("name\ttype\tunits\nC4\tEEG\tuV\nA1\tEEG\tuV\n")
    rec = E.Recording("r", tmp_path / "r_eeg.edf", "r_eeg.edf", "scan", channels_tsv=tmp_path / "r_channels.tsv")
    raw = E.open_raw(rec.path)
    notes = E.set_units(raw, rec)
    assert notes and np.abs(E.read_window(raw, [0], 0, 2560)).max() == pytest.approx(50, rel=0.02)
    rec.channels_tsv = None
    raw = E.open_raw(rec.path)
    E.set_units(raw, rec)  # no channels.tsv: amplitude decides (50 "V" is implausible, so µV)
    assert np.abs(E.read_window(raw, [0], 0, 2560)).max() == pytest.approx(50, rel=0.02)


def test_stream_100hz_matches_direct_resampling(fixture_dir):
    from scipy.signal import resample_poly

    rec = [r for r in E.find_recordings(fixture_dir / "eeg")[0] if r.recording_id == REC1][0]
    raw = E.open_raw(rec.path)
    idx = {n: i for i, n in enumerate(raw.ch_names)}
    d = Z.Derivation("C4-A1", [(idx["C4"], 1.0), (idx["A1"], -1.0)], "")
    x = Z.stream_100hz(raw, [d], 0, 1800, chunk_s=300)[0]
    full = (raw.get_data(picks=[idx["C4"]], stop=1800 * 256) - raw.get_data(picks=[idx["A1"]], stop=1800 * 256))[0]
    y = resample_poly(full * 1e6, 25, 64)
    assert len(x) == len(y) == 180000
    inner = slice(500, -500)
    assert np.max(np.abs(x[inner] - y[inner])) < 0.01 * np.std(y)


def test_candidates_order(fixture_dir):
    recs = {r.recording_id: r for r in E.find_recordings(fixture_dir / "eeg")[0]}
    for rid, first in ((REC1, ["C4-A1", "C3-A2", "C4", "C3", "Cz"]), (REC2, ["F4-P4", "F3-P3", "Fz-Pz", "C4-P4"])):
        raw = E.open_raw(recs[rid].path)
        ch = E.channels(recs[rid], raw.ch_names, raw.get_channel_types())
        good = [c for c in ch if c.status == "good"]
        idx = {n: i for i, n in enumerate(raw.ch_names)}
        labels = [d.label for d in Z.staging_candidates(good, idx, E.montage_of(recs[rid], good))]
        assert labels[:len(first)] == first
