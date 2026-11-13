# eeg-sleep

This app is built on **YASA** (Yet Another Spindle Algorithm), developed by **Raphael Vallat and
Matthew P. Walker** at the Center for Human Sleep Science, University of California, Berkeley
(https://github.com/raphaelvallat/yasa; Vallat & Walker 2021, *eLife*,
doi:10.7554/eLife.70092). Its sleep-staging classifiers were trained by Raphael Vallat on
polysomnography from the **National Sleep Research Resource** (NSRR, https://sleepdata.org,
funded by the National Heart, Lung and Blood Institute; Dean et al. 2016, Zhang et al. 2018), and
run on **LightGBM** (Ke et al. 2017; Microsoft and the LightGBM developers). Signals are read with
**MNE-Python** (Alexandre Gramfort and the MNE community; Gramfort et al. 2013). **Pennsieve
packages these methods for clinical scalp EEG; it did not develop them.** The demo results below
use EEG from **OpenNeuro ds003555** (Dorottya Cserpan, Johannes Sarnthein, Georgia Ramantani and
colleagues, University Children's Hospital and University Hospital Zurich) and the **Siena Scalp
EEG Database** (Paolo Detti and colleagues, University of Siena; the BIDS conversion by Jonathan
Dan and Paolo Detti is published by EPFL on Zenodo) and the **CHB-MIT Scalp EEG Database** (Ali Shoeb and John Guttag, MIT, with
Children's Hospital Boston).

A step of both scalp EEG workflows:

- **Scalp EEG Interictal Biomarkers:** `eeg-prep → {eeg-spectral, eeg-connectivity, eeg-sleep → eeg-spikes} → eeg-report`
- **Scalp EEG Seizure Detection & Monitoring:** `eeg-prep → {eeg-seizures, eeg-sleep → eeg-spikes} → eeg-report`

**For the epileptologist.** Interictal spikes increase with the depth of NREM sleep, and spike
rates in NREM sleep localise better than rates in wake, so a hypnogram makes spike rates from a
long-term or overnight EEG comparable between patients and between nights. This step scores
every 30-s epoch as wake, N1, N2, N3 or REM automatically, with the classifier's confidence, and
writes the hypnogram that **eeg-spikes** uses to report spike rates per sleep stage. It also
reports sleep statistics (total sleep time, sleep efficiency, sleep-onset latency, time and
percentage per stage) and finds **sleep spindles** and **slow waves** in N2/N3 on every channel,
with a head map of their density: an asymmetry can point to a lesion, a focal epilepsy or a bad
electrode. The staging uses **EEG only** (clinical EEG exports rarely keep EOG and chin EMG), on
one central derivation: a hypnogram to review, not a polysomnography scoring. Recordings shorter
than an hour (a routine EEG) are skipped on purpose.

**For the researcher.** `yasa.SleepStaging` with YASA's bundled EEG-only LightGBM classifier
(`clf_eeg_lgb_0.5.0`) on one derivation chosen by a fixed rule (C4-M1 > C3-M2 > central electrode
in the recording's reference; on bipolar-only recordings F4-P4 summed from the double banana),
resampled to 100 Hz with a polyphase filter and streamed, so 24 h recordings never sit in memory;
`Hypnogram.sleep_statistics` (AASM definitions as implemented by YASA, between lights-off and
lights-on events when there are any); `yasa.spindles_detect` and `yasa.sw_detect` with YASA's
defaults on N2/N3 samples, against linked ears/mastoids when the recording has them, with bad
segments, artifacts and seizures (± 60 s) left out. Tables are long-format CSV keyed by
`recording_id` (workflow-scalp-eeg `docs/conventions.md`); per-channel densities also go to the
shared `features.csv`.

## Ports

| Port | Direction | Placement | What |
|---|---|---|---|
| `eeg` | input | root | The BIDS-EEG derivative of eeg-prep: `recordings.csv`, `*_desc-preproc_eeg.edf` with `_channels.tsv`, `_events.tsv`, `_eeg.json`. Also any BIDS-EEG dataset (its `participants.tsv` gives age and sex; a `derivatives/` folder next to raw recordings is ignored), or plain EDF/BDF/BrainVision/FIF/EEGLAB files, referential or bipolar-only, found anywhere under `INPUT_DIR` (folders and symlinks followed). |
| `sleep` | output | `OUTPUT_DIR/eeg-sleep/` (or `OUTPUT_DIR/eeg-sleep/<LABEL>/`) | the tables below, `figures/`, `summary.json`, `provenance.json` |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `STAGING_CHANNEL` | `auto` | The derivation YASA stages on (see *Staging channel*). Or a channel name (`C4`, `C4-P4`), or the difference of two channels (`C4-A1`). |
| `USE_DEMOGRAPHICS` | `false` | `true`: use YASA's classifier that also takes age and sex (from `recordings.csv` or `participants.tsv`) when both are known. Off by default, so the reported accuracy (EEG only) applies. |
| `MIN_SLEEP_MINUTES` | `60` | Recordings shorter than this are not staged: status `skipped: X min < MIN_SLEEP_MINUTES 60` in `provenance.json`, no rows, and the run still succeeds. At least 10 (YASA smooths its features over 7.5 min). |
| `DETECT_CHANNELS` | `all` | Channels for spindles and slow waves: `all` good EEG channels (the head map needs them), or `frontal-central`. |
| `DETECT_REFERENCE` | `auto` | `auto`: in a referential recording with both ear or mastoid electrodes (A1/A2, M1/M2), detect against their mean (linked ears), as sleep EEG is read; else as recorded. `as-recorded`: always the recording's own reference. |
| `SPINDLE_FREQ` | `12,15` | Sigma band of `spindles_detect` (YASA's default). Young children's spindles are slower: see *Results*. |
| `SW_AMP_PTP` | `75,350` | Peak-to-peak amplitude range (µV) of a slow wave (YASA's default, for adults referred to the mastoids). |
| `EXCLUDE_SEIZURE` | `true` | Leave annotated seizures (`seizure` events, or HED-SCORE `sz…` events of SzCORE-style BIDS) out of spindle and slow-wave detection, `SEIZURE_MARGIN_S` on both sides. `bad_segment` and `artifact` events are always left out. Every epoch is staged regardless. |
| `SEIZURE_MARGIN_S` | `60` | Seconds left out before each seizure's onset and after its end. |
| `WINDOW_S` | `7200` | Detection windows (30 s of context on each side): YASA's spindle RMS threshold is computed per window over its N2/N3 time. |
| `THREADS` | `0` | Channels analysed in parallel (processes); `0` = one per vCPU the container is given (cgroup CPU quota, else CPU affinity, else the CPU count), capped so that ~450 MB per worker plus ~900 MB for the main process fit the memory limit. The number and why are logged and kept in `provenance.json`. |
| `LABEL` | (empty) | When set, write to `OUTPUT_DIR/eeg-sleep/<LABEL>/`. |

## Outputs

All under `eeg-sleep/`. Times are seconds from the recording start.

**`hypnogram.csv`** (read by eeg-spikes; columns exactly as in conventions.md): `recording_id`,
`epoch` (0, 1, …), `start_s`, `duration_s` (30), `stage` (`W`, `N1`, `N2`, `N3`, `R`),
`confidence` (the classifier's probability of that stage, 0-1), `channel` (the staging
derivation, e.g. `C4-A1`). One row per full 30-s epoch; a final partial epoch is not staged.

**`sleep_stats.csv`**: `recording_id, metric, value`, YASA's `sleep_statistics` over the epochs
from lights-off to lights-on (`lights_off` / `lights_on` events), else the whole recording:
`TIB` (time in bed), `SPT` (sleep period time), `WASO`, `TST`, `SOL` (sleep-onset latency),
`SOL_5min`, `Lat_REM` (from lights-off; AASM REM latency = `Lat_REM − SOL`), `WAKE`, `N1`, `N2`,
`N3`, `REM` in **minutes**; `SE` (sleep efficiency), `SME` (sleep maintenance efficiency), `%N1`,
`%N2`, `%N3`, `%REM`, `%NREM` in **percent of TST**; `SFI` (sleep-to-wake transitions per hour of
TST). Plus `confidence_mean` (%), `confidence_ge80_pct` (percent of epochs with confidence ≥ 0.8),
`analysis_start_s`, `analysis_end_s`. Metrics YASA leaves undefined (no REM: `Lat_REM`) are absent
or empty.

**`spindles.csv`**: `recording_id, channel, start_s, end_s, duration_s, frequency_hz` (median
instantaneous frequency), `amplitude_uv` (peak-to-peak, broadband 1-30 Hz, detrended), `stage`
(`N2`/`N3`).

**`slow_waves.csv`**: `recording_id, channel, start_s, end_s, amplitude_uv` (peak-to-peak),
`frequency_hz`, `stage`.

**`features.csv`** (shared shape `recording_id, channel, feature, value`): `spindle_density` and
`sw_density` (events per minute of N2+N3 analysed), `spindle_amplitude`, `spindle_frequency`,
`sw_amplitude` (means), `nrem_minutes_analysed`.

**`figures/`**: per recording `<id>_hypnogram.png` (stages, low-confidence epochs, time left out,
stage probabilities; clock time when the start time is known), `<id>_spectrogram.png` (staging
channel, 0.5-25 Hz) and `<id>_density.png` (spindle and slow-wave density head maps; bipolar pairs
at their midpoints).

**`summary.json`**: a `notice` that the report shows at the top of this step (EEG-only staging, not
validated against expert scoring on these recordings, unreliable under 5 years); headline numbers per recording (`TST_min`, `SE_pct`, `SOL_min`, `WASO_min`,
stage percentages, `confidence_mean_pct`, `confidence_ge80_pct`, `n_spindles`, `n_slow_waves`,
`spindle_density_central`, `sw_density_frontal`, `nrem_minutes_analysed`), the figures in report
order with captions, the tables, and warnings (low confidence, a non-mastoid staging derivation,
bipolar-only input, a high-pass above 0.5 Hz, age under 5).

**`provenance.json`** (keys as in conventions.md): tool versions, parameters, inputs, times, peak
memory, threads; per recording the status (`ok`, `failed` with the error, `skipped: …`), montage,
staging derivation and the alternatives, the classifier file, detection channels and reference,
time left out and why, lights, high-pass, notes.

## How it works

1. **Finds the recordings** (every `recordings.csv`, plus signal files no CSV lists, with their
   sidecars and `participants.tsv`) and handles them one at a time, opened lazily with MNE.
   Channel names are canonicalised to MNE's 10-20/10-10 spelling (`EEG Fp1-Ref`, `Fp1-Avg` →
   `Fp1`; `T3/T4/T5/T6` → `T7/T8/P7/P8`; `FP1-F7` → `Fp1-F7`), duplicate and reversed pairs
   (CHB-MIT's `T8-P8-1`, `P7-T7`) are left out. EDF channels without a physical unit (OpenNeuro
   ds003555) take their unit from `_channels.tsv`, else from their amplitude.
2. **Staging channel** (below), read in 10-min chunks, combined (e.g. C4 − A1), resampled to
   100 Hz with `scipy.signal.resample_poly` and kept (a 24 h trace is 35 MB). A candidate that is
   flat in more than half of its epochs is passed over for the next.
3. **Staging**: `yasa.SleepStaging(raw, eeg_name=…)` computes YASA's features per 30-s epoch
   (time-domain, spectral, entropy and fractal features, smoothed over 7.5 min and 2 min, robust
   z-scored per recording) and predicts each epoch with the bundled classifier, loaded from the
   read-only `/opt/eeg-sleep/classifiers/` (copied from the YASA wheel at build time, checked by
   SHA-256; no network).
4. **Sleep statistics** from the hypnogram, between lights-off and lights-on when annotated.
5. **Spindles and slow waves** per `WINDOW_S` window of all detection channels at 100 Hz, each
   channel in a worker process, with YASA's integer hypnogram as the mask: N2 and N3 samples
   count, excluded time (bad segments, artifacts, seizures ± margin) is set to YASA's artefact
   code. Events are kept when they start inside the window's core.
6. Figures, tables, `summary.json`, `provenance.json`.

### Staging channel

YASA's classifiers were trained on one central EEG derivation, "C4-M1 or C4-Fpz depending on the
dataset" (Vallat & Walker 2021): the SHHS and MESA recordings have mastoid-referred central leads,
the CCSHS, CFS, CHAT, HomePAP and MrOS recordings C3/C4-Fpz. `STAGING_CHANNEL=auto` picks, in order:

1. **Referential recordings**: `C4-M1` (or `C4-A1`), then `C3-M2` (`C3-A2`), computed as the
   difference of the two electrodes, which is exact whatever the recording's reference (also in
   eeg-prep's average reference). Then `C4`, `C3`, `Cz` in the recording's own reference (usually
   the common average).
2. **Bipolar-only recordings** (CHB-MIT style): `F4-P4` = (`F4-C4`) + (`C4-P4`) (exact: the
   central electrode cancels), then `F3-P3`, `Fz-Pz`, then the single pairs `C4-P4`, `C3-P3`,
   `F4-C4`, `F3-C3`, `Cz-Pz`, `Fz-Cz`.

The order of the fallbacks is from data: on the three ds003555 nights (which have A1/A2) each
alternative was staged and compared with staging on C4-A1 (Cohen's kappa per night): C4 in average
reference 0.62 / 0.55 / 0.62; C4-O1 0.61 / 0.49 / 0.67; C4-Fp1 0.49 / 0.55 / 0.65; C3-A2 0.47 /
0.15 / 0.63; Cz in average reference 0.56 / 0.27 / 0.45; bipolar F4-P4 0.58 / 0.43 / 0.46, C4-P4
0.48 / 0.47 / 0.53, F4-C4 0.57 / 0.42 / 0.48. So the derivation matters a lot (even C4-A1 and its
mirror C3-A2 agree with kappa 0.15-0.63 in these children), and staging on anything other than a
central-mastoid or central-Fpz derivation should be read as approximate; the app warns when it did.

### Reported accuracy

Vallat & Walker 2021 report, on 585 held-out nights of the NSRR cohorts (testing set 1), a
median accuracy of **85.12 % (median Cohen's kappa 0.782) with a single EEG channel**, against
87.5 % for the full model with EEG, EOG, EMG, age and sex; epochs the classifier scored with
≥ 80 % confidence were 95.90 % accurate, those below 62.91 %. N1 is the weakest stage (sensitivity
45.4 % for the full model), and the commonest errors were N1 scored as N2 and N3 scored as N2.
Those figures are for the central EEG of polysomnography (C4-M1 or C4-Fpz) in the training population (children
from 5 years, adolescents and adults, many with sleep apnoea); they are **not** validated for
clinical EEG in epilepsy, in average reference, on bipolar pairs, in children under 5, during
frequent epileptiform activity or under sedation. This app adds no validation of staging accuracy:
the demo data have no expert hypnograms.

## Validation and results

**Synthetic fixture** (`tests/`, 27 tests): two 65-min "nights" with planted 13 Hz spindles and
0.8 Hz slow waves (one referential in average reference with A1/A2, one bipolar-only). With the
hypnogram YASA produced, ≥ 99 % of the planted spindles in N2/N3 epochs are found at C4 and Cz
(precision 1.00) and ≥ 98 % on the bipolar central pairs (precision 0.97); slow-wave recall at Fz
is 0.74 against linked ears, but 0.16 in average reference (the common average removes much of a
diffuse slow wave, so YASA's absolute amplitude criteria miss them: hence `DETECT_REFERENCE=auto`);
no event is reported in a bad segment or a seizure. Staging tests check the contract only.

**OpenNeuro ds003555** (3 h of sleep EEG per child, 1024 Hz, 10-20 with A1/A2, staged on C4-A1),
eeg-prep derivative (`data/work/ds003555/eeg-prep/eeg`, average reference, units fixed by
eeg-prep), in `data/work/ds003555/eeg-sleep/`:

| Recording | Age | TST (min) | SE (%) | N2 / N3 / REM (% TST) | Mean confidence (%) | Epochs ≥ 0.8 (%) | Spindles /min N2+N3, C3/C4/Cz | Slow waves /min N2+N3, frontal |
|---|---|---|---|---|---|---|---|---|
| sub-01 | 5.6 y | 167.0 | 92.8 | 40.4 / 42.8 / 15.6 | 62.3 | 18.9 | 0.04 / 0.19 / 0.12 | 26.3 |
| sub-02 | 1.6 y | 46.0 | 25.6 | 56.5 / 0 / 37.0 | 56.7 | 8.1 | 0.66 / 0.27 / 0.97 | 2.1 |
| sub-03 | 4.8 y | 141.0 | 78.3 | 14.5 / 59.9 / 12.4 | 71.7 | 42.5 | 0 / 0 / 0 | 32.9 |

The same nights run on the raw BIDS files (`data/work/ds003555/eeg-sleep-rawbids/`, units from
`channels.tsv`) have identical recording ids and epoch boundaries, and 89 % / 84 % / 87 % of
epochs get the same stage (TST 168.0 / 47.5 / 138.0 min): eeg-prep's filtering, ICA and average
reference change the staging only a little, since C4-A1 is the same difference in either reference.

sub-01's hypnogram is plausible for the first 3 h of a child's night (N3 early, the first REM
about an hour in, cycles). sub-02, a toddler, comes out mostly as wake in what is a recording of
sleep: below YASA's training ages, and the app says so. Spindles are almost absent at YASA's adult
default band (12-15 Hz): the spectrogram shows NREM sigma power around 9-11 Hz instead, as
expected for slow spindles in young children. With `SPINDLE_FREQ=10,15` (raw-BIDS run) sub-01 has 3.1 / 3.8 / 2.3
spindles per minute at C3 / C4 / Cz (sub-02 1.0-1.2, sub-03 none). Choose the band for the
population, and say so.

**Siena Scalp EEG** (adults in video-EEG monitoring, average reference, no ear electrodes, staged
on C4 in average reference): the eeg-prep derivative of all 41 runs (`data/work/siena/eeg-prep/eeg`), in
`data/work/siena/eeg-sleep/`. 35 runs were staged and 6 shorter than an hour were skipped; none
failed. Mean staging confidence per run: median 69 % (55-82 %). The three whole nights:

| Recording | Hours | TST (min) | SE (%) | N2 / N3 / REM (% TST) | Mean confidence (%) | Spindles /min N2+N3, central | Slow waves /min N2+N3, frontal |
|---|---|---|---|---|---|---|---|
| sub-01 run-00 | 13.5 | 376.0 | 46.5 | 52.8 / 15.2 / 15.2 | 76.1 | 0.63 | 0.01 |
| sub-03 run-00 | 13.0 | 381.0 | 49.0 | 67.5 / 5.1 / 14.0 | 81.7 | 0.41 | 0.64 |
| sub-14 run-02 | 11.7 | 313.0 | 44.7 | 83.9 / 0 / 12.5 | 78.0 | 6.26 | 0.02 |

The nights have a plausible shape (wake in the evening, sleep from around 23:00-00:00, NREM-REM
cycles, waking in the morning; SE is low because the recordings start hours before sleep and there
are no lights-off events), but N3 is scarce or absent (none at all in sub-14). But the Siena source was **high-pass filtered at 1.59 Hz** before export (its
sidecar's `SoftwareFilters`; the app warns on every run): slow waves are nearly absent (≤ 3 per
minute of N2+N3 on every run, against ~21 in the unfiltered ds003555 children) and N3 is likely
under-scored. Without ear electrodes the staging falls back to C4 in average reference, and the
derivation matters: on PN06's run-01 (raw BIDS), TST was 88 min on C4, 66 min on C3 and 49 min on
C4-O1. Shorter daytime runs give low TST with mostly N2 and REM at lower confidence, as expected
for naps in a monitoring unit or misclassified drowsy wake; review them against the EEG.

**CHB-MIT chb01** (eeg-prep derivative of 42 one-hour files, bipolar-only, staged on F4-P4), in
`data/work/chbmit/eeg-sleep/`: 39 staged, 3 shorter than an hour skipped, none failed; median
mean confidence 64 %, median TST 26 min per hour; median 0.02 spindles and 6.9 slow waves per
minute of N2+N3 on the pairs touching central and frontal electrodes. Each hour is staged on its own (YASA's
smoothing and robust scaling see one hour, not the night), so treat these as rough.
On the raw file chb01_03 the repeated `T8-P8` and reversed `P7-T7` channels are left out.

## Runtime

`computeTypes: [standard]`, **2 vCPU / 4 GiB**, 2 h timeout. The staging trace is streamed at
100 Hz, detection runs in 2-h windows, channels in parallel; memory is bounded by the window and
the channel count, not by the recording length. With a 2 h timeout (sized for 24 h recordings and
more than one per run) the platform runs it as an ECS task, not a Lambda step; the image is
Lambda-safe all the same (read-only file system, any uid, caches under `$TMPDIR`; tested with
`docker run --read-only --user 65534:65534`).

Measured in Docker (linux/amd64 **emulated** on an M-series Mac), defaults, `--cpus 2`, on the
final eeg-prep outputs with `--memory 2g` while four other apps ran on the same Docker (these are
the runs in `data/work/<demo>/eeg-sleep/`; wall time from the container, the app's own
`wall_seconds` in `provenance.json` is ~2 s less):

| Run | Recording time | Wall time | CPU time | Peak memory (cgroup `memory.peak`, includes the page cache of the files read) | Peak RSS |
|---|---|---|---|---|---|
| ds003555, 3 recordings, 1024 Hz, 23 channels | 9 h | 70 s | 88 s | 1.81 GB | 1.05 GB |
| Siena, 41 runs (35 staged, 6 skipped), 256 Hz, 19 channels | 128 h | 323 s | 464 s | 2.09 GB (near the 2 GiB limit: page cache) | 1.18 GB |
| CHB-MIT chb01, 42 files (39 staged, 3 skipped), 256 Hz, bipolar | 42 h | 147 s | 205 s | 1.69 GB | 0.79 GB |

Earlier, single-recording measurements with `--memory 3g` and nothing else running: Siena sub-01
run-00 (13.5 h, raw BIDS) 82 s wall, 117 s CPU, 1.89 GB cgroup peak, 1.01 GB RSS; the synthetic
fixture Lambda-style (`--read-only --user 65534`) 41 s, 0.69 GB RSS.

About 10 s of every run is importing YASA and compiling AntroPy's numba functions under
emulation. Time grows with recording length × channels × sampling rate (reading and resampling)
plus about 6 CPU-s per channel-hour of detection; a 24 h, 25-channel, 512 Hz recording should take
a few minutes on 2 vCPU and stay under 4 GiB (not measured). The image is ~243 MB compressed
(~940 MB unpacked).

## Limitations

- **EEG-only, single-derivation staging**, without EOG and chin EMG: REM and wake are told apart
  from EEG alone, and N1 is unreliable. The accuracy quoted above is for polysomnography, not for
  clinical EEG with epileptiform activity; nothing here is validated against expert scoring of
  these recordings. Treat the hypnogram as a review candidate.
- **Average reference and bipolar pairs** change the amplitude and spectrum YASA was trained on;
  on Siena the staging changed substantially with the derivation chosen (*Results*). Keep the ear
  or mastoid electrodes in the export when you can.
- **High-pass filtering above 0.5 Hz** (Siena's source was filtered at 1.59 Hz) attenuates delta
  and slow waves: N3 is under-scored and slow waves under-counted. The app reads the filter from
  the sidecars and warns.
- **Children**: YASA's training data start at 5 years; under 5 the hypnogram is unreliable (the app
  warns), and spindles are slower than YASA's default band at least up to school age.
- **Long-term EEG** without lights-off / lights-on events: sleep statistics cover the whole
  recording (daytime included), so SE and SOL are not night measures. Annotate `lights_off` /
  `lights_on` (eeg-prep carries them through) for night statistics.
- Spindle and slow-wave thresholds are YASA's defaults for adult PSG; the RMS threshold is
  computed per 2-h window. Densities are per minute of N2+N3 *as staged*, so staging errors carry
  over.
- Seizures, sedation, encephalopathy and status epilepticus in sleep are outside what any of this
  was trained or tested on.

## Licences

The app is Apache-2.0. YASA, AntroPy, MNE-Python, SciPy, scikit-learn: BSD-3-Clause; LightGBM: MIT.
YASA's classifiers ship inside its BSD-3-Clause wheel with no separate weights licence or gate.
Every bundled component and its licence: `THIRD_PARTY.md`. Demo data: ds003555 CC0; Siena
CC BY 4.0 (the BIDS conversion's `dataset_description.json` says ODC-By 1.0); CHB-MIT ODC-By 1.0.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full
author lists in CITATION.cff.

- Vallat R, Walker MP. An open-source, high-performance tool for automated sleep staging.
  *eLife* 2021;10:e70092. doi:10.7554/eLife.70092. PMID 34648426. (YASA; the method.)
- Dean DA, et al. Scaling Up Scientific Discovery in Sleep Medicine: The National Sleep Research
  Resource. *Sleep* 2016;39(5):1151-1164. doi:10.5665/sleep.5774. PMID 27070134. (Training data of
  YASA's classifiers, as YASA asks.)
- Zhang GQ, et al. The National Sleep Research Resource: towards a sleep data commons. *J Am Med
  Inform Assoc* 2018;25(10):1351-1358. doi:10.1093/jamia/ocy064. PMID 29860441.
- Ke G, et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural
  Information Processing Systems* 30 (2017), 3149-3157.
  https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- Gramfort A, et al. MEG and EEG data analysis with MNE-Python. *Front Neurosci* 2013;7:267.
  doi:10.3389/fnins.2013.00267. PMID 24431986.
- Virtanen P, et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nat
  Methods* 2020;17:261-272. doi:10.1038/s41592-019-0686-2. PMID 32015543.
- Software: YASA 0.8.1 (Raphael Vallat, https://github.com/raphaelvallat/yasa), the classifier
  training code (https://github.com/raphaelvallat/yasa_classifier), AntroPy 0.2.3
  (https://github.com/raphaelvallat/antropy), LightGBM 4.7.0 (https://github.com/lightgbm-org/LightGBM).

Demo data:

- Cserpan D, Boran E, Rosch R, Lo Biundo SP, Ramantani G, Sarnthein J. Dataset of EEG recordings of
  pediatric patients with epilepsy based on the 10-20 system. OpenNeuro ds003555 v1.0.1 (2021).
  doi:10.18112/openneuro.ds003555.v1.0.1. With: Cserpan D, et al. Scalp high-frequency oscillation
  rates are higher in younger children. *Brain Commun* 2021;3(2):fcab052.
  doi:10.1093/braincomms/fcab052. PMID 33870193.
- Dan J, Detti P. BIDS Siena Scalp EEG Database. Zenodo (2024). doi:10.5281/zenodo.10640762.
  Detti P. Siena Scalp EEG Database. PhysioNet (2020). doi:10.13026/5d4a-j060. Detti P, Vatti G,
  Zabalo Manrique de Lara G. EEG Synchronization Analysis for Seizure Prediction: A Study on Data of
  Noninvasive Recordings. *Processes* 2020;8(7):846. doi:10.3390/pr8070846. Goldberger AL, et al.
  PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 2000;101(23).
  doi:10.1161/01.CIR.101.23.e215. PMID 10851218.
- Shoeb A. CHB-MIT Scalp EEG Database. PhysioNet (2010). doi:10.13026/C2K01R; Shoeb AH.
  Application of machine learning to epileptic seizure onset detection and treatment. PhD thesis,
  MIT (2009). https://dspace.mit.edu/handle/1721.1/54669.

## Acknowledgements

YASA and its classifiers are the work of Raphael Vallat and Matthew P. Walker (UC Berkeley). The
classifiers were trained on data from the National Sleep Research Resource, which is funded by
the National Heart, Lung and Blood Institute, and its cohorts (CCSHS, CFS, CHAT, HomePAP, MESA,
MrOS, SHHS). Thanks to the authors of OpenNeuro ds003555 (University Children's Hospital Zurich and
University Hospital Zurich) and of the Siena Scalp EEG Database (University of Siena), and to
PhysioNet and OpenNeuro for hosting them.
