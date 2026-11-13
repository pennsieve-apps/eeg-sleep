# sample-input

Real input for this app is the `eeg` output of **eeg-prep**: a BIDS-EEG derivative with
`recordings.csv` and, per recording, `*_desc-preproc_eeg.edf`, `_channels.tsv`, `_events.tsv`
(see `docs/conventions.md` in workflow-scalp-eeg). Any BIDS-EEG dataset or loose EDF files work too,
referential or bipolar-only. The app stages recordings of at least `MIN_SLEEP_MINUTES` (60); a
routine EEG of 20-30 minutes is skipped by design.

Public recordings with sleep, as used for the README results:

- **OpenNeuro ds003555 v1.0.1** (CC0), *Dataset of EEG recordings of pediatric patients with
  epilepsy based on the 10-20 system*, doi:10.18112/openneuro.ds003555.v1.0.1: the first 3 h of
  sleep EEG of children with epilepsy, 10-20 with A1/A2 (so the app stages on C4-A1), 1024 Hz, EDF.
  `sub-01/ses-01/eeg/sub-01_ses-01_task-hfo_eeg.edf` with its `_channels.tsv` (the EDFs carry no
  unit; `channels.tsv` says µV, which the app reads). The `derivatives/` folder holds 5-min clips;
  the app ignores it when the raw recordings are next to it.
- **BIDS Siena Scalp EEG Database** (Zenodo doi:10.5281/zenodo.10640762, CC BY 4.0, from PhysioNet
  doi:10.13026/5d4a-j060): adults in video-EEG monitoring, 19 channels in average reference at
  256 Hz, runs of tens of minutes to 13.5 h; `sub-01_ses-01_task-szMonitoring_run-00` is a whole
  night. The source was high-pass filtered at 1.59 Hz before export, which attenuates slow waves
  (the app warns).
- CHB-MIT (PhysioNet doi:10.13026/C2K01R) is bipolar-only, in one-hour files: each is just long
  enough for the default `MIN_SLEEP_MINUTES` (60) and is staged on F4-P4 (= F4-C4 + C4-P4); its
  repeated `T8-P8-1` and reversed `P7-T7` channels are left out as duplicates. An hour is short for
  staging, and the hours of one night are staged independently.

```sh
cd ../workflow-scalp-eeg && scripts/fetch-sample-data.sh                 # -> data/
# run eeg-prep on it (-> data/work/<demo>/eeg-prep/eeg), then:
cd ../eeg-sleep && make run-demo DEMO=../workflow-scalp-eeg/data/work/siena/eeg-prep/eeg
# or straight on the raw BIDS dataset:
make run-demo DEMO=../workflow-scalp-eeg/data/ds003555
```

For tests and `make run`, `tests/make_fixture.py` writes a small synthetic derivative instead
(`make fixture` -> `data/fixture/eeg/`): two 65-min "nights" with planted spindles and slow waves
(one referential in average reference with A1/A2 at 256 Hz, with lights off/on, a bad segment and
a seizure; one bipolar-only at 128 Hz) and a 5-min recording that is skipped.
`data/fixture/truth.json` holds the planted times.
