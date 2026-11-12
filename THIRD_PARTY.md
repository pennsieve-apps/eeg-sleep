# Third-party software and models

This app is Apache-2.0 (see `LICENSE`). The image it builds contains the following third-party
components, installed from PyPI as binary wheels at the versions pinned in `requirements.txt`,
plus one Debian package. All are permissively licensed (BSD, MIT, Apache-2.0, PSF-based, MPL-2.0
for pure-Python helpers); nothing GPL is linked in-process.

## Model weights

| Model | Where it comes from | Licence | Use here |
|---|---|---|---|
| YASA sleep-staging classifiers `clf_eeg_lgb_0.5.0.joblib` (EEG only, the default) and `clf_eeg+demo_lgb_0.5.0.joblib` (EEG + age and sex, `USE_DEMOGRAPHICS=true`) | Inside the `yasa` 0.8.1 wheel from PyPI (`yasa/classifiers/`, with six other combinations this app does not use). LightGBM gradient-boosted trees trained by Raphael Vallat on ~3,000 polysomnography nights of the National Sleep Research Resource (NSRR; CCSHS, CFS, CHAT, HomePAP, MESA, MrOS, SHHS), Vallat & Walker 2021; training code at https://github.com/raphaelvallat/yasa_classifier (BSD-3-Clause, Copyright (c) 2021 Raphael Vallat) | BSD-3-Clause, the licence of the `yasa` package that ships them (Copyright (c) 2018, Raphael Vallat; copied to `/opt/eeg-sleep/classifiers/LICENSE-yasa` in the image). No separate weights licence and no access gate | The Dockerfile checks both files against their SHA-256 (`330c1afd…6e31`, `686fd99a…e86b`, the same as the wheel's `RECORD`) and copies them to the read-only `/opt/eeg-sleep/classifiers/` (`YASA_CLASSIFIER_DIR`); the app loads them from there with no network. They were pickled with scikit-learn 0.24.2; only their `LabelEncoder` is a scikit-learn object, and it loads unchanged in 1.9.1 (scikit-learn warns about the version; tests and real-data runs confirm the predictions) |

The NSRR data the classifiers were trained on are not redistributed with them. YASA's documentation
asks users of its default classifiers to cite the two NSRR papers (Dean et al. 2016; Zhang et al.
2018); they are in `CITATION.cff` and the README.

## Software

| Component | Version | What it does here | Licence | Source |
|---|---|---|---|---|
| YASA | 0.8.1 | `SleepStaging` (features + bundled classifiers), `spindles_detect`, `sw_detect`, `Hypnogram.sleep_statistics` | BSD-3-Clause (Copyright (c) 2018, Raphael Vallat) | https://github.com/raphaelvallat/yasa |
| LightGBM | 4.7.0 | Runs the classifiers. The wheel links GNU OpenMP (`libgomp1`, below) | MIT (Copyright (c) Microsoft Corporation; Copyright (c) The LightGBM developers) | https://github.com/lightgbm-org/LightGBM |
| scikit-learn, threadpoolctl, narwhals | 1.9.1, 3.7.0, 2.28.0 | YASA's feature scaling; unpickling the classifiers; LightGBM/scikit-learn dependencies. The scikit-learn wheel bundles libgomp (GPL-3.0 with the GCC Runtime Library Exception) | BSD-3-Clause; BSD-3-Clause; MIT | https://scikit-learn.org |
| AntroPy | 0.2.3 | Entropy and fractal-dimension features of YASA's staging | BSD-3-Clause (Copyright (c) 2018-2021, Raphael Vallat) | https://github.com/raphaelvallat/antropy |
| numba, llvmlite | 0.68.1, 0.50.0 | JIT compiler used by AntroPy | BSD-2-Clause; BSD-2-Clause AND Apache-2.0 WITH LLVM-exception (bundled LLVM) | https://numba.pydata.org |
| lspopt | 1.4.0 | Imported by YASA (multitaper spectrograms; not used here) | MIT | https://github.com/hbldh/lspopt |
| seaborn | 0.13.2 | Imported by YASA's plotting module | BSD-3-Clause | https://seaborn.pydata.org |
| MNE-Python | 1.13.2 | Reads EDF/BDF/BrainVision/FIF/EEGLAB lazily, one window at a time; 10-05 template positions; head maps | BSD-3-Clause | https://mne.tools |
| NumPy | 2.5.4 | Arrays. The wheel bundles OpenBLAS (BSD-3-Clause) and the GCC Fortran runtime (GPL-3.0 with the GCC Runtime Library Exception, which places no obligations on code linked against it) | BSD-3-Clause | https://numpy.org |
| SciPy | 1.18.1 | Polyphase resampling to 100 Hz, Welch spectra, YASA's filters. Bundles OpenBLAS and the GCC Fortran runtime as above | BSD-3-Clause | https://scipy.org |
| pandas, python-dateutil, six, tzdata | 3.0.6, 3.0.0, 1.17.0, 2026.6 | YASA's tables | BSD-3-Clause; Apache-2.0/BSD dual; MIT; Apache-2.0 | https://pandas.pydata.org |
| joblib, cloudpickle | 1.6.0, 3.1.2 | Channels in parallel (process pool); loading the classifiers | BSD-3-Clause | https://joblib.readthedocs.io |
| Matplotlib | 3.11.2 | Figures | Matplotlib License (PSF-based, BSD-compatible); bundled DejaVu/STIX fonts under their own free licences | https://matplotlib.org |
| contourpy, cycler, kiwisolver | 1.4.0, 0.12.1, 1.5.1 | Matplotlib dependencies | BSD-3-Clause | PyPI |
| fonttools, pyparsing | 4.68.0, 3.3.3 | Matplotlib dependencies | MIT | PyPI |
| pillow | 13.0.0 | Matplotlib dependency | MIT-CMU | https://python-pillow.org |
| decorator, lazy-loader, packaging | 5.3.1, 0.6, 26.3 | MNE dependencies | BSD-2-Clause; BSD-3-Clause; Apache-2.0 OR BSD-2-Clause | PyPI |
| Jinja2, MarkupSafe | 3.1.6, 3.0.4 | MNE dependencies (HTML reports, unused) | BSD-3-Clause | https://palletsprojects.com |
| pooch, platformdirs, requests, certifi, charset-normalizer, idna, urllib3 | 1.9.0, 4.14.0, 2.34.2, 2026.10.23, 3.5.2, 3.20, 2.8.0 | MNE / YASA dependencies (dataset downloads, never used: no network at run time) | BSD-3-Clause; MIT; Apache-2.0; MPL-2.0; MIT; BSD-3-Clause; MIT | PyPI |
| tqdm | 4.70.1 | MNE dependency (progress bars) | MPL-2.0 AND MIT | https://tqdm.github.io |
| libgomp1 (Debian 12) | 12.2.0-14+deb12u1 | GNU OpenMP runtime for LightGBM | GPL-3.0 with the GCC Runtime Library Exception | https://packages.debian.org/bookworm/libgomp1 |
| python:3.12.16-slim-bookworm | digest `sha256:935c3c5b…03df` | Base image (Debian 12 + CPython 3.12, pip) | PSF-2.0 (Python); Debian packages under their own free licences; pip MIT | https://hub.docker.com/_/python |

## Data used for testing (not in the image or the repository)

- OpenNeuro ds003555 v1.0.1, *Dataset of EEG recordings of pediatric patients with epilepsy based
  on the 10-20 system* (Cserpan D, Boran E, Rosch R, Lo Biundo SP, Ramantani G, Sarnthein J;
  doi:10.18112/openneuro.ds003555.v1.0.1), CC0.
- BIDS Siena Scalp EEG Database (Dan J, Detti P; Zenodo doi:10.5281/zenodo.10640762), from the
  Siena Scalp EEG Database (Detti P; PhysioNet doi:10.13026/5d4a-j060), CC BY 4.0.
