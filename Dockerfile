FROM --platform=linux/amd64 python:3.12.16-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

WORKDIR /processor

# LightGBM's wheel links the GNU OpenMP runtime, which the slim image does not have.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1=12.2.0-14+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

# Pinned Python dependencies, binary wheels only, no dependency resolution at build time
# (requirements.txt lists every package).
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --no-deps --only-binary=:all: --root-user-action=ignore -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12 -depth -type d -name __pycache__ -exec rm -rf {} + \
    && find /usr/local/lib/python3.12/site-packages -depth -type d \( -name tests -o -name testing \) -path "*/mne/*" -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/pandas/tests /usr/local/lib/python3.12/site-packages/scipy/*/tests \
              /usr/local/lib/python3.12/site-packages/numpy/*/tests /usr/local/lib/python3.12/site-packages/matplotlib/tests \
              /usr/local/lib/python3.12/site-packages/sklearn/*/tests

# YASA's pre-trained LightGBM sleep-staging classifiers ship inside the yasa 0.8.1 wheel
# (yasa/classifiers/, BSD-3-Clause, see THIRD_PARTY.md). The two this app uses (EEG only, and EEG +
# age/sex) are checked against their SHA-256 and copied to a read-only folder; nothing is downloaded
# at run time.
RUN mkdir -p /opt/eeg-sleep/classifiers \
    && CLF=/usr/local/lib/python3.12/site-packages/yasa/classifiers \
    && echo "330c1afdc86396e0b6d9f1294d4c4796beb5f7e7cd38be424c10feae98867e31  $CLF/clf_eeg_lgb_0.5.0.joblib" > /tmp/clf.sha256 \
    && echo "686fd99ad834d657833d7d7c2b3256f9689c023cf13aee16381b2b2cd6f0e86b  $CLF/clf_eeg+demo_lgb_0.5.0.joblib" >> /tmp/clf.sha256 \
    && sha256sum -c /tmp/clf.sha256 \
    && cp "$CLF/clf_eeg_lgb_0.5.0.joblib" "$CLF/clf_eeg+demo_lgb_0.5.0.joblib" /opt/eeg-sleep/classifiers/ \
    && cp /usr/local/lib/python3.12/site-packages/yasa-0.8.1.dist-info/licenses/LICENSE /opt/eeg-sleep/classifiers/LICENSE-yasa \
    && rm /tmp/clf.sha256 \
    && MPLCONFIGDIR=/opt/eeg-sleep/matplotlib python3 -c "import matplotlib.font_manager" \
    && YASA_CLASSIFIER_DIR=/opt/eeg-sleep/classifiers python3 -c "import joblib, lightgbm, yasa, sys; \
from pathlib import Path; [joblib.load(p) for p in Path('/opt/eeg-sleep/classifiers').glob('*.joblib')]; print('classifiers ok')" \
    && find /usr/local/lib/python3.12 -depth -type d -name __pycache__ -exec rm -rf {} + \
    && chmod -R a+rX /opt/eeg-sleep

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

# Caches that must be written (Matplotlib, numba, MNE) go under $TMPDIR at start-up
# (processor/main.py), because only INPUT_DIR, OUTPUT_DIR, HOME and TMPDIR are writable on Lambda.
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YASA_CLASSIFIER_DIR=/opt/eeg-sleep/classifiers \
    MNE_DONTWRITE_HOME=true \
    MNE_USE_CUDA=false

# Runs as uid 1000 on ECS and as an arbitrary user on Lambda; nothing here needs root or a HOME.
RUN echo "app:x:1000:1000:app:/tmp:/usr/sbin/nologin" >> /etc/passwd
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
