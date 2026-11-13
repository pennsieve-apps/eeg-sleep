.PHONY: help test fixture run run-demo clean

SERVICE_NAME ?= eeg-sleep
PYTHON ?= python3
IMAGE ?= eeg-sleep:dev
DEMO ?= ../workflow-scalp-eeg/data/work/siena/eeg-prep/eeg

# On macOS, LightGBM's wheel needs an OpenMP runtime; scikit-learn's wheel ships one.
ifeq ($(shell uname -s),Darwin)
export DYLD_FALLBACK_LIBRARY_PATH := $(shell $(PYTHON) -c "import os, sklearn; print(os.path.join(os.path.dirname(sklearn.__file__), '.dylibs'))" 2>/dev/null)
endif

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make test     - pytest: synthetic BIDS-EEG fixture through the processor (no Docker)"
	@echo "make fixture  - write the synthetic fixture to data/fixture/ (eeg/ derivative + truth.json)"
	@echo "make run      - build the image and run it in Docker over data/fixture/eeg, writing to data/output/"
	@echo "make run-demo - run the image over an eeg-prep output (DEMO=$(DEMO)) into data/output-demo/"
	@echo "make clean    - remove data/output*/ and data/fixture/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

test:
	$(PYTHON) -m pytest -q tests

fixture:
	$(PYTHON) -m tests.make_fixture data/fixture

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -type f | sort

run-demo:
	docker build --platform linux/amd64 -t $(IMAGE) .
	rm -rf data/output-demo && mkdir -p data/output-demo
	docker run --rm --platform linux/amd64 --cpus 2 --memory 3g --env-file dev.env \
		-v "$(abspath $(DEMO))":/data/input:ro -v "$(CURDIR)/data/output-demo":/data/output $(IMAGE)

clean:
	rm -rf data/output data/output-demo data/fixture
