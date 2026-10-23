# Locus packaging targets.  Nothing here uploads anywhere: no registry push, no PyPI.
PYTHON  ?= python3
IMAGE   ?= locus:local
ENGINE  ?= $(shell command -v docker >/dev/null 2>&1 && echo docker || echo podman)
# podman/buildah default to OCI images, which drop HEALTHCHECK; build Docker-format images.
BUILD_FMT := $(if $(filter podman,$(notdir $(ENGINE))),--format docker,)
VERSION := $(shell $(PYTHON) -c "import re,pathlib;print(re.search(r'^version\s*=\s*\"([^\"]+)\"',pathlib.Path('pyproject.toml').read_text(),re.M).group(1))")
BUNDLE  := locus-offline-$(VERSION)

.PHONY: dist offline-bundle image image-save test clean

## dist: build wheel + sdist into dist/ (needs the `build` package or `uv`; build backend: hatchling)
dist:
	rm -rf dist
	@if $(PYTHON) -c "import build" 2>/dev/null; then \
	  $(PYTHON) -m build --sdist --wheel --outdir dist . ; \
	else \
	  uv build --sdist --wheel --out-dir dist . ; \
	fi
	cd dist && sha256sum * > SHA256SUMS

## offline-bundle: dist/ + installer in one tarball; installs with no network
offline-bundle: dist
	rm -rf build/$(BUNDLE) && mkdir -p build/$(BUNDLE)
	cp dist/*.whl dist/*.tar.gz build/$(BUNDLE)/
	cp scripts/offline-install.sh build/$(BUNDLE)/install.sh
	cp LICENSE build/$(BUNDLE)/
	cd build/$(BUNDLE) && sha256sum *.whl *.tar.gz install.sh LICENSE > SHA256SUMS
	tar -C build --sort=name --owner=0 --group=0 --numeric-owner \
	    --mtime='2026-01-01 00:00:00Z' -czf dist/$(BUNDLE).tar.gz $(BUNDLE)
	cd dist && sha256sum $(BUNDLE).tar.gz > $(BUNDLE).tar.gz.sha256
	@echo "bundle: dist/$(BUNDLE).tar.gz"

## image: build the container image locally from dist/*.whl (never pushed)
image: dist
	$(ENGINE) build $(BUILD_FMT) -t $(IMAGE) .

## image-save: export the image for air-gapped hosts (docker/podman load -i ...)
image-save: image
	$(ENGINE) save -o dist/locus-image-$(VERSION).tar $(IMAGE)

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	rm -rf dist build
