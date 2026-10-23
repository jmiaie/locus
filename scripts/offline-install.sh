#!/bin/sh
# Install Locus from this bundle with no network access.
# Usage: ./install.sh [python-executable]        (default: python3)
# Installs into the given interpreter's environment; use a venv:
#   python3 -m venv /opt/locus && ./install.sh /opt/locus/bin/python
set -eu
PY="${1:-python3}"
cd "$(dirname "$0")"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum -c SHA256SUMS
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), "Locus needs Python >= 3.11"'
# Locus has zero runtime dependencies, so --no-index never needs PyPI.
"$PY" -m pip install --no-index --find-links . locus-rag
"$PY" -m locus --help >/dev/null && echo "Locus installed: $("$PY" -c 'import locus; print(locus.__version__)')"
