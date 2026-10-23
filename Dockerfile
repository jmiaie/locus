# syntax=docker/dockerfile:1
#
# Locus on-prem image: slim, non-root, read-only-rootfs friendly, zero runtime deps.
#
# Build (no network needed beyond the pinned base image; the wheel comes from dist/):
#   make dist && docker build -t locus:local .        # or: make image
#
# Security model (same as `locus serve`, see README "Security"):
#   * HTTP binds 127.0.0.1 *inside the container* by default -> not reachable from
#     outside even if a port is published.  Use `docker exec` or MCP stdio
#     (`docker run -i --rm ... locus mcp`) for local use.
#   * Remote use: set LOCUS_HTTP_HOST=0.0.0.0 AND give a bearer token, preferably
#     as a file/secret via LOCUS_HTTP_TOKEN_FILE.  Locus refuses to bind a
#     non-loopback address without a token.  Publish the port on the host's
#     loopback (127.0.0.1:7391:7391) and put TLS in front (reverse proxy) for
#     anything beyond the host.

# python:3.12-slim (Debian trixie), multi-arch index digest, pinned 2026-10-21.
ARG BASE=docker.io/library/python:3.12-slim@sha256:2259debdc6af85626058a15a7bf44ad6a68b8c98c9ae82a02cf5cdb92f096d47

FROM ${BASE}

ARG LOCUS_UID=10001
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOCUS_STORE=/data/.locus \
    LOCUS_HTTP_HOST=127.0.0.1 \
    LOCUS_HTTP_PORT=7391

# Install the locally built wheel only: Locus has no runtime dependencies, so
# --no-index guarantees nothing is fetched from PyPI.
COPY dist/*.whl /tmp/wheels/
RUN python -m pip install --no-index --no-compile /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels \
 && python -m pip uninstall -y pip setuptools wheel >/dev/null 2>&1 || true \
 && groupadd --system --gid ${LOCUS_UID} locus \
 && useradd --system --uid ${LOCUS_UID} --gid locus --home-dir /data --shell /usr/sbin/nologin locus \
 && mkdir -p /data /docs \
 && chown locus:locus /data

USER ${LOCUS_UID}:${LOCUS_UID}
WORKDIR /data
# /data holds the index (mount a volume); /docs is for source documents (mount read-only).
VOLUME ["/data"]
EXPOSE 7391

# /health is unauthenticated and returns only {"ok": true, "version": ...}.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('LOCUS_HTTP_PORT','7391'), timeout=2).status == 200 else 1)"]

ENTRYPOINT ["locus"]
CMD ["serve"]
