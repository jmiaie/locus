"""Regression: the HTTP transport must not expose an unauthenticated bind-all.

Guards the fix for the `--host` default. Before it, `locus serve` bound 0.0.0.0 with
no token, so every Locus tool was reachable by anything that could route to the port.

The case worth naming: "" is NOT loopback. bind(("", port)) resolves to 0.0.0.0, so
an empty host in the allow-list re-opens exactly the hole the guard closes.
"""
import http.server

import pytest

from locus.mcp import http_server as hs


class _StubServer(http.server.HTTPServer):
    """Real HTTPServer; we only stop it before it blocks."""

    def serve_forever(self, *a, **kw):
        raise KeyboardInterrupt

    def server_close(self):
        pass


@pytest.fixture(autouse=True)
def _no_block(monkeypatch):
    monkeypatch.setattr(hs, "HTTPServer", _StubServer)


class _Engine:
    pass


@pytest.mark.parametrize("host", ["0.0.0.0", "", "::", "192.168.1.10"])
def test_non_loopback_without_token_is_refused(host):
    """An unauthenticated permissive bind must raise, not silently serve."""
    with pytest.raises(SystemExit):
        hs.serve(_Engine(), host=host, port=0, token=None)


def _ipv6_loopback_available() -> bool:
    """True if ::1 can actually be served here.

    Probe with HTTPServer, NOT a raw AF_INET6 socket. HTTPServer defaults to
    address_family=AF_INET, so it resolves "::1" as IPv4 and raises gaierror even on a
    host whose IPv6 stack works fine. A raw-socket probe returns True and the test then
    fails -- the probe must exercise the same mechanism the code does.
    """
    try:
        srv = http.server.HTTPServer(("::1", 0), http.server.BaseHTTPRequestHandler)
    except OSError:
        return False
    srv.server_close()
    return True


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_loopback_without_token_is_allowed(host):
    """Local-only access stays the easy path."""
    hs.serve(_Engine(), host=host, port=0, token=None)


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="no IPv6 loopback on this host")
def test_ipv6_loopback_without_token_is_allowed():
    """::1 is loopback and must be allowed, where the host has IPv6 at all."""
    hs.serve(_Engine(), host="::1", port=0, token=None)


@pytest.mark.parametrize("host", ["0.0.0.0", ""])
def test_non_loopback_with_token_is_allowed(host):
    """A remote bind IS possible -- when auth is deliberately paired with it."""
    hs.serve(_Engine(), host=host, port=0, token="sekrit")


def test_default_is_loopback_not_bind_all():
    """The default must stay safe even if every guard above is removed."""
    import inspect

    sig = inspect.signature(hs.serve)
    assert sig.parameters["host"].default == "127.0.0.1", (
        "serve() default drifted off loopback -- this is the line that was 0.0.0.0"
    )
    assert "" not in hs._LOOPBACK, (
        "'' resolves to 0.0.0.0 in bind(); it must never be treated as loopback"
    )
