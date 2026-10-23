#!/usr/bin/env bash
# Smoke-test a locally built Locus image.  Never pushes anything.
#   make image && scripts/container-smoke-test.sh [image]
# Uses docker if present, else podman.  Needs curl and python3 on the host.
set -euo pipefail
IMAGE="${1:-locus:local}"
ENGINE="${ENGINE:-$(command -v docker >/dev/null 2>&1 && echo docker || echo podman)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOCS="$ROOT/examples/construction/docs"
WORK="$(mktemp -d)"; VOL="locus-smoke-$$"; NAME="locus-smoke-$$"; PORT="${PORT:-17391}"
pass=0; fail=0
ok()   { echo "PASS  $*"; pass=$((pass+1)); }
bad()  { echo "FAIL  $*"; fail=$((fail+1)); }
check(){ local want="$1" got="$2" what="$3"; [[ "$got" == "$want" ]] && ok "$what ($got)" || bad "$what (want $want, got $got)"; }
cleanup(){ $ENGINE rm -f "$NAME" "$NAME-lo" >/dev/null 2>&1 || true; $ENGINE volume rm -f "$VOL" >/dev/null 2>&1 || true; rm -rf "$WORK"; }
trap cleanup EXIT
HARDEN=(--read-only --tmpfs /tmp:size=16m --cap-drop ALL --security-opt no-new-privileges --memory 512m --pids-limit 128)
run(){ $ENGINE run --rm --network=none "${HARDEN[@]}" -v "$VOL:/data" -v "$DOCS:/docs:ro" "$IMAGE" "$@"; }

$ENGINE volume rm -f "$VOL" >/dev/null 2>&1 || true
echo "== image: $IMAGE ($ENGINE)"
$ENGINE image inspect "$IMAGE" --format 'size={{.Size}} user={{.Config.User}}'
check "10001:10001" "$($ENGINE image inspect "$IMAGE" --format '{{.Config.User}}')" "runs as non-root user"
[[ -n "$($ENGINE image inspect "$IMAGE" --format '{{json .Config.Healthcheck}}' | grep -v null || true)" ]] && ok "HEALTHCHECK present" || bad "HEALTHCHECK missing"
check "uid=10001" "$($ENGINE run --rm --network=none --entrypoint python "$IMAGE" -c 'import os;print("uid=%d"%os.getuid())')" "process uid"

echo "== index + query (read-only rootfs, no network, all caps dropped)"
$ENGINE volume create "$VOL" >/dev/null
if out="$(run index /docs 2>&1)"; then echo "$out" | tail -3; ok "index /docs"; else bad "index /docs: $out"; fi
q="$(run retrieve "Saturday work hours permit amendment" --limit 3 --raw)"
echo "$q" | python3 -c 'import json,sys; r=json.load(sys.stdin); r=r if isinstance(r,list) else r.get("results",r); print("top:", r[0]["doc_path"])'
[[ "$q" == *"BP-2026-0418-amendment-1"* ]] && ok "retrieve returns the permit amendment with its source path" || bad "retrieve"

echo "== MCP stdio"
mcp="$(printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"locus_retrieve","arguments":{"query":"RFI-019 delayed egress Stair 2","limit":2,"track":false}}}' \
  | $ENGINE run -i --rm --network=none "${HARDEN[@]}" -v "$VOL:/data" "$IMAGE" mcp)"
n_tools="$(echo "$mcp" | python3 -c 'import json,sys
for l in sys.stdin:
    l=l.strip()
    if l.startswith("{"):
        m=json.loads(l)
        if m.get("id")==2: print(len(m["result"]["tools"]))')"
[[ "${n_tools:-0}" -gt 0 ]] && ok "MCP tools/list ($n_tools tools)" || bad "MCP tools/list"
[[ "$mcp" == *"RFI-019"* ]] && ok "MCP tools/call locus_retrieve cites rfis/RFI-019" || bad "MCP tools/call"

echo "== HTTP, default bind (127.0.0.1 inside the container)"
$ENGINE run -d --name "$NAME-lo" "${HARDEN[@]}" -p "127.0.0.1:$((PORT+1)):7391" -v "$VOL:/data" "$IMAGE" >/dev/null
sleep 3
inside="$($ENGINE exec "$NAME-lo" python -c 'import urllib.request as u;print(u.urlopen("http://127.0.0.1:7391/health").status)')"
check 200 "$inside" "health from inside the container"
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$((PORT+1))/health" || true)"
check 000 "$code" "published port is NOT reachable when bound to container loopback"
if [[ "$ENGINE" == podman ]]; then
  $ENGINE healthcheck run "$NAME-lo" >/dev/null 2>&1 && ok "podman healthcheck run" || bad "podman healthcheck run"
fi
lo_rpc="$($ENGINE exec "$NAME-lo" python -c '
import urllib.request as u, urllib.error as e
r=u.Request("http://127.0.0.1:7391/rpc", data=b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\"}", headers={"Content-Type":"application/json","Host":"evil.example"})
try: print(u.urlopen(r).status)
except e.HTTPError as x: print(x.code)')"
check 403 "$lo_rpc" "no-token server rejects a non-loopback Host header (DNS-rebinding guard)"
$ENGINE rm -f "$NAME-lo" >/dev/null

echo "== HTTP, remote mode refuses to start without a token"
set +e; $ENGINE run --rm "${HARDEN[@]}" -e LOCUS_HTTP_HOST=0.0.0.0 -v "$VOL:/data" "$IMAGE" >"$WORK/notoken.log" 2>&1; rc=$?; set -e
check 2 "$rc" "exit code without token on 0.0.0.0"; grep -q "Refusing to bind" "$WORK/notoken.log" && ok "refusal message" || bad "refusal message"

echo "== HTTP, remote mode with a token file (secret)"
python3 -c 'import secrets;print(secrets.token_hex(32))' > "$WORK/token"; chmod 644 "$WORK/token"
TOKEN="$(cat "$WORK/token")"
$ENGINE run -d --name "$NAME" "${HARDEN[@]}" -p "127.0.0.1:$PORT:7391" -e LOCUS_HTTP_HOST=0.0.0.0 \
  -e LOCUS_HTTP_TOKEN_FILE=/run/secrets/locus_token -v "$WORK/token:/run/secrets/locus_token:ro" \
  -v "$VOL:/data" "$IMAGE" >/dev/null
for i in $(seq 1 20); do curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 0.5; done
U="http://127.0.0.1:$PORT"; J='Content-Type: application/json'
BODY='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"locus_retrieve","arguments":{"query":"CO-003 roof membrane upgrade cost","limit":2,"track":false}}}'
check 200 "$(curl -s -o /dev/null -w '%{http_code}' $U/health)" "GET /health (unauthenticated)"
check 401 "$(curl -s -o /dev/null -w '%{http_code}' -H "$J" -d "$BODY" $U/rpc)" "POST /rpc without token"
check 401 "$(curl -s -o /dev/null -w '%{http_code}' -H "$J" -H 'Authorization: Bearer wrong' -d "$BODY" $U/rpc)" "POST /rpc with wrong token"
check 401 "$(curl -s -o /dev/null -w '%{http_code}' $U/tools)" "GET /tools without token"
check 403 "$(curl -s -o /dev/null -w '%{http_code}' -H "$J" -H "Authorization: Bearer $TOKEN" -H 'Origin: https://evil.example' -d "$BODY" $U/rpc)" "POST /rpc with token but foreign Origin"
check 200 "$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" $U/tools)" "GET /tools with token"
resp="$(curl -s -H "$J" -H "Authorization: Bearer $TOKEN" -d "$BODY" $U/rpc)"
[[ "$resp" == *"CO-003"* ]] && ok "POST /rpc with token returns a cited result (CO-003)" || bad "POST /rpc with token: ${resp:0:200}"
$ENGINE logs "$NAME" 2>&1 | grep -q "$TOKEN" && bad "token appears in container logs" || ok "token not in container logs"
$ENGINE inspect "$NAME" --format '{{.Config.Cmd}} {{.Config.Env}}' | grep -q "$TOKEN" && bad "token visible in container config" || ok "token not in container config/env"

t0=$(date +%s%N); $ENGINE stop -t 10 "$NAME" >/dev/null; t1=$(date +%s%N)
ms=$(( (t1-t0)/1000000 )); rc="$($ENGINE inspect "$NAME" --format '{{.State.ExitCode}}')"
[[ $ms -lt 5000 && "$rc" == 0 ]] && ok "clean stop on SIGTERM (${ms} ms, exit $rc)" || bad "stop took ${ms} ms, exit $rc"

echo "== offline install from the bundle (no network)"
BUNDLE="$(ls "$ROOT"/dist/locus-offline-*.tar.gz 2>/dev/null | head -1 || true)"
if [[ -n "$BUNDLE" ]]; then
  BASE="$(sed -n 's/^ARG BASE=//p' "$ROOT/Dockerfile")"
  inst="$($ENGINE run --rm --network=none -v "$BUNDLE:/b.tgz:ro" "$BASE" sh -c \
    'cd /tmp && tar xzf /b.tgz && python -m venv /tmp/v && ./locus-offline-*/install.sh /tmp/v/bin/python 2>&1 | tail -2 && /tmp/v/bin/locus --help >/dev/null && echo INSTALL_OK')"
  echo "$inst" | tail -2
  [[ "$inst" == *INSTALL_OK* ]] && ok "offline bundle installs with --network=none" || bad "offline bundle install"
else
  echo "SKIP  offline bundle (run make offline-bundle first)"
fi

echo "== result: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
