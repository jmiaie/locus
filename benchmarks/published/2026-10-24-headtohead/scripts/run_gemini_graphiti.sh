#!/bin/bash
# Detached supervisor for the Graphiti-on-Gemini side-by-side run (multi-key). Resumes from results/lme30__graphiti_gemini.jsonl.
# usage: run_gemini_graphiti.sh [LIMIT]   (LIMIT = total target incl. done; default 30)
cd /workspace/llmbench
LIMIT=${1:-30}; OUTF=results/lme30__graphiti_gemini.jsonl; LOG=p2_graphiti_gemini.log
export GM_RPM=${GM_RPM:-13} GM_TPM=${GM_TPM:-230000}
MAXW=${GM_MAX_WORKERS:-4}; MB_PER_WORKER=${GM_MB_PER_WORKER:-900}; MB_RESERVE=${GM_MB_RESERVE:-1500}
ndone() { [ -f $OUTF ] && grep -vc '"error"' $OUTF || echo 0; }
while true; do
  [ -f GEMINI_STOP ] && { echo "$(date) supervisor: GEMINI_STOP present, exiting" >> $LOG; exit 3; }
  [ "$(ndone)" -ge "$LIMIT" ] && { echo "$(date) supervisor: done $(ndone)/$LIMIT" >> $LOG; exit 0; }
  if ! curl -sf localhost:8766/health >/dev/null; then
    (export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false DO_NOT_TRACK=1; unset OPENROUTER_API_KEY; nohup venvs/v-p2/bin/python gateway_gemini.py >> gateway_gemini.log 2>&1 &)
    for i in $(seq 1 100); do sleep 3; curl -sf localhost:8766/health >/dev/null && break; done
  fi
  # workers = independent free-tier pools (keys minus detected shared projects), capped by free memory and MAXW
  NPOOL=$(curl -sf localhost:8766/health | python3 -c "import sys,json; print(len(json.load(sys.stdin)['pools']))" 2>/dev/null || echo 1)
  AVAIL=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo); MEMW=$(( (AVAIL - MB_RESERVE) / MB_PER_WORKER ))
  W=$NPOOL; [ $W -gt $MAXW ] && W=$MAXW; [ $W -gt $MEMW ] && W=$MEMW; [ $W -lt 1 ] && W=1
  # wipe partial graph DBs of unfinished questions (a partial Kuzu DB would contaminate a re-run)
  mkdir -p graphiti_db_gemini; for d in graphiti_db_gemini/*; do [ -d "$d" ] || continue; q=$(basename $d); grep -q "\"qid\": \"$q\", \"qtype\"" $OUTF 2>/dev/null || rm -rf "$d"; done
  echo "$(date) supervisor: starting p2_run (done $(ndone)/$LIMIT, pools $NPOOL, MemAvailable ${AVAIL}MB, workers $W)" >> $LOG
  (for v in $(env | grep -o '^GEMINI_API_KEY[_0-9]*'); do unset $v; done; source p2_common.sh
   export P2_PROVIDER=gemini P2_GEMINI_MODEL=${P2_GEMINI_MODEL:-gemini-3.5-flash-lite}
   venvs/v-p2/bin/python p2_run.py graphiti --workers $W --order lme_subset30_order.json --limit $LIMIT >> $LOG 2>&1)
  echo "$(date) supervisor: p2_run exited $?" >> $LOG
  sleep 60
done
