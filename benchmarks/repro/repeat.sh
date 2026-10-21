#!/bin/bash
# Run the repo's wiki and simple-100 benchmarks 5x each, each in a fresh process with a
# different PYTHONHASHSEED, to demonstrate determinism. usage: wiki_repeat.sh REPO_DIR TAG
REPO=$(cd "$1" && pwd); TAG=$2; PY=${PYTHON:-python}
OUT=$(mkdir -p "${LOCUS_BENCH_OUT:-benchmarks/results/repro}/repeat_$TAG" && cd "${LOCUS_BENCH_OUT:-benchmarks/results/repro}/repeat_$TAG" && pwd)
cd $REPO
for i in 1 2 3 4 5; do
  PYTHONHASHSEED=$((i*7919)) $PY benchmarks/run.py --no-latency --out $OUT/wiki_run$i.json > $OUT/wiki_run$i.txt 2>&1
  PYTHONHASHSEED=$((i*7919)) $PY benchmarks/run.py --simple --docs 100 --no-latency --out $OUT/simple100_run$i.json > $OUT/simple100_run$i.txt 2>&1
done
echo done
