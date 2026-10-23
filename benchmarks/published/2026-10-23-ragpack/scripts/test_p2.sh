P=/workspace/stem/venv/bin/python
run() { n=$1; c=$2; for b in scifact nfcorpus; do $P run.py $n $b --splits test --cfg "$c"; done; $P run.py $n construction --cfg "$c"; $P run.py $n lme --cfg "$c"; }
run t_base '{}'
run t_ch200h '{"chunking":{"chunk_words":200,"overlap_words":0,"heading_context":true}}'
run t_ch400hl '{"chunking":{"chunk_words":400,"overlap_words":50,"heading_context":true,"boundaries":"lines"}}'
run t_pc100 '{"chunking":{"chunk_words":100,"overlap_words":0,"heading_context":true},"retrieve":{"parent":"doc","chunk_limit":10}}'
run t_rm3 '{"retrieve":{"expansion":"rm3","rm3":{"fb_docs":10,"fb_terms":20,"orig_weight":0.9}}}'
echo P2_DONE
