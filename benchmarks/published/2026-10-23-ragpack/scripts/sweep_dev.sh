P=/workspace/stem/venv/bin/python
for ds in scifact nfcorpus; do
 for cw in 100 200; do for ov in 0 q; do for h in false true; do
  o=0; [ $ov = q ] && o=$((cw/4))
  $P run.py ch_c${cw}_o${o}_h${h} $ds --splits dev --cfg "{\"chunking\":{\"chunk_words\":$cw,\"overlap_words\":$o,\"heading_context\":$h}}"
 done; done; done
 for fd in 5 10; do for ft in 10 20; do for lam in 0.5 0.7 0.9; do
  $P run.py rm3_d${fd}_t${ft}_l${lam} $ds --splits dev --cfg "{\"retrieve\":{\"expansion\":\"rm3\",\"rm3\":{\"fb_docs\":$fd,\"fb_terms\":$ft,\"orig_weight\":$lam}}}"
 done; done; done
done
echo SWEEP_DONE
