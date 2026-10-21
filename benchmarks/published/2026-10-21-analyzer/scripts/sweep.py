"""Dev-split sweep (doc-level BM25 sim with FTS5's IDF) — tokenizer variants x k1/b. Dev only."""
import json, csv, math, sys, re
from collections import Counter, defaultdict
sys.path.insert(0, '/workspace/scorecard-20261020/repos/locus')
from locus.memory.corpus import STOPWORDS
from locus.memory.stemmer import stem
D='/workspace/ompa-locus-20261020/data'
EXT = STOPWORDS | set("""i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself they them their theirs themselves what which who whom am about against between into through during before after above below up down out off over under again further once here there when where why how all any both each few most other some such nor only own same too very can just don now also into onto upon via""".split())
def tok(text, st=False, sw=STOPWORDS):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    out = [w for w in text.split() if w not in sw and len(w) > 1]
    return [stem(w) for w in out] if st else out
def load(ds, split):
    corpus=[(json.loads(l)) for l in open(f'{D}/{ds}/corpus.jsonl')]
    qs={json.loads(l)['_id']:json.loads(l)['text'] for l in open(f'{D}/{ds}/queries.jsonl')}
    qrels=defaultdict(dict)
    r=csv.reader(open(f'{D}/{ds}/qrels/{split}.tsv'),delimiter='\t'); next(r)
    for q,d,s in r:
        if int(s)>0: qrels[q][d]=int(s)
    return corpus,qs,{q:v for q,v in qrels.items() if q in qs}
def ndcg(ranked, rel, k=10):
    dcg=sum((2**rel.get(d,0)-1)/math.log2(i+2) for i,d in enumerate(ranked[:k]))
    ideal=sorted(rel.values(),reverse=True)[:k]
    idcg=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
    return dcg/idcg if idcg else 0
def run(ds, split, variants, grid):
    corpus,qs,qrels=load(ds,split)
    res={}
    for vname,(st,sw) in variants.items():
        docs=[tok('# '+(c.get('title') or '')+'\n\n'+c['text'],st,sw) for c in corpus]
        ids=[c['_id'] for c in corpus]; N=len(docs); dl=[len(d) for d in docs]; avg=sum(dl)/N
        post=defaultdict(list)
        for i,d in enumerate(docs):
            for t,c in Counter(d).items(): post[t].append((i,c))
        qtoks={q:tok(qs[q],st,sw) for q in qrels}
        for k1,b in grid:
            sc=[]
            for q,rel in qrels.items():
                s=defaultdict(float)
                for t in qtoks[q]:
                    pl=post.get(t)
                    if not pl: continue
                    n=len(pl); idf=math.log((N-n+0.5)/(n+0.5)); idf=max(idf,1e-6)
                    for i,c in pl:
                        s[i]+=idf*c*(k1+1)/(c+k1*(1-b+b*dl[i]/avg))
                top=sorted(s,key=lambda i:(-s[i],i))[:10]
                sc.append(ndcg([ids[i] for i in top],rel))
            res[(vname,k1,b)]=sum(sc)/len(sc)
    return res, len(qrels)
if __name__=='__main__':
    ds,split=sys.argv[1],sys.argv[2]
    V={'base':(False,STOPWORDS),'stem':(True,STOPWORDS),'stem_ext':(True,EXT),'ext':(False,EXT)}
    grid=[(1.2,0.75)]+[(k,b) for k in (0.6,0.9,1.2,1.5,2.0) for b in (0.3,0.5,0.75,0.9) if (k,b)!=(1.2,0.75)]
    r,n=run(ds,split,V,grid)
    print(ds,split,'nq',n)
    for v in V:
        best=max(((k,b) for (vv,k,b) in r if vv==v), key=lambda kb:r[(v,)+kb])
        print(f"{v:9s} default(1.2,0.75)={r[(v,1.2,0.75)]:.4f}  best{best}={r[(v,)+best]:.4f}")
    json.dump({f"{a}|{b}|{c}":v for (a,b,c),v in r.items()}, open(f'/workspace/stem/sweep_{ds}_{split}.json','w'),indent=0)
