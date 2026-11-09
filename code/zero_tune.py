import numpy as np, json
from sim import e1_batch, outage_e1
sd=[-5,0,5,10,15,20]; m=9.0; R={}
for N in [16,64,1024]:
    R[N]={}
    for meth in ['raw','debias','confirm','gap']:
        per=[]
        for i,sdb in enumerate(sd):
            s=10**(sdb/10); o=[];g=[]
            for r in range(20):
                rng=np.random.default_rng(900+r*31+i+N)
                est,ssel,_=e1_batch(rng,N,s,3000)
                e=est[meth]; oo,gg=outage_e1(np.maximum(e,0)/m,ssel); o.append(oo.mean()); g.append(gg.mean())
            o=np.array(o);g=np.array(g)
            per.append((float(o.mean()),float(2.093*o.std(ddof=1)/np.sqrt(20)),float(g.mean())))
        R[N][meth]=per
    print(N,{k:[round(x[0],3) for x in v] for k,v in R[N].items()})
json.dump(R,open('zero_tune.json','w'))
