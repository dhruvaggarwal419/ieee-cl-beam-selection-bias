import numpy as np, json
rng=np.random.default_rng(5); out={}
m=9.0
for N in [16,64,256]:
  for sdb in [-3,0,5,10,20]:
    s=10**(sdb/10); B=400000
    y=rng.exponential(1,(B,N)); y[:,0]*=1+s
    w=y.argmax(1)==0
    ys=np.sort(y,1); M=ys[:,-2]; yw=y.max(1)
    shat=np.maximum(yw-M-1,0)[w]/m
    sinst=rng.exponential(s,w.sum())
    o=np.mean(shat>sinst); se=np.sqrt(o*(1-o)/w.sum())
    mu=1+s; f=np.exp(-1/mu)*mu/(mu+m*s)
    bias=np.mean((yw-M-1)[w]-s); sb=np.std((yw-M-1)[w])/np.sqrt(w.sum())
    out[f'{N}_{sdb}']=dict(mc=float(o),se=float(se),formula=float(f),bias=float(bias),bias_se=float(sb),pwin=float(w.mean()))
    print(N,sdb,'mc %.4f±%.4f formula %.4f bias %.3f±%.3f pwin %.3f'%(o,se,f,bias,sb,w.mean()))
json.dump(out,open('prop_check.json','w'),indent=1)
