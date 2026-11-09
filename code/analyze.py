import numpy as np, json
Z=np.load('arrays.npz'); MARG=Z['MARG']; sd=Z['sd']
METH=['raw','debias','confirm','gap_noguard','gap']
def pick(kind,N,m,target=0.1):
    o=Z[f'{kind}_{N}_cal_{m}_out'].mean(0); g=Z[f'{kind}_{N}_cal_{m}_gp'].mean(0)   # S,K
    worst=o.max(0); feas=np.where(worst<=target)[0]
    if len(feas)==0: return None
    return feas[np.argmax(g[:,feas].mean(0))]
def ev(kind,N,m,k):
    o=Z[f'{kind}_{N}_ev_{m}_out'][:,:,k]; g=Z[f'{kind}_{N}_ev_{m}_gp'][:,:,k]   # reps,S
    go=g.mean(1); wo=o.max(1)
    ci=lambda a:(a.mean(),2.093*a.std(ddof=1)/np.sqrt(len(a)))
    return dict(gp=ci(go),worst=ci(wo),out_s=o.mean(0))
out={}
for m in METH:
    k=pick(1,64,m)
    print(m,'tuned on N=64 k=',k,None if k is None else MARG[k])
    if k is None: continue
    for N in [16,64,256,1024]:
        r=ev(1,N,m,k); print('   N=%d gp=%.3f±%.3f worstout=%.3f±%.3f'%(N,*r['gp'],*r['worst']), np.round(r['out_s'],3))
print('E2')
for m in METH:
    k=pick(2,64,m)
    if k is None: print(m,'infeasible'); continue
    r=ev(2,64,m,k); print(m,MARG[k],'gp=%.3f±%.3f worstout=%.3f±%.3f'%(*r['gp'],*r['worst']),np.round(r['out_s'],3))
