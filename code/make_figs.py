import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
plt.rcParams.update({'font.family':'serif','font.size':7,'pdf.fonttype':42})
Z=np.load('arrays.npz'); MARG=Z['MARG']; B=json.load(open('bias.json'))
METH=['raw','debias','confirm','gap']
LAB={'raw':'Raw max','debias':'Mean-offset','confirm':'Confirm slot','gap':'Gap (proposed)'}
COL={'raw':'#c0392b','debias':'#8e6bbf','confirm':'#2e86c1','gap':'#1e8449'}
MK={'raw':'o','debias':'s','confirm':'^','gap':'D'}
def ci(a): return a.mean(), 2.093*a.std(ddof=1)/np.sqrt(len(a))
def pick(kind,N,m,target=0.1):
    o=Z[f'{kind}_{N}_cal_{m}_out'].mean(0); g=Z[f'{kind}_{N}_cal_{m}_gp'].mean(0)
    feas=np.where(o.max(0)<=target)[0]; return feas[np.argmax(g[:,feas].mean(0))]
res={}
for m in METH+['gap_noguard']:
    k=pick(1,64,m); res[m]={'k':int(k),'m':float(MARG[k][0]),'thr':float(MARG[k][1])}
    for N in [16,64,256,1024]:
        o=Z[f'1_{N}_ev_{m}_out'][:,:,k]; g=Z[f'1_{N}_ev_{m}_gp'][:,:,k]
        res[m][N]={'worst':ci(o.max(1)),'gp':ci(g.mean(1))}
    k2=pick(2,64,m); o=Z[f'2_64_ev_{m}_out'][:,:,k2]; g=Z[f'2_64_ev_{m}_gp'][:,:,k2]
    res[m]['E2']={'m':float(MARG[k2][0]),'thr':float(MARG[k2][1]),'worst':ci(o.max(1)),'gp':ci(g.mean(1))}
json.dump(res,open('final.json','w'),indent=1,default=float)
# ---- results figure
fig,ax=plt.subplots(1,2,figsize=(7.0,2.2))
sd=[-10,-5,0,5,10,15,20]
for m in METH:
    d=B['E1_bias_N64'][m]; y=[d[str(s)][0] for s in sd]; e=[d[str(s)][1] for s in sd]
    ax[0].errorbar(sd,y,yerr=e,color=COL[m],marker=MK[m],ms=3.2,lw=1,capsize=1.5,label=LAB[m])
ax[0].axhline(0,color='k',lw=.5)
ax[0].set_xlabel('True SNR of the best beam (dB)'); ax[0].set_ylabel('Mean report error (noise units)')
ax[0].set_title('(a) Bias of the reported SNR, N = 64',fontsize=7.5); ax[0].legend(frameon=False,fontsize=6.5); ax[0].grid(alpha=.3)
Ns=[16,64,256,1024]
for m in METH:
    y=[res[m][N]['worst'][0] for N in Ns]; e=[res[m][N]['worst'][1] for N in Ns]
    ax[1].errorbar(range(4),y,yerr=e,color=COL[m],marker=MK[m],ms=3.2,lw=1,capsize=1.5,label=LAB[m])
ax[1].axhline(0.1,color='k',ls='--',lw=.7); ax[1].text(3.02,0.102,'target',fontsize=6,ha='right',va='bottom')
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(Ns); ax[1].set_xlabel('Number of beams N at deployment (tuned at N = 64)')
ax[1].set_ylabel('Worst outage over SNR'); ax[1].set_title('(b) Outage after tuning at N = 64',fontsize=7.5); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig('fig_results.pdf'); 
# ---- overview figure
fig,ax=plt.subplots(figsize=(3.45,2.55)); ax.set_xlim(0,100); ax.set_ylim(0,74); ax.axis('off')
def box(x,y,w,h,t,c,ec,fs=6.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.5',fc=c,ec=ec,lw=.8))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=.8,color='#333',shrinkA=0,shrinkB=0))
ax.add_patch(FancyBboxPatch((1,42.5),98,29.5,boxstyle='round,pad=0.3,rounding_size=2',fc='#f4f6f7',ec='#999',lw=.6,ls='--'))
ax.text(3,69,'Beam sweep (existing, no change)',fontsize=5.8,va='center',style='italic')
box(4,46,26,17,'$N$ beams,\none power\nsample\neach','#d6eaf8','#2e86c1')
box(37,46,26,17,'Winner $w$:\nlargest power\n$y_w$','#d6eaf8','#2e86c1')
box(70,46,26,17,'Runner-up $M$\noutside guard\nbins of $w$','#fdebd0','#d68910')
arr(30,54.5,37,54.5); arr(63,54.5,70,54.5)
box(4,16,43,17,'Raw report $y_w-\\sigma^2$\nbias up, about $\\ln N$','#fadbd8','#c0392b')
box(53,16,43,17,'Gap report $y_w-M-\\sigma^2$\nbias zero, no extra slot','#d5f5e3','#1e8449')
ln=lambda xs,ys:ax.plot(xs,ys,color='#333',lw=.8,solid_capstyle='butt')
ln([50,50],[46,38]); ln([25,70],[38,38]); arr(25,38,25,33); arr(70,38,70,33); ln([88,88],[46,43.2]); arr(88,43.2,88,33)
box(20,1,60,9,'Rate choice with one margin for every $N$','#e8daef','#7d3c98')
ln([25,25],[16,13]); ln([75,75],[16,13]); ln([25,75],[13,13]); arr(50,13,50,10)
ax.legend(handles=[Patch(fc='#d6eaf8',ec='#2e86c1',label='measure'),Patch(fc='#fdebd0',ec='#d68910',label='side data'),Patch(fc='#fadbd8',ec='#c0392b',label='prior report'),Patch(fc='#d5f5e3',ec='#1e8449',label='proposed')],loc='upper center',bbox_to_anchor=(.5,-0.0),ncol=4,fontsize=5.2,frameon=False,handlelength=1.2,columnspacing=.8)
plt.savefig('fig_overview.pdf',bbox_inches='tight',pad_inches=.02)
for m in METH+['gap_noguard']: print(m,res[m]['m'],res[m]['thr'],[ (N,'%.3f±%.3f'%res[m][N]['worst'],'%.3f'%res[m][N]['gp'][0]) for N in Ns], res[m]['E2'])
