import json,numpy as np,sys
sys.path.insert(0,'.')
from diagram_kit import *
A=json.load(open('r2_A_guard.json')); G3=A['A1_guard_width']['3']; CFG=A['A5_correction']
B=json.load(open('r2_B_outage.json'))['B1_closed_form']['m9_N64']
rng=np.random.default_rng(4)
snr=[-5,0,5,10,15,20]
D=Diagram(7.16,3.45,100,48)
Y0,YH=10.5,34
D.group(0.8,Y0,19.5,YH,'1. Beam sweep',c='in',fs=6.4)
D.group(22.0,Y0,31.0,YH,'2. Reports of link quality',c='warn',fs=6.2)
D.group(54.7,Y0,18.0,YH,'3. Rate rule',c='stat',fs=6.4)
D.group(74.4,Y0,24.8,YH,'4. What the array shows',c='out',fs=6.2)
# sweep thumbnail
N=64; y=rng.exponential(1.0,N); y[20]=rng.exponential(1+3.16); w=int(np.argmax(y)); guard=3
ys=y.copy(); ys[max(0,w-guard):w+guard+1]=-1; ru=int(np.argmax(ys))
ax=D.inset(2.6,19.0,15.2,17.0); cols=['#9DB9D8']*N; cols[w]='#B85450'; cols[ru]='#D79B00'
ax.bar(range(N),y,color=cols,width=0.9); ax.axvspan(w-guard-0.5,w+guard+0.5,color='#B85450',alpha=0.12); ax.set_xticks([]); ax.set_yticks([0,4,8]); ax.set_title('power of $N=64$ beams',fontsize=5,pad=1)
D.text(11.3,16.3,'red: winner, orange: runner-up\noutside the guard zone',fs=4.5)
D.text(11.3,12.6,'the winner of noisy\nvalues is biased high',fs=4.7,color='#B85450',bold=True)
# reports boxes
names=[('Raw','$y_w-1$'),('Offset','$y_w-1-H_{N-1}$'),('Gap','$y_w-M_g-1$'),('Confirm','fresh slot')]
for i,(n,f) in enumerate(names):
    D.box(23.4,34.5-i*6.3,12.8,5.2,n+'\n'+f,'white',4.9)
ax=D.inset(39.2,16.0,13.0,19.8)
for k,l,c,ls in [('raw','raw','#1565C0','-'),('debias','offset','#2E7D32','-'),('gap','gap','#B85450','-'),('confirm','confirm','#9E9E9E','--')]:
    ax.plot(snr,[G3[k][str(s)]['bias'][0] for s in snr],ls,color=c,lw=1.0,label=l)
ax.axhline(0,color='#555',lw=0.4); ax.set_title('bias vs SNR (dB)',fontsize=5,pad=1); ax.legend(fontsize=4.0,frameon=False,loc='lower left',handlelength=1.1,borderpad=0.1,labelspacing=0.15); ax.set_xticks([0,10,20]); ax.set_yticks([-8,0,4])
D.text(37.5,13.0,'noise units, 64-antenna array',fs=4.4)
# rate rule
D.box(56.0,30.5,15.4,7.0,'$\\log_2(1+\\hat\\rho/m)$\nsend only if $\\hat\\rho>\\theta$','white',4.9)
ax=D.inset(57.2,13.6,13.4,13.4)
sn=[-10,-5,0,5,10,15,20,25]
v=lambda s,k: (lambda x: x[0] if isinstance(x,list) else x)(B[str(s)][k])
ax.plot(sn,[v(s,'closed_form') for s in sn],'--',color='#555',lw=1.0,label='formula'); ax.plot(sn,[v(s,'emp_total') for s in sn],color='#B85450',lw=1.2,label='all events'); ax.plot(sn,[v(s,'emp_out_given_correct') for s in sn],':',color='#2E7D32',lw=1.2,label='right beam')
ax.set_title('outage vs SNR (dB)',fontsize=5,pad=1); ax.legend(fontsize=4.0,frameon=False,loc='upper right',handlelength=1.1,borderpad=0.1,labelspacing=0.15); ax.set_xticks([-10,10,25]); ax.set_yticks([0.1,0.3])
# outcome thumbnails
ax=D.inset(76.4,24.0,10.6,12.2); ks=['base','side20','side50','paths6','N128']; vals=[CFG[k]['20']['gap_bias'][0] for k in ks]
ax.bar(range(5),vals,color='#B85450',width=0.7); ax.set_xticks(range(5)); ax.set_xticklabels(['base','s.2','s.5','6p','N128'],fontsize=4.0); ax.set_yticks([-40,-20,0]); ax.set_title('gap bias at 20 dB',fontsize=5,pad=1)
Cc=json.load(open('r2_C_tradeoff.json'))
ax=D.inset(89.2,24.0,8.2,12.2); gp={a:np.mean([Cc[k]['methods'][a]['gp'][0] for k in ['base','side20','side50','paths6','M32','N128','N256']]) for a in ('raw','debias','confirm','gap')}
ax.bar(range(4),[gp[a] for a in ('raw','debias','confirm','gap')],color=['#1565C0','#2E7D32','#9E9E9E','#B85450'],width=0.7); ax.set_ylim(1.5,1.95); ax.set_xticks(range(4)); ax.set_xticklabels(['R','O','C','G'],fontsize=4.2); ax.set_yticks([1.6,1.8]); ax.set_title('goodput',fontsize=5,pad=1)
D.text(86.8,19.4,'guard zone halves the bias but\nnothing removes it; the offset\nbeats the gap in 6 of 7 settings',fs=4.6,color='#B85450',bold=True)
D.text(86.8,13.2,'R raw, O offset, C confirm, G gap',fs=4.0)
# arrows (axis-aligned)
D.arrow(18.0,27.5,23.3,27.5)
for yc in (37.1,30.8,24.5,18.2): D.line([36.3,37.7],[yc,yc])
D.line([37.7,37.7],[18.2,37.1]); D.arrow(37.7,26.0,39.1,26.0)
D.arrow(52.3,34.0,55.9,34.0); D.arrow(58.2,30.4,58.2,27.3); D.arrow(71.5,34.0,76.3,34.0)
D.legend([('in','Sweep'),('warn','Reports'),('stat','Rate rule'),('out','Array results')],y=5.0,x0=3,dx=24)
D.text(50,1.6,'The exact bias formulas hold in the exponential model. In a real array the runner-up gap is biased and a harmonic-number offset does at least as well.',fs=5.0,style='italic')
D.save('../fig_overview.pdf','/tmp/l2v2_ov.png')
