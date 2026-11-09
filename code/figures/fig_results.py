import json,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='serif'; plt.rcParams['font.size']=7.5
A=json.load(open('r2_A_guard.json'))
fig,ax=plt.subplots(1,2,figsize=(3.5,2.35))
a=ax[0]; gs=['0','1','2','3','4','6','8']
for s,c in [('10','#1565C0'),('15','#F9A825'),('20','#C62828')]:
    a.plot([int(g) for g in gs],[A['A1_guard_width'][g]['gap'][s]['bias'][0] for g in gs],'-o',ms=2.5,lw=1.2,color=c,label=f'{s} dB')
a.set_xlabel('Guard width $g$ (beams)'); a.set_ylabel('Gap bias (noise units)'); a.grid(alpha=0.25,lw=0.4); a.legend(fontsize=6,frameon=False,loc='lower right'); a.set_title('(a)',fontsize=8)
b=ax[1]; ss=['0.25','0.5','1.0','2.0','4.0','8.0','16.0']
for s,c in [('15','#F9A825'),('20','#C62828')]:
    b.semilogx([float(x) for x in ss],[A['A2_separation'][x]['gap'][s]['bias'][0] for x in ss],'-o',ms=2.5,lw=1.2,color=c,label=f'{s} dB')
b.axvline(3.5,color='#555',lw=0.6,ls=':'); b.text(3.9,-11.0,'guard\nedge',fontsize=5.5,color='#555',va='top')
b.set_xlabel('Separation (beams)'); b.grid(alpha=0.25,lw=0.4); b.legend(fontsize=6,frameon=False,loc='lower left'); b.set_title('(b)',fontsize=8)
fig.tight_layout(pad=0.4,w_pad=0.8); fig.savefig('../fig_results.pdf',bbox_inches='tight',pad_inches=0.02); fig.savefig('/tmp/l2v2_res.png',dpi=130)
