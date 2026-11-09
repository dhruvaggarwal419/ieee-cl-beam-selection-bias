"""(D) Unknown noise power. sigma^2 is replaced by an estimate with
relative error (zero-mean random, or systematic)."""
import numpy as np, json, sys, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('R2OUT', HERE)
sys.path.insert(0, HERE)
import r2lib as L

Q = len(sys.argv) > 1
REPS, B = (3, 2000) if Q else (10, 20000)
SD = [-5, 0, 5, 10, 15, 20]
KEYS = ['out', 'gp', 'ptx', 'false', 'missed']
CASES = [('clean', None),
         ('rand1', ('rand', 0.01)), ('rand5', ('rand', 0.05)),
         ('rand10', ('rand', 0.10)), ('rand20', ('rand', 0.20)),
         ('bias+5', ('bias', 0.05)), ('bias-5', ('bias', -0.05)),
         ('bias+10', ('bias', 0.10)), ('bias-10', ('bias', -0.10)),
         ('bias+20', ('bias', 0.20)), ('bias-20', ('bias', -0.20))]
CFGS = [('base', dict(M=64, N=64, nside=2, pside=0.05)),
        ('N256', dict(M=64, N=256, nside=2, pside=0.05))]
t0 = time.time()
R = {'meta': {'reps': REPS, 'B': B, 'sdbs': SD}}
for cname, cfg in CFGS:
    Bc = int(min(B, 1.3e6 // cfg['N']))
    # knobs tuned with a perfect noise estimate (the realistic deployment case)
    cal0, _ = L.run_cfg(cfg, SD, REPS, Bc, 1000, keys=KEYS)
    K0 = L.tune(cal0)
    sec = {'knobs_clean': {m: (None if K0[m] is None else list(map(float, L.MARG[K0[m]])))
                           for m in L.METH}}
    for nname, mode in CASES:
        ev, aux = L.run_cfg(cfg, SD, REPS, Bc, 2000, keys=KEYS, s2mode=mode)
        cal, _ = L.run_cfg(cfg, SD, REPS, Bc, 1000, keys=KEYS, s2mode=mode)
        K1 = L.tune(cal)
        d = {'knobs_retuned': {m: (None if K1[m] is None else list(map(float, L.MARG[K1[m]])))
                               for m in L.METH}, 'methods': {}}
        for m in L.METH:
            e = {'bias': {str(s): L.ci(aux['bias_' + m][:, i]) for i, s in enumerate(SD)}}
            for tag, k in [('transfer', K0[m]), ('retuned', K1[m])]:
                if k is None:
                    e[tag] = None
                    continue
                e[tag] = {'gp': L.ci(ev[m]['gp'][:, :, k].mean(1)),
                          'worst_out': L.ci(ev[m]['out'][:, :, k].max(1)),
                          'p_tx': L.ci(ev[m]['ptx'][:, :, k].mean(1))}
            d['methods'][m] = e
        sec[nname] = d
        print('D %s %s %.0fs' % (cname, nname, time.time() - t0), flush=True)
        R[cname] = sec
        json.dump(R, open(OUT + '/r2_D_noise.json', 'w'), indent=1)
print('D done %.0fs' % (time.time() - t0), flush=True)
