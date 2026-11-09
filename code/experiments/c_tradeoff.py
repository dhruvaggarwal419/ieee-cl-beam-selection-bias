"""(C) Full trade-off table per array config: goodput, outage (uncond and
conditional on transmitting), false-transmission, missed-transmission,
beam-selection error, and calibration cost."""
import numpy as np, json, sys, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('R2OUT', HERE)
sys.path.insert(0, HERE)
import r2lib as L

Q = len(sys.argv) > 1
REPS, B = (3, 2000) if Q else (10, 20000)
NCALS = [125, 250, 500, 1000, 2000, 4000, 8000] if not Q else [125, 500]
NDRAW = 4 if Q else 20
SD = [-5, 0, 5, 10, 15, 20]
KEYS = ['out', 'gp', 'ptx', 'false', 'missed', 'lost_gp', 'out_cond']
CFGS = [('base', dict(M=64, N=64, nside=2, pside=0.05)),
        ('side20', dict(M=64, N=64, nside=2, pside=0.2)),
        ('side50', dict(M=64, N=64, nside=2, pside=0.5)),
        ('paths6', dict(M=64, N=64, nside=6, pside=0.05)),
        ('M32', dict(M=32, N=64, nside=2, pside=0.05)),
        ('N128', dict(M=64, N=128, nside=2, pside=0.05)),
        ('N256', dict(M=64, N=256, nside=2, pside=0.05))]
t0 = time.time()
R = {'meta': {'reps': REPS, 'B': B, 'sdbs': SD, 'target': 0.1, 'ncals': NCALS, 'ndraw': NDRAW}}

for name, cfg in CFGS:
    Bc = int(min(B, 1.3e6 // cfg['N']))
    cal, _ = L.run_cfg(cfg, SD, REPS, Bc, 1000, keys=KEYS)
    ev, aux = L.run_cfg(cfg, SD, REPS, Bc, 2000, keys=KEYS)
    K = L.tune(cal)
    row = {'B': Bc,
           'beamerr': {str(s): L.ci(aux['beamerr'][:, i]) for i, s in enumerate(SD)},
           'beamloss_dB': {str(s): L.ci(aux['beamloss_dB'][:, i]) for i, s in enumerate(SD)},
           'knobs': {m: (None if K[m] is None else list(map(float, L.MARG[K[m]]))) for m in L.METH},
           'methods': {}}
    for m in L.METH:
        k = K[m]
        if k is None:
            row['methods'][m] = None
            continue
        row['methods'][m] = {
            'gp': L.ci(ev[m]['gp'][:, :, k].mean(1)),
            'worst_out': L.ci(ev[m]['out'][:, :, k].max(1)),
            'mean_out': L.ci(ev[m]['out'][:, :, k].mean(1)),
            'false_tx': L.ci(ev[m]['false'][:, :, k].mean(1)),
            'worst_false_tx': L.ci(ev[m]['false'][:, :, k].max(1)),
            'out_cond_tx': L.ci(ev[m]['out_cond'][:, :, k].mean(1)),
            'p_tx': L.ci(ev[m]['ptx'][:, :, k].mean(1)),
            'missed_tx': L.ci(ev[m]['missed'][:, :, k].mean(1)),
            'lost_gp': L.ci(ev[m]['lost_gp'][:, :, k].mean(1)),
            'bias': {str(s): L.ci(aux['bias_' + m][:, i]) for i, s in enumerate(SD)},
            'per_snr_out': [float(ev[m]['out'][:, i, k].mean()) for i in range(len(SD))],
            'per_snr_gp': [float(ev[m]['gp'][:, i, k].mean()) for i in range(len(SD))]}
    # ---- calibration cost: how much tuning data each method needs ----------
    cc = {m: {} for m in L.METH}
    for nc in NCALS:
        picks = {m: [] for m in L.METH}
        for dr in range(NDRAW):
            c1, _ = L.run_cfg(cfg, SD, 1, min(nc, Bc), 7000 + 97 * dr + nc, keys=['out', 'gp'])
            kk = L.tune(c1)
            for m in L.METH:
                picks[m].append(kk[m])
        for m in L.METH:
            ok, gps, outs = [], [], []
            for k in picks[m]:
                if k is None:
                    ok.append(0.0)
                    continue
                wo = float(ev[m]['out'][:, :, k].max(1).mean())
                ok.append(1.0 if wo <= 0.11 else 0.0)
                outs.append(wo)
                gps.append(float(ev[m]['gp'][:, :, k].mean()))
            cc[m][str(nc)] = {'frac_meeting_target': float(np.mean(ok)),
                              'mean_eval_worst_out': float(np.mean(outs)) if outs else None,
                              'mean_eval_gp': float(np.mean(gps)) if gps else None,
                              'n_infeasible': int(sum(k is None for k in picks[m]))}
    for m in L.METH:
        need = [nc for nc in NCALS if cc[m][str(nc)]['frac_meeting_target'] >= 0.9]
        cc[m]['n_cal_needed'] = need[0] if need else None
    row['calibration'] = cc
    R[name] = row
    json.dump(R, open(OUT + '/r2_C_tradeoff.json', 'w'), indent=1)
    print('C %s %.0fs' % (name, time.time() - t0), flush=True)
print('C done %.0fs' % (time.time() - t0), flush=True)
