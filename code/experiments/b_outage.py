"""(B) Outage decomposition incl. wrong-beam selection, Rician fading,
and a direct test of the paper's closed form P_out = e^{-1/(1+r)}(1+r)/(1+r+m r)."""
import numpy as np, json, sys, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('R2OUT', HERE)
sys.path.insert(0, HERE)
import r2lib as L

Q = len(sys.argv) > 1
REPS, B = (3, 5000) if Q else (12, 60000)
REPS2, B2 = (3, 2000) if Q else (10, 20000)
t0 = time.time()
R = {'meta': {'reps': REPS, 'B_max': B, 'reps2': REPS2, 'B2': B2}}


# ================= B1: E1 model, closed form vs empirical ==================
def e1(rng, N, s, Bn):
    y = rng.exponential(1.0, (Bn, N))
    y[:, 0] *= (1 + s)
    w = y.argmax(1)
    r = np.arange(Bn)
    yw = y[r, w]
    M2 = np.partition(y, N - 2, axis=1)[:, N - 2]
    ssel = np.where(w == 0, s, 0.0)
    est = {'raw': yw - 1, 'debias': yw - 1 - L.H(N - 1),
           'confirm': rng.exponential(1.0, Bn) * (1 + ssel) - 1,
           'gap': yw - M2 - 1}
    est['gap_noguard'] = est['gap']
    return est, ssel, (w == 0)


def closed(r, m):
    return float(np.exp(-1.0 / (1 + r)) * (1 + r) / (1 + r + m * r))


SD1 = [-10, -5, 0, 5, 10, 15, 20, 25]
R['B1_closed_form'] = {}
for m in [4.0, 9.0, 16.0]:
    for N in [16, 64, 256, 1024]:
        Bn = int(min(B, 1e7 // N))
        rows = {}
        for sdb in SD1:
            s = 10 ** (sdb / 10.0)
            tot, oc, ow, pw, ptx = [], [], [], [], []
            for sq in np.random.SeedSequence(11 + N + int(m)).spawn(REPS):
                rng = np.random.default_rng(sq)
                est, ssel, corr = e1(rng, N, s, Bn)
                e = est['gap']
                x = np.where(e >= 0, np.maximum(e, 0.0), 0.0) / m
                # data fading Rayleigh with mean ssel
                o = np.where(ssel > 0, 1 - np.exp(-x / np.maximum(ssel, 1e-12)),
                             (x > 0).astype(float))
                o = np.where(x > 0, o, 0.0)
                tot.append(float(o.mean()))
                oc.append(float(o[corr].mean()) if corr.any() else 0.0)
                ow.append(float(o[~corr].mean()) if (~corr).any() else 0.0)
                pw.append(float((~corr).mean()))
                ptx.append(float((x > 0).mean()))
            rows[str(sdb)] = {'emp_total': L.ci(tot), 'emp_out_given_correct': L.ci(oc),
                              'emp_out_given_wrong': L.ci(ow), 'p_wrong': L.ci(pw),
                              'p_tx': L.ci(ptx), 'closed_form': closed(s, m),
                              'err_vs_total': L.ci(tot)[0] - closed(s, m),
                              'err_vs_correct': L.ci(oc)[0] - closed(s, m),
                              'wrong_share_of_outage': (L.ci(pw)[0] * L.ci(ow)[0]
                                                        / max(L.ci(tot)[0], 1e-12))}
        R['B1_closed_form']['m%g_N%d' % (m, N)] = rows
        print('B1 m=%g N=%d %.0fs' % (m, N, time.time() - t0), flush=True)

# raw report under the same zero-tune rule, for reference
R['B1_raw_ref'] = {}
for m in [9.0]:
    for N in [64, 1024]:
        Bn = int(min(B, 1e7 // N))
        rows = {}
        for sdb in SD1:
            s = 10 ** (sdb / 10.0)
            tot = []
            for sq in np.random.SeedSequence(31 + N).spawn(REPS):
                rng = np.random.default_rng(sq)
                est, ssel, corr = e1(rng, N, s, Bn)
                x = np.maximum(est['raw'], 0.0) / m
                o = np.where(ssel > 0, 1 - np.exp(-x / np.maximum(ssel, 1e-12)), (x > 0).astype(float))
                tot.append(float(np.where(x > 0, o, 0.0).mean()))
            rows[str(sdb)] = {'emp_total': L.ci(tot)}
        R['B1_raw_ref']['m%g_N%d' % (m, N)] = rows

# ================= B2: array model, Rician + decomposition =================
SD = [-5, 0, 5, 10, 15, 20]
KEYS = ['out', 'gp', 'ptx', 'out_correct', 'out_wrong', 'p_wrong', 'out_cond']
KCASES = [('rayleigh', -100.0), ('K0dB', 0.0), ('K3dB', 3.0), ('K6dB', 6.0),
          ('K10dB', 10.0), ('LOSfixed', None)]
R['B2_array'] = {}
for name, Kdb in KCASES:
    cfg = dict(M=64, N=64, nside=2, pside=0.05, Kdb=Kdb)
    cal, _ = L.run_cfg(cfg, SD, REPS2, B2, 1000, keys=KEYS)
    ev, aux = L.run_cfg(cfg, SD, REPS2, B2, 2000, keys=KEYS)
    K = L.tune(cal)
    row = {'knobs': {m: (None if K[m] is None else list(map(float, L.MARG[K[m]])))
                     for m in L.METH},
           'beamerr': {str(s): L.ci(aux['beamerr'][:, i]) for i, s in enumerate(SD)},
           'bias': {m: {str(s): L.ci(aux['bias_' + m][:, i]) for i, s in enumerate(SD)}
                    for m in L.METH}}
    for m in L.METH:
        k = K[m]
        if k is None:
            row[m] = None
            continue
        d = {'gp': L.ci(ev[m]['gp'][:, :, k].mean(1)),
             'worst_out': L.ci(ev[m]['out'][:, :, k].max(1)), 'per_snr': {}}
        for i, s in enumerate(SD):
            pw = ev[m]['p_wrong'][:, i, k].mean()
            oc = ev[m]['out_correct'][:, i, k].mean()
            ow = ev[m]['out_wrong'][:, i, k].mean()
            tot = ev[m]['out'][:, i, k].mean()
            d['per_snr'][str(s)] = {
                'total': L.ci(ev[m]['out'][:, i, k]),
                'p_wrong': float(pw), 'out_given_correct': float(oc),
                'out_given_wrong': float(ow),
                'contrib_correct': float((1 - pw) * oc), 'contrib_wrong': float(pw * ow),
                'wrong_share': float(pw * ow / max(tot, 1e-12)),
                'p_tx': float(ev[m]['ptx'][:, i, k].mean()),
                'out_cond_tx': float(ev[m]['out_cond'][:, i, k].mean()),
                'gp': L.ci(ev[m]['gp'][:, i, k])}
        row[m] = d
    R['B2_array'][name] = row
    print('B2 %s %.0fs' % (name, time.time() - t0), flush=True)
    json.dump(R, open(OUT + '/r2_B_outage.json', 'w'), indent=1)

json.dump(R, open(OUT + '/r2_B_outage.json', 'w'), indent=1)
print('B done %.0fs' % (time.time() - t0), flush=True)
