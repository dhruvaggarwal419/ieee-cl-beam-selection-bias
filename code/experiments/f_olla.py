"""(F) Outer-loop link adaptation (OLLA) baseline against the five reports.

OLLA keeps a per-link offset (dB) on top of the EXISTING raw report: it steps
the offset up by du on a NACK and down by du*target/(1-target) on an ACK, so
the steady-state failure rate converges to `target` with no new report format.
"""
import numpy as np, json, sys, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('R2OUT', HERE)
sys.path.insert(0, HERE)
import r2lib as L

Q = len(sys.argv) > 1
T, B, REPS = (300, 1000, 2) if Q else (1500, 3000, 4)
SD = [-5, 0, 5, 10, 15, 20]
TARGET = 0.1
GUARD = 3
CFG = dict(M=64, N=64, nside=2, pside=0.05)
RPT = ['raw', 'debias', 'gap']
DUS = [0.1, 0.3, 1.0]
THRS = [0, 1, 2, 4, 8]
KN = [(d, t) for d in DUS for t in THRS]
nK = len(KN)
DU = np.array([k[0] for k in KN])[None, :, None]
TH = np.array([k[1] for k in KN])[None, :, None]
DD = DU * TARGET / (1 - TARGET)
t0 = time.time()


def olla(sdb, seed, T=T, B=B, cfg=CFG):
    """returns out (T,R,K), gp (T,R,K), final offsets (R,K)"""
    rng = np.random.default_rng(seed)
    s0 = 10 ** (sdb / 10.0)
    off = np.zeros((len(RPT), nK, B))
    OUT = np.zeros((T, len(RPT), nK))
    GP = np.zeros((T, len(RPT), nK))
    M, N = cfg['M'], cfg['N']
    for t in range(T):
        TH_, LOS, DIF = L.draw_paths(rng, B, cfg)
        sig = L.sig_of(TH_, LOS + DIF, s0, M, N)
        y = L.measure(rng, sig)
        est, w, _, _ = L.estimates(rng, y, sig, GUARD)
        ssel = np.abs(sig[np.arange(B), w]) ** 2
        E = np.stack([est[m] for m in RPT])[:, None, :]          # R,1,B
        x = np.where(E >= TH, np.maximum(E, 0.0) * 10 ** (-off / 10.0), 0.0)
        tx = x > 0
        fail = tx & (x > ssel[None, None, :])
        OUT[t] = fail.mean(2)
        GP[t] = (np.log2(1 + x) * (tx & ~fail)).mean(2)
        off = np.clip(off + np.where(tx, np.where(fail, DU, -DD), 0.0), -10.0, 40.0)
    return OUT, GP, off.mean(2)


def sweep(seed):
    ss = T // 4
    out = np.zeros((len(SD), len(RPT), nK))
    gp = np.zeros((len(SD), len(RPT), nK))
    offs = np.zeros((len(SD), len(RPT), nK))
    traj = {}
    for i, sdb in enumerate(SD):
        O, G, of = olla(sdb, seed + 13 * i)
        out[i] = O[-ss:].mean(0)
        gp[i] = G[-ss:].mean(0)
        offs[i] = of
        traj[str(sdb)] = {'out_first50': O[:50].mean(0).tolist(),
                          'out_50_200': O[50:200].mean(0).tolist(),
                          'out_last': out[i].tolist(),
                          'gp_first50': G[:50].mean(0).tolist(),
                          'gp_last': gp[i].tolist()}
        print('  olla sdb=%d %.0fs' % (sdb, time.time() - t0), flush=True)
    return out, gp, offs, traj


R = {'meta': {'T': T, 'B': B, 'reps': REPS, 'sdbs': SD, 'target': TARGET,
              'knobs_grid': [list(map(float, k)) for k in KN], 'reports': RPT,
              'steady_state_window': 'last T/4 slots'}}

# ---- calibrate OLLA knobs -------------------------------------------------
cout, cgp, _, _ = sweep(1000)
Kpick = {}
for ri, m in enumerate(RPT):
    w = cout[:, ri, :].max(0)
    feas = np.where(w <= TARGET)[0]
    Kpick[m] = int(feas[np.argmax(cgp[:, ri, feas].mean(0))]) if len(feas) else None
R['olla_knobs'] = {m: (None if Kpick[m] is None else
                       {'du_dB': KN[Kpick[m]][0], 'thr': KN[Kpick[m]][1]}) for m in RPT}
print('olla knobs', R['olla_knobs'], flush=True)

# ---- evaluate OLLA on fresh seeds ----------------------------------------
EO, EG, EF = [], [], []
trajs = []
for rp in range(REPS):
    o, g, f, tj = sweep(2000 + 101 * rp)
    EO.append(o); EG.append(g); EF.append(f); trajs.append(tj)
EO = np.array(EO); EG = np.array(EG); EF = np.array(EF)
R['olla'] = {}
for ri, m in enumerate(RPT):
    k = Kpick[m]
    if k is None:
        R['olla'][m] = None
        continue
    R['olla'][m] = {
        'gp': L.ci(EG[:, :, ri, k].mean(1)),
        'worst_out': L.ci(EO[:, :, ri, k].max(1)),
        'per_snr_out': [float(EO[:, i, ri, k].mean()) for i in range(len(SD))],
        'per_snr_gp': [float(EG[:, i, ri, k].mean()) for i in range(len(SD))],
        'converged_offset_dB': [float(EF[:, i, ri, k].mean()) for i in range(len(SD))],
        'transient_out_first50': [float(np.mean([tj[str(s)]['out_first50'][ri][k] for tj in trajs]))
                                  for s in SD],
        'transient_out_50_200': [float(np.mean([tj[str(s)]['out_50_200'][ri][k] for tj in trajs]))
                                 for s in SD],
        'transient_gp_first50': [float(np.mean([tj[str(s)]['gp_first50'][ri][k] for tj in trajs]))
                                 for s in SD]}
json.dump(R, open(OUT + '/r2_F_olla.json', 'w'), indent=1)

# ---- static (no feedback) baselines on the same config --------------------
KEYS = ['out', 'gp', 'ptx']
cal, _ = L.run_cfg(CFG, SD, 10, 20000, 1000, keys=KEYS)
ev, aux = L.run_cfg(CFG, SD, 10, 20000, 2000, keys=KEYS)
Ks = L.tune(cal, TARGET)
R['static'] = {'knobs': {m: (None if Ks[m] is None else list(map(float, L.MARG[Ks[m]])))
                         for m in L.METH}, 'methods': {}}
for m in L.METH:
    k = Ks[m]
    R['static']['methods'][m] = None if k is None else {
        'gp': L.ci(ev[m]['gp'][:, :, k].mean(1)),
        'worst_out': L.ci(ev[m]['out'][:, :, k].max(1)),
        'per_snr_out': [float(ev[m]['out'][:, i, k].mean()) for i in range(len(SD))],
        'per_snr_gp': [float(ev[m]['gp'][:, i, k].mean()) for i in range(len(SD))]}
json.dump(R, open(OUT + '/r2_F_olla.json', 'w'), indent=1)
print('F done %.0fs' % (time.time() - t0), flush=True)
