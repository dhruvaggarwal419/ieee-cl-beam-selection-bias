"""(A) Guard-zone bias characterisation + an explicit bias correction."""
import numpy as np, json, sys, time
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('R2OUT', HERE)
sys.path.insert(0, HERE)
import r2lib as L

Q = len(sys.argv) > 1
REPS, B = (3, 2000) if Q else (10, 20000)
SD = [-5, 0, 5, 10, 15, 20]
t0 = time.time()
R = {'meta': {'reps': REPS, 'B': B, 'sdbs': SD}}


def bias_only(cfg, guard, sdbs, reps, B, seed):
    """mean/std of (est - ssel) per method per SNR; no rate rule."""
    M, N, sll = cfg['M'], cfg['N'], cfg.get('sll', 0.0)
    acc = {m: {s: [] for s in sdbs} for m in L.METH}
    sd_ = {m: {s: [] for s in sdbs} for m in L.METH}
    extra = {s: [] for s in sdbs}
    for sq in np.random.SeedSequence(seed).spawn(reps):
        rng = np.random.default_rng(sq)
        for sdb in sdbs:
            s0 = 10 ** (sdb / 10.0)
            TH, LOS, DIF = L.draw_paths(rng, B, cfg)
            sig = L.sig_of(TH, LOS + DIF, s0, M, N, sll)
            y = L.measure(rng, sig)
            est, w, yw, Mg = L.estimates(rng, y, sig, guard)
            r = np.arange(B)
            ssel = np.abs(sig[r, w]) ** 2
            best = (np.abs(sig) ** 2).argmax(1)
            for m in L.METH:
                acc[m][sdb].append(float(np.mean(est[m] - ssel)))
                sd_[m][sdb].append(float(np.std(est[m] - ssel)))
            extra[sdb].append([float((w != best).mean()), float(np.mean(Mg)), float(ssel.mean())])
    out = {}
    for m in L.METH:
        out[m] = {str(s): {'bias': L.ci(acc[m][s]), 'std': L.ci(sd_[m][s])} for s in sdbs}
    out['_aux'] = {str(s): {'beamerr': L.ci([e[0] for e in extra[s]]),
                            'EMg': L.ci([e[1] for e in extra[s]]),
                            'Essel': L.ci([e[2] for e in extra[s]])} for s in sdbs}
    return out


# ---- A1 guard width -------------------------------------------------------
base = dict(M=64, N=64, nside=2, pside=0.05)
R['A1_guard_width'] = {}
for g in [0, 1, 2, 3, 4, 6, 8]:
    R['A1_guard_width'][str(g)] = bias_only(base, g, SD, REPS, B, 100 + g)
    print('A1 g=%d %.0fs' % (g, time.time() - t0), flush=True)

# ---- A2 angular separation of nearest side path ---------------------------
# beamwidth in u units = 2/N
R['A2_separation'] = {}
for bw in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
    cfg = dict(base, pside=0.2, sep=bw * 2.0 / 64)
    R['A2_separation'][str(bw)] = bias_only(cfg, 3, SD, REPS, B, 200 + int(bw * 10))
print('A2 %.0fs' % (time.time() - t0), flush=True)

# ---- A3 angular spread of the dominant cluster ----------------------------
R['A3_spread'] = {}
for sp in [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]:
    cfg = dict(base, spread=sp * 2.0 / 64, nsub=8)
    R['A3_spread'][str(sp)] = bias_only(cfg, 3, SD, REPS, B, 300 + int(sp * 100))
print('A3 %.0fs' % (time.time() - t0), flush=True)

# ---- A4 codebook sidelobe level (Chebyshev taper) -------------------------
R['A4_sll'] = {}
for sll in [0.0, 20.0, 25.0, 30.0, 40.0]:   # 0.0 = untapered rectangular (-13.3 dB)
    cfg = dict(base, sll=sll)
    R['A4_sll'][str(sll)] = bias_only(cfg, 3, SD, REPS, B, 400 + int(sll))
print('A4 %.0fs' % (time.time() - t0), flush=True)


# ---- A5 explicit bias correction ------------------------------------------
def leak_table(M, N, g, sll=0.0, pside=0.0, nside=0, n=40000, seed=7):
    """Noiseless expected strongest out-of-guard leakage, relative to the
    power captured by the winning beam. Depends only on (M,N,g,sll[,pside])."""
    rng = np.random.default_rng(seed)
    nn = np.arange(M)
    W, _ = L.codebook(M, N, sll)
    th0 = rng.uniform(-1, 1, n)
    A = np.exp(1j * np.pi * th0[:, None] * nn) / np.sqrt(M)
    c = np.abs(A @ W.conj().T) ** 2
    tot = c.copy()
    for _ in range(nside):
        ts = rng.uniform(-1, 1, n)
        As = np.exp(1j * np.pi * ts[:, None] * nn) / np.sqrt(M)
        tot = tot + pside * np.abs(As @ W.conj().T) ** 2
    w = c.argmax(1)
    d = np.abs(np.arange(N)[None, :] - w[:, None])
    d = np.minimum(d, N - d)
    mx = np.where(d > g, tot, -np.inf).max(1)
    cw = c[np.arange(n), w]
    return float(mx.mean() / cw.mean())


A5cfgs = [('base', dict(M=64, N=64, nside=2, pside=0.05)),
          ('side20', dict(M=64, N=64, nside=2, pside=0.2)),
          ('side50', dict(M=64, N=64, nside=2, pside=0.5)),
          ('paths6', dict(M=64, N=64, nside=6, pside=0.05)),
          ('N128', dict(M=64, N=128, nside=2, pside=0.05)),
          ('sll30', dict(M=64, N=64, nside=2, pside=0.05, sll=30.0))]
R['A5_correction'] = {}
R['A5_rho'] = {}
for ci5, (name, cfg) in enumerate(A5cfgs):
    g = 3
    M, N, sll = cfg['M'], cfg['N'], cfg.get('sll', 0.0)
    rho_main = leak_table(M, N, g, sll)                                   # needs (g,N,M,SLL) only
    rho_full = leak_table(M, N, g, sll, cfg['pside'], cfg['nside'])       # also needs path stats
    R['A5_rho'][name] = {'rho_main': rho_main, 'rho_full': rho_full,
                         'rho_by_guard': {str(gg): leak_table(M, N, gg, sll) for gg in [0, 1, 2, 3, 4, 6, 8]}}
    res = {}
    for sdb in SD:
        s0 = 10 ** (sdb / 10.0)
        bm, bM, bF, sM, sF, sg, ess, esh = [], [], [], [], [], [], [], []
        for sq in np.random.SeedSequence(500 + 17 * ci5).spawn(REPS):
            rng = np.random.default_rng(sq)
            TH, LOS, DIF = L.draw_paths(rng, B, cfg)
            sig = L.sig_of(TH, LOS + DIF, s0, M, N, sll)
            y = L.measure(rng, sig)
            est, w, yw, Mg = L.estimates(rng, y, sig, g)
            ssel = np.abs(sig[np.arange(B), w]) ** 2
            shat = np.maximum(est['debias'], 0.0)
            cm = est['gap'] + rho_main * shat
            cf = est['gap'] + rho_full * shat
            bm.append(float(np.mean(est['gap'] - ssel))); sg.append(float(np.std(est['gap'] - ssel)))
            bM.append(float(np.mean(cm - ssel))); sM.append(float(np.std(cm - ssel)))
            bF.append(float(np.mean(cf - ssel))); sF.append(float(np.std(cf - ssel)))
            ess.append(float(ssel.mean())); esh.append(float(shat.mean()))
        res[str(sdb)] = {'gap_bias': L.ci(bm), 'gap_std': L.ci(sg),
                         'corr_main_bias': L.ci(bM), 'corr_main_std': L.ci(sM),
                         'corr_full_bias': L.ci(bF), 'corr_full_std': L.ci(sF),
                         'Essel': L.ci(ess), 'Eshat': L.ci(esh)}
    R['A5_correction'][name] = res
    print('A5 %s %.0fs' % (name, time.time() - t0), flush=True)

json.dump(R, open(OUT + '/r2_A_guard.json', 'w'), indent=1)
print('A done %.0fs' % (time.time() - t0), flush=True)
