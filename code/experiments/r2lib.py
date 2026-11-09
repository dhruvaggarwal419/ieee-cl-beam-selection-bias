"""Shared library for reviewer-round-2 beam-report experiments.

Model (all powers in NOISE units, noise variance 1 per beam measurement):
  codebook: N unit-norm beamformers on an M-element half-wavelength ULA,
            pointing at u_j = -1 + 2j/N, optionally Chebyshev-tapered (sll dB).
  channel : P paths, angle th_p in [-1,1), complex gain g_p.
            received signal on beam j: sig_j = sqrt(s0) * sum_p g_p * (w_j^H a(th_p))
            with ||w_j|| = ||a|| = 1, so s0 = peak SNR when perfectly aligned and |g|=1.
  measure : y_j = |sig_j + n_j|^2, n_j ~ CN(0,1).
  truth   : ssel = |sig_w|^2 on the selected beam w = argmax_j y_j.
"""
import numpy as np
from scipy.signal.windows import chebwin

METH = ['raw', 'debias', 'confirm', 'gap_noguard', 'gap']


def H(n):
    n = int(n)
    return float(np.sum(1.0 / np.arange(1, n + 1))) if n >= 1 else 0.0


_WC = {}


def codebook(M, N, sll=0.0):
    key = (M, N, round(float(sll), 4))
    if key not in _WC:
        n = np.arange(M)
        u = -1.0 + 2.0 * np.arange(N) / N
        W = np.exp(1j * np.pi * np.outer(u, n))
        if sll and sll > 0:
            W = W * chebwin(M, at=float(sll))[None, :]
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        _WC[key] = (W, u)
    return _WC[key]


def draw_paths(rng, B, cfg):
    """Return TH (B,P), LOS (B,P), DIF (B,P). Path 0 (or the first nsub
    sub-paths) is the dominant cluster; the rest are side paths."""
    nside = cfg.get('nside', 2)
    pside = cfg.get('pside', 0.05)
    Kdb = cfg.get('Kdb', None)          # None -> unit-modulus LOS (paper default)
    sep = cfg.get('sep', None)          # angular sep (u units) of nearest side path
    spread = cfg.get('spread', 0.0)     # rms angular spread of dominant cluster (u units)
    nsub = cfg.get('nsub', 1) if spread > 0 else 1

    th0 = rng.uniform(-1, 1, B)
    TH = [np.mod(th0[:, None] + spread * rng.standard_normal((B, nsub)) + 1, 2) - 1
          if nsub > 1 or spread > 0 else th0[:, None]]
    # dominant cluster gains: total power 1 split equally over nsub sub-paths
    if Kdb is None:
        los0 = np.exp(1j * rng.uniform(0, 2 * np.pi, (B, nsub))) / np.sqrt(nsub)
        dif0 = np.zeros((B, nsub), complex)
    else:
        K = 10.0 ** (float(Kdb) / 10.0)
        los0 = np.sqrt(K / (K + 1.0) / nsub) * np.exp(1j * rng.uniform(0, 2 * np.pi, (B, nsub)))
        dif0 = np.sqrt(1.0 / (2 * (K + 1.0) * nsub)) * (
            rng.standard_normal((B, nsub)) + 1j * rng.standard_normal((B, nsub)))
    LOS = [los0]
    DIF = [dif0]
    if nside > 0:
        ths = rng.uniform(-1, 1, (B, nside))
        if sep is not None:
            ths[:, 0] = np.mod(th0 + float(sep) + 1, 2) - 1
        TH.append(ths)
        LOS.append(np.zeros((B, nside), complex))
        DIF.append(np.sqrt(pside / 2) * (rng.standard_normal((B, nside))
                                         + 1j * rng.standard_normal((B, nside))))
    return np.concatenate(TH, 1), np.concatenate(LOS, 1), np.concatenate(DIF, 1)


def sig_of(TH, G, s0, M, N, sll=0.0):
    W, _ = codebook(M, N, sll)
    A = np.exp(1j * np.pi * TH[:, :, None] * np.arange(M)[None, None, :]) / np.sqrt(M)
    C = A @ W.conj().T                       # B,P,N
    return np.sqrt(s0) * np.einsum('bp,bpn->bn', G, C)


def measure(rng, sig):
    B, N = sig.shape
    n = (rng.standard_normal((B, N)) + 1j * rng.standard_normal((B, N))) / np.sqrt(2)
    return np.abs(sig + n) ** 2


def estimates(rng, y, sig, guard=3, s2hat=1.0):
    """All five reports, plus the selected index and the true selected power."""
    B, N = y.shape
    r = np.arange(B)
    w = y.argmax(1)
    yw = y[r, w]
    idx = np.arange(N)[None, :]
    d = np.abs(idx - w[:, None])
    d = np.minimum(d, N - d)
    Mg = np.where(d > guard, y, -np.inf).max(1)
    M2 = np.partition(y, N - 2, axis=1)[:, N - 2]
    nz = (rng.standard_normal(B) + 1j * rng.standard_normal(B)) / np.sqrt(2)
    y2 = np.abs(sig[r, w] + nz) ** 2
    s2 = np.asarray(s2hat, float)
    est = {'raw': yw / s2 - 1.0,
           'debias': yw / s2 - 1.0 - H(N - 2 * guard - 1),
           'confirm': y2 / s2 - 1.0,
           'gap_noguard': (yw - M2) / s2 - 1.0,
           'gap': (yw - Mg) / s2 - 1.0}
    return est, w, yw, Mg


MARG = np.array([(m, t) for m in np.geomspace(0.5, 16, 10)
                 for t in [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16]])


def rule(est, thr, margin):
    x = np.maximum(est, 0.0) / margin
    return np.where(est >= thr, x, 0.0)


def grid_metrics(est, ssel_data, correct=None, MG=None):
    """est (B,), ssel_data (B,) true data-time SNR of the selected beam.
    Returns dict of (K,) arrays over the (margin,threshold) grid."""
    MG = MARG if MG is None else MG
    e = est[None, :]
    m = MG[:, 0][:, None]
    t = MG[:, 1][:, None]
    x = np.where(e >= t, np.maximum(e, 0.0) / m, 0.0)      # K,B
    tx = x > 0
    fail = tx & (x > ssel_data[None, :])
    out = {'out': fail.mean(1),                             # unconditional (paper's)
           'gp': (np.log2(1 + x) * (tx & ~fail)).mean(1),
           'ptx': tx.mean(1),
           'false': fail.mean(1),
           'out_cond': np.where(tx.mean(1) > 0, fail.sum(1) / np.maximum(tx.sum(1), 1), 0.0)}
    could = ssel_data >= 1.0
    out['missed'] = ((~tx) & could[None, :]).mean(1)
    out['lost_gp'] = (np.log2(1 + ssel_data)[None, :] * (~tx)).mean(1)
    if correct is not None:
        c = correct[None, :]
        nc = max(correct.sum(), 1)
        nw = max((~correct).sum(), 1)
        out['out_correct'] = (fail & c).sum(1) / nc
        out['out_wrong'] = (fail & ~c).sum(1) / nw
        out['p_wrong'] = np.full(len(MG), float((~correct).mean()))
    return out


def ci(a):
    a = np.asarray(a, float)
    n = len(a)
    return [float(a.mean()), float(2.093 * a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0]


def run_cfg(cfg, sdbs, reps, B, seed, guard=3, s2mode=None, rho_t=1.0, keys=None):
    """Full sweep over SNR grid and the (margin,threshold) grid.
    s2mode: None, or ('rand', r) zero-mean lognormal-free relative error,
            or ('bias', r) systematic sigma^2 scaled by (1+r).
    rho_t : AR(1) correlation between training and data channel (diffuse part).
    Returns res[m][key] shape (reps,S,K) and aux arrays (reps,S)."""
    M, N = cfg['M'], cfg['N']
    sll = cfg.get('sll', 0.0)
    keys = keys or ['out', 'gp']
    S, K = len(sdbs), len(MARG)
    res = {m: {k: np.zeros((reps, S, K)) for k in keys} for m in METH}
    aux = {'bias_' + m: np.zeros((reps, S)) for m in METH}
    aux.update({'std_' + m: np.zeros((reps, S)) for m in METH})
    aux['beamerr'] = np.zeros((reps, S))
    aux['beamloss_dB'] = np.zeros((reps, S))
    for rp, sq in enumerate(np.random.SeedSequence(seed).spawn(reps)):
        rng = np.random.default_rng(sq)
        for i, sdb in enumerate(sdbs):
            s0 = 10 ** (sdb / 10.0)
            TH, LOS, DIF = draw_paths(rng, B, cfg)
            sig = sig_of(TH, LOS + DIF, s0, M, N, sll)
            y = measure(rng, sig)
            if s2mode is None:
                s2 = 1.0
            elif s2mode[0] == 'rand':
                s2 = np.maximum(1.0 + s2mode[1] * rng.standard_normal(B), 0.05)
            else:
                s2 = 1.0 + s2mode[1]
            est, w, _, _ = estimates(rng, y, sig, guard, s2)
            r = np.arange(B)
            ssel = np.abs(sig[r, w]) ** 2
            best = (np.abs(sig) ** 2).argmax(1)
            correct = (w == best)
            aux['beamerr'][rp, i] = float((~correct).mean())
            pb = (np.abs(sig) ** 2).max(1)
            aux['beamloss_dB'][rp, i] = float(10 * np.log10(ssel.mean() / max(pb.mean(), 1e-12)))
            if rho_t >= 1.0:
                ssel_d = ssel
            else:
                DIF2 = rho_t * DIF + np.sqrt(1 - rho_t ** 2) * _redraw(rng, DIF, cfg)
                sig2 = sig_of(TH, LOS + DIF2, s0, M, N, sll)
                ssel_d = np.abs(sig2[r, w]) ** 2
            for m in METH:
                aux['bias_' + m][rp, i] = float(np.mean(est[m] - ssel))
                aux['std_' + m][rp, i] = float(np.std(est[m] - ssel))
                g = grid_metrics(est[m], ssel_d, correct)
                for k in keys:
                    res[m][k][rp, i] = g[k]
    return res, aux


def _redraw(rng, DIF, cfg):
    """Fresh diffuse gains with the same per-path variances as DIF."""
    B, P = DIF.shape
    nsub = cfg.get('nsub', 1) if cfg.get('spread', 0.0) > 0 else 1
    Kdb = cfg.get('Kdb', None)
    pside = cfg.get('pside', 0.05)
    out = np.zeros((B, P), complex)
    if Kdb is not None:
        K = 10.0 ** (float(Kdb) / 10.0)
        s = np.sqrt(1.0 / (2 * (K + 1.0) * nsub))
        out[:, :nsub] = s * (rng.standard_normal((B, nsub)) + 1j * rng.standard_normal((B, nsub)))
    if P > nsub:
        ns = P - nsub
        out[:, nsub:] = np.sqrt(pside / 2) * (rng.standard_normal((B, ns))
                                              + 1j * rng.standard_normal((B, ns)))
    return out


def tune(res_cal, target=0.1, keys_gp='gp'):
    """Worst-case-over-SNR outage <= target, then best mean goodput."""
    K = {}
    for m in METH:
        o = res_cal[m]['out'].mean(0)      # S,K
        g = res_cal[m][keys_gp].mean(0)
        feas = np.where(o.max(0) <= target)[0]
        K[m] = int(feas[np.argmax(g[:, feas].mean(0))]) if len(feas) else None
    return K
