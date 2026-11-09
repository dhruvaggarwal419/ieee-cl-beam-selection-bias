"""(E) Mobility / channel aging between the training measurement and the data
transmission. AR(1) correlation rho on the diffuse channel component."""
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
RHOS = [1.0, 0.99, 0.95, 0.9]
SCEN = [('LOS_K10dB', dict(M=64, N=64, nside=2, pside=0.05, Kdb=10.0)),
        ('NLOS_rayleigh', dict(M=64, N=64, nside=2, pside=0.05, Kdb=-100.0)),
        ('LOS_fixed_paper', dict(M=64, N=64, nside=2, pside=0.05, Kdb=None))]
t0 = time.time()
R = {'meta': {'reps': REPS, 'B': B, 'sdbs': SD, 'rhos': RHOS,
              'note': 'AR(1) applies to the diffuse component; LOS component and angles are fixed'}}
for sname, cfg in SCEN:
    cal0, _ = L.run_cfg(cfg, SD, REPS, B, 1000, keys=KEYS, rho_t=1.0)
    K0 = L.tune(cal0)
    sec = {'knobs_static': {m: (None if K0[m] is None else list(map(float, L.MARG[K0[m]])))
                            for m in L.METH}}
    for rho in RHOS:
        ev, aux = L.run_cfg(cfg, SD, REPS, B, 2000, keys=KEYS, rho_t=rho)
        cal, _ = L.run_cfg(cfg, SD, REPS, B, 1000, keys=KEYS, rho_t=rho)
        K1 = L.tune(cal)
        d = {'knobs_retuned': {m: (None if K1[m] is None else list(map(float, L.MARG[K1[m]])))
                               for m in L.METH}, 'methods': {}}
        for m in L.METH:
            e = {}
            for tag, k in [('transfer_from_static', K0[m]), ('retuned', K1[m])]:
                if k is None:
                    e[tag] = None
                    continue
                e[tag] = {'gp': L.ci(ev[m]['gp'][:, :, k].mean(1)),
                          'worst_out': L.ci(ev[m]['out'][:, :, k].max(1)),
                          'per_snr_out': [float(ev[m]['out'][:, i, k].mean()) for i in range(len(SD))],
                          'p_tx': L.ci(ev[m]['ptx'][:, :, k].mean(1))}
            d['methods'][m] = e
        sec['rho%g' % rho] = d
        print('E %s rho=%g %.0fs' % (sname, rho, time.time() - t0), flush=True)
        R[sname] = sec
        json.dump(R, open(OUT + '/r2_E_mobility.json', 'w'), indent=1)
print('E done %.0fs' % (time.time() - t0), flush=True)
