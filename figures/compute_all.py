#!/usr/bin/env python
"""compute_all.py - R3 figure/table/appendix data for the CADCON paper.

Recomputes EVERY figure/table/appendix number from the frozen raw backup
(cspike_frozen/raw/) using the frozen spike_analysis machinery, asserts the
main-text values (EXPECTED, at printed precision), and writes figures_data.json.
This script doubles as the paper's final numeric verification pass.

Run:  <sandbox>/.venv/bin/python compute_all.py
"""
import json, sys, math, collections
from pathlib import Path
import numpy as np
from scipy import stats

SANDBOX = Path(__file__).resolve().parent.parent / 'analysis'
RAW = Path(__file__).resolve().parent.parent / 'data'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SANDBOX))
import spike_analysis as SA  # frozen recall machinery

SEEDS = (0, 1, 2)
frozen = json.load(open(SANDBOX / 'spike_frozen_lists.json'))
N76 = set(frozen['analysis_idx_n76'])
PROFILES = frozen['profiles_4tag']

report = []          # (name, ok, expected, got)
def check(name, got, expected, tol=5e-4):
    ok = (abs(got - expected) <= tol) if isinstance(expected, float) else (got == expected)
    report.append((name, ok, expected, got))
    return ok

# ---------------- load & score ----------------
def load(p):
    return [json.loads(l) for l in open(p)]

def build(rows):
    """cond -> seed -> idx -> {rec, ex, gset} on N76 (exec-inclusive recall)."""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r['idx'] not in N76:
            continue
        rec = SA.recall_4tag(r['intended'], set(r['geom_produced']) if r['executable'] else set())
        if rec is None:
            continue
        d[r['condition']][r['seed']][r['idx']] = {
            'rec': 0.0 if not r['executable'] else rec,
            'ex': bool(r['executable']),
            'gset': frozenset(r['geom_produced']) if r['executable'] else None,
        }
    return d

spike_rows = load(RAW / 'spike_results/geom_scores.jsonl')
f4_rows = load(RAW / 'f4_results/geom_scores.jsonl')
S = build(spike_rows)
F = build(f4_rows)
regex_rj = json.load(open(RAW / 'spike_results/results.json'))

def arm(store, name_fn):
    return {s: store.get(name_fn(s), {}).get(s, {}) for s in SEEDS}

def amean(a, key='rec'):
    vals = [v[key] for s in SEEDS for v in a.get(s, {}).values()]
    return float(np.mean(vals)) if vals else float('nan')

def exrate(a):
    vals = [1.0 if v['ex'] else 0.0 for s in SEEDS for v in a.get(s, {}).values()]
    return float(np.mean(vals)) if vals else float('nan')

def paired(a, b, seed, key='rec'):
    pa, pb = a.get(seed, {}), b.get(seed, {})
    idx = sorted(set(pa) & set(pb))
    return np.array([pa[i][key] - pb[i][key] for i in idx]), idx

def wil(d, alt):
    d = np.asarray(d, float)
    if d.size == 0 or np.all(d == 0):
        return float('nan')
    return float(stats.wilcoxon(d, zero_method='pratt', alternative=alt, method='asymptotic').pvalue)

# arms
M_c   = arm(S, lambda s: 'spike_text_correct_p04')
M_w   = arm(S, lambda s: 'spike_text_shuffled_p04')
M_m   = arm(S, lambda s: 'spike_text_masked_p04')
Mk_c  = arm(S, lambda s: 'spike_token_correct_p04')
Mk_w  = arm(S, lambda s: 'spike_token_shuffled_p04')
Mk_m  = arm(S, lambda s: 'spike_token_masked_p04')
B04   = arm(S, lambda s: 'spike_baseline_p04')
B00   = arm(S, lambda s: 'spike_baseline_p00')
M_c0  = arm(S, lambda s: 'spike_text_correct_p00')
M_w0  = arm(S, lambda s: 'spike_text_shuffled_p00')
M_m0  = arm(S, lambda s: 'spike_text_masked_p00')
Mk_c0 = arm(S, lambda s: 'spike_token_correct_p00')
Mk_w0 = arm(S, lambda s: 'spike_token_shuffled_p00')
Mk_m0 = arm(S, lambda s: 'spike_token_masked_p00')
R_c   = arm(F, lambda s: f'f4_R_correct_p04_s{s}')
R_w   = arm(F, lambda s: f'f4_R_shuffled_p04_s{s}')
R_m   = arm(F, lambda s: f'f4_R_masked_p04_s{s}')
R_c0  = arm(F, lambda s: f'f4_R_correct_p00_s{s}')
M_cor = arm(F, lambda s: f'f4_M_corrupted_p04_s{s}')
M_rnd = arm(F, lambda s: f'f4_M_random_p04_s{s}')

D = {}  # output payload

# ---------------- A. Table 2 (geom + regex) ----------------
geom_cells = {
    'baseline_p04': amean(B04), 'text_correct_p04': amean(M_c), 'text_wrong_p04': amean(M_w),
    'text_masked_p04': amean(M_m), 'token_correct_p04': amean(Mk_c), 'token_wrong_p04': amean(Mk_w),
    'token_masked_p04': amean(Mk_m), 'baseline_p00': amean(B00), 'text_correct_p00': amean(M_c0),
    'text_wrong_p00': amean(M_w0), 'text_masked_p00': amean(M_m0), 'token_correct_p00': amean(Mk_c0),
    'token_wrong_p00': amean(Mk_w0), 'token_masked_p00': amean(Mk_m0),
}
def regex_mean(cond):
    by = regex_rj[cond]
    return float(np.mean([by[str(s)]['adherence_recall'] for s in SEEDS]))
regex_map = {
    'baseline_p04': 'spike_baseline_p04', 'text_correct_p04': 'spike_text_correct_p04',
    'text_wrong_p04': 'spike_text_shuffled_p04', 'text_masked_p04': 'spike_text_masked_p04',
    'token_correct_p04': 'spike_token_correct_p04', 'token_wrong_p04': 'spike_token_shuffled_p04',
    'token_masked_p04': 'spike_token_masked_p04', 'baseline_p00': 'spike_baseline_p00',
    'text_correct_p00': 'spike_text_correct_p00', 'text_wrong_p00': 'spike_text_shuffled_p00',
    'text_masked_p00': 'spike_text_masked_p00', 'token_correct_p00': 'spike_token_correct_p00',
    'token_wrong_p00': 'spike_token_shuffled_p00', 'token_masked_p00': 'spike_token_masked_p00',
}
regex_cells = {k: regex_mean(v) for k, v in regex_map.items()}
D['table2'] = {'geom': geom_cells, 'regex': regex_cells}

EXP_GEOM = {'baseline_p04': .425, 'text_correct_p04': .557, 'text_wrong_p04': .305, 'text_masked_p04': .263,
            'token_correct_p04': .443, 'token_wrong_p04': .215, 'token_masked_p04': .237, 'baseline_p00': .000,
            'text_correct_p00': .776, 'text_wrong_p00': .160, 'text_masked_p00': .000,
            'token_correct_p00': .500, 'token_wrong_p00': .077, 'token_masked_p00': .000}
for k, v in EXP_GEOM.items():
    check(f'T2.geom.{k}', round(geom_cells[k], 3), v, 1e-9)
EXP_REGEX = {'baseline_p04': .451, 'text_correct_p04': .686, 'text_wrong_p04': .300, 'text_masked_p04': .200,
             'token_correct_p04': .658, 'token_wrong_p04': .261, 'token_masked_p04': .181, 'baseline_p00': .000,
             'text_correct_p00': .677, 'text_wrong_p00': .149, 'text_masked_p00': .000,
             'token_correct_p00': .628, 'token_wrong_p00': .139, 'token_masked_p00': .000}
for k, v in EXP_REGEX.items():
    check(f'T2.regex.{k}', round(regex_cells[k], 3), v, 1e-9)

# ---------------- B. §4.2 wrong<none ----------------
sec42 = {'token_p': [], 'text_p': [], 'wlt_token': [], 'wlt_text': [], 'pc_token_wl': [],
         'vsub_text': [], 'vsub_text_p': [], 'vsub_token': []}
for s in SEEDS:
    d, _ = paired(Mk_w, B04, s); sec42['token_p'].append(wil(d, 'less'))
    sec42['wlt_token'].append([int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())])
    d, _ = paired(M_w, B04, s); sec42['text_p'].append(wil(d, 'less'))
    sec42['wlt_text'].append([int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())])
    d, _ = paired(Mk_c, B04, s)
    sec42['pc_token_wl'].append([int((d > 0).sum()), int((d < 0).sum())])
    # validity subset (both executable)
    for arm_w, key in ((M_w, 'vsub_text'), (Mk_w, 'vsub_token')):
        pw, pb = arm_w.get(s, {}), B04.get(s, {})
        idx = [i for i in set(pw) & set(pb) if pw[i]['ex'] and pb[i]['ex']]
        diffs = np.array([pw[i]['rec'] - pb[i]['rec'] for i in sorted(idx)])
        sec42[key].append(float(diffs.mean()))
        if key == 'vsub_text':
            sec42['vsub_text_p'].append(wil(diffs, 'less'))
D['sec42'] = sec42
for i, v in enumerate([6.8e-6, 1.8e-5, 8.0e-4]):
    check(f'42.token_p.s{i}', sec42['token_p'][i], v, v * 0.06)
for i, v in enumerate([0.073, 0.014, 0.003]):
    check(f'42.text_p.s{i}', round(sec42['text_p'][i], 3), v, 1e-9)
for i, v in enumerate([[19, 0, 57], [20, 1, 55], [19, 4, 53]]):
    check(f'42.wlt_token.s{i}', sec42['wlt_token'][i], v)
for i, v in enumerate([[12, 6, 58], [16, 6, 54], [16, 4, 56]]):
    check(f'42.wlt_text.s{i}', sec42['wlt_text'][i], v)
for i, v in enumerate([-.155, -.230, -.273]):
    check(f'42.vsub_text.s{i}', round(sec42['vsub_text'][i], 3), v, 1e-9)
for i, v in enumerate([-.288, -.250, -.284]):
    check(f'42.vsub_token.s{i}', round(sec42['vsub_token'][i], 3), v, 1e-9)

# ---------------- C. gate ----------------
gate = {
    'R_c_seed': [float(np.mean([v['rec'] for v in R_c[s].values()])) for s in SEEDS],
    'R_w_seed': [float(np.mean([v['rec'] for v in R_w[s].values()])) for s in SEEDS],
    'R_exec_pooled': float(np.mean([1.0 if v['ex'] else 0.0 for a in (R_c, R_w) for s in SEEDS for v in a[s].values()])),
    'baseline_exec': exrate(B04),
}
D['gate'] = gate
for i, v in enumerate([.355, .408, .355]):
    check(f'gate.Rc.s{i}', round(gate['R_c_seed'][i], 3), v, 1e-9)
for i, v in enumerate([.362, .414, .414]):
    check(f'gate.Rw.s{i}', round(gate['R_w_seed'][i], 3), v, 1e-9)
check('gate.R_exec', round(gate['R_exec_pooled'], 3), .535, 1e-9)
check('gate.base_exec', round(gate['baseline_exec'], 3), .553, 1e-9)

# ---------------- D. Test D + guardrails ----------------
testd = {'drop_M': [], 'drop_R': [], 'inter_mean': [], 'p': [], 'nonzero': []}
inter_by_seed = {}
for s in SEEDS:
    dM, _ = paired(M_c, M_w, s)
    dR, _ = paired(R_c, R_w, s)
    pM, pMn, pR, pRn = M_c[s], M_w[s], R_c[s], R_w[s]
    idx = sorted(set(pM) & set(pMn) & set(pR) & set(pRn))
    diff = np.array([(pM[i]['rec'] - pMn[i]['rec']) - (pR[i]['rec'] - pRn[i]['rec']) for i in idx])
    inter_by_seed[s] = (diff, idx)
    testd['drop_M'].append(float(dM.mean())); testd['drop_R'].append(float(dR.mean()))
    testd['inter_mean'].append(float(diff.mean())); testd['p'].append(wil(diff, 'greater'))
    testd['nonzero'].append(int((diff != 0).sum()))
# guard_a: all four arms executable
ga_diffs, ga_n = [], []
for s in SEEDS:
    idx = [i for i in N76 if all(a[s].get(i, {'ex': False})['ex'] for a in (M_c, M_w, R_c, R_w))]
    d = [(M_c[s][i]['rec'] - M_w[s][i]['rec']) - (R_c[s][i]['rec'] - R_w[s][i]['rec']) for i in idx]
    ga_diffs += d; ga_n.append(len(idx))
testd['guard_a_mean'] = float(np.mean(ga_diffs)); testd['guard_a_n'] = ga_n
# guard_b: per-profile effects
prof_eff = []
for prof in PROFILES:
    pid = set(prof['idx']) & N76
    effs = []
    for s in SEEDS:
        vals = [(M_c[s][i]['rec'] - M_w[s][i]['rec']) - (R_c[s][i]['rec'] - R_w[s][i]['rec'])
                for i in pid if i in M_c[s] and i in R_c[s]]
        if vals:
            effs.append(float(np.mean(vals)))
    prof_eff.append({'profile': prof['profile'], 'n': len(pid), 'effect': float(np.mean(effs))})
testd['guard_b_unweighted'] = float(np.mean([p['effect'] for p in prof_eff]))
testd['guard_b_weighted'] = float(np.average([p['effect'] for p in prof_eff], weights=[p['n'] for p in prof_eff]))
testd['per_profile'] = prof_eff
D['testd'] = testd
for i, v in enumerate([.184, .289, .283]):
    check(f'D.dropM.s{i}', round(testd['drop_M'][i], 3), v, 1e-9)
for i, v in enumerate([-.007, -.007, -.059]):
    check(f'D.dropR.s{i}', round(testd['drop_R'][i], 3), v, 1e-9)
for i, v in enumerate([.191, .296, .342]):
    check(f'D.inter.s{i}', round(testd['inter_mean'][i], 3), v, 1e-9)
for i, v in enumerate([4.2e-3, 6.7e-6, 3.6e-6]):
    check(f'D.p.s{i}', testd['p'][i], v, v * 0.06)
check('D.nonzero', testd['nonzero'], [26, 26, 32])
check('D.guard_a', round(testd['guard_a_mean'], 3), .391, 1e-9)
check('D.guard_a_n', testd['guard_a_n'], [20, 33, 34])
check('D.guard_b_u', round(testd['guard_b_unweighted'], 3), .310, 1e-9)
check('D.guard_b_w', round(testd['guard_b_weighted'], 3), .276, 1e-9)

# ---------------- E. exec-only competence ----------------
def exec_only(a):
    return np.array([v['rec'] for s in SEEDS for v in a[s].values() if v['ex']])
eo_R, eo_B, eo_M = exec_only(R_c), exec_only(B04), exec_only(M_c)
ec = {'R': float(eo_R.mean()), 'B': float(eo_B.mean()), 'M': float(eo_M.mean()),
      'nR': len(eo_R), 'nB': len(eo_B), 'nM': len(eo_M),
      'p_RB': float(stats.mannwhitneyu(eo_R, eo_B, alternative='two-sided').pvalue),
      'p_RM': float(stats.mannwhitneyu(eo_R, eo_M, alternative='two-sided').pvalue)}
# per-seed exec-only interaction: (Mc-Mw)-(Rc-Rw), each arm's mean over its own executable outputs
def eo_mean(a, s):
    vals = [v['rec'] for v in a[s].values() if v['ex']]
    return float(np.mean(vals))
gap = [(eo_mean(M_c, s) - eo_mean(M_w, s)) - (eo_mean(R_c, s) - eo_mean(R_w, s)) for s in SEEDS]
ec['gap_MR_per_seed'] = gap
D['exec_only'] = ec
for i, v in enumerate([.47, .51, .49]):
    check(f'E.gapMR.s{i}', round(ec['gap_MR_per_seed'][i], 2), v, 1e-9)
check('E.R', round(ec['R'], 2), .71, 1e-9); check('E.B', round(ec['B'], 2), .77, 1e-9)
check('E.M', round(ec['M'], 2), .94, 1e-9); check('E.nR', ec['nR'], 120)
check('E.p_RB', round(ec['p_RB'], 2), .24, 1e-9)
check('E.p_RM', ec['p_RM'], 1.6e-8, 0.2e-8)

# ---------------- F. output identity ----------------
def identity(a_c, a_w):
    out = []
    for s in SEEDS:
        idx = [i for i in set(a_c[s]) & set(a_w[s]) if a_c[s][i]['ex'] and a_w[s][i]['ex']]
        same = sum(1 for i in idx if a_c[s][i]['gset'] == a_w[s][i]['gset'])
        out.append([same, len(idx)])
    return out
ident = {'R': identity(R_c, R_w), 'M': identity(M_c, M_w)}
D['identity'] = ident
check('F.R', ident['R'], [[28, 30], [30, 42], [41, 42]])
check('F.M', ident['M'], [[13, 33], [15, 43], [18, 42]])

# ---------------- G. within-M ----------------
# masked p00 executability on the FULL 100-program set
mask_ex = []
full = collections.defaultdict(lambda: collections.defaultdict(list))
for r in spike_rows:
    if r['condition'] == 'spike_text_masked_p00':
        full[r['seed']]['ex'].append(bool(r['executable']))
for s in SEEDS:
    mask_ex.append(float(np.mean(full[s]['ex'])))
wm = {'M_c_p00': amean(M_c0), 'M_m_p00': amean(M_m0), 'masked_exec_full100': mask_ex}
D['within_M'] = wm
check('G.c', round(wm['M_c_p00'], 3), .776, 1e-9)
check('G.m', round(wm['M_m_p00'], 3), .000, 1e-9)
check('G.ex.s0', round(mask_ex[0], 2), 0.0, 1e-9)
report.append(('G.ex.s1s2_95_100', 0.95 <= mask_ex[1] <= 1.0 and 0.95 <= mask_ex[2] <= 1.0,
               '0.95-1.00', f'{mask_ex[1]:.2f}/{mask_ex[2]:.2f}'))

# ---------------- H. OOD panel ----------------
ood = {'wrong_exec': exrate(M_w), 'base_exec': exrate(B04), 'inter_diff': [], 'inter_p': []}
for s in SEEDS:
    pw, pb = M_w[s], B04[s]
    idx = [i for i in set(pw) & set(pb) if pw[i]['ex'] and pb[i]['ex']]
    d = np.array([pw[i]['rec'] - pb[i]['rec'] for i in sorted(idx)])
    ood['inter_diff'].append(float(d.mean())); ood['inter_p'].append(wil(d, 'less'))
D['ood'] = ood
check('H.wrong_exec', round(ood['wrong_exec'], 3), .645, 1e-9)
for i, v in enumerate([0.061, 0.005, 2e-4]):
    check(f'H.p.s{i}', ood['inter_p'][i], v, max(v * 0.25, 5e-4))

# ---------------- I. F4b dose ----------------
dose = {'means': {'correct': amean(M_c), 'corrupted': amean(M_cor), 'wrong': amean(M_w), 'random': amean(M_rnd)},
        'c_gt_cor_p': [], 'cor_gt_w_p': [], 'w_gt_rnd_p': []}
for s in SEEDS:
    d, _ = paired(M_c, M_cor, s); dose['c_gt_cor_p'].append(wil(d, 'greater'))
    d, _ = paired(M_cor, M_w, s); dose['cor_gt_w_p'].append(wil(d, 'greater'))
    d, _ = paired(M_w, M_rnd, s); dose['w_gt_rnd_p'].append(wil(d, 'greater'))
D['dose'] = dose
check('I.corrupted', round(dose['means']['corrupted'], 3), .393, 1e-9)
check('I.random', round(dose['means']['random'], 3), .219, 1e-9)
for i, v in enumerate([0.091, 3e-4, 1e-3]):
    check(f'I.c_cor.s{i}', dose['c_gt_cor_p'][i], v, max(v * 0.2, 2e-4))
for i, v in enumerate([0.074, 0.056, 0.038]):
    check(f'I.cor_w.s{i}', round(dose['cor_gt_w_p'][i], 3), v, 2e-3)
for i, v in enumerate([0.011, 0.006, 0.040]):
    check(f'I.w_rnd.s{i}', round(dose['w_gt_rnd_p'][i], 3), v, 2e-3)

# ---------------- J. App B: zero-handling robustness ----------------
rng = np.random.default_rng(20260709)
rob = []
for s in SEEDS:
    diff, _ = inter_by_seed[s]
    nz = diff[diff != 0]
    perm_stats = (rng.choice([-1., 1.], size=(200_000, len(nz))) * nz).sum(axis=1)
    p_perm = float(((perm_stats >= nz.sum()).sum() + 1) / 200_001)
    p_sign = float(stats.binomtest(int((nz > 0).sum()), len(nz), alternative='greater').pvalue)
    try:
        p_exact = float(stats.wilcoxon(nz, alternative='greater', method='exact').pvalue)
    except Exception:
        p_exact = float('nan')
    rob.append({'seed': s, 'pratt': testd['p'][s], 'drop_zero': wil(nz, 'greater'),
                'exact_dropzero': p_exact, 'perm_200k': p_perm, 'sign': p_sign})
D['robustness'] = rob

# ---------------- K. App B: diff-mass decomposition ----------------
mass = []
for s in SEEDS:
    dM, idxM = paired(M_c, M_w, s)
    dR, _ = paired(R_c, R_w, s)
    diff, idx = inter_by_seed[s]
    dMv = {i: v for i, v in zip(idxM, dM)}
    dRv = {i: v for i, v in zip(idxM, dR)}
    both_flat = sum(1 for i in idx if dMv[i] == 0 and dRv[i] == 0)
    m_only = sum(1 for i in idx if dMv[i] != 0 and dRv[i] == 0)
    r_only = sum(1 for i in idx if dMv[i] == 0 and dRv[i] != 0)
    cancel = sum(1 for i in idx if dMv[i] != 0 and dRv[i] != 0 and (dMv[i] - dRv[i]) == 0)
    mass.append({'seed': s, 'both_flat': both_flat, 'M_moves_R_flat': m_only,
                 'R_moves_M_flat': r_only, 'both_move_cancel': cancel,
                 'pos': int((diff > 0).sum()), 'neg': int((diff < 0).sum())})
D['diff_mass'] = mass

# ---------------- L. App B: leave-one-profile-out Test D ----------------
loo = []
for prof in PROFILES + [{'profile': ['<ALL-CIRCLE-IDX>'], 'idx': [i for p in PROFILES if '<CIRCLE>' in p['profile'] for i in p['idx']]}]:
    drop = set(prof['idx'])
    ps, means = [], []
    for s in SEEDS:
        diff, idx = inter_by_seed[s]
        keep = np.array([d for d, i in zip(diff, idx) if i not in drop])
        ps.append(wil(keep, 'greater')); means.append(float(keep.mean()))
    loo.append({'dropped': '+'.join(prof['profile']), 'n_dropped': len(drop & N76),
                'mean': means, 'p': ps,
                'n_dir': sum(1 for m in means if m > 0),
                'n_sig': sum(1 for p in ps if not math.isnan(p) and p < .05)})
D['leave_one_out'] = loo
ngon_row = [r for r in loo if r['dropped'] == '<NGON>'][0]
report.append(('L.dropNGON_pass', ngon_row['n_dir'] >= 2 and ngon_row['n_sig'] >= 2,
               'n_dir>=2,n_sig>=2', f"n_dir={ngon_row['n_dir']},n_sig={ngon_row['n_sig']}"))

# ---------------- M. App B: M1 probes ----------------
m1 = {'content_incl': [], 'content_incl_p': [], 'presence_incl': [], 'content_inter': []}
for s in SEEDS:
    d, _ = paired(R_c, R_w, s)
    m1['content_incl'].append(float(d.mean())); m1['content_incl_p'].append(wil(d, 'less'))
    d, _ = paired(R_c, R_m, s)
    m1['presence_incl'].append(float(d.mean()))
    pw, pb = R_c[s], R_w[s]
    idx = [i for i in set(pw) & set(pb) if pw[i]['ex'] and pb[i]['ex']]
    d = np.array([pw[i]['rec'] - pb[i]['rec'] for i in sorted(idx)])
    m1['content_inter'].append(float(d.mean()))
# seed2 artifact idx: R_correct non-executable & zero-filled while R_wrong executable, nonzero diff
art = [i for i in sorted(N76) if not R_c[2][i]['ex'] and R_w[2][i]['ex'] and R_w[2][i]['rec'] > 0]
m1['seed2_artifact_idx'] = art
pooled = np.concatenate([paired(R_c, R_w, s)[0][np.nonzero(paired(R_c, R_w, s)[0])] for s in SEEDS]) \
    if any(np.any(paired(R_c, R_w, s)[0]) for s in SEEDS) else np.array([])
D['m1'] = m1
check('M.content.s2', round(m1['content_incl'][2], 3), -.059, 1e-9)
check('M.artifact', m1['seed2_artifact_idx'], [56, 76, 77, 97])

# ---------------- N. App B: header-length confound ----------------
try:
    hlen = collections.defaultdict(dict)
    for r in spike_rows:
        if r['condition'] in ('spike_text_correct_p04', 'spike_text_shuffled_p04') and r['idx'] in N76:
            h = r.get('header_injected')
            hlen[(r['condition'], r['seed'])][r['idx']] = len(str(h)) if h is not None else 0
    xs, ys = [], []
    for s in SEEDS:
        dM, idxM = paired(M_c, M_w, s)
        lc = hlen[('spike_text_correct_p04', s)]
        lw = hlen[('spike_text_shuffled_p04', s)]
        for d, i in zip(dM, idxM):
            if i in lc and i in lw:
                xs.append(lw[i] - lc[i]); ys.append(d)
    r_len = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float('nan')
    D['length_confound_r'] = r_len
    report.append(('N.len_r~0.21', abs(r_len - 0.21) < 0.06, '~0.21', f'{r_len:.3f}'))
except Exception as e:  # noqa: BLE001
    D['length_confound_r'] = None
    report.append(('N.len_r', False, '~0.21', f'ERROR {e}'))

# ---------------- O. App C: per-profile drops + non-tied (already have) ----------------
per_prof = []
for prof in PROFILES:
    pid = set(prof['idx']) & N76
    row = {'profile': '+'.join(t.strip('<>') for t in prof['profile']), 'n': len(pid)}
    for nm, (ac, aw) in (('M', (M_c, M_w)), ('R', (R_c, R_w))):
        drops = []
        for s in SEEDS:
            vals = [ac[s][i]['rec'] - aw[s][i]['rec'] for i in pid if i in ac[s] and i in aw[s]]
            if vals:
                drops.append(float(np.mean(vals)))
        row[f'drop_{nm}'] = float(np.mean(drops))
    row['interaction'] = row['drop_M'] - row['drop_R']
    # baseline/wrong floor status
    row['base_mean'] = float(np.mean([B04[s][i]['rec'] for s in SEEDS for i in pid if i in B04[s]]))
    row['wrong_mean'] = float(np.mean([M_w[s][i]['rec'] for s in SEEDS for i in pid if i in M_w[s]]))
    row['correct_mean'] = float(np.mean([M_c[s][i]['rec'] for s in SEEDS for i in pid if i in M_c[s]]))
    per_prof.append(row)
D['per_profile'] = per_prof

# ---------------- write ----------------
json.dump(D, open(OUT / 'figures_data.json', 'w'), indent=1)
n_fail = sum(1 for _, ok, *_ in report if not ok)
lines = [f"{'PASS' if ok else 'FAIL'}  {name}: expected={exp} got={got}" for name, ok, exp, got in report]
(OUT / 'verification_report.txt').write_text('\n'.join(lines) + f'\n\n{len(report)-n_fail}/{len(report)} PASS\n')
print('\n'.join(l for l in lines if l.startswith('FAIL')) or '(no failures)')
print(f'== {len(report)-n_fail}/{len(report)} checks PASS -> figures_data.json ==')
sys.exit(1 if n_fail else 0)
