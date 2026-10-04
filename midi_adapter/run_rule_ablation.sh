#!/usr/bin/env bash
# =============================================================================
#  Rule-model ablation: which compiled program, and does the cross-attention
#  earn its place?
#
#  The diagnosis from the first no-input run was that KEY INFERENCE is the
#  bottleneck (key_found 0.24-0.41, and halfbar_acc tracks it almost exactly),
#  not the phase schedule. These variants attack that directly:
#
#    retrieve   Aggregate only; the adapter does the modular arithmetic
#    full       conventional TracR: Aggregate + SequenceMap(MLP)
#    full/triad same, with a chromagram in and out instead of a root
#    mlp_only   NO head -- the projection estimates the key itself from the
#               full causal context, the MLP applies the rule
#
#  and, orthogonally, whether the cross-attention does any routing:
#
#    CR=none    pure positional Q/K -> query t reads rule position t, so the
#               attention is a per-position read plus a global mean
#    CR=q       Q += q_proj(h): the music can steer where it looks
#
#  Usage:
#    bash midi_adapter/run_rule_ablation.sh              # train + eval all
#    bash midi_adapter/run_rule_ablation.sh report       # just re-read logs
#    VARIANTS="mlp_only mlp_only_crq" bash ... # a subset
# =============================================================================

set -uo pipefail
RASP_REPO=/l/users/xinyue.li/rasp
cd "$RASP_REPO" || exit 1

# label                 PROGRAM    PROXY_ACT  CR    RULE_IN
read -r -d '' SPECS <<'SPEC'
retrieve_h1           retrieve   none       none  root
retrieve_h2           retrieve   none       none  root
full                  full       hard       none  root
full_triad            full       hard       none  triad
mlp_only              mlp_only   hard       none  root
mlp_only_crq          mlp_only   hard       q     root
full_crq              full       hard       q     root
SPEC

suffix_for() {   # suffix_for PROGRAM PROXY CR RULE_IN HEADS -> the run-name suffix
    local prog=$1 act=$2 cr=$3 rin=$4 heads=$5 suf=""
    [ "$prog" = retrieve ] && [ "$heads" != 1 ] && suf="${suf}_h${heads}"
    [ "$cr"  != none ]     && suf="${suf}_cr${cr}"
    [ "$act" != none ]     && suf="${suf}_${act}"
    [ "$prog" != retrieve ] && suf="${suf}_${prog}"
    [ "$rin" != root ]      && suf="${suf}_${rin}"
    echo "$suf"
}

WANT="${VARIANTS:-}"
MODE="${1:-run}"

if [ "$MODE" != report ]; then
    while read -r label prog act cr rin; do
        [ -z "$label" ] && continue
        [ -n "$WANT" ] && ! grep -qw "$label" <<<"$WANT" && continue
        heads=1; [ "$label" = retrieve_h2 ] && heads=2
        echo -e "\n══════════ $label ══════════"
        PROGRAM="$prog" PROXY_ACT="$act" CR="$cr" RULE_IN="$rin" HEADS="$heads" \
            bash midi_adapter/run_adapter_tracr.sh direct
    done <<<"$SPECS"
fi

# ── report ──────────────────────────────────────────────────────────────
echo -e "\n══════════ ABLATION SUMMARY (direct test, SEEN keys) ══════════"
python3 - "$RASP_REPO" <<'PY'
import os, re, sys
repo = sys.argv[1]
specs = [l.split() for l in """
retrieve_h1           retrieve   none       none  root  1
retrieve_h2           retrieve   none       none  root  2
full                  full       hard       none  root  1
full_triad            full       hard       none  triad 1
mlp_only              mlp_only   hard       none  root  1
mlp_only_crq          mlp_only   hard       q     root  1
full_crq              full       hard       q     root  1
""".strip().splitlines()]

def suffix(prog, act, cr, rin, heads):
    s = ''
    if prog == 'retrieve' and heads != '1': s += f'_h{heads}'
    if cr  != 'none': s += f'_cr{cr}'
    if act != 'none': s += f'_{act}'
    if prog != 'retrieve': s += f'_{prog}'
    if rin != 'root': s += f'_{rin}'
    return s

def grab(path):
    """First (SEEN) block: halfbar POOL + the diagnostics."""
    if not os.path.exists(path): return None
    txt = open(path).read().split('Loading UNSEEN')[0]
    out = {}
    m = re.search(r'^\s*POOL\s+\d+\s+([\d.]+)', txt, re.M)
    if m: out['halfbar'] = float(m.group(1))
    for key, pat in (('I', r'I slots.*?([\d.]+)'), ('IVV', r'IV/V slots.*?([\d.]+)'),
                     ('parked', r'parked on one chord\s+([\d.]+)'),
                     ('key', r'modal root == key\s+([\d.]+)'),
                     ('cond', r'halfbar \| key found\s+([\d.]+)')):
        m = re.search(pat, txt)
        if m: out[key] = float(m.group(1))
    return out or None

print(f'  {"variant":<16}{"halfbar":>9}{"key_found":>11}{"IV/V":>8}{"I":>8}'
      f'{"parked":>9}{"hb|key":>9}')
print('  ' + '-' * 70)
print(f'  {"constant tonic":<16}{0.500:>9.3f}{"—":>11}{"0.000":>8}{"1.000":>8}'
      f'{"1.000":>9}{"—":>9}')
for label, prog, act, cr, rin, heads in specs:
    r = grab(os.path.join(repo, 'eval_logs_grid',
                          f'adapter_tracr_direct{suffix(prog,act,cr,rin,heads)}_direct.log'))
    if not r:
        print(f'  {label:<16}{"not run":>9}')
        continue
    g = lambda k: f'{r[k]:.3f}' if k in r else '—'
    print(f'  {label:<16}{g("halfbar"):>9}{g("key"):>11}{g("IVV"):>8}{g("I"):>8}'
          f'{g("parked"):>9}{g("cond"):>9}')
print()
print('  key_found is the metric: halfbar_acc tracked it to within 0.002 on the')
print('  first run, so a variant that does not move key_found will not move the')
print('  headline either. hb|key is halfbar restricted to windows where the key')
print('  was right — it says how good the schedule is once perception succeeds.')
PY
