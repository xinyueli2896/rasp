"""Mirror ChordTracrRuleModel's matrix construction in numpy and check the
compiled head actually retrieves the tonic for every key and every position."""
import numpy as np

V, P, D, SPC, SCALE = 12, 4, 28, 8, 20.0
OFFSETS = [0, 5, 7, 0]
PB = D - P   # 24

W_E = np.zeros((V, D)); W_E[:V, :V] = np.eye(V)
W_pos = np.zeros((P, D))
for i in range(P): W_pos[i, PB + i] = 1.0
W_Q = np.zeros((D, D))
for i in range(P): W_Q[PB + 0, PB + i] = 1.0
W_K = np.zeros((D, D))
for i in range(P): W_K[PB + i, PB + i] = 1.0
W_V = np.zeros((D, D))
for i in range(V): W_V[V + i, i] = 1.0
W_O = np.zeros((D, D))
for i in range(V): W_O[V + i, V + i] = 1.0

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def run(T, key):
    t = np.arange(T); phase = (t // SPC) % P
    root = (key + np.array([OFFSETS[p] for p in phase])) % V
    x = W_E[root] + W_pos[phase]                       # (T, D)
    Q, K, Vv = x @ W_Q.T, x @ W_K.T, x @ W_V.T
    s = (Q @ K.T) * SCALE
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    a = softmax(s)
    return x + (a @ Vv) @ W_O.T, a, phase, root

T = 64
print(f'{"key":>4}{"tonic recovered":>18}{"min mass on phase-0":>22}{"min margin":>12}')
worst_mass, all_ok = 1.0, True
for key in range(12):
    h, a, phase, root = run(T, key)
    tonic = h[:, V:2*V]
    ok = (tonic.argmax(-1) == key).all()
    # how much attention mass lands on phase-0 positions, per query
    mass = a[:, phase == 0].sum(-1).min()
    # margin between the retrieved tonic and the runner-up
    srt = np.sort(tonic, axis=-1)
    margin = (srt[:, -1] - srt[:, -2]).min()
    all_ok &= bool(ok); worst_mass = min(worst_mass, mass)
    print(f'{key:>4}{str(bool(ok)):>18}{mass:>22.6f}{margin:>12.6f}')

print()
print('all keys recovered exactly :', all_ok)
print('worst-case phase-0 mass    :', f'{worst_mass:.6f}')

# The head must NOT be solving the rule -- dims 0-11 still hold the current
# root, and the adapter has to do (key + OFFSETS[phase]) % 12 itself.
h, a, phase, root = run(T, 7)
print()
print('dims 0-11 argmax (per slot):', h[::SPC, :V].argmax(-1).tolist(), '= current root')
print('dims 12-23 argmax          :', h[::SPC, V:2*V].argmax(-1).tolist(), '= tonic (constant)')
print('dims 24-27 argmax          :', h[::SPC, PB:].argmax(-1).tolist(), '= phase')
print('expected roots for key 7   :', [(7 + OFFSETS[p]) % 12 for p in phase[::SPC]])

# Causality: query t must never attend to a position > t.
print()
print('causal (no future mass)    :', bool((np.triu(a, 1) == 0).all()))

# Position 0..7 (slot 0) can only see slot 0, which IS phase 0 -> still correct.
print('slot-0 queries correct     :', bool((h[:SPC, V:2*V].argmax(-1) == 7).all()))
