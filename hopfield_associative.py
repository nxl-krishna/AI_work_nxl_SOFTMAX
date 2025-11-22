import numpy as np
import random


def bipolarize(p):
    p = np.array(p)
    if set(np.unique(p)) <= {0,1}:
        return (p * 2 - 1).astype(int)
    return p.astype(int)


def hebbian_weights(patterns, normalize=True):
    N = len(patterns[0])
    W = np.zeros((N, N), dtype=float)
    for p in patterns:
        p = bipolarize(p)
        W += np.outer(p, p)
    np.fill_diagonal(W, 0.0)
    if normalize and len(patterns) > 0:
        W = W / len(patterns)
    return W


def async_update(state, W, max_iters=2000):
    N = len(state)
    s = state.copy()
    for it in range(max_iters):
        i = random.randrange(N)
        raw = np.dot(W[i], s)
        s_new = 1 if raw >= 0 else -1
        s[i] = s_new
        # quick check for stability every N steps
        if it % N == 0:
            sync = np.sign(W.dot(s))
            sync[sync == 0] = 1
            if np.array_equal(sync, s):
                break
    return s


def random_patterns(P, N=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return [np.random.choice([-1, 1], size=N) for _ in range(P)]


def flip_fraction(pat, frac, seed=None):
    if seed is not None:
        np.random.seed(seed)
    p = pat.copy()
    N = len(p)
    k = int(round(frac * N))
    idx = np.random.choice(N, size=k, replace=False)
    p[idx] *= -1
    return p


def test_error_correction():
    N = 100
    P = 10
    patterns = random_patterns(P, N=N, seed=1)
    W = hebbian_weights(patterns)
    target = patterns[0]
    fracs = [0.0, 0.05, 0.1, 0.2, 0.3]
    print("Error-correction test (pattern 0):")
    for f in fracs:
        succ = 0
        trials = 20
        for t in range(trials):
            noisy = flip_fraction(target, f)
            out = async_update(noisy, W)
            if np.array_equal(out, target):
                succ += 1
        print(f"flip {int(100*f)}%: {succ}/{trials} success -> {succ/trials*100:.1f}%")


def capacity_sweep(max_P=25):
    N = 100
    print("Capacity sweep (recall with 10% noise):")
    for P in range(1, max_P+1):
        pats = random_patterns(P, N=N)
        W = hebbian_weights(pats)
        correct = 0
        for p in pats:
            noisy = flip_fraction(p, 0.10)
            out = async_update(noisy, W)
            if np.array_equal(out, p):
                correct += 1
        print(f"P={P:2d}, recalled {correct}/{P}")


if __name__ == '__main__':
    print("--- Hopfield associative memory ---")
    test_error_correction()
    print('\n--- Capacity sweep ---')
    capacity_sweep(20)