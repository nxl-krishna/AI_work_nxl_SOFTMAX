import numpy as np


def build_weights(dist, A=500.0, B=500.0, C=200.0):
    N = dist.shape[0]
    M = N*N
    W = np.zeros((M, M))
    def idx(i,p):
        return i*N + p
    # city constraint: each city appears once
    for i in range(N):
        for p in range(N):
            for q in range(N):
                if p != q:
                    W[idx(i,p), idx(i,q)] -= A
    # position constraint: each position has exactly one city
    for p in range(N):
        for i in range(N):
            for j in range(N):
                if i != j:
                    W[idx(i,p), idx(j,p)] -= B
    # cost term
    for p in range(N):
        q = (p + 1) % N
        for i in range(N):
            for j in range(N):
                W[idx(i,p), idx(j,q)] -= C * dist[i,j]
    np.fill_diagonal(W, 0.0)
    return W


if __name__ == '__main__':
    N = 10
    rng = np.random.default_rng(2)
    pts = rng.random((N,2))
    dist = np.sqrt(((pts[:,None,:] - pts[None,:,:])**2).sum(axis=2))
    W = build_weights(dist)
    neurons = N*N
    unique_weights = neurons*(neurons-1)//2
    print(f"Cities: {N}, neurons: {neurons}, unique pairwise weights: {unique_weights}")
    print('Weight matrix shape:', W.shape)
