import numpy as np
import random


def init_random_board():
    board = np.zeros((8,8), dtype=int)
    # start with one rook per row (simple init)
    for i in range(8):
        j = np.random.randint(0,8)
        board[i,j] = 1
    return board


def energy(board, A=6.0, B=6.0):
    rows = board.sum(axis=1)
    cols = board.sum(axis=0)
    E = (A/2.0) * np.sum((rows - 1.0)**2) + (B/2.0) * np.sum((cols - 1.0)**2)
    return E


def greedy_lower_energy(board, A=6.0, B=6.0, iterations=3000):
    b = board.copy().astype(float)
    for _ in range(iterations):
        i = np.random.randint(0,8)
        j = np.random.randint(0,8)
        old = b[i,j]
        b[i,j] = 1
        e1 = energy(b, A, B)
        b[i,j] = 0
        e0 = energy(b, A, B)
        b[i,j] = 1 if e1 < e0 else 0
    return b.astype(int)


def print_board(board):
    for r in board:
        print(''.join('R' if x==1 else '.' for x in r))


if __name__ == '__main__':
    print('--- Eight-Rook Hopfield-style solver ---')
    b0 = init_random_board()
    print('Initial board:')
    print_board(b0)
    print('Initial energy:', energy(b0))
    bsol = greedy_lower_energy(b0, A=6.0, B=6.0, iterations=5000)
    print('\nAfter energy minimization:')
    print_board(bsol)
    print('Final energy:', energy(bsol))
    print('Row sums:', bsol.sum(axis=1))
    print('Col sums:', bsol.sum(axis=0))

