import torch 
import numpy as np

def gray_code(n):
    g0 = np.array([[0], [1]])
    g  = g0

    while g.shape[0] < n:
        g = np.hstack([np.kron(g0 ,  np.ones( (g.shape[0], 1) )  ) , np.vstack([g, g[::-1, :]])])
    return g


def cake_cutting_seq(i, p):
    """Sequence of i-th"""
    step = -i*(-1)**(np.mod(i, 2))

    seq = None
    # if i is odd
    if np.mod(i, 2) == 1:
        seq = list(range(i, i*p + 1, step))
    else:
        seq = list(range(i*p , i - 1, step))

    return seq


def cake_cutting_order(n):
    """Cake cutting order"""
    p = int(np.sqrt(n))
    seq = [ cake_cutting_seq(i, p) for i in range(1, p+1) ]
    seq = [item for sublist in seq for item in sublist]
    return np.argsort(seq)


def hadamard(n):
    if n == 1:
        return np.array([[1]])
    else:
        h = hadamard(n // 2)
        return np.block([[h, h], [h, -h]])

def hadamard_walsh(n):
    H = hadamard(n)
    G = gray_code(n)
    G = G[:, ::-1]
    G = np.dot(G, 2**np.arange(G.shape[1]-1, -1, -1)).astype(np.int32)
    return H[G]

def hadamard_cake_cutting(n):
    H = hadamard_walsh(n)
    CC = cake_cutting_order(n)
    return H[CC]

def matmul(H, x):
    if len(H.shape) == 2:
        H = H.unsqueeze(0)
    return torch.einsum('bij,bjk->bik', H, x)

def left_matmul(H, x):
    x = torch.transpose(x, 1, 2)
    x = matmul(H, x)
    return torch.transpose(x, 1, 2)

def ht2D(H, x):
    x = matmul(H, x)
    x = left_matmul(H, x)
    return x

def iht2D(H, x):
    x = left_matmul(H.T, x)
    x = matmul(H.T, x)
    x = x / (H.shape[0] * H.shape[1])
    return x


def ht1D(H, x):
    c, h, w = x.shape
    x = x.reshape(c, h*w).transpose(0, 1)
    x = torch.matmul(H, x)
    x = x.transpose(0, 1).reshape(c, h, w)
    return x

def iht1D(H, x):
    x = ht1D(H.T, x)
    x = x / H.shape[0]
    return x