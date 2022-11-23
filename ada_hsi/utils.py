from colorsys import yiq_to_rgb
from operator import xor
from turtle import forward
import torch 
import math
from torch.autograd import Function, Variable
from skimage.segmentation import slic
import numpy as np 

def superpixels(img, N):
  numSegments = N
  D = slic(img, n_segments=numSegments, compactness=0.1, start_label=0)
  D = D / np.max(D)
  D = np.floor(D*(numSegments-1)).astype(np.int64)
  D =  torch.tensor(D)
  return D


def dec2bit(dec, n):

    bit = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        bit[i] = torch.tensor( dec % 2 , dtype=torch.long)
        dec = dec // 2
    return bit

def Dt2Dmap(Dt):
    Dmap   = Dt[0].permute(1, 2, 0).detach().cpu().numpy()
    indxs  = np.random.permutation(Dmap.shape[-1])[None, None, ...]
    Dmap = np.multiply(Dmap, indxs)
    Dmap = np.sum(Dmap, axis=-1)
    return Dmap

  
H = torch.tensor([[1, 1], [1, -1]])

def hadamard_row(row, n):

    bits = dec2bit(row, n)
    h = torch.ones(1, 1)

    for i in range(n):
        hi = H[bits[i]]
        h = torch.kron(h, hi)
    return h


def uniform_sampling(size, down_factor):

    A = torch.arange(0, size**2).reshape(size, size)
    A = torch.kron(A, torch.ones(down_factor, down_factor))
    return A.type(torch.int64)


def generate_code(row, n, D):

    [H, W] = D.shape
    Hrow = hadamard_row(row, n)
    D = D.reshape(1, H*W)
    Code = torch.gather(Hrow, 1, D)
    Code = Code.reshape(H, W, 1)
    return Code


def spc_forward_step(x, row, n, D):
    C = generate_code(row, n, D)
    y = torch.sum(  torch.multiply(x, C) , dim=(0, 1))
    return y


def spc_forward(x, n, D):
    total = 2**n
    y = torch.stack([ spc_forward_step(x, i, n, D) for i in range(total) ], dim=0)
    return y


def ideal_spc(x, D):
    
    y = torch.zeros_like(x)
    Dsum = torch.sum( D, dim=(2, 3), keepdim=True)
    w = torch.maximum( torch.ones_like(Dsum), Dsum )

    for i in range(x.shape[1]):
        yi = x[:, i, :, :]
        yi = torch.unsqueeze(yi, dim=1)
        yi = torch.sum( torch.multiply(yi, D), dim=(2, 3), keepdim=True) / w
        yi = torch.multiply(yi, D)
        yi = torch.sum(yi, dim=1)
        y[:, i, :, :] = yi

    return y


import torch.nn.functional as F

def matrixSPC(x_tensor, H, rI=None):
    # H is a matrix of size (S, H*W)
    # HinvH = torch.matmul( torch.pinverse(H) , H)
    HtH = torch.matmul( H.t() , H)
    if rI is not None:
        Hinv = HtH + rI
    else:
        Hinv = HtH

    Hinv = torch.inverse(Hinv)
    Hinv = torch.matmul(Hinv, HtH)
    Hinv = Hinv.unsqueeze(0)    
    x_est = torch.flatten(x_tensor, start_dim=2)             # [B, C, H*W]
    x_est = torch.transpose(x_est, 2, 1)    
    x_est = torch.einsum('bij,bjk->bik', Hinv, x_est)       # [B, H*W, C]
    x_est = torch.transpose(x_est, 2, 1)                        # [B, C, H*W]
    x_est = torch.reshape(x_est, x_tensor.shape)                # [B, C, H, W]

    return x_est    

def matrixHSI(x_tensor, H, D):
    # H is a matrix of size (S, H*W)
    # HinvH = torch.matmul( torch.pinverse(H) , H)
    M = H.shape[0]
    A = torch.sum(D, dim=1) * M
    A = 1 / torch.maximum( A, torch.ones_like(A) )
    A = torch.diag(A)                                       # Dp = diag( 1 /   M @ D @ 1 )
    A = torch.matmul( D.t(), A )                            # D.T @ Dp
    A = torch.matmul( A, H.t() )                            # D.T @ Dp @ H.T
    A = torch.matmul( A, torch.matmul( H, D) )              # D.T @ Dp @ H.T @ H @ D

    A = A.unsqueeze(0)
    x_est = torch.flatten(x_tensor, start_dim=2)            # [B, C, H*W]
    x_est = torch.transpose(x_est, 2, 1)
    x_est = torch.einsum('bij,bjk->bik', A, x_est)          # [B, H*W, C]
    x_est = torch.transpose(x_est, 2, 1)                    # [B, C, H*W]
    x_est = torch.reshape(x_est, x_tensor.shape)            # [B, C, H, W]

    return x_est



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def recons_loss(y_true, prob, downsize, loss_fn):

    # prob  = binary_softmax(prob)
    y_est = poolfeat(y_true, prob, downsize, downsize)
    y_est = upfeat(y_est, prob, downsize, downsize)
    loss  = loss_fn(y_true, y_est)
    
    return loss


def matrixSPC(x_tensor, H, rI=None):
    # H is a matrix of size (S, H*W)
    # HinvH = torch.matmul( torch.pinverse(H) , H)
    HtH = torch.matmul( H.t() , H)
    if rI is not None:
        Hinv = HtH + rI
    else:
        Hinv = HtH

    Hinv = torch.inverse(Hinv)
    Hinv = torch.matmul(Hinv, HtH)
    Hinv = Hinv.unsqueeze(0)
    x_est = torch.flatten(x_tensor, start_dim=2)                # [B, C, H*W]
    x_est = torch.transpose(x_est, 2, 1)    
    x_est = torch.einsum('bij,bjk->bik', Hinv, x_est)           # [B, H*W, C]
    x_est = torch.transpose(x_est, 2, 1)                        # [B, C, H*W]
    x_est = torch.reshape(x_est, x_tensor.shape)                # [B, C, H, W]
    return x_est