#%%
import os, sys
current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(current_path)
sys.path.append(SRC_ROOT)

#%%
import torch
import numpy as np
import torch.nn as nn
from libs.my_utils import assertEqual
from libs.common_utils import make_tensor
# %%
def _check_einsum(*args, np_args=None):
    if np_args is None:
        np_args = [arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
    res = torch.einsum(*args)
    ref = np.einsum(*np_args)
    assertEqual(torch.from_numpy(np.array(ref)), res)
# %%
device = 'cpu'
dtype = torch.float

x = make_tensor((5,), device, dtype)
y = make_tensor((7,), device, dtype)
A = make_tensor((3, 5), device, dtype)
B = make_tensor((2, 5), device, dtype)
C = make_tensor((2, 3, 5), device, dtype)
D = make_tensor((2, 5, 7), device, dtype)
E = make_tensor((7, 9), device, dtype)
F = make_tensor((2, 3, 3, 5), device, dtype)
G = make_tensor((5, 4, 6), device, dtype)
H = make_tensor((4, 4), device, dtype)
I = make_tensor((2, 3, 2), device, dtype)

# %%
_check_einsum('i->', x)                     # sum
_check_einsum('i,i->', x, x)                # dot
_check_einsum('i,i->i', x, x)               # vector element-wisem mul
_check_einsum('i,j->ij', x, y)              # outer

# Matrix operations
_check_einsum("ij->ji", A)                  # transpose
_check_einsum("ij->j", A)                   # row sum
_check_einsum("ij->i", A)                   # col sum
_check_einsum("ij,ij->ij", A, A)            # matrix element-wise mul
_check_einsum("ij,j->i", A, x)              # matrix vector multiplication
_check_einsum("ij,kj->ik", A, B)            # matmul
_check_einsum("ij,ab->ijab", A, E)          # matrix outer product

# Tensor operations
_check_einsum("Aij,Ajk->Aik", C, D)         # batch matmul
_check_einsum("ijk,jk->i", C, A)            # tensor matrix contraction
_check_einsum("aij,jk->aik", D, E)          # tensor matrix contraction
_check_einsum("abCd,dFg->abCFg", F, G)      # tensor tensor contraction
_check_einsum("ijk,jk->ik", C, A)           # tensor matrix contraction with double indices
_check_einsum("ijk,jk->ij", C, A)           # tensor matrix contraction with double indices
_check_einsum("ijk,ik->j", C, B)            # non contiguous
_check_einsum("ijk,ik->jk", C, B)           # non contiguous with double indices

# Test diagonals
_check_einsum("ii", H)                      # trace
_check_einsum("ii->i", H)                   # diagonal
_check_einsum('iji->j', I)                  # non-contiguous trace
_check_einsum('ngrg...->nrg...', make_tensor((2, 1, 3, 1, 4), device, dtype))

# Test ellipsis
_check_einsum("i...->...", H)
_check_einsum("ki,...k->i...", A.t(), B)
_check_einsum("k...,jk->...", A.t(), B)
_check_einsum('...ik, ...j -> ...ij', C, x)
_check_einsum('Bik,k...j->i...j', C, make_tensor((5, 3), device, dtype))
_check_einsum('i...j, ij... -> ...ij', C, make_tensor((2, 5, 2, 3), device, dtype))

# torch.bilinear with noncontiguous tensors
l = make_tensor((5, 10), device, dtype, noncontiguous=True)
r = make_tensor((5, 20), device, dtype, noncontiguous=True)
w = make_tensor((15, 10, 20), device, dtype)
_check_einsum("bn,anm,bm->ba", l, w, r)

# with strided tensors
_check_einsum("bn,Anm,bm->bA", l[:, ::2], w[:, ::2, ::2], r[:, ::2])
# %%
