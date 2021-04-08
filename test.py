import numpy as np
import torch
from torch import nn
from libs.activation import *
from libs.transformer import *
import libs.functional as F

_MAX_LENGTH = 80

def assertEqual(a, b):
    for ai, bi in zip(a,b):
        assert ai == bi

def assertTrue(expr, msg=None):
    """Check that the expression is true."""
    if not expr:
        msg = f"{safe_repr(expr)} is not true"
        raise msg


def safe_repr(obj, short=False):
    try:
        result = repr(obj)
    except Exception:
        result = object.__repr__(obj)
    if not short or len(result) < _MAX_LENGTH:
        return result
    return result[:_MAX_LENGTH] + ' [truncated]...'
