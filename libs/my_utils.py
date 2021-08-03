_MAX_LENGTH = 80

def assertEqual_simple(a, b):
    for ai, bi in zip(a,b):
        assert ai == bi

def barf():
    import pdb
    pdb.set_trace()

def assertEqual(tensor, expected, threshold=0.001):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        for t, e in zip(tensor, expected):
            assertEqual(t, e)
    else:
        if (tensor - expected).abs().max() > threshold:
            # barf()
            raise AssertionError(f"{safe_repr(tensor)} != {safe_repr(expected)}")

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