import numpy as np


def isZero(x, atol):
    """Determine if a floating point number is equal to zero."""
    return abs(x) <= atol


def areEqual(x, y, atol):
    """Determine if two floating point numbers are equal."""
    return abs(x - y) <= atol


def myArrayEq(x, y, atol):
    """Determine if two numpy arrays of floating point numbers are equal."""
    return np.allclose(x, y, atol=atol, rtol=0)


# Update the alias
myPointEq = myArrayEq


def myPointsEq(x, y, atol):
    """Determine if two lists of numpy arrays are equal."""
    if len(x) != len(y):
        return False
    # Convert lists to numpy arrays for vectorized operations
    x_array = np.array(x)
    y_array = np.array(y)
    return np.allclose(x_array, y_array, atol=atol, rtol=0)


def ListHasPoint(L, P, atol):
    """Determine if a list of numpy arrays contains a specific point."""
    # Handle empty list case
    if len(L) == 0:
        return False
    # Convert list to numpy array for vectorized operations
    L_array = np.array(L)
    # Broadcasting the comparison across all points
    return np.any(np.all(np.abs(L_array - P) <= atol, axis=1))
