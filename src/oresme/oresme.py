# oresme.py
"""
A module for generating Oresme numbers (harmonic series partial sums)
"""

from fractions import Fraction
import math
import numpy as np
from typing import List, Union, Generator

def oresme_sequence(n_terms: int) -> List[float]:
    """Generates the first n terms of the Oresme sequence (1/2^n series)."""
    return [i / (2 ** i) for i in range(1, n_terms + 1)]

def harmonic_numbers(n_terms: int, start_index: int = 1) -> List[Fraction]:
    """Generates the first n terms of harmonic numbers (H_n = 1 + 1/2 + ... + 1/n)."""
    sequence = []
    current_sum = Fraction(0)
    for i in range(start_index, start_index + n_terms):
        current_sum += Fraction(1, i)
        sequence.append(current_sum)
    return sequence

def harmonic_number(n: int) -> float:
    """Calculates the nth harmonic number (H_n = 1 + 1/2 + ... + 1/n)."""
    return sum(1/k for k in range(1, n + 1))

def harmonic_number_approx(n: int) -> float:
    """Approximates the nth harmonic number using Euler-Mascheroni constant."""
    gamma = 0.57721566490153286060  # Euler-Mascheroni constant
    return math.log(n) + gamma + 1/(2*n)

def harmonic_generator(n: int) -> Generator[float, None, None]:
    """Generates harmonic numbers H_1 to H_n."""
    total = 0.0
    for k in range(1, n + 1):
        total += 1/k
        yield total

def harmonic_numbers_numpy(n: int) -> np.ndarray:
    """Calculates harmonic numbers H_1 to H_n using NumPy for vectorization."""
    return np.cumsum(1/np.arange(1, n + 1))

def is_in_hilbert(sequence: Union[List[float], np.ndarray, Generator[float, None, None]], 
                 max_terms: int = 10000, 
                 tolerance: float = 1e-6) -> bool:
    """
    Determines if a given sequence belongs to the Hilbert space ℓ².
    A sequence {a_n} is in ℓ² (Hilbert space) if the sum of the squares of its terms is finite:
        Σ |a_n|² < ∞
    This function computes the partial sum of squared terms up to `max_terms` and checks
    whether the sum converges within a given tolerance (i.e., the increments become negligible).
    Parameters
    ----------
    sequence : list, np.ndarray, or generator
        The input sequence to test (e.g., [1, 1/2, 1/3, ...]).
    max_terms : int, optional
        Maximum number of terms to consider for convergence check. Default is 10,000.
    tolerance : float, optional
        The threshold for determining convergence. If the increment in cumulative sum
        falls below this value for consecutive steps, the series is considered convergent.
        Default is 1e-6.
    Returns
    -------
    bool
        True if the sequence is likely in ℓ² (sum of squares converges), False otherwise.
    Examples
    --------
    >>> from oresmen import harmonic_numbers_numba, is_in_hilbert
    >>> import numpy as np
    # Harmonic terms: a_n = 1/n → sum(1/n²) converges → in Hilbert space
    >>> n = 1000
    >>> harmonic_terms = 1 / np.arange(1, n+1)
    >>> is_in_hilbert(harmonic_terms)
    True
    # Constant terms: a_n = 1 → sum(1²) = ∞ → not in Hilbert space
    >>> constant_terms = np.ones(1000)
    >>> is_in_hilbert(constant_terms)
    False
    Notes
    -----
    - This is a numerical approximation. True mathematical convergence may require
      analytical proof, but this function provides a practical check for common sequences.
    - Sequences like 1/n, 1/n^(0.6), log(n)/n are tested implicitly via their decay rate.
    """
    # Convert generator to list if needed
    if isinstance(sequence, Generator):
        sequence = list(sequence)
    arr = np.array(sequence, dtype=float)
    squares = arr ** 2

    # Compute cumulative sum of squares
    cumsum = np.cumsum(squares)
    
    # If we have fewer than 2 terms, can't check convergence
    if len(cumsum) < 2:
        return bool(np.isfinite(cumsum[0]))
    
    # Check if increments in cumulative sum become smaller than tolerance
    increments = np.diff(cumsum)
    recent_increments = increments[-100:]  # Last 100 increments for stability
    
    # If all recent increments are below tolerance, assume convergence
    if np.all(recent_increments < tolerance):
        return True
    else:
        return False

# Example usage when module is run directly
if __name__ == "__main__":
    print("Oresme sequence (first 5 terms):", oresme_sequence(5))
    print("Harmonic numbers (H1-H5):", harmonic_numbers(5))
    print("5th harmonic number:", harmonic_number(5))
    print("Approximation of 1000th harmonic number:", harmonic_number_approx(1000))
    print("Harmonic generator (first 3 terms):", list(harmonic_generator(3)))
    print("NumPy harmonic numbers (H1-H5):", harmonic_numbers_numpy(5))
