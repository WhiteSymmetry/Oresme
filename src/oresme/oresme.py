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

# Example usage when module is run directly
if __name__ == "__main__":
    print("Oresme sequence (first 5 terms):", oresme_sequence(5))
    print("Harmonic numbers (H1-H5):", harmonic_numbers(5))
    print("5th harmonic number:", harmonic_number(5))
    print("Approximation of 1000th harmonic number:", harmonic_number_approx(1000))
    print("Harmonic generator (first 3 terms):", list(harmonic_generator(3)))
    print("NumPy harmonic numbers (H1-H5):", harmonic_numbers_numpy(5))
