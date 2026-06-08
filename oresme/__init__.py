# __init__.py
"""
Oresme, Harmonic Series and Hilbert Space Module
Oresme, Harmonik Seri ve Hilbert Uzayı Modülü

This package provides three computation backends:
- Pure Python + NumPy (default)
- Numba-accelerated (oresme.numba)
- JAX-accelerated (oresme.jax)

All backends expose the same mathematical tools:
- Harmonic number calculations (exact fractions and floating point)
- Oresme sequence (n / 2^n) generation
- Hilbert space (ℓ²) membership tests (mathematically sound)
- Sequence analysis and comparison utilities
- Approximation methods and convergence analysis

Bu paket üç hesaplama arka ucu sunar:
- Saf Python + NumPy (varsayılan)
- Numba hızlandırmalı (oresme.numba)
- JAX hızlandırmalı (oresme.jax)

Tüm arka uçlar aynı matematiksel araçları sağlar:
- Harmonik sayı hesaplamaları (kesirli tam sonuçlar ve kayan noktalı)
- Oresme dizisi (n / 2^n) üretimi
- ℓ² (Hilbert uzayı) aidiyet testleri (matematiksel olarak doğru)
- Dizi analizi ve karşılaştırma yardımcıları
- Yaklaştırım yöntemleri ve yakınsama analizi
"""

import importlib
import warnings

__version__ = "0.1.6"

# ----------------------------------------------------------------------
# Default backend: pure Python + NumPy (always available)
# ----------------------------------------------------------------------
from .oresme import (
    # Core
    oresme_sequence,
    harmonic_numbers,
    harmonic_number,
    harmonic_numbers_numpy,
    harmonic_generator,
    # Approximation
    EULER_MASCHERONI,
    EULER_MASCHERONI_FRACTION,
    ApproximationMethod,
    harmonic_number_approx,
    bernoulli_number,
    harmonic_sum_approx,
    # Hilbert test
    is_in_hilbert,
    # Utilities
    harmonic_sequence,
    p_series,
    geometric_sequence,
    analyze_sequence,
    compare_sequences,
    # Performance & analysis
    benchmark_harmonic,
    compare_with_approximation,
    harmonic_convergence_analysis,
    plot_comparative_performance,
    _run_tests,
    main,
)

# ----------------------------------------------------------------------
# Optional Numba backend (oresme.numba)
# ----------------------------------------------------------------------
try:
    from . import oresmen as _numba_module
    # Attach as a submodule so users can do: oresme.numba.harmonic_number_numba(...)
    import sys
    sys.modules[__name__ + '.numba'] = _numba_module
except ImportError:
    _numba_module = None
    warnings.warn("Numba backend not available. Install numba for acceleration.", ImportWarning)

# ----------------------------------------------------------------------
# Optional JAX backend (oresme.jax)
# ----------------------------------------------------------------------
try:
    from . import oresmej as _jax_module
    import sys
    sys.modules[__name__ + '.jax'] = _jax_module
except ImportError:
    _jax_module = None
    warnings.warn("JAX backend not available. Install jax for acceleration.", ImportWarning)

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
__all__ = [
    # Core
    'oresme_sequence',
    'harmonic_numbers',
    'harmonic_number',
    'harmonic_numbers_numpy',
    'harmonic_generator',
    # Constants & Enums
    'EULER_MASCHERONI',
    'EULER_MASCHERONI_FRACTION',
    'ApproximationMethod',
    # Approximation
    'harmonic_number_approx',
    'bernoulli_number',
    'harmonic_sum_approx',
    # Hilbert test
    'is_in_hilbert',
    # Utility sequences
    'harmonic_sequence',
    'p_series',
    'geometric_sequence',
    # Analysis
    'analyze_sequence',
    'compare_sequences',
    # Performance
    'benchmark_harmonic',
    'compare_with_approximation',
    'harmonic_convergence_analysis',
    'plot_comparative_performance',
    # Optional submodules
    'numba',
    'jax',
    # test
    '_run_tests',
    'main',
]


def get_backend(name: str):
    """
    Return the requested backend module.
    İstenen arka uç modülünü döndürür.

    Parameters
    ----------
    name : str
        One of 'pure', 'numba', 'jax'.
    Returns
    -------
    module
    Raises
    ------
    ValueError if backend is unknown or not installed.
    """
    name = name.lower()
    if name in ('pure', 'default'):
        return importlib.import_module('.oresme', __package__)
    elif name == 'numba':
        if _numba_module is None:
            raise ImportError("Numba backend is not installed.")
        return _numba_module
    elif name == 'jax':
        if _jax_module is None:
            raise ImportError("JAX backend is not installed.")
        return _jax_module
    else:
        raise ValueError(f"Unknown backend: {name}. Choose 'pure', 'numba', or 'jax'.")
