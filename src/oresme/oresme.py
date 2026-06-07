# oresme.py
"""
Oresme, Harmonic Series and Hilbert Space Module
Oresme, Harmonik Seri ve Hilbert Uzayı Modülü

This module provides (without Numba/JAX):
- Harmonic number calculations (exact fractions and floating point)
- Oresme sequence (n / 2^n) generation
- Hilbert space (ℓ²) membership tests (mathematically sound)
- Sequence analysis and comparison utilities
- Approximation methods and convergence analysis

Bu modül (Numba/JAX olmadan) şunları sağlar:
- Harmonik sayı hesaplamaları (kesirli tam sonuçlar ve kayan noktalı)
- Oresme dizisi (n / 2^n) üretimi
- ℓ² (Hilbert uzayı) aidiyet testleri (matematiksel olarak doğru)
- Dizi analizi ve karşılaştırma yardımcıları
- Yaklaştırım yöntemleri ve yakınsama analizi
"""

import math
import time
import logging
from enum import Enum, auto
from functools import lru_cache
from fractions import Fraction
from typing import List, Union, Generator, Tuple, Optional

import numpy as np

__version__ = "0.3.0"

# -----------------------------
# Logging Configuration / Loglama Yapılandırması
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('harmonic')
logger.propagate = False

# -----------------------------
# Constants and Enums / Sabitler ve Enum'lar
# -----------------------------
class ApproximationMethod(Enum):
    """Harmonic number approximation methods / Harmonik sayı yaklaştırma yöntemleri"""
    EULER_MASCHERONI = auto()
    EULER_MACLAURIN = auto()
    ASYMPTOTIC = auto()

EULER_MASCHERONI = 0.5772156649015328606065120900824024310421
EULER_MASCHERONI_FRACTION = Fraction(303847, 562250)

# Optional scipy for Bernoulli numbers / Bernoulli sayıları için opsiyonel scipy
try:
    from scipy.special import bernoulli as _scipy_bernoulli
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    logger.info("scipy not found, fallback for Bernoulli numbers / scipy bulunamadı, yedek Bernoulli yöntemi kullanılacak")

# -----------------------------
# Basic Functions / Temel Fonksiyonlar
# -----------------------------
def oresme_sequence(n_terms: int, start: int = 1) -> List[float]:
    """
    Generate the Oresme sequence: a_i = i / 2^i
    Oresme dizisi: a_i = i / 2^i
    """
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    return [i / (2 ** i) for i in range(start, start + n_terms)]


@lru_cache(maxsize=128)
def harmonic_numbers(n_terms: int, start_index: int = 1) -> Tuple[Fraction]:
    """Exact fractional harmonic numbers (cached) / Kesirli tam harmonik sayılar (önbellekli)"""
    if n_terms <= 0:
        raise ValueError("n_terms must be positive / n_terms pozitif olmalıdır")
    if start_index <= 0:
        raise ValueError("start_index must be positive / start_index pozitif olmalıdır")
    sequence = []
    current_sum = Fraction(0)
    for i in range(start_index, start_index + n_terms):
        current_sum += Fraction(1, i)
        sequence.append(current_sum)
    return tuple(sequence)


def harmonic_number(n: int) -> float:
    """n-th harmonic number (float) / n-inci harmonik sayı (float)"""
    if n <= 0:
        raise ValueError("n must be positive / n pozitif olmalıdır")
    return sum(1.0 / k for k in range(1, n + 1))


def harmonic_numbers_numpy(n: int) -> np.ndarray:
    """Array of harmonic numbers H_1 … H_n (NumPy) / H_1 … H_n harmonik sayılar dizisi (NumPy)"""
    return np.cumsum(1.0 / np.arange(1, n + 1))


def harmonic_generator(n: int) -> Generator[float, None, None]:
    """Generator of harmonic numbers / Harmonik sayı üreteci"""
    total = 0.0
    for k in range(1, n + 1):
        total += 1.0 / k
        yield total

# -----------------------------
# Approximation Functions / Yaklaştırma Fonksiyonları
# -----------------------------
@lru_cache(maxsize=32)
def bernoulli_number(n: int) -> float:
    """Compute Bernoulli numbers (cached). / Bernoulli sayılarını hesaplar (önbellekli)."""
    if n == 0:
        return 1.0
    if n == 1:
        return -0.5
    if n % 2 != 0:
        return 0.0
    if _HAS_SCIPY:
        return _scipy_bernoulli(n)[n]
    # Fallback for even B_n up to 12 / 12'ye kadar çift Bernoulli sayıları için yedek
    even_bernoulli = {
        2: 1/6, 4: -1/30, 6: 1/42, 8: -1/30, 10: 5/66, 12: -691/2730,
    }
    if n in even_bernoulli:
        return even_bernoulli[n]
    raise NotImplementedError(
        f"Bernoulli number B_{n} requires scipy. / Bernoulli sayısı B_{n} için scipy gereklidir."
    )


def harmonic_number_approx(
    n: int,
    method: ApproximationMethod = ApproximationMethod.EULER_MASCHERONI,
    k: int = 2
) -> float:
    """Approximate harmonic number / Yaklaşık harmonik sayı"""
    if n <= 0:
        raise ValueError("n must be positive / n pozitif olmalıdır")
    if method == ApproximationMethod.EULER_MASCHERONI:
        return math.log(n) + EULER_MASCHERONI + 1/(2*n) - 1/(12*n**2)
    elif method == ApproximationMethod.EULER_MACLAURIN:
        result = math.log(n) + EULER_MASCHERONI + 1/(2*n)
        for i in range(1, k+1):
            B = bernoulli_number(2*i)
            term = B / (2*i) * (1/n)**(2*i)
            result -= term
        return result
    elif method == ApproximationMethod.ASYMPTOTIC:
        return math.log(n) + EULER_MASCHERONI + 1/(2*n)
    else:
        raise ValueError("Unknown approximation method / Bilinmeyen yaklaştırma yöntemi")


def harmonic_sum_approx(n: Union[float, np.ndarray],
                        method: ApproximationMethod = ApproximationMethod.EULER_MACLAURIN,
                        order: int = 4) -> Union[float, np.ndarray]:
    """
    Advanced harmonic series approximation (Euler-Maclaurin).
    Args: n (scalar or array), method, order (2,4,6)
    """
    scalar = np.isscalar(n)
    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    log_n = np.log(n_arr)
    inv_n = 1.0 / n_arr
    result = EULER_MASCHERONI + log_n + 0.5 * inv_n
    if method in (ApproximationMethod.EULER_MACLAURIN,):
        if order >= 2:
            result -= 1/(12 * n_arr**2)
        if order >= 4:
            result += 1/(120 * n_arr**4)
        if order >= 6:
            result -= 1/(252 * n_arr**6)
    if scalar:
        return float(result[0])
    return result


# -----------------------------
# ℓ² (Hilbert Space) Membership Test / ℓ² (Hilbert Uzayı) Aidiyet Testi
# -----------------------------
def is_in_hilbert(
    sequence: Union[List[float], np.ndarray, Generator[float, None, None]],
    max_terms: int = 10000,
    tolerance: float = 1e-8
) -> bool:
    """
    Test whether a sequence belongs to ℓ² (Hilbert space).
    Bir dizinin ℓ² (Hilbert) uzayında olup olmadığını test eder.
    """
    if isinstance(sequence, Generator):
        sequence = list(sequence)
    arr = np.array(sequence, dtype=float)
    if arr.size == 0:
        return True
    if not np.all(np.isfinite(arr)):
        return False

    n_terms = min(len(arr), max_terms)
    test_seq = arr[:n_terms]
    squares = test_seq ** 2
    cumsum = np.cumsum(squares)
    total_sum = cumsum[-1]

    if not np.isfinite(total_sum):
        return False

    # p-series heuristic for positive terms / Pozitif terimler için p-serisi sezgisi
    if n_terms > 500 and np.all(test_seq[100:] > 0):
        log_terms = np.log(test_seq[100:] + 1e-12)
        log_n = np.log(np.arange(100, n_terms))
        try:
            alpha = -np.polyfit(log_n, log_terms, 1)[0]
            if alpha > 0.5:
                return True
            elif 0 < alpha <= 0.5:
                return False
            elif alpha > 10:   # exponential decay / üstel sönüm
                return True
        except Exception:
            pass

    # Tail contribution check / Kuyruk katkısı kontrolü
    if n_terms > 1000:
        last_contribution = squares[-1000:]
        if np.sum(last_contribution) < tolerance:
            return True

    # Ratio test for exponential decay / Üstel sönüm için oran testi
    if n_terms > 100:
        ratios = np.abs(test_seq[1:100] / (test_seq[:99] + 1e-12))
        avg_ratio = np.mean(ratios)
        if avg_ratio < 0.95:
            return True

    return np.isfinite(total_sum)


# -----------------------------
# Utility Functions / Yardımcı Fonksiyonlar
# -----------------------------
def harmonic_sequence(n_terms: int, start: int = 1) -> np.ndarray:
    """Generate harmonic sequence terms: a_n = 1/n / Harmonik dizi terimlerini üretir: a_n = 1/n"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    indices = np.arange(start, start + n_terms, dtype=float)
    return 1.0 / indices


def p_series(p: float, n_terms: int, start: int = 1) -> np.ndarray:
    """Generate p-series: a_n = 1/n^p / p-serisi üretir: a_n = 1/n^p"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    indices = np.arange(start, start + n_terms, dtype=float)
    return 1.0 / (indices ** p)


def geometric_sequence(ratio: float, n_terms: int, start: int = 1) -> np.ndarray:
    """Generate geometric sequence: a_n = ratio^n / Geometrik dizi üretir: a_n = ratio^n"""
    if n_terms <= 0:
        raise ValueError("Number of terms must be positive / Terim sayısı pozitif olmalıdır")
    exponents = np.arange(start, start + n_terms, dtype=float)
    return ratio ** exponents


def analyze_sequence(
    sequence: Union[List[float], np.ndarray],
    name: str = "Sequence / Dizi",
    n_display: int = 5
) -> dict:
    """Detailed analysis of a sequence / Bir dizinin detaylı analizi"""
    seq = np.array(sequence, dtype=float)
    squares = seq ** 2
    cumsum = np.cumsum(squares)
    results = {
        'name': name,
        'first_terms': seq[:n_display].tolist(),
        'n_terms': len(seq),
        'sum_of_squares': cumsum[-1] if np.isfinite(cumsum[-1]) else np.inf,
        'in_hilbert': is_in_hilbert(seq),
        'max_term': float(np.max(np.abs(seq))),
        'decay_rate': None
    }
    if len(seq) > 100 and np.all(seq[100:] > 0):
        log_terms = np.log(seq[100:] + 1e-12)
        log_n = np.log(np.arange(100, len(seq)))
        try:
            alpha = -np.polyfit(log_n, log_terms, 1)[0]
            results['decay_rate'] = alpha
            results['decay_description'] = f"~ 1/n^{alpha:.2f}"
        except Exception:
            pass
    return results


def compare_sequences(sequences: dict, n_test: int = 5000) -> None:
    """Compare multiple sequences / Birden fazla diziyi karşılaştırır"""
    results = []
    for name, seq in sequences.items():
        if len(seq) < n_test:
            # Auto-extend / Otomatik uzat
            if "n/2" in name or "Oresme" in name:
                indices = np.arange(1, n_test + 1)
                seq = indices / (2.0 ** indices)
            elif "1/n" in name and "1/n²" not in name and "1/n³" not in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / indices
            elif "1/n²" in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / (indices ** 2)
            elif "1/√n" in name:
                indices = np.arange(1, n_test + 1)
                seq = 1.0 / np.sqrt(indices)
            elif "e⁻ⁿ" in name or "exp" in name:
                indices = np.arange(1, n_test + 1)
                seq = np.exp(-indices)
        test_seq = seq[:n_test]
        squares_sum = np.sum(test_seq ** 2)
        in_hilbert = is_in_hilbert(test_seq)
        results.append({
            "Sequence / Dizi": name,
            "First 5 terms / İlk 5 terim": str(test_seq[:5].tolist())[:60],
            "∑ a_n²": f"{squares_sum:.6f}" if np.isfinite(squares_sum) else "∞",
            "In ℓ²? / ℓ²'de mi?": "✓ Yes / Evet" if in_hilbert else "✗ No / Hayır"
        })
    try:
        from tabulate import tabulate
        print(tabulate(results, headers="keys", tablefmt="grid", stralign="left"))
    except ImportError:
        for row in results:
            print(row)


# -----------------------------
# Performance Analysis / Performans Analizi
# -----------------------------
def benchmark_harmonic(n: int, runs: int = 10) -> dict:
    """Compare different computation methods / Farklı hesaplama yöntemlerini karşılaştırır"""
    results = {}

    # Pure Python
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number(n)
    results['pure_python'] = (time.perf_counter() - start) / runs

    # NumPy vectorized
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_numbers_numpy(n)
    results['numpy'] = (time.perf_counter() - start) / runs

    # Approximation
    start = time.perf_counter()
    for _ in range(runs):
        _ = harmonic_number_approx(n)
    results['approximate'] = (time.perf_counter() - start) / runs

    return results


def compare_with_approximation(n: int) -> dict:
    """Compare exact and approximate values / Tam ve yaklaşık değerleri karşılaştırır"""
    exact = harmonic_number(n)
    approx = harmonic_number_approx(n)
    error = abs(exact - approx)
    relative_error = error / exact if exact != 0 else 0.0
    return {
        'exact': exact,
        'approximate': approx,
        'absolute_error': error,
        'relative_error': relative_error,
        'percentage_error': relative_error * 100
    }


# -----------------------------
# Convergence Analysis / Yakınsama Analizi
# -----------------------------
def harmonic_convergence_analysis(n_values: np.ndarray) -> dict:
    """Analyze harmonic series convergence / Harmonik seri yakınsamasını analiz eder"""
    exact = harmonic_numbers_numpy(n_values[-1])[n_values - 1]  # 0-based indexing
    approx = harmonic_sum_approx(n_values.astype(float), order=4)
    return {
        'exact_sums': exact,
        'approx_sums': approx,
        'errors': np.abs(exact - approx),
        'log_fit': np.polyfit(np.log(n_values), exact, 1)
    }


# -----------------------------
# Visualization / Görselleştirme
# -----------------------------
def plot_comparative_performance(max_n=50000, step=5000, runs=10):
    """Comparative performance plot / Karşılaştırmalı performans grafiği"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for plotting / Grafik için matplotlib gereklidir")
        return

    n_values = list(range(5000, max_n + 1, step))
    results = {'python': [], 'numpy': [], 'approx': []}

    for n in n_values:
        # Python
        py_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = harmonic_number(n)
            py_times.append(time.perf_counter() - t0)

        # NumPy
        np_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = harmonic_numbers_numpy(n)
            np_times.append(time.perf_counter() - t0)

        # Approx
        approx_times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = harmonic_number_approx(n)
            approx_times.append(time.perf_counter() - t0)

        results['python'].append(np.mean(py_times) * 1000)
        results['numpy'].append(np.mean(np_times) * 1000)
        results['approx'].append(np.mean(approx_times) * 1000)

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, results['python'], 'b-o', label='Pure Python')
    plt.plot(n_values, results['numpy'], 'r-s', label='NumPy')
    plt.plot(n_values, results['approx'], 'g-^', label='Approximate')
    plt.title('Performance Comparison (oresme.py)')
    plt.xlabel('n')
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main Program / Ana Program
# -----------------------------
def main():
    """Main test routine / Ana test rutini"""
    logger.info("=" * 70)
    logger.info(" ORESME MODULE TEST (PURE PYTHON/NUMPY) / ORESME MODÜL TESTİ (SAF PYTHON/NUMPY)")
    logger.info("=" * 70)

    logger.info("Oresme sequence (first 5) / Oresme dizisi (ilk 5): %s", oresme_sequence(5))
    logger.info("Fractional harmonic numbers H1-H3 / Kesirli harmonik sayılar H1-H3: %s", harmonic_numbers(3))
    logger.info("5th harmonic number / 5. harmonik sayı: %.4f", harmonic_number(5))
    logger.info("NumPy harmonic numbers H1-H5 / NumPy harmonik sayılar H1-H5: %s", harmonic_numbers_numpy(5))

    logger.info("-" * 50)
    logger.info("ℓ² (Hilbert space) membership tests / ℓ² (Hilbert uzayı) aidiyet testleri:")
    n_test = 10000
    harmonic_seq = 1 / np.arange(1, n_test + 1)
    logger.info("  1/n in ℓ²? / 1/n ℓ²'de mi? %s", is_in_hilbert(harmonic_seq))
    slow_seq = 1 / np.sqrt(np.arange(1, n_test + 1))
    logger.info("  1/√n in ℓ²? / 1/√n ℓ²'de mi? %s", is_in_hilbert(slow_seq))
    oresme_seq = np.array([i / (2 ** i) for i in range(1, n_test + 1)])
    logger.info("  n/2ⁿ in ℓ²? / n/2ⁿ ℓ²'de mi? %s", is_in_hilbert(oresme_seq))

    n_perf = 100000
    logger.info("-" * 50)
    logger.info("Performance test (n=%d) / Performans testi (n=%d):", n_perf, n_perf)
    bench = benchmark_harmonic(n_perf)
    for method, t in bench.items():
        logger.info("%15s: %.6f s/run", method, t)

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
