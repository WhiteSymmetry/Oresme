# __init__.py
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

from __future__ import annotations  # Gelecekteki özellikler için (Python 3.7+)
import importlib
import os
import warnings

# Paket sürüm numarası
__version__ = "0.1.4"

# if os.getenv("DEVELOPMENT") == "true":
    # importlib.reload(oresme)

# Göreli modül içe aktarmaları
# F401 hatasını önlemek için sadece kullanacağınız şeyleri dışa aktarın
# Aksi halde linter'lar "imported but unused" uyarısı verir
try:
    #from .oresme import *  # gerekirse burada belirli fonksiyonları seçmeli yapmak daha güvenlidir
    #from . import oresme  # Modülün kendisine doğrudan erişim isteniyorsa
    from .oresme import oresme_sequence, harmonic_numbers, harmonic_number, harmonic_number_approx, harmonic_generator, harmonic_numbers_numpy, is_in_hilbert
except ImportError as e:
    warnings.warn(f"Gerekli modül yüklenemedi: {e}", ImportWarning)

# Eski bir fonksiyonun yer tutucusu - gelecekte kaldırılacak
def eski_fonksiyon():
    """
    Kaldırılması planlanan eski bir fonksiyondur.
    Lütfen alternatif fonksiyonları kullanın.
    """
    warnings.warn(
        "eski_fonksiyon() artık kullanılmamaktadır ve gelecekte kaldırılacaktır. "
        "Lütfen yeni alternatif fonksiyonları kullanın. "
        category=DeprecationWarning,
        stacklevel=2
    )

__all__ = ["oresme_sequence", "harmonic_numbers", "harmonic_number", "harmonic_number_approx", "harmonic_generator", "harmonic_numbers_numpy", "is_in_hilbert"]

# Geliştirme sırasında test etmek için
if __name__ == "__main__":
    eski_fonksiyon()
