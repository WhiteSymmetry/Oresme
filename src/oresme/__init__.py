# __init__.py
# Bu dosya paketin başlangıç noktası olarak çalışır.
# Alt modülleri yükler, sürüm bilgileri tanımlar ve geriye dönük uyumluluk için uyarılar sağlar.

from __future__ import annotations  # Gelecekteki özellikler için (Python 3.7+)

import importlib
import warnings
import os
if os.getenv("DEVELOPMENT") == "true":
    importlib.reload(oresme)


# Göreli modül içe aktarmaları
# F401 hatasını önlemek için sadece kullanacağınız şeyleri dışa aktarın
# Aksi halde linter'lar "imported but unused" uyarısı verir
try:
    from .oresme import *  # gerekirse burada belirli fonksiyonları seçmeli yapmak daha güvenlidir
    from . import oresme  # Modülün kendisine doğrudan erişim isteniyorsa
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
        "Oresme Python 3.9-3.14 sürümlerinde desteklenmektedir.",
        category=DeprecationWarning,
        stacklevel=2
    )


# Paket sürüm numarası
__version__ = "0.1.0"


# Geliştirme sırasında test etmek için
if __name__ == "__main__":
    eski_fonksiyon()
