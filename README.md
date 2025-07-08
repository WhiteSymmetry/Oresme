# Oresme

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15833238.svg)](https://doi.org/10.5281/zenodo.15833238)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI-10.48546/workflowhub.datafile.18.1-blue)](https://doi.org/10.48546/workflowhub.datafile.18.1)

[![figshare DOI](https://img.shields.io/badge/DOI-10.6084/m9.figshare.29504708-blue)](https://doi.org/10.6084/m9.figshare.29504708)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/Oresme/badges/version.svg)](https://anaconda.org/bilgi/Oresme)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/Oresme/badges/latest_release_date.svg)](https://anaconda.org/bilgi/Oresme)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/Oresme/badges/platforms.svg)](https://anaconda.org/bilgi/Oresme)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/Oresme/badges/license.svg)](https://anaconda.org/bilgi/Oresme)
[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Oresme numbers refer to the sums related to the harmonic series.

---
### **Türkçe Tanım:**
**Oresme Sayıları**, 14. yüzyılda Nicole Oresme tarafından incelenen matematiksel serilerdir. Oresme sayıları harmonik seriye ait toplamları ifade eder. İki türü vardır:  
1. **\( \frac{n}{2^n} \) serisi** (Oresme'nin orijinal çalışması),  
2. **Harmonik sayılar** (\( H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n} \)).  
Bu sayılar, analiz ve sayı teorisinde önemli rol oynar.

---

### **English Definition:**
**Oresme Numbers** are mathematical series studied by Nicole Oresme in the 14th century. Oresme numbers refer to the sums related to the harmonic series. They include two types:  
1. The **\( \frac{n}{2^n} \) sequence** (Oresme's original work),  
2. **Harmonic numbers** (\( H_n = 1 + \frac{1}{2} + \cdots + \frac{1}{n} \)).  
These numbers play a key role in analysis and number theory.

---

### **Fark/Karşılaştırma (Difference):**
- **Oresme'nin \( \frac{n}{2^n} \) serisi** ıraksaklık kanıtları için önemlidir.  
- **Harmonik sayılar** (\( H_n \)) ise logaritmik büyüme gösterir ve \( n \to \infty \) iken ıraksar.  
- Modern literatürde "Oresme numbers" terimi daha çok tarihsel bağlamda kullanılır.

---

## Kurulum (Türkçe) / Installation (English)

### Python ile Kurulum / Install with pip, conda, mamba
```bash
pip install Oresme -U
python -m pip install -U Oresme
conda install bilgi::Oresme -y
mamba install bilgi::Oresme -y
```

```diff
- pip uninstall Oresme -y
+ pip install -U Oresme
+ python -m pip install -U Oresme
```

[PyPI](https://pypi.org/project/Oresme/)

### Test Kurulumu / Test Installation

```bash
pip install -i https://test.pypi.org/simple/ Oresme -U
```

### Github Master Kurulumu / GitHub Master Installation

**Terminal:**

```bash
pip install git+https://github.com/WhiteSymmetry/Oresme.git
```

**Jupyter Lab, Notebook, Visual Studio Code:**

```python
!pip install git+https://github.com/WhiteSymmetry/Oresme.git
# or
%pip install git+https://github.com/WhiteSymmetry/Oresme.git
```

---

## Kullanım (Türkçe) / Usage (English)

```python
import oresme as ore 

# Example 1: Generate Oresme sequence
print(ore.oresme_sequence(5))  # [0.5, 0.5, 0.375, 0.25, 0.15625]

# Example 2: Get exact harmonic numbers as fractions
print(ore.harmonic_numbers(3))  # [Fraction(1, 1), Fraction(3, 2), Fraction(11, 6)]

# Example 3: Calculate single harmonic number
print(ore.harmonic_number(5))  # 2.283333333333333

# Example 4: Approximate large harmonic number
print(ore.harmonic_number_approx(1_000_000))  # ≈14.392726722865724

# Example 5: Use generator
for i, h in enumerate(ore.harmonic_generator(3), 1):
    print(f"H_{i} = {h}")

# Example 6: NumPy vectorized version
print(ore.harmonic_numbers_numpy(5))  # [1. 1.5 1.833... 2.083... 2.283...]

[0.5, 0.5, 0.375, 0.25, 0.15625]
[Fraction(1, 1), Fraction(3, 2), Fraction(11, 6)]
2.283333333333333
14.392726722865808
H_1 = 1.0
H_2 = 1.5
H_3 = 1.8333333333333333
[1.         1.5        1.83333333 2.08333333 2.28333333]

```

```python
import oresme
oresme.__version__
```
---

### Development
```bash
# Clone the repository
git clone https://github.com/WhiteSymmetry/Oresme.git
cd Oresme

# Install in development mode
python -m pip install -ve . # Install package in development mode

# Run tests
pytest

Notebook, Jupyterlab, Colab, Visual Studio Code
!python -m pip install git+https://github.com/WhiteSymmetry/Oresme.git
```
---

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX


### APA

```
Keçeci, M. (2025). Dynamic vs Static Number Sequences: The Case of Keçeci and Oresme Numbers. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15833351

Keçeci, M. (2025). Oresme. figshare. https://doi.org/10.6084/m9.figshare.29504708

Keçeci, M. (2025). Oresme [Data set]. WorkflowHub. https://doi.org/10.48546/workflowhub.datafile.18.1 

Keçeci, M. (2025). Oresme (0.1.0). Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15833238
```

### Chicago

```
Keçeci, Mehmet. Dynamic vs Static Number Sequences: The Case of Keçeci and Oresme Numbers. Open Science Articles (OSAs), Zenodo, 2025. https://doi.org/10.5281/zenodo.15833351

Keçeci, Mehmet. Oresme. figshare, 2025. https://doi.org/10.6084/m9.figshare.29504708

Keçeci, Mehmet. Oresme [Data set]. WorkflowHub, 2025. https://doi.org/10.48546/workflowhub.datafile.18.1 

Keçeci, Mehmet. Oresme. Open Science Articles (OSAs), Zenodo, 2025. https://doi.org/10.5281/zenodo.15833238

```


### Lisans (Türkçe) / License (English)

```
This project is licensed under the MIT License.
```
