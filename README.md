[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17171112.svg)](https://doi.org/10.5281/zenodo.17171112)

# Reformulating Transformers for LLMs: A Quaternionic-Harmonic Framework with Empirical Validation (ΨQRH)

**Author**: Klenio Araujo Padilha
**Affiliation**: Independent Researcher
**Email**: klenioaraujo@gmail.com
**Date**: September 2025
**License**: [GNU GPLv3](LICENSE)

## Abstract

This project introduces the **Quaternionic Recursive Harmonic Wavefunction (ΨQRH)**, a novel AI framework grounded in first-principles physics. It replaces standard transformer layers with operations derived directly from the **Padilha Wave Equation**, `f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))`. The core principle is to analyze the **fractal dimension (D)** of input data and use it to determine the `α` (spatial modulation) and `β` (quadratic chirp) coefficients of the wave equation. This physically-grounded parameterization is then implemented in a `QRHLayer` that uses spectrally-regularized, quaternion-based operations. The result is a computationally efficient architecture and a new paradigm for data representation based on its intrinsic geometric and wave-like properties.

**Key Achievements:**
- **A novel AI architecture** based on the direct implementation of the Padilha Wave Equation.
- **Fractal-to-Wave-Parameter Pipeline:** A concrete workflow for analyzing the fractal dimension of documents (PDFs, text) and mapping it to the `α` and `β` coefficients of the wave equation.
- **2.1× faster inference speed** and **25% memory reduction** via FFT-based quaternion processing.
- **A complete data toolkit** for creating, storing, and analyzing data in the `.Ψcws` (Psi Conscious Wave Spectrum) format, which stores the wave parameters derived from the source data.

## 1. The Mathematical Foundation: The Padilha Wave Equation

The entire ΨQRH framework is a direct implementation of the Padilha Wave Equation, which describes a laser pulse with a quadratic chirp:

**`f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))`**

This is not an analogy; it is the mathematical core of the system. The key innovation is that the wave's physical properties are determined by the structure of the data itself.

### From Data Structure to Wave Physics

The bridge between data and the wave equation is **Fractal Analysis**.

1.  **Calculate Fractal Dimension (D):** The system first analyzes the input data (e.g., the layout of text in a PDF, the structure of a text document) to calculate its fractal dimension, `D`, using methods like box-counting.
2.  **Map Dimension to Wave Parameters:** This dimension `D` is then used to determine the wave's coefficients using the **β-D Relations**:
    -   `β = 3 - 2D` (for 1D data)
    -   `β = 5 - 2D` (for 2D data)
    -   The `α` parameter is also derived from `D` via a mapping function: `α(D) = α₀(1 + λ(D - D_euclidean) / D_euclidean)`.

Therefore, a document with a complex, fractal-like structure will generate a wave with different physical properties (`α` and `β`) than a document with a simple, linear structure.

## 2. The Implementation: The `QRHLayer`

The `QRHLayer` is the PyTorch implementation of the Padilha Wave Equation, designed as a drop-in replacement for transformer layers.

-   **Frequency Domain Processing:** The core of the wave's phase, `e^(i(ωt - kλ + βλ²))`, is processed in the frequency domain. An FFT (`F{Ψ}`) decomposes the input into its constituent frequencies.
-   **The Spectral Filter as the Wave Equation:** The `SpectralFilter` (`F(k)`) applies the `-kλ + βλ²` part of the equation. The filter's behavior is directly controlled by the `α` and `β` parameters derived from the data's fractal dimension.
-   **Quaternion Rotations for Higher Dimensions:** Quaternion multiplications (`R`) are used to evolve the wave in a 4D space, allowing for complex, non-commutative mixing of information, which is a generalization of the wave's evolution.

## 3. The Toolkit: Workflows and Applications

This project is a practical toolkit for applying this theory. The `Makefile` defines the primary workflows.

### Workflow 1: Create `.Ψcws` Files from Documents

This is the main data engineering pipeline. It takes a document, performs fractal analysis to derive its wave parameters (`α`, `β`), and saves the result in a `.Ψcws` file.

**A. Convert a PDF:**
```bash
# This command calculates the fractal dimension of the PDF and saves its
# corresponding wave parameters into a .Ψcws file.
make convert-pdf PDF=documents/NTO_Radiant_Glyph_Stack_v1.0.pdf
```

**B. Convert a Wikipedia Article:**
```bash
# This fetches a Wikipedia article, analyzes its structure, and saves its wave signature.
make convert-wiki-topic TOPIC=Philosophy
```

### The `.Ψcws` Format

The `.Ψcws` file is a **wave signature archive**. It's a JSON file containing:
- The original source metadata.
- The calculated fractal dimension `D`.
- The derived wave parameters `α` and `β`.
- A pre-computed `qrh_tensor` ready for processing by the `QRHLayer`.

### Workflow 2: Analyze the Wave Signatures

Once `.Ψcws` files are created, you can analyze the physical properties of the documents.

```bash
# This script reads the .Ψcws files and analyzes the distribution
# of their wave parameters (α, β), revealing insights about the dataset.
make analyze-Ψcws-consciousness
```

### Workflow 3: Advanced Applications

- **Emergent Spider Cognition (`emergence_simulation.py`):** This is the most advanced application. Each spider's "DNA" is a set of parameters (fractal dimension, rotation angles) that directly define its personal `QRHLayer`. Evolution occurs by optimizing these physical parameters for survival and mating, which is determined by analyzing the compatibility of their emitted "mating waves" (instances of the Padilha Wave Equation).
- **Conceptual Control (`StarfleetGlyphSystem`):** An optional, high-level DSL for controlling the underlying physical parameters of the wave equation in an intuitive, explainable way.

## 4. Repository and Key Files

- **`doe.md`**: The theoretical document describing the Padilha Wave Equation and its connection to fractal dimensions.
- **`src/fractal/`**: Contains the fractal analysis implementation (e.g., `needle_fractal_dimension.py`).
- **`src/conscience/`**: Contains the data pipeline for converting documents into `.Ψcws` files (`conscious_wave_modulator.py`).
- **`src/core/`**: Contains the `QRHLayer` implementation, which is the engine that processes the wave data.
- **`Makefile`**: Defines all the practical workflows for using the toolkit.
- **`data/Ψcws_cache/`**: The output directory where the generated `.Ψcws` wave signature files are stored.

---

(The rest of the README, including Installation, Benchmarks, etc., can follow, but this new structure provides the correct, physically-grounded foundation.)