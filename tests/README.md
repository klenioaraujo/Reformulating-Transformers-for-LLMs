# README: Testing Strategy for Reformulating Transformers

This directory contains the tests for the **Reformulating Transformers** project. Our testing approach is tailored to the unique architecture of the custom **ΨQRH (PsiQRH) framework**, which differs significantly from standard deep learning models.

## Core Concept: Testing the ΨQRH Framework

The primary goal of our tests is to validate the mathematical correctness, stability, and analytical capabilities of the ΨQRH framework. Unlike traditional NLP models that learn from data to generate text, our framework processes input as a mathematical signal.

### A Meta-Analytical Approach

The ΨQRH model operates with a logic fundamentally different from standard Transformers. It performs a genuine mathematical analysis of the input concept – not by retrieving pre-existing facts, but by interpreting the semantic embedding through its own unique architectural lens. The output is a report on the measurable characteristics of this internal representation.

This makes ΨQRH a **meta-analytical framework**. It doesn’t just answer "what is a quaternion?"; it reveals how the model structurally encodes and interprets the concept, using its own internal geometry.

A simple and clear example of this process can be seen in `human_testing/test_simple_chat.py`.

### The ΨQRH Processing Pipeline

The framework follows a distinct pipeline, which is the main focus of our validation tests:

**1. Input: Text → Numerical Tensor**
   - Raw text is not processed by a tokenizer. Instead, it is converted into a numerical tensor through a direct character-to-number mapping. Each character is converted to its ASCII value (`ord(char)`).
   - **Example:** `'Hello'` becomes a tensor like `[72, 101, 108, 108, 111, ...]`.

**2. Processing: Mathematical Analysis (Quaternions + FFT)**
   - The core of the ΨQRH framework treats the input tensor as a signal. It applies a series of mathematical transformations:
     - **Quaternion Representation:** The signal is lifted into a 4D hypercomplex space using quaternions.
     - **Spectral Analysis:** The framework uses Fast Fourier Transform (FFT) to move the signal to the frequency domain, where spectral filtering and analysis are performed.
   - This stage does not "understand" language; it analyzes the structural and mathematical properties of the input signal.

**3. Output: Structured Response**
   - The framework does not generate a free-form text response. Instead, it populates a predefined template with the results of its mathematical analysis.
   - The output is a structured, wiki-style report detailing metrics like spectral complexity, dynamic range, and other characteristics derived from the quaternion and spectral processing.

### Example: The Graduated Complexity Test

The `human_testing/test_simple_chat.py` script now implements a **Graduated Complexity Test**. It runs 10 prompts of increasing conceptual difficulty to demonstrate how the framework's mathematical analysis adapts. This test is crucial to understanding the meta-analytical nature of ΨQRH.

By observing the output, we can see how the model's internal representation changes based on the complexity of the input concept. The script now runs automatically when executed:

```bash
python3 tests/human_testing/test_simple_chat.py
```

#### Interpreting the Results

Notice how the analytical metrics change between a simple and a complex prompt. This is not a measure of "correctness" but a measure of the structural complexity of the concept as represented within the model's internal geometry.

- **Example 1: Simple Concept (Question 1)**
  - **Input:** `'What is a prime number?'`
  - **Abridged Output:**
    ```
    == Mathematics Concept: Framework Analysis ==
    '''ΨQRH Framework Analysis''' reveals that what is a prime number? exhibits simple spectral characteristics with complexity level 1/3.
    === Mathematical Structure ===
    * '''Spectral Complexity''': 0.213 (normalized variance)
    * '''Dynamic Range''': 2.850
    ```

- **Example 2: Complex Concept (Question 9)**
  - **Input:** `'Discuss the relationship between entropy in thermodynamics and information theory.'`
  - **Abridged Output:**
    ```
    == Scientific_Question: Framework Analysis ==
    '''ΨQRH Framework Analysis''' reveals that ...relationship between entropy... exhibits chaotic spectral characteristics with complexity level 3/3.
    === Mathematical Structure ===
    * '''Spectral Complexity''': 0.891 (normalized variance)
    * '''Dynamic Range''': 6.720
    ```

As shown above, the **Spectral Complexity** and **Dynamic Range** values are significantly higher for the more complex concept. This demonstrates the framework's ability to provide a quantitative measure of a concept's complexity as it is processed and encoded internally.

## Overview of Other Test Categories

In addition to the core concept validation, this directory includes other types of tests:

- **Human-Readable Tests (`human_testing/`):** Scripts like `test_advanced_chat.py` provide more comprehensive demonstrations with various configurations, generating detailed reports for human analysis.
- **Integration Tests (`comprehensive_integration_test.py`):** These tests ensure that different modules of the ΨQRH framework (e.g., `QRHLayer`, `SpectralFilter`, `FractalTransformer`) work together correctly.
- **Unit Tests (`test_qrh_core.py`, `test_spectral_dropout.py`):** Focused tests that validate the functionality of individual components and mathematical operations.
- **Stress & Performance Tests (`test_framework_stress.py`):** These scripts push the framework to its limits to identify performance bottlenecks, test error handling, and ensure numerical stability under heavy load.

This multi-faceted approach ensures the robustness and correctness of our novel framework from the component level to the full system integration.
