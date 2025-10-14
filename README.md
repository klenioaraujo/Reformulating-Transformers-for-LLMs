# Reformulating Transformers for LLMs: The ΨQRH Project

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17171112.svg)](https://doi.org/10.5281/zenodo.17171112)

---

## ❤️ Support Independent Research

This project is the result of independent research by **Klenio Araujo Padilha**. The development of ambitious projects like ΨQRH is driven by a passion for science and the pursuit of new frontiers of knowledge.

By donating, you are not just supporting this project, but also enabling the continuation of free and open research that seeks to make significant contributions to the field of Artificial Intelligence. Your help is crucial.

[![Donate with PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate/?hosted_button_id=YOUR_DONATE_BUTTON_ID_HERE)

Or use the direct link:
[**https://paypal.me/kleniopadilha**](https://paypal.me/kleniopadilha?locale.x=pt_PT&country.x=PT)

---

**ΨQRH (Quantum Quaternion Reaction-diffusion Hypernet)** is a fundamental reimagining of the Transformer architecture. More than just a model, it is a research journey to build a language system that not only learns from data but operates on a foundation of rigorous physical and mathematical principles, aiming for a more authentic and interpretable form of "reasoning."

## 🌀 Project Status: Active and in Constant Evolution

This is a living research project. It never stops and will always be undergoing new improvements. The philosophy of ΨQRH is one of continuous exploration, refining and expanding the architecture as new insights are discovered.

Interaction with the system can be done in two ways:
1.  **Main API**: The primary and most robust way to interact with the system is through its **API**. It offers scalable and optimized access to all of ΨQRH's functionalities.
2.  **Demonstration Pipeline**: The `psiqrh_pipeline.py` file remains an excellent starting point for understanding the system's core concepts in isolation and for running quick tests.

## 🎯 The Project's Goal

The objective of ΨQRH is to build a new type of Transformer that is:

*   **Mathematically Faithful**: Operations are based on robust mathematical structures like **Quaternions**, enabling richer geometric and rotational representations in a **Hilbert Space**.
*   **Physically Rigorous**: The model incorporates principles from quantum physics and dynamic systems theory. Concepts like the "Padilha Equation," wave-particle duality, and fractal fields are used to define the semantic space.
*   **Model-Agnostic**: The architecture is designed to use external LLMs as a semantic foundation. It can ingest models like **`gpt2`** or **`deepseek-ai/deepseek-coder-6.7b-instruct`**, convert them into its native "semantic format," and use them as a knowledge base for its physical simulations.
*   **Authentic in its Reasoning**: The heart of the system is the **Fractal Consciousness Dynamics (DCF)**, a semantic consensus layer that uses **Kuramoto oscillators**. This allows the answer to emerge from a "debate" among candidate tokens, a process analogous to a thought process.

## 🔬 A New Approach to LLMs

This project represents a different approach to studying and working with language models, founded on four theoretical pillars:

1.  **Quantum Representation & Hilbert Space**: Word and token vectors are treated as quantum states in a high-dimensional **Hilbert Space**, allowing the use of concepts like superposition, entanglement, and unitary evolution to represent semantics.
2.  **Fractal Dynamics**: The complexity and self-similarity of language are modeled through fractal geometry. Parameters like the fractal dimension are used to characterize and generate the semantic space.
3.  **Spectral Analysis**: The system analyzes signals in the frequency domain. **Spectral Attention** and spectral filtering allow the model to focus on deep rhythmic and structural patterns in the data.
4.  **Quaternion Algebra**: To overcome the limitations of complex numbers, the system uses Quaternions to represent quantum states. This enables true 4D rotations (SO(4)), preserving norms and representing complex spatial relationships between concepts more faithfully.

## 🛠️ Installation and Execution

### Using Docker (Recommended)
To ensure environment consistency and avoid dependency conflicts, using Docker is strongly recommended.

```bash
docker build -t psiqrh .
docker run -it psiqrh /bin/bash
```

### Using `make`
The project includes a `Makefile` that automates the most common tasks. Run `make help` to see a full list of available commands.

#### Model Management
```bash
# Download and cache a model from Hugging Face
make download-model SOURCE_MODEL=gpt2

# List all locally downloaded models
make list-downloaded-models
```

#### Semantic Model Management
The "semantic" format is ΨQRH's proprietary format, optimized for use within the system.

```bash
# Convert a downloaded model to the semantic format
make convert-to-semantic SOURCE_MODEL=gpt2
# Or for other models:
make convert-to-semantic SOURCE_MODEL=deepseek-ai/deepseek-coder-6.7b-instruct
```

```bash
# List available semantic models
make list-semantic-models
```
*Example output:*
```
🧠 Models in semantic format:
   📁 psiqrh_semantic_deepseek-ai_deepseek-coder-6.7b-instruct (4020.99 MB)
   📁 psiqrh_semantic_gpt2 (2187.80 MB)
```

```bash
# Full workflow: download and convert
make semantic-workflow SOURCE_MODEL=gpt2
```

## 🌌 The ΨQRH Philosophy: A Framework for Rigorous Exploration

This repository is more than a model; it is an **experimental research framework**. It is designed for developers and researchers who are free to explore, modify, and test new ideas. The architecture is built on a foundation of advanced, verifiable features, governed by a **ZERO FALLBACK POLICY**—if a physical principle cannot be correctly applied, the system fails, ensuring that no non-physical shortcuts are taken.

Key architectural features include:

*   **Model-Agnostic Core**: The system is designed to be a reasoning layer on top of existing knowledge. It can load and convert various external models, such as **`gpt2`** and **`deepseek-ai/deepseek-coder-6.7b-instruct`**, into its native "semantic format," using their learned knowledge as the foundation for its own physical simulations.

*   **Dynamic Auto-Calibration**: The system is not static. It uses a `CompleteAutoCalibrationSystem` which includes components like a `QuantumTemperatureCalculator` and `OpticalCoherenceCalculator`. Before processing, it dynamically adjusts its own internal physical parameters (`alpha`, `beta`, fractal dimension) to ensure the simulation is always in a valid and optimal state for the given input.

*   **Harmonic Orchestration and Energy Control**: A `PhysicalHarmonicOrchestrator` manages the pipeline's transformations. It applies physical corrections and ensures that fundamental properties like **energy conservation** are strictly maintained, in line with the project's mandatory mathematical validation requirements.

*   **Semantic Harmonization**: At its core, the `DCF` engine uses Kuramoto dynamics to achieve consensus. This is not a simple selection but a process of **harmonization**, where candidate concepts (oscillators) influence each other based on their semantic proximity and synchronize to form a coherent response.

*   **A Learnable, End-to-End Architecture**: The main generation loop is a three-stage process (`Context Funnel` → `Cognitive Processor (DCF)` → `Inverse Cognitive Projector`) designed for end-to-end learning. The system learns not only to reason but also how to perceive context and translate its abstract "thoughts" back into language.

*   **Comprehensive Validation Suite**: The project is supported by a robust set of tests (`make test-physics`, `make test-echo`). This validation suite is crucial for ensuring that the complex physical simulations are implemented correctly and behave as expected.

We invite researchers and developers to dive into this different approach, contribute new ideas, and help explore the future of physically-grounded artificial intelligence.

## 📜 License and Authorship

This project is the result of independent research and contains numerous original concepts, algorithms, and implementations by **Klenio Araujo Padilha**. The goal is to advance the frontier of AI knowledge openly and transparently.

All source code and associated concepts are made available under the **GNU General Public License v3.0 (GPLv3)**.

## 🔗 Contact and Social Networks

To follow the project's development, discuss ideas, or for professional inquiries, connect with the author:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Klenio_Padilha-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/kleniopadilha/)

## 📚 How to Cite

If you use this work in your research, please cite it using the DOI provided by Zenodo:

> Klenio Araujo Padilha. (2025). *Reformulating Transformers for LLMs: The ΨQRH Project (Version 1.0.0)* [Software]. Zenodo. https://doi.org/10.5281/zenodo.17171112