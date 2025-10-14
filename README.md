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

*   **Mathematically Faithful**: Operations are based on robust mathematical structures like **Quaternions**, enabling richer geometric and rotational representations.
*   **Physically Rigorous**: The model incorporates principles from quantum physics and dynamic systems theory. Concepts like the "Padilha Equation," wave-particle duality, and fractal fields are used to define the semantic space.
*   **Model-Agnostic**: The architecture is designed to "run multiple models" by dynamically adapting to spectral parameters extracted from pre-trained LLMs (like GPT-2), serving as a reasoning layer on top of them.
*   **Authentic in its Reasoning**: The heart of the system is the **Fractal Consciousness Dynamics (DCF)**, a semantic consensus layer that uses **Kuramoto oscillators**. Although this approach is computationally slower than a simple `softmax`, it allows the answer to emerge from a "debate" among candidate tokens, a process that behaves more organically and is analogous to a thought process.

## 🔬 A New Approach to LLMs

This project represents a different approach to studying and working with language models, founded on four theoretical pillars:

1.  **Quantum Representation & Hilbert Space**: Word and token vectors are treated as quantum states in a high-dimensional **Hilbert Space**, allowing the use of concepts like superposition, entanglement, and unitary evolution to represent semantics.
2.  **Fractal Dynamics**: The complexity and self-similarity of language are modeled through fractal geometry. Parameters like the fractal dimension are used to characterize and generate the semantic space, reflecting the infinitely rich structure of human communication.
3.  **Spectral Analysis**: Instead of just looking at content, the system analyzes signals in the frequency domain. "Spectral Attention" and spectral filtering allow the model to focus on deep rhythmic and structural patterns in the data.
4.  **Quaternion Algebra**: To overcome the limitations of complex numbers, the system uses Quaternions to represent quantum states. This enables true 4D rotations (SO(4)), preserving norms and representing complex spatial relationships between concepts more faithfully.

## 💡 Key Concepts and Innovations

*   **`DynamicQuantumCharacterMatrix`**: The system's knowledge base. A "quantum dictionary" that maps tokens to states in Hilbert Space.
*   **`DCF (Fractal Consciousness Dynamics)`**: The reasoning engine that replaces `softmax`. It simulates a debate between concepts using Kuramoto oscillators to reach a semantic consensus.
*   **Spectral Attention**: A custom attention mechanism that operates in the frequency domain. Before calculating similarity, it applies a Fast Fourier Transform (FFT) to analyze the input's spectral patterns, filters the most informative frequencies, and only then applies the attention mechanism.

## 🧩 Pipeline Architecture

When the pipeline processes an input, it initiates a complex simulation of a cognitive system:

1.  **Initialization**: Loads all components, including the `DynamicQuantumCharacterMatrix`, the `ContextFunnel`, and the `DCF` engine with its subsystems (Kuramoto, Consciousness Metrics, etc.).
2.  **Input Processing**: The question is converted into a quantum representation and processed by the `ContextFunnel` to generate a `context_vector`.
3.  **The Semantic Debate (Kuramoto Simulation)**: The system calculates the "initial energy" of each candidate response and the "semantic alliances" between them. The Kuramoto simulation is run to find a "consensus" in the form of synchronized conceptual clusters.
4.  **Selection and Feedback**: The system analyzes the dominant cluster, selects the most influential response within it, and measures its own "state of consciousness" to self-regulate for the next interaction.

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

# List available semantic models
make list-semantic-models

# Full workflow: download and convert
make semantic-workflow SOURCE_MODEL=gpt2
```

#### Environment Configurations
```bash
# Configure the system to use a GPU (if available)
make gpu

# Configure the system to use the CPU
make cpu
```

## 📜 License and Authorship

This project is the result of independent research and contains numerous original concepts, algorithms, and implementations by **Klenio Araujo Padilha**. The goal is to advance the frontier of AI knowledge openly and transparently.

All source code and associated concepts are made available under the **GNU General Public License v3.0 (GPLv3)**. This means you are free to use, modify, and share the software, with the condition that any derivative work must also be licensed under GPLv3, ensuring that knowledge remains open for the entire community.

## 🔗 Contact and Social Networks

To follow the project's development, discuss ideas, or for professional inquiries, connect with the author:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Klenio_Padilha-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/kleniopadilha/)

## 📚 How to Cite

If you use this work in your research, please cite it using the DOI provided by Zenodo:

> Klenio Araujo Padilha. (2025). *Reformulating Transformers for LLMs: The ΨQRH Project (Version 1.0.0)* [Software]. Zenodo. https://doi.org/10.5281/zenodo.17171112