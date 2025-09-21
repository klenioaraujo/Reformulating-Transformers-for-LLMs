
# Emergent Behavior Simulation: A Genetic Algorithm with ΨQRH Agents

This document outlines the architecture of the agent-based simulation, where the principles of the ΨQRH framework are used to create a system of evolution and natural selection.

## Core Philosophy

The simulation adheres to a strict principle: **behavior is not programmed, it emerges.** There are no pre-trained models. Instead, each agent's processing core (its `QRHLayer`) is a direct expression of its digital DNA. The simulation itself is the training environment, and evolution through natural selection is the learning algorithm.

This fulfills the vision of a system where generation, genetics, behavioral principles, environment, and reproduction are all intertwined and emerge from a chaotic, interactive ecosystem.

## 1. The Agent's DNA: `AraneaeDNA`

The genetic code of each agent is defined in the `AraneaeDNA` class (`dna.py`). This is the core of the agent's being and is composed of two primary genes:

1.  **Fractal Genes (IFS Coefficients):** A list of coefficients for an Iterated Function System (IFS). This gene defines a unique fractal shape associated with the agent.
    *   **Emergent Property:** The **fractal dimension (D)** is calculated from this gene using the `FractalGenerator` (from `needle_fractal_dimension.py`). This dimension `D` is then mapped to the `alpha` parameter of the agent's `QRHLayer`, directly linking its geometric DNA to its wave processing physics.

2.  **4D Rotation Genes (Rotation Angles):** A list of 6 angles.
    *   **Emergent Property:** These angles define two unique quaternions, `q_left` and `q_right`. These are used to perform the full SO(4) unitary rotation (`Ψ' = q_left · Ψ · q_right`) as described in the `README_4D_Unitary_Layer.md`. This gives each agent a unique way of transforming information.

## 2. The Agnostic Model: `QRHLayer`

The `QRHLayer` remains unchanged and agnostic, as per the design principles. It knows nothing of DNA or spiders. It is a pure processing layer that accepts a `QRHConfig` object upon initialization.

Each agent uses its `AraneaeDNA` to generate a personal `QRHConfig` and then instantiates its own private `QRHLayer`. The agent's mind is, therefore, a direct product of its DNA.

## 3. The Agent: `Araneae_PsiQRH`

The spider agent encapsulates the entire system:

-   **Initialization:** It is born with a specific `AraneaeDNA`.
-   **Personal Physics:** It creates its own `QRHConfig` and `QRHLayer` from its DNA.
-   **Health & Fitness:** The agent has a `health` attribute determined by the numerical stability of its personal `QRHLayer` (inspired by the project's `tests`). Unstable DNA (e.g., parameters causing energy loss) leads to poor health, reducing the chance of reproduction.
-   **Emergent Readiness:** The desire to reproduce (`mating_readiness`) is not random. It emerges from a combination of factors: age, health, and time since last reproduction.

## 4. The Ecosystem: `emergence_simulation.py`

The simulation orchestrates the genetic algorithm:

1.  **Genesis:** An initial population is created with random `AraneaeDNA`.
2.  **Communication & Chaos:**
    *   Healthy, mature males emit a `PadilhaWave`, a complex signal whose properties are defined by their unique DNA.
    *   The environment introduces a `chaos_factor` that distorts these waves as they propagate.
3.  **Selection:**
    *   Healthy, mature females listen for these waves.
    *   A female uses her **personal `QRHLayer`** to analyze the distorted wave. She attempts to find a correlation between the received signal and her own internal expectations.
    *   A successful correlation (a high similarity score after ΨQRH processing) triggers a `REPRODUCE` action. This is natural selection in action: only agents whose DNA is effective at both sending and decoding signals under chaotic conditions will be selected for reproduction.
4.  **Reproduction & Evolution:**
    *   When reproduction is triggered, a new `AraneaeDNA` is created for the offspring via **crossover** (mixing the parents' DNA) and **mutation** (small random changes).
    *   A new agent is born with this new DNA.
    *   The population grows, and the cycle continues.

Over generations, the population evolves. The DNA that produces more stable and effective `QRHLayers` will lead to healthier agents that reproduce more successfully, propagating those beneficial genes throughout the population.
