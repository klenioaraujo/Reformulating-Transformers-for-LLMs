import random
import numpy as np
from models.insect_specimens.dna import AraneaeDNA
from models.insect_specimens.araneae import Araneae_PsiQRH

def run_emergent_simulation():
    """
    Orchestrates a full genetic algorithm simulation where agents evolve.
    Their behavior, survival, and reproduction emerge from their DNA, which
    defines their personal QRHLayer, fractal signature, and 4D rotation.
    """
    print("=" * 80)
    print("      ΨQRH AGENT-BASED EVOLUTIONARY SIMULATION (GENETIC ALGORITHM)      ")
    print("=" * 80)

    # --- 1. Initial Population (Genesis) ---
    population_size = 6
    population = [Araneae_PsiQRH(dna=AraneaeDNA()) for _ in range(population_size)]
    print(f"\n--- Initial Population (Generation 0) ---")
    for agent in population:
        print(f"  - Agent {id(agent)} created. Gender: {agent.gender}, DNA Alpha: {agent.config.alpha:.2f}")

    # --- 2. Simulation Loop (Evolution over Time) ---
    num_generations = 15
    for gen in range(num_generations):
        print(f"\n{'-'*30} Generation {gen + 1} {'-'*30}")
        print(f"Population: {len(population)}")

        environment = {
            "chaos_factor": max(0, 0.1 + np.sin(gen / 3) * 0.2)
        }
        emitted_waves = []
        reproduction_pairs = []

        # --- Agent Actions & Communication ---
        for agent in population:
            if agent.gender == 'male':
                action = agent.forward(environment)
                if action["type"] == "EMIT_MATING_WAVE":
                    print(f"Event: Male {id(agent)} (Health: {agent.health:.2f}) emits mating wave.")
                    emitted_waves.append({"emitter_id": id(agent), "wave": action["wave"]})

        # --- Female Analysis & Mating Selection ---
        if emitted_waves:
            environment["waves"] = emitted_waves # Add waves to the environment for females
            for agent in population:
                if agent.gender == 'female':
                    action = agent.forward(environment)
                    if action["type"] == "REPRODUCE":
                        # Find the partner agent from the ID
                        partner = next((p for p in population if id(p) == action["partner_id"]), None)
                        if partner:
                            print(f"Event: Female {id(agent)} accepts mate {id(partner)}.")
                            reproduction_pairs.append((agent, partner))

        # --- Resolution: Reproduction & New Generation ---
        newly_born = []
        if reproduction_pairs:
            for parent1, parent2 in reproduction_pairs:
                print(f"*** Reproduction Occurs! Offspring from {id(parent1)} and {id(parent2)} ***")
                child = Araneae_PsiQRH.reproduce(parent1, parent2)
                newly_born.append(child)

        if newly_born:
            print(f"---> {len(newly_born)} new agent(s) born! Population growing. <---")
            population.extend(newly_born)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print(f"Final population size: {len(population)}")
    # Print DNA of the final generation for analysis
    print("\n--- Final Generation DNA Samples ---")
    for i, agent in enumerate(random.sample(population, min(5, len(population)))):
        print(f"  Sample {i+1}: Alpha={agent.config.alpha:.3f}, Angles={np.round(agent.dna.rotation_angles, 2).tolist()}")
    print("=" * 80)

if __name__ == "__main__":
    run_emergent_simulation()