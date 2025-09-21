
import torch
import random
import numpy as np
from models.insect_specimens.araneae import Araneae_PsiQRH

def simulate_emergence():
    """
    A simulation demonstrating emergent reproductive behavior in a dynamic
    population of spider agents, driven by ΨQRH wave communication.
    """
    print("-" * 80)
    print("ΨQRH SPIDER POPULATION SIMULATION (REPRODUCTION, WAVE COMMUNICATION & CHAOS)")
    print("-" * 80)

    # 1. Environment Setup
    environment = {
        "locations": {
            "sheltered_branch": {"prey_traffic": 0.7, "wind_exposure": 0.2, "anchor_points": 3},
            "exposed_leaf": {"prey_traffic": 0.9, "wind_exposure": 0.8, "anchor_points": 1},
        },
        "chaos_factor": 0.1 # Initial environmental chaos
    }

    # 2. Initial Population
    spider_population = [Araneae_PsiQRH(maturity_age=3) for _ in range(3)]
    # Ensure at least one male and one female for reproduction demonstration
    guaranteed_female = Araneae_PsiQRH(maturity_age=3)
    guaranteed_female.gender = 'female'
    spider_population.append(guaranteed_female)
    random.shuffle(spider_population)

    print(f"Initial population: {len(spider_population)} spiders.")
    for spider in spider_population: 
        print(f"- Spider {id(spider)} ({spider.gender}) created with signature α={spider.signature[0]}, β={spider.signature[1]}.")

    # 3. Main Simulation Loop
    num_steps = 15
    for i in range(num_steps):
        print(f"\n{'='*30} TIME STEP {i + 1} {'='*30}")
        print(f"Population: {len(spider_population)} | Environmental Chaos: {environment['chaos_factor']:.2f}")

        # Fluctuate chaos
        environment["chaos_factor"] = max(0, 0.1 + np.sin(i / 3) * 0.2)

        emitted_waves = []
        reproduction_events = []
        newly_born = []

        # --- Action & Communication Phase ---
        # First, determine actions without immediate interaction
        for spider in spider_population:
            # Spiders perceive an environment without waves first
            environmental_data = {"locations": environment["locations"], "waves": []}
            action = spider.forward(environmental_data)

            if action["type"] == "EMIT_MATING_WAVE":
                print(f"Event: Male {id(spider)} emits a mating wave.")
                emitted_waves.append({
                    "emitter_id": id(spider),
                    "wave_form_base": action["wave"],
                    "signature": spider.signature
                })

        # --- Interaction & Analysis Phase ---
        if emitted_waves:
            propagated_waves = []
            for wave_packet in emitted_waves:
                distorted_form = wave_packet["wave_form_base"].propagate(environment["chaos_factor"])
                propagated_waves.append({
                    "emitter_id": wave_packet["emitter_id"],
                    "wave_form": distorted_form,
                    "signature": wave_packet["signature"]
                })

            for spider in spider_population:
                if spider.gender == 'female' and spider.mating_readiness > spider.reproduction_threshold:
                    # This female is now listening to the propagated waves
                    action = spider.forward({"locations": environment["locations"], "waves": propagated_waves})
                    if action["type"] == "REPRODUCE":
                        reproduction_events.append((spider, action["partner_id"]))

        # --- Resolution Phase ---
        if reproduction_events:
            for female, male_id in reproduction_events:
                print(f"*** SUCCESS! Female {id(female)} correlated with Male {male_id}. Reproduction occurs! ***")
                num_offspring = random.randint(1, 3)
                for _ in range(num_offspring):
                    newly_born.append(Araneae_PsiQRH(maturity_age=female.age + 5))
            
            if newly_born:
                print(f"---> {len(newly_born)} spiderling(s) were born! <---")
                spider_population.extend(newly_born)

    print("\n" + "-" * 80)
    print("SIMULATION COMPLETE")
    print(f"Final population: {len(spider_population)} spiders.")
    print("-" * 80)

if __name__ == "__main__":
    simulate_emergence()
