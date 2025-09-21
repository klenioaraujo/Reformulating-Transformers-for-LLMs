import torch
import random
from models.insect_specimens import Chrysopidae_PsiQRH, Tettigoniidae_PsiQRH

def simulate_emergence():
    """
    A simulation to demonstrate the 'emergence' of behaviors from the ΨQRH insect specimens.
    """
    print("-" * 50)
    print("ΨQRH INSECT EMERGENCE SIMULATION")
    print("-" * 50)

    # 1. Instantiate the emergent specimens
    chrysopidae = Chrysopidae_PsiQRH(attack_threshold=0.8)
    tettigoniidae = Tettigoniidae_PsiQRH()

    print("Specimens emerged:")
    print(f"- {chrysopidae.__class__.__name__} (Heuristic: {chrysopidae.heuristic})")
    print(f"- {tettigoniidae.__class__.__name__} (Heuristic: {tettigoniidae.heuristic})")
    print("\n" + "-" * 50)

    # 2. Simulation loop for a few time steps
    num_steps = 3
    for i in range(num_steps):
        print(f"\nTIME STEP {i + 1}")

        # --- Chrysopidae Simulation (Tensor-based) ---
        print("\n--- Chrysopidae's Environment ---")
        # Simulate sensory input with random tensors
        vision_input = torch.randn(10)
        vibration_input = torch.randn(5)
        odor_input = torch.randn(5)

        # To demonstrate the logic, we'll manually add a strong signal
        # in some steps to push the score above the threshold.
        if i % 2 == 0:
            print("A strong prey signature is detected nearby!")
            odor_input = odor_input * 3 # Amplify odor signal
            vision_input = vision_input * 2 # Amplify vision signal
        else:
            print("The environment is calm.")
        
        action = chrysopidae.forward(vision=vision_input, vibration=vibration_input, odor=odor_input)
        print(f"--> Chrysopidae's action: {action}")

        # --- Tettigoniidae Simulation (String-based) ---
        print("\n--- Tettigoniidae's Environment ---")
        # Simulate acoustic environment with string placeholders
        possible_sounds = ["wind_rustling", "predator_frequency", "mate_call"]
        acoustic_input = random.choice(possible_sounds)
        tactile_input = "leaf_surface"
        vision_input = "dappled_light"

        action = tettigoniidae.forward(acoustic=acoustic_input, tactile=tactile_input, vision=vision_input)
        print(f"--> Tettigoniidae's action: {action}")

    print("\n" + "-" * 50)
    print("SIMULATION COMPLETE")
    print("-" * 50)

if __name__ == "__main__":
    simulate_emergence()