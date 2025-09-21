import torch
from .base_specimen import PsiQRHBase

class Chrysopidae_PsiQRH(PsiQRHBase):
    """
    ΨQRH specimen for Chrysopidae (Green Lacewings), predators of aphids.
    """
    def __init__(self, attack_threshold=0.75):
        super().__init__()
        # 🧬 Sensory Profile
        # Sharp vision for colors, antennae for vibrations and odors, tactile receptors.
        self.sensory_input = ["vision_spectrum", "vibration_antennae", "plant_odor"]

        # 🌀 ΨQRH Training Profile
        # Ψ: wavefunction that prioritizes "slow movement + greenish-yellow color + sweet honeydew odor"
        self.collapse_function = "predator_focus_collapse"
        
        # Q: quantum processing of visual patterns (micro-motion discrimination)
        self.quantum_basis = "micro_motion_entanglement"
        
        # R: relationship with host plant, aphid prey, and competitor ants
        self.relational_graph = [("plant", "prey"), ("prey", "competitor_ant")]
        
        # H: heuristic of "active hunting on leaves with high chemical entropy"
        self.heuristic = "maximize_prey_capture_per_energy"

        # Threshold for triggering an attack, making the decision continuous
        self.attack_threshold = attack_threshold

    def forward(self, vision, vibration, odor):
        """
        Perception-action cycle for the Green Lacewing.
        The model logic to process sensory inputs and decide on an action
        (e.g., move, attack) would be implemented here.
        """
        # Add sensory noise for biological realism and robust training
        vision = vision + torch.randn_like(vision) * 0.1  # Simulates neural noise
        vibration = vibration + torch.randn_like(vibration) * 0.05
        odor = odor + torch.randn_like(odor) * 0.05

        # The wavefunction collapse and decision logic is now based on a score
        action = self._decide_action(vision, vibration, odor)
        return action

    def _compute_prey_score(self, vision, vibration, odor):
        """
        Computes a continuous score indicating the likelihood of prey being present.
        This is a placeholder for a more complex function that would be learned.
        """
        # Use the mean of the absolute values to get a more stable signal magnitude.
        vision_signal = torch.mean(torch.abs(vision)) * 1.0
        odor_signal = torch.mean(torch.abs(odor)) * 1.5
        vibration_signal = torch.mean(torch.abs(vibration)) * 0.5

        score = vision_signal + odor_signal + vibration_signal
        
        # Scale the score before the sigmoid to make it more sensitive
        return torch.sigmoid(score - 2.5) # Subtracting a value centers the sensitive range

    def _decide_action(self, vision, vibration, odor):
        """
        Makes a probabilistic decision based on a continuous prey score.
        """
        prey_score = self._compute_prey_score(vision, vibration, odor)
        
        print(f"Chrysopidae perceives a prey score of: {prey_score.item():.2f}")

        if prey_score > self.attack_threshold:
            return "ATTACK"
        else:
            return "SEARCH"