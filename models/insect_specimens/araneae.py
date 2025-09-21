
import torch
import random
import numpy as np
from .base_specimen import PsiQRHBase
from .dna import AraneaeDNA
from qrh_layer import QRHLayer
from .communication import PadilhaWave

class Araneae_PsiQRH(PsiQRHBase):
    """
    An agent whose behavior and processing capabilities emerge from its DNA.
    The DNA defines the configuration of its personal QRHLayer.
    """
    def __init__(self, dna: AraneaeDNA, maturity_age: int = 5, device: str = 'cpu'):
        super().__init__()
        self.heuristic = "survive_and_reproduce"
        self.dna = dna
        self.device = device

        # Create the agent's personal QRHLayer from its DNA
        self.config = self.dna.create_config(embed_dim=64, device=self.device)
        self.qrh_layer = QRHLayer(config=self.config)
        
        # --- Attributes derived from DNA and State ---
        self.age = 0
        self.gender = random.choice(['male', 'female'])
        self.life_stage = "spiderling"
        self.maturity_age = maturity_age
        self.state = "IDLE"
        self.health = 1.0 # Represents the stability of its DNA/QRHLayer
        self.mating_readiness = 0.0
        self.reproduction_threshold = 0.6 # Lowered threshold
        self.last_reproduced_age = -10

    @staticmethod
    def reproduce(parent1: 'Araneae_PsiQRH', parent2: 'Araneae_PsiQRH', mutation_rate: float = 0.1) -> 'Araneae_PsiQRH':
        """Creates a new agent via genetic crossover and mutation."""
        child_dna = AraneaeDNA.crossover(parent1.dna, parent2.dna)
        child_dna.mutate(mutation_rate=mutation_rate)
        # The child inherits the maturity age from its parents, plus some delay
        child_maturity_age = max(parent1.age, parent2.age) + 5
        return Araneae_PsiQRH(dna=child_dna, maturity_age=child_maturity_age)

    def analyze_wave(self, received_wave: PadilhaWave) -> float:
        """Uses the agent's own QRHLayer to analyze a received wave."""
        # This replaces the old WaveAnalyzer, making the analysis personal to the agent's DNA
        ideal_wave = PadilhaWave(emitter_signature=(self.config.alpha, 0)) # Simplified expected signature
        
        # Convert waves to tensors
        # This logic needs to be robust and match the layer's expectation
        seq_len = len(received_wave.wave_shape)
        received_tensor = torch.zeros(1, seq_len, 4 * self.config.embed_dim, device=self.device)
        ideal_tensor = torch.zeros(1, seq_len, 4 * self.config.embed_dim, device=self.device)
        
        received_tensor[0, :, 0] = torch.from_numpy(np.real(received_wave.wave_shape)).float()
        received_tensor[0, :, 1] = torch.from_numpy(np.imag(received_wave.wave_shape)).float()
        ideal_tensor[0, :, 0] = torch.from_numpy(np.real(ideal_wave.wave_shape)).float()
        ideal_tensor[0, :, 1] = torch.from_numpy(np.imag(ideal_wave.wave_shape)).float()

        with torch.no_grad():
            processed_received = self.qrh_layer(received_tensor)
            processed_ideal = self.qrh_layer(ideal_tensor)
        
        similarity = torch.nn.functional.cosine_similarity(processed_received.flatten(), processed_ideal.flatten(), dim=0)
        return max(0, similarity.item())

    def forward(self, environmental_data: dict):
        self.age += 1
        if self.age >= self.maturity_age and self.life_stage == "spiderling":
            self.life_stage = "adult"
            print(f"*** Spider {id(self)} has matured into an ADULT ({self.gender})! ***")

        # Health check: Unstable DNA can lead to poor health
        health_check = self.qrh_layer.check_health(torch.randn(1, 256, 4 * self.config.embed_dim, device=self.device))
        self.health = 0.9 * self.health + 0.1 * (1.0 if health_check.get('is_stable', True) else 0.0)

        # Mating readiness depends on age, health, and time since last reproduction
        if self.life_stage == 'adult' and self.health > 0.7 and (self.age - self.last_reproduced_age > 5):
            self.mating_readiness = min(1.0, self.mating_readiness + 0.2)
        else:
            self.mating_readiness = max(0.0, self.mating_readiness - 0.1)

        action = {"type": "IDLE"}

        if self.life_stage == 'spiderling':
            action["type"] = "HIDING"
        elif self.state == "IDLE" and self.mating_readiness > self.reproduction_threshold:
            self.state = "SEEKING_MATE"

        if self.state == "SEEKING_MATE":
            if self.gender == 'male':
                action["type"] = "EMIT_MATING_WAVE"
                action["wave"] = PadilhaWave(emitter_signature=(self.config.alpha, 0))
                self.state = "IDLE" # Conserve energy
            else: # Female is listening
                for wave_packet in environmental_data.get("waves", []):
                    correlation = self.analyze_wave(wave_packet["wave"])
                    print(f"Female {id(self)} analyzed wave from {wave_packet['emitter_id']} with correlation: {correlation:.2f}")
                    if correlation > 0.9: # High correlation needed
                        action["type"] = "REPRODUCE"
                        action["partner_id"] = wave_packet['emitter_id']
                        self.mating_readiness = 0.0
                        self.last_reproduced_age = self.age
                        self.state = "IDLE"
                        break
        
        print(f"Spider {id(self)} ({self.gender}, Age {self.age}, State {self.state}, Health {self.health:.2f}, Readiness {self.mating_readiness:.2f}) -> Action: {action['type']}")
        return action
