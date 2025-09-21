
import torch
import random
from .base_specimen import PsiQRHBase
from .communication import PadilhaWave, WaveAnalyzer

class Araneae_PsiQRH(PsiQRHBase):
    """
    Enhanced spider agent with a full lifecycle, environmental intelligence,
    and a reproductive cycle based on ΨQRH wave communication.
    """
    def __init__(self, maturity_age: int = 4, device: str = 'cpu'):
        super().__init__()
        self.heuristic = "maximize_prey_capture_minimize_energy_cost_and_reproduce"
        
        # Core Attributes
        self.age = 0
        self.gender = random.choice(['male', 'female'])
        self.device = device

        # Lifecycle & State
        self.life_stage = "spiderling"
        self.maturity_age = maturity_age
        self.state = "SCOUTING"
        self.location = None

        # Resources
        self.silk_reserves = 1.0
        self.web_exists = False
        self.web_integrity = 0.0
        self.web_repair_threshold = 0.7

        # Reproduction Attributes
        self.mating_readiness = 0.0
        self.reproduction_threshold = 0.8
        self.last_reproduced_age = -10
        self.signature = self._generate_signature()
        self.wave_analyzer = WaveAnalyzer(embed_dim=128, device=self.device)

    def _generate_signature(self) -> tuple:
        """Generates a unique (alpha, beta) signature for the spider's wave.
           Conceptually, this is part of its 'DNA'."""
        # Use object id for a unique, stable seed
        random.seed(id(self))
        alpha = round(random.uniform(1.2, 2.5), 3)
        beta = round(random.uniform(0.5, 1.5), 3)
        return (alpha, beta)

    def _update_internal_state(self):
        """Handles all internal state changes per time step."""
        self.age += 1
        if self.age >= self.maturity_age and self.life_stage == "spiderling":
            self.life_stage = "adult"
            self.state = "SCOUTING"
            print(f"*** Spider {id(self)} has matured into an ADULT ({self.gender})! ***")

        self.silk_reserves = min(1.0, self.silk_reserves + 0.05)
        if self.web_exists:
            self.web_integrity = max(0.0, self.web_integrity - 0.02)

        # Update mating readiness based on health, age, and successful living
        if self.life_stage == 'adult' and self.web_exists and self.web_integrity > 0.5 and (self.age - self.last_reproduced_age > 5):
            self.mating_readiness = min(1.0, self.mating_readiness + 0.15)
        else:
            self.mating_readiness = max(0.0, self.mating_readiness - 0.1)

    def _evaluate_locations(self, locations: dict) -> str:
        best_location = None
        max_score = -float('inf')
        print(f"Spider {id(self)} is evaluating locations:")
        for name, props in locations.items():
            score = (props['prey_traffic'] * 1.5) - (props['wind_exposure'] * 1.2) + (props['anchor_points'] * 0.5)
            print(f"  - {name}: Score {score:.2f}")
            if score > max_score:
                max_score = score
                best_location = name
        return best_location

    def forward(self, environmental_data: dict):
        self._update_internal_state()

        action = {"type": "WAIT"} # Default action is a dictionary

        # --- Primary State Machine ---
        if self.mating_readiness > self.reproduction_threshold and self.gender == 'male' and self.state != 'SEEKING_MATE':
             self.state = 'SEEKING_MATE'

        # --- Behavior based on State ---
        if self.state == "SCOUTING":
            if self.life_stage == "spiderling":
                action["type"] = "HIDING_AND_GROWING"
            else:
                best_location_name = self._evaluate_locations(environmental_data['locations'])
                self.location = best_location_name
                self.state = "IDLE"
                action["type"] = f"CHOSE_LOCATION '{self.location}'"

        elif self.state == "IDLE":
            if self.life_stage == "adult" and self.silk_reserves >= 0.5:
                self.state = "BUILDING"
                action["type"] = "START_BUILDING_WEB"
            else:
                action["type"] = "RESTING"

        elif self.state == "BUILDING":
            self.web_exists = True
            self.web_integrity = 1.0
            self.silk_reserves -= 0.5
            self.state = "WAITING"
            action["type"] = "FINISH_BUILDING_WEB"

        elif self.state == "WAITING":
            # Females listen for mating calls when ready
            if self.mating_readiness > self.reproduction_threshold and self.gender == 'female':
                for wave_packet in environmental_data.get("waves", []):
                    correlation = self.wave_analyzer.analyze_correlation(wave_packet["wave_form"], wave_packet["signature"])
                    print(f"Female {id(self)} analyzed a wave with correlation: {correlation:.2f}")
                    if correlation > 0.85: # High correlation needed
                        action["type"] = "REPRODUCE"
                        action["partner_id"] = wave_packet["emitter_id"]
                        self.mating_readiness = 0.0
                        self.last_reproduced_age = self.age
                        break # Found a mate
            # Standard waiting behavior if not mating
            if action["type"] == "WAIT":
                # Handle prey, repairs etc.
                pass # Simplified for clarity

        elif self.state == "SEEKING_MATE":
            if self.gender == 'male':
                action["type"] = "EMIT_MATING_WAVE"
                action["wave"] = PadilhaWave(self.signature)
                # After emitting, go back to waiting to conserve energy
                self.state = "WAITING"

        # Simplified print statement
        print(f"Spider {id(self)} ({self.gender}, Age {self.age}, State {self.state}, Readiness {self.mating_readiness:.2f}) -> Action: {action['type']}")
        return action
