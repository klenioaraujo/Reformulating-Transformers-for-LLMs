from dataclasses import dataclass
from typing import List, Dict, Any
import yaml
import os

@dataclass
class ModelConfig:
    embed_dim: int = 64
    max_history: int = 10
    vocab_size: int = 256
    num_heads: int = 4
    hidden_dim: int = 128
    num_layers: int = 3

@dataclass
class PhysicsConfig:
    I0: float = 1.0
    alpha: float = 1.0
    beta: float = 0.5
    k: float = 2.0
    omega: float = 1.0

@dataclass
class SystemConfig:
    model: ModelConfig
    physics: PhysicsConfig
    device: str = "auto"
    enable_components: List[str] = None
    validation: Dict[str, bool] = None

    def __post_init__(self):
        if self.enable_components is None:
            self.enable_components = ["quantum_memory", "auto_calibration", "physical_harmonics"]
        if self.validation is None:
            self.validation = {
                "energy_conservation": True,
                "unitarity": True,
                "numerical_stability": True
            }

    @classmethod
    def from_yaml(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from YAML file with legacy support"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Extract nested configs
        model_config = ModelConfig(**config_data.get('model', {}))
        physics_config = PhysicsConfig(**config_data.get('physics', {}))
        system_config = config_data.get('system', {})

        # Legacy compatibility mapping
        enable_components = system_config.get('enable_components', [])

        # Map legacy boolean flags to component list
        if system_config.get('enable_auto_calibration', False):
            if 'auto_calibration' not in enable_components:
                enable_components.append('auto_calibration')

        if system_config.get('enable_cognitive_priming', False):
            if 'cognitive_priming' not in enable_components:
                enable_components.append('cognitive_priming')

        if system_config.get('enable_noncommutative', False):
            if 'noncommutative' not in enable_components:
                enable_components.append('noncommutative')

        # Default components if none specified
        if not enable_components:
            enable_components = ["quantum_memory", "auto_calibration", "physical_harmonics"]

        return cls(
            model=model_config,
            physics=physics_config,
            device=system_config.get('device', 'auto'),
            enable_components=enable_components,
            validation=system_config.get('validation', {
                "energy_conservation": True,
                "unitarity": True,
                "numerical_stability": True
            })
        )

    @classmethod
    def default(cls) -> 'SystemConfig':
        """Cria configuração padrão para testes e uso básico"""
        return cls(
            model=ModelConfig(),
            physics=PhysicsConfig()
        )

    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_data = {
            'model': {
                'embed_dim': self.model.embed_dim,
                'max_history': self.model.max_history,
                'vocab_size': self.model.vocab_size,
                'num_heads': self.model.num_heads,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers
            },
            'physics': {
                'I0': self.physics.I0,
                'alpha': self.physics.alpha,
                'beta': self.physics.beta,
                'k': self.physics.k,
                'omega': self.physics.omega
            },
            'system': {
                'device': self.device,
                'enable_components': self.enable_components,
                'validation': self.validation
            }
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)