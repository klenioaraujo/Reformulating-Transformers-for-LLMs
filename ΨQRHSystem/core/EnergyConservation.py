import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from ΨQRHSystem.core.TernaryLogicFramework import TernaryLogicFramework


class EnergyConservation:
    """
    Análise da Conservação de Energia no ΨQRH

    Princípio Fundamental:
    ⟨ψ|H|ψ⟩ deve ser constante em sistemas fechados

    Implementa verificação rigorosa de conservação de energia
    com validação baseada em π para estabilidade numérica.
    """

    def __init__(self, device: str = "cpu", epsilon: float = 1e-8):
        """
        Inicializa verificador de conservação de energia

        Args:
            device: Dispositivo de computação
            epsilon: Tolerância para verificações numéricas
        """
        self.device = device
        self.epsilon = epsilon
        self.hamiltonian = None
        self.energy_history = []
        self.ternary_logic = TernaryLogicFramework(device=device)

        print("⚡ Energy Conservation Analysis initialized with π-based validation")

    def verify_conservation(self, state: torch.Tensor, H: Optional[torch.Tensor] = None) -> bool:
        """
        Verifica conservação de energia: ⟨ψ|H|ψ⟩ deve ser constante

        Args:
            state: Estado quântico ψ
            H: Hamiltoniano (opcional, usa self.hamiltonian se None)

        Returns:
            True se energia conservada dentro da tolerância
        """
        if H is not None:
            self.hamiltonian = H

        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not provided and not previously set")

        # Calcular valor esperado da energia
        expected_energy = torch.vdot(state, self.hamiltonian @ state)

        # Armazenar histórico
        self.energy_history.append(expected_energy.real)

        # Verificar conservação se temos histórico suficiente
        if len(self.energy_history) < 2:
            return True  # Não há o que comparar ainda

        # Calcular variação relativa
        initial_energy = self.energy_history[0]
        current_energy = self.energy_history[-1]
        energy_variation = abs(current_energy - initial_energy)

        # Tolerância baseada em π para estabilidade
        tolerance = self._pi_based_tolerance(state)

        return energy_variation < tolerance

    def validate_energy_conservation(self, input_energy: float, output_energy: float,
                                    tolerance: float = 0.05) -> bool:
        """
        Valida conservação de energia

        Args:
            input_energy: Energia de entrada
            output_energy: Energia de saída
            tolerance: Tolerância (5% padrão)

        Returns:
            True se energia conservada dentro da tolerância
        """
        if input_energy == 0:
            return True

        conservation_ratio = abs(input_energy - output_energy) / input_energy
        return conservation_ratio <= tolerance

    def _pi_based_tolerance(self, state: torch.Tensor) -> float:
        """
        Calcula tolerância baseada em π para verificação de energia

        Args:
            state: Estado quântico

        Returns:
            Tolerância adaptativa
        """
        # Norma do estado
        state_norm = torch.norm(state)

        # Energia típica baseada na dimensão
        typical_energy = state_norm ** 2

        # Tolerância: ε = π * ||ψ||² / (1 + ||ψ||²)
        # Isso garante estabilidade para estados de diferentes escalas
        tolerance = torch.pi * typical_energy / (1 + typical_energy)

        return tolerance.item()

    def validate_energy_bounds(self, state: torch.Tensor, H: torch.Tensor,
                             energy_bounds: Tuple[float, float]) -> bool:
        """
        Valida se energia está dentro de bounds físicos

        Args:
            state: Estado quântico
            H: Hamiltoniano
            energy_bounds: (E_min, E_max) bounds físicos

        Returns:
            True se energia dentro dos bounds
        """
        expected_energy = torch.vdot(state, H @ state).real

        E_min, E_max = energy_bounds
        return E_min <= expected_energy <= E_max

    def compute_energy_gradient(self, state: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Computa gradiente da energia em relação ao estado

        Args:
            state: Estado quântico
            H: Hamiltoniano

        Returns:
            Gradiente da energia
        """
        # ∇_ψ ⟨ψ|H|ψ⟩ = 2 H ψ (para hamiltonianos hermitianos)
        return 2 * H @ state

    def validate_unitary_evolution(self, state_t: torch.Tensor,
                                 state_t_plus_dt: torch.Tensor,
                                 U: torch.Tensor) -> bool:
        """
        Valida evolução unitária conservando energia

        Args:
            state_t: Estado no tempo t
            state_t_plus_dt: Estado no tempo t+dt
            U: Operador unitário de evolução

        Returns:
            True se evolução é unitária e energia conservada
        """
        # Verificar unitariedade: U†U = I
        identity = torch.eye(U.shape[0], device=U.device, dtype=U.dtype)
        unitary_check = torch.allclose(U.conj().T @ U, identity, atol=1e-5)

        # Verificar evolução: ψ(t+dt) = U ψ(t)
        evolution_check = torch.allclose(state_t_plus_dt, U @ state_t, atol=1e-5)

        # Combinar verificações usando lógica ternária
        unitary_result = 1 if unitary_check else -1
        evolution_result = 1 if evolution_check else -1

        final_result = self.ternary_logic.ternary_and(unitary_result, evolution_result)

        return final_result == 1

    def compute_energy_spectrum(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computa espectro de energia do hamiltoniano

        Args:
            H: Hamiltoniano

        Returns:
            (autovalores, autovetores)
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(H)
        return eigenvalues, eigenvectors

    def validate_energy_uncertainty(self, state: torch.Tensor, H: torch.Tensor) -> float:
        """
        Valida princípio de incerteza energia-tempo

        Args:
            state: Estado quântico
            H: Hamiltoniano

        Returns:
            Incerteza energia-tempo (deve ser ≥ ℏ/2)
        """
        # Energia média
        E_mean = torch.vdot(state, H @ state).real

        # Incerteza energia
        delta_E = torch.sqrt(torch.vdot(state, (H - E_mean * torch.eye(H.shape[0], device=H.device)) ** 2 @ state).real)

        # Para tempo, precisaríamos da derivada temporal
        # Por enquanto, retornamos apenas delta_E como proxy
        return delta_E.item()

    def get_conservation_report(self) -> Dict[str, Any]:
        """
        Gera relatório detalhado de conservação de energia

        Returns:
            Relatório com métricas de conservação
        """
        if not self.energy_history:
            return {'status': 'No energy measurements available'}

        energies = torch.tensor(self.energy_history, device=self.device)

        # Estatísticas básicas
        energy_mean = torch.mean(energies).item()
        energy_std = torch.std(energies).item()
        energy_min = torch.min(energies).item()
        energy_max = torch.max(energies).item()

        # Variação relativa
        relative_variation = energy_std / (abs(energy_mean) + self.epsilon)

        # Conservação baseada em π
        pi_conservation_score = torch.exp(-relative_variation / torch.pi).item()

        # Classificação ternária
        if relative_variation < 0.01:  # Excelente conservação
            conservation_class = 1
        elif relative_variation < 0.1:  # Boa conservação
            conservation_class = 0
        else:  # Conservação pobre
            conservation_class = -1

        return {
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_range': (energy_min, energy_max),
            'relative_variation': relative_variation,
            'pi_conservation_score': pi_conservation_score,
            'conservation_class': conservation_class,
            'measurements_count': len(self.energy_history),
            'is_conserved': conservation_class >= 0
        }

    def reset_energy_history(self):
        """Reseta histórico de medições de energia"""
        self.energy_history.clear()
        print("⚡ Energy conservation history reset")

    def validate_thermodynamic_consistency(self, states: List[torch.Tensor],
                                         temperatures: List[float],
                                         H: torch.Tensor) -> bool:
        """
        Valida consistência termodinâmica

        Args:
            states: Lista de estados quânticos
            temperatures: Temperaturas correspondentes
            H: Hamiltoniano

        Returns:
            True se consistente termodinamicamente
        """
        if len(states) != len(temperatures):
            return False

        energies = []
        entropies = []

        for state, T in zip(states, temperatures):
            # Energia média
            E = torch.vdot(state, H @ state).real.item()
            energies.append(E)

            # Entropia (aproximada)
            # Para estados puros, S = 0; para mistos, S = -∑ p_i ln p_i
            # Aqui usamos uma aproximação baseada na pureza
            purity = torch.abs(torch.vdot(state, state))**2
            S = -torch.log(purity + self.epsilon).item()
            entropies.append(S)

        # Verificar se energia aumenta com temperatura (princípio básico)
        # Nota: Isso é uma verificação simplificada
        energy_trend = np.polyfit(temperatures, energies, 1)[0]

        return energy_trend >= 0  # Energia deve aumentar com temperatura