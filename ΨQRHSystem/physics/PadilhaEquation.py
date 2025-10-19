import torch
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List


class PadilhaEquation:
    """
    Padilha Equation Implementation

    f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

    Implementa a equação completa de Padilha para física quântica-fractal-óptica.
    """

    def __init__(self, I0: float = 1.0, alpha: float = 1.0, beta: float = 0.5,
                 k: float = 2.0, omega: float = 1.0, device: str = "cpu"):
        """
        Inicializa equação de Padilha

        Args:
            I0: Amplitude base
            alpha: Parâmetro de dispersão linear
            beta: Parâmetro de dispersão quadrática
            k: Número de onda
            omega: Frequência angular
            device: Dispositivo de computação
        """
        self.I0 = I0
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.omega = omega
        self.device = device

        print(f"🔬 Padilha Equation inicializada: f(λ,t) = {I0} sin({omega}t + {alpha}λ) e^(i({omega}t - {k}λ + {beta}λ²))")

    def compute_wave_function(self, wavelength: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Computa função de onda completa usando equação de Padilha

        Args:
            wavelength: Comprimento de onda λ [..., N]
            time: Tempo t [..., N]

        Returns:
            Função de onda f(λ,t) [..., N]
        """
        # f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))

        # Parte real (onda senoidal)
        real_part = self.I0 * torch.sin(self.omega * time + self.alpha * wavelength)

        # Parte imaginária (fase complexa)
        phase = self.omega * time - self.k * wavelength + self.beta * wavelength**2
        complex_part = torch.exp(1j * phase)

        # Função de onda completa
        wave_function = real_part * complex_part

        return wave_function

    def compute_spectral_components(self, wavelength: torch.Tensor, time: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computa componentes espectrais da equação de Padilha

        Args:
            wavelength: Comprimento de onda λ
            time: Tempo t

        Returns:
            Dicionário com componentes espectrais
        """
        # Componentes individuais
        oscillatory_term = torch.sin(self.omega * time + self.alpha * wavelength)
        dispersion_linear = self.alpha * wavelength
        dispersion_quadratic = self.beta * wavelength**2
        wave_number_term = self.k * wavelength

        # Fase total
        total_phase = self.omega * time - wave_number_term + dispersion_quadratic

        # Função de onda completa
        wave_function = self.compute_wave_function(wavelength, time)

        return {
            'wave_function': wave_function,
            'oscillatory_term': oscillatory_term,
            'dispersion_linear': dispersion_linear,
            'dispersion_quadratic': dispersion_quadratic,
            'wave_number_term': wave_number_term,
            'total_phase': total_phase,
            'real_part': wave_function.real,
            'imag_part': wave_function.imag,
            'magnitude': torch.abs(wave_function),
            'phase': torch.angle(wave_function)
        }

    def compute_fractal_dimension(self, wave_function: torch.Tensor) -> float:
        """
        Computa dimensão fractal da função de onda

        Args:
            wave_function: Função de onda

        Returns:
            Dimensão fractal D
        """
        # Análise de power-law: P(k) ~ k^(-β) → D = (3 - β) / 2

        # Computar power spectrum
        spectrum = torch.abs(torch.fft.fft(wave_function.flatten()))
        k = torch.arange(1, len(spectrum) + 1, dtype=torch.float32, device=wave_function.device)

        # Power-law fitting
        log_k = torch.log(k + 1e-10)
        log_P = torch.log(spectrum + 1e-10)

        # Regressão linear
        n = len(log_k)
        sum_x = log_k.sum()
        sum_y = log_P.sum()
        sum_xy = (log_k * log_P).sum()
        sum_x2 = (log_k ** 2).sum()

        # Coeficiente angular β
        beta = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        # Dimensão fractal
        D = (3.0 - beta.item()) / 2.0

        # Clamping para valores físicos
        D = max(1.0, min(D, 2.0))

        return D

    def validate_energy_conservation(self, input_energy: float, output_wave: torch.Tensor) -> bool:
        """
        Valida conservação de energia

        Args:
            input_energy: Energia de entrada
            output_wave: Função de onda de saída

        Returns:
            True se energia conservada
        """
        output_energy = torch.sum(torch.abs(output_wave) ** 2).item()
        conservation_ratio = abs(input_energy - output_energy) / input_energy

        return conservation_ratio <= 0.05  # 5% tolerância

    def get_optical_probe_output(self, wave_function: torch.Tensor) -> str:
        """
        Converte função de onda para saída óptica (texto)

        Args:
            wave_function: Função de onda

        Returns:
            Texto gerado via sonda óptica
        """
        # Usar magnitude e fase para gerar caracteres
        magnitude = torch.abs(wave_function)
        phase = torch.angle(wave_function)

        # Normalizar magnitude para range ASCII
        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-10)
        char_codes = 32 + (mag_norm * 95).long()  # Range printable ASCII

        # Modulação por fase
        phase_modulation = ((phase + torch.pi) / (2 * torch.pi) * 95).long()
        final_codes = torch.clamp(char_codes + phase_modulation, 32, 126)

        # Converter para caracteres
        text = ''.join(chr(int(code)) for code in final_codes.flatten()[:50])  # Limitar tamanho

        return text if text.strip() else "Ψ"  # Fallback

    def update_parameters(self, new_params: Dict[str, float]):
        """
        Atualiza parâmetros da equação

        Args:
            new_params: Novos parâmetros
        """
        for param, value in new_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
                print(f"🔧 Atualizado {param} = {value}")

    def get_parameters(self) -> Dict[str, float]:
        """
        Retorna parâmetros atuais

        Returns:
            Parâmetros da equação
        """
        return {
            'I0': self.I0,
            'alpha': self.alpha,
            'beta': self.beta,
            'k': self.k,
            'omega': self.omega
        }

    def compute_temporal_evolution(self, wavelength_range: Tuple[float, float],
                                 time_range: Tuple[float, float],
                                 num_points: int = 100) -> Dict[str, torch.Tensor]:
        """
        Computa evolução temporal da equação de Padilha

        Args:
            wavelength_range: Range de comprimento de onda (λ_min, λ_max)
            time_range: Range temporal (t_min, t_max)
            num_points: Número de pontos para discretização

        Returns:
            Evolução temporal completa
        """
        # Criar grids de wavelength e time
        wavelength = torch.linspace(wavelength_range[0], wavelength_range[1], num_points,
                                  device=self.device)
        time = torch.linspace(time_range[0], time_range[1], num_points,
                            device=self.device)

        # Criar meshgrid
        lambda_grid, t_grid = torch.meshgrid(wavelength, time, indexing='ij')

        # Computar função de onda
        wave_function = self.compute_wave_function(lambda_grid, t_grid)

        # Computar componentes espectrais
        spectral_components = self.compute_spectral_components(lambda_grid, t_grid)

        return {
            'wavelength': lambda_grid,
            'time': t_grid,
            'wave_function': wave_function,
            **spectral_components
        }