#!/usr/bin/env python3
"""
Chaos-Driven Visual Perspective for ΨQRH Spider Agents

This module creates a visual perspective where the environment's chaos_factor
defines how we perceive the quartz processor field. Each pixel represents
the physical state of a processor modulated by spider DNA.

The spatial relationship is not fixed - it is determined by environmental chaos.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import yaml
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from models.insect_specimens.dna import AraneaeDNA
from models.insect_specimens.araneae import Araneae_PsiQRH
from quartz_light_prototype import QuartzOpticalProcessor, CrystalProperties, LaserProperties

class ChaosVisualizer:
    """
    Visualizer that uses chaos_factor to modulate the visual perspective
    of the quartz processor field controlled by spider DNA.
    """

    def __init__(self, config_path=None):
        # Load configurations
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "qrh_config.yaml"
        self.config = self._load_config(config_path)

        # Parâmetros do campo visual (grade de processadores)
        self.field_size = (16, 16)  # 256 processadores

        # Estado dos processadores de quartzo
        self.processor_field = np.zeros(self.field_size, dtype=complex)

        # Mapeamento chaos -> distorção espacial
        self.chaos_history = []

        print(f"Chaos Visual Perspective initialized")
        print(f"Field size: {self.field_size[0]}×{self.field_size[1]} processors")
        print(f"Config loaded: {config_path}")

    def _load_config(self, config_path):
        """Load existing YAML configurations."""
        with open(str(config_path), 'r') as f:
            return yaml.safe_load(f)

    def chaos_spatial_transform(self, chaos_factor, base_positions):
        """
        Aplica transformação espacial baseada no chaos_factor.
        O caos distorce como percebemos o espaço dos processadores.

        Args:
            chaos_factor: Fator de caos do ambiente (0.0 a 1.0)
            base_positions: Posições base dos processadores

        Returns:
            Posições transformadas pelo caos
        """
        # Múltiplos modos de distorção baseados no caos

        # Modo 1: Distorção sinusoidal (caos baixo)
        if chaos_factor < 0.3:
            distortion = np.sin(base_positions * np.pi * chaos_factor * 4)
            scale = chaos_factor * 0.2

        # Modo 2: Distorção fractal (caos médio)
        elif chaos_factor < 0.7:
            # Usa o mesmo padrão fractal dos DNA das aranhas
            distortion = np.sin(base_positions * 3.14159 * chaos_factor * 8) * np.cos(base_positions * 2.71828 * chaos_factor * 6)
            scale = chaos_factor * 0.4

        # Modo 3: Distorção quântica (caos alto)
        else:
            # Distorção baseada em interferência quântica
            wave1 = np.exp(1j * base_positions * chaos_factor * 10)
            wave2 = np.exp(1j * base_positions * chaos_factor * 13)
            distortion = np.real(wave1 * np.conj(wave2))
            scale = chaos_factor * 0.6

        return base_positions + distortion * scale

    def map_spider_dna_to_processor(self, spider, chaos_factor):
        """
        Mapeia o DNA da aranha para o estado físico de um processador de quartzo.
        O chaos_factor modula essa transformação.

        Args:
            spider: Instância Araneae_PsiQRH
            chaos_factor: Fator de caos atual

        Returns:
            Estado complexo do processador
        """
        # Extrair parâmetros do DNA
        alpha = spider.config.alpha
        rotation_angles = spider.dna.rotation_angles
        health = spider.health

        # Aplicar modulação pelo caos
        chaos_modulated_alpha = alpha * (1 + chaos_factor * 0.3)

        # Simular processador de quartzo
        crystal = CrystalProperties()
        laser = LaserProperties()
        processor = QuartzOpticalProcessor(crystal, laser)

        # Criar quaternion baseado no DNA
        q_dna = np.array([
            1.0,  # Componente real
            rotation_angles[0] * 0.5,  # x
            rotation_angles[1] * 0.5,  # y
            rotation_angles[2] * 0.5   # z
        ])
        q_dna = q_dna / np.linalg.norm(q_dna)

        # Voltagem controlada pelo DNA modulado pelo caos
        control_voltage = (chaos_modulated_alpha - 1.0) * 10 * health
        control_voltage = np.clip(control_voltage, 0, 10)

        # Processar através do hardware de quartzo
        output_q = processor.propagate_quaternion_state(
            q_dna,
            control_voltage=control_voltage,
            crystal_angle=rotation_angles[3]
        )

        # Estado complexo: amplitude modulada pelo caos
        amplitude = health * (1 + chaos_factor * np.sin(alpha * 3.14159))
        phase = np.arctan2(output_q[1], output_q[0]) + chaos_factor * rotation_angles[4]

        return amplitude * np.exp(1j * phase)

    def generate_visual_field(self, spider_population, chaos_factor):
        """
        Gera o campo visual completo baseado na população de aranhas e caos.

        Args:
            spider_population: Lista de aranhas Araneae_PsiQRH
            chaos_factor: Fator de caos do ambiente

        Returns:
            Campo visual 2D (complex array)
        """
        # Resetar campo
        self.processor_field.fill(0)

        # Base positions para transformação
        y_base, x_base = np.mgrid[0:self.field_size[0], 0:self.field_size[1]]

        # Aplicar transformação espacial do caos
        x_transformed = self.chaos_spatial_transform(chaos_factor, x_base)
        y_transformed = self.chaos_spatial_transform(chaos_factor, y_base)

        # Para cada aranha, computar seu estado e distribuir no campo
        for i, spider in enumerate(spider_population):
            # Mapear DNA para estado do processador
            processor_state = self.map_spider_dna_to_processor(spider, chaos_factor)

            # Distribuição gaussiana no campo (influência da aranha)
            center_x = (i * 7 + int(chaos_factor * 50)) % self.field_size[1]
            center_y = (i * 5 + int(chaos_factor * 30)) % self.field_size[0]

            # Raio de influência modulado pelo health e caos
            radius = spider.health * (2 + chaos_factor * 3)

            # Aplicar influência gaussiana
            for y in range(self.field_size[0]):
                for x in range(self.field_size[1]):
                    # Distância com transformação do caos
                    dx = x_transformed[y, x] - center_x
                    dy = y_transformed[y, x] - center_y
                    dist = np.sqrt(dx*dx + dy*dy)

                    if dist < radius:
                        # Contribuição gaussiana modulada pelo caos
                        gaussian = np.exp(-dist*dist / (2 * radius*radius))
                        chaos_modulation = 1 + chaos_factor * np.sin(dist * 0.5)

                        self.processor_field[y, x] += processor_state * gaussian * chaos_modulation

        # Normalizar o campo
        max_val = np.max(np.abs(self.processor_field))
        if max_val > 0:
            self.processor_field = self.processor_field / max_val

        return self.processor_field

    def visualize_chaos_perspective(self, spider_population, chaos_factor, generation=0):
        """
        Cria visualização da perspectiva visual modulada pelo caos.

        Args:
            spider_population: Lista de aranhas
            chaos_factor: Fator de caos atual
            generation: Número da geração
        """
        # Gerar campo visual
        visual_field = self.generate_visual_field(spider_population, chaos_factor)

        # Armazenar histórico de caos
        self.chaos_history.append(chaos_factor)

        # Criar figura com múltiplas perspectivas
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Chaos-Driven Visual Perspective - Generation {generation}\n' +
                    f'Chaos Factor: {chaos_factor:.3f}', fontsize=16, fontweight='bold')

        # 1. Campo visual principal (intensidade)
        ax1 = axes[0, 0]
        intensity = np.abs(visual_field)
        im1 = ax1.imshow(intensity, cmap='plasma', interpolation='bilinear')
        ax1.set_title(f'Processor Intensity Field\n(Chaos Modulated)')
        ax1.set_xlabel('Processor X')
        ax1.set_ylabel('Processor Y')
        plt.colorbar(im1, ax=ax1, label='Light Intensity')

        # 2. Campo de fase (informação quântica)
        ax2 = axes[0, 1]
        phase = np.angle(visual_field)
        im2 = ax2.imshow(phase, cmap='hsv', interpolation='bilinear')
        ax2.set_title('Quantum Phase Field\n(DNA Rotation Information)')
        ax2.set_xlabel('Processor X')
        ax2.set_ylabel('Processor Y')
        plt.colorbar(im2, ax=ax2, label='Phase (radians)')

        # 3. Distorção espacial do caos
        ax3 = axes[0, 2]
        y_base, x_base = np.mgrid[0:8, 0:8]  # Grade reduzida para visualização
        x_dist = self.chaos_spatial_transform(chaos_factor, x_base)
        y_dist = self.chaos_spatial_transform(chaos_factor, y_base)

        ax3.quiver(x_base, y_base, x_dist - x_base, y_dist - y_base,
                  scale=20, color='blue', alpha=0.7)
        ax3.set_title(f'Chaos Spatial Distortion\n(Factor: {chaos_factor:.3f})')
        ax3.set_xlabel('Base X')
        ax3.set_ylabel('Base Y')
        ax3.grid(True, alpha=0.3)

        # 4. DNA dos agentes
        ax4 = axes[1, 0]
        alphas = [spider.config.alpha for spider in spider_population]
        healths = [spider.health for spider in spider_population]
        genders = [spider.gender for spider in spider_population]

        colors = ['blue' if g == 'male' else 'red' for g in genders]
        scatter = ax4.scatter(alphas, healths, c=colors, s=100, alpha=0.7, edgecolors='black')
        ax4.set_xlabel('DNA Alpha (Fractal Parameter)')
        ax4.set_ylabel('Health (System Stability)')
        ax4.set_title('Spider Population DNA Profile')
        ax4.grid(True, alpha=0.3)

        # 5. Evolução do caos
        ax5 = axes[1, 1]
        if len(self.chaos_history) > 1:
            ax5.plot(self.chaos_history, 'g-', linewidth=2, marker='o')
            ax5.axhline(y=chaos_factor, color='red', linestyle='--', alpha=0.7,
                       label=f'Current: {chaos_factor:.3f}')
        else:
            ax5.bar([0], [chaos_factor], color='green', alpha=0.7)
        ax5.set_title('Chaos Factor Evolution')
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('Chaos Factor')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Mapa espectral (análise de frequência)
        ax6 = axes[1, 2]
        # FFT 2D do campo visual
        fft_field = np.fft.fft2(visual_field)
        power_spectrum = np.abs(np.fft.fftshift(fft_field))**2

        # Log scale para melhor visualização
        log_spectrum = np.log10(power_spectrum + 1e-10)
        im6 = ax6.imshow(log_spectrum, cmap='viridis', interpolation='bilinear')
        ax6.set_title('Spectral Power Map\n(Frequency Domain)')
        ax6.set_xlabel('Frequency X')
        ax6.set_ylabel('Frequency Y')
        plt.colorbar(im6, ax=ax6, label='Log Power')

        plt.tight_layout()

        # Save visualization in images directory
        images_dir = Path(__file__).parent.parent / "images"
        images_dir.mkdir(exist_ok=True)
        filename = images_dir / f"chaos_perspective_gen_{generation:03d}.png"
        plt.savefig(str(filename), dpi=300, bbox_inches='tight')
        print(f"Chaos perspective saved: {filename}")

        return fig, visual_field


def run_chaos_visual_simulation():
    """
    Executa simulação visual usando o sistema ΨQRH existente
    com modulação pelo chaos_factor.
    """
    print("=" * 70)
    print("    CHAOS-DRIVEN VISUAL PERSPECTIVE SIMULATION")
    print("=" * 70)

    # Criar visualizador
    visualizer = ChaosVisualizer()

    # Criar população inicial (usar sistema existente)
    population = [Araneae_PsiQRH(dna=AraneaeDNA()) for _ in range(6)]

    print(f"\nPopulation created: {len(population)} spiders")
    for i, spider in enumerate(population):
        print(f"  Spider {i}: α={spider.config.alpha:.3f}, health={spider.health:.2f}, gender={spider.gender}")

    # Simular gerações com diferentes níveis de caos
    chaos_scenarios = [
        (0.1, "Low Chaos - Ordered Environment"),
        (0.4, "Medium Chaos - Dynamic Environment"),
        (0.8, "High Chaos - Turbulent Environment"),
        (0.95, "Extreme Chaos - Chaotic Environment")
    ]

    for gen, (chaos_factor, description) in enumerate(chaos_scenarios):
        print(f"\n--- Generation {gen+1}: {description} ---")

        # Ambiente com o fator de caos específico
        environment = {
            "chaos_factor": chaos_factor,
            "waves": []  # Sem ondas para focar na visualização
        }

        # Executar step dos agentes
        for spider in population:
            spider.forward(environment)

        # Gerar visualização
        fig, field = visualizer.visualize_chaos_perspective(
            population, chaos_factor, generation=gen+1
        )

        # Estatísticas da geração
        avg_alpha = np.mean([s.config.alpha for s in population])
        avg_health = np.mean([s.health for s in population])
        field_complexity = np.std(np.abs(field))

        print(f"  Chaos Factor: {chaos_factor:.3f}")
        print(f"  Average DNA Alpha: {avg_alpha:.3f}")
        print(f"  Average Health: {avg_health:.3f}")
        print(f"  Visual Field Complexity: {field_complexity:.3f}")

    # Mostrar última visualização
    plt.show()

    print("\n" + "=" * 70)
    print("CHAOS VISUAL SIMULATION COMPLETE")
    print("Generated 4 perspective views showing chaos modulation")
    print("=" * 70)

    return visualizer, population


if __name__ == "__main__":
    visualizer, population = run_chaos_visual_simulation()