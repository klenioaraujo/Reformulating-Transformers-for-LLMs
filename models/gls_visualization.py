#!/usr/bin/env python3
"""
GLS Visualization System - ΨQRH Framework

Visualization tools for Genetic Light Spectral ecosystem data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from .gls_data_models import GLSHabitatModel, ColonyAnalysis, SpectralSignature
from .gls_analysis import GLSAnalyzer


class GLSVisualizer:
    """
    Main visualization engine for GLS ecosystem data
    """

    def __init__(self, habitat_model: GLSHabitatModel, analyzer: GLSAnalyzer):
        self.habitat_model = habitat_model
        self.analyzer = analyzer
        self.color_scheme = {
            'Araneae': '#FF4136',      # Red - Spiders
            'Chrysopidae': '#2ECC40',  # Green - Lacewings
            'Apis': '#0074D9',         # Blue - Bees
            'Scutigera': '#FF851B',    # Orange - Centipedes
            'background': '#0F0F23',   # Dark blue background
            'accent': '#00FF88',       # Bright green accent
            'secondary': '#4ECDC4'     # Teal secondary
        }

    def create_4d_habitat_projection(self,
                                   positions: List[Dict],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create 4D habitat projection with w-dimension as color
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Set dark theme
        fig.patch.set_facecolor(self.color_scheme['background'])
        ax.set_facecolor(self.color_scheme['background'])

        # Group positions by species
        species_data = {}
        for pos in positions:
            species = pos['species']
            if species not in species_data:
                species_data[species] = {'x': [], 'y': [], 'z': [], 'w': [], 'ids': []}

            species_data[species]['x'].append(pos['x'])
            species_data[species]['y'].append(pos['y'])
            species_data[species]['z'].append(pos['z'])
            species_data[species]['w'].append(pos.get('w', 0.0))
            species_data[species]['ids'].append(pos['id'])

        # Plot each species
        for species, data in species_data.items():
            if len(data['x']) > 0:
                scatter = ax.scatter(
                    data['x'], data['y'], data['z'],
                    c=data['w'],  # w-dimension as color
                    s=80,
                    alpha=0.8,
                    cmap='viridis',
                    label=species,
                    edgecolors=self.color_scheme.get(species, '#FFFFFF'),
                    linewidth=2
                )

        # Customize plot
        ax.set_xlabel('X (meters)', color='white', fontsize=12)
        ax.set_ylabel('Y (meters)', color='white', fontsize=12)
        ax.set_zlabel('Z (height)', color='white', fontsize=12)
        ax.set_title('4D Unitary Habitat Projection\n(w-dimension as color)',
                    color=self.color_scheme['accent'], fontsize=14, fontweight='bold')

        # Set axis limits
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 15)
        ax.set_zlim(0, 10)

        # Color scheme
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', facecolor='black', edgecolor='white')

        # Add colorbar for w-dimension
        if positions:
            plt.colorbar(scatter, ax=ax, label='Unitary Phase (w)', shrink=0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, facecolor=self.color_scheme['background'],
                       edgecolor='none', dpi=300)

        return fig

    def create_colony_dynamics_plot(self,
                                  historical_data: List[Dict],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Create colony population dynamics visualization
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.patch.set_facecolor(self.color_scheme['background'])

        if len(historical_data) < 2:
            return fig

        # Extract time series data
        timestamps = []
        species_populations = {species: [] for species in ['Araneae', 'Chrysopidae', 'Apis']}
        species_health = {species: [] for species in ['Araneae', 'Chrysopidae', 'Apis']}

        for entry in historical_data:
            timestamps.append(entry.get('timestamp', datetime.now()))
            data = entry.get('data', {})
            colonies = data.get('colonies', {})

            for species in species_populations.keys():
                if species in colonies:
                    colony = colonies[species]
                    species_populations[species].append(getattr(colony, 'population', 0))
                    species_health[species].append(getattr(colony, 'health_score', 0.0))
                else:
                    species_populations[species].append(0)
                    species_health[species].append(0.0)

        # Population dynamics plot
        ax1.set_facecolor(self.color_scheme['background'])
        for species, populations in species_populations.items():
            if len(populations) > 0:
                ax1.plot(range(len(populations)), populations,
                        color=self.color_scheme.get(species, '#FFFFFF'),
                        linewidth=3, marker='o', markersize=6,
                        label=f'{species} Population')

        ax1.set_title('Colony Population Dynamics Over Time',
                     color=self.color_scheme['accent'], fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Steps', color='white', fontsize=12)
        ax1.set_ylabel('Population Count', color='white', fontsize=12)
        ax1.grid(True, alpha=0.3, color='white')
        ax1.tick_params(colors='white')
        ax1.legend(facecolor='black', edgecolor='white')

        # Health scores plot
        ax2.set_facecolor(self.color_scheme['background'])
        for species, health_scores in species_health.items():
            if len(health_scores) > 0:
                ax2.plot(range(len(health_scores)), health_scores,
                        color=self.color_scheme.get(species, '#FFFFFF'),
                        linewidth=3, marker='s', markersize=6,
                        label=f'{species} Health')

        ax2.set_title('Colony Health Scores Over Time',
                     color=self.color_scheme['accent'], fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Steps', color='white', fontsize=12)
        ax2.set_ylabel('Health Score', color='white', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, color='white')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='black', edgecolor='white')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, facecolor=self.color_scheme['background'],
                       edgecolor='none', dpi=300)

        return fig

    def create_spectral_emergence_map(self,
                                    coherence_data: Dict[str, float],
                                    emergence_level: float,
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create spectral coherence and emergence visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor(self.color_scheme['background'])

        # Spectral coherence radar chart
        ax1.set_facecolor(self.color_scheme['background'])

        if coherence_data:
            species = list(coherence_data.keys())
            coherences = list(coherence_data.values())

            # Create polar plot
            angles = np.linspace(0, 2 * np.pi, len(species), endpoint=False).tolist()
            coherences += coherences[:1]  # Complete the circle
            angles += angles[:1]

            ax1 = plt.subplot(121, projection='polar')
            ax1.plot(angles, coherences, color=self.color_scheme['accent'],
                    linewidth=3, marker='o', markersize=8)
            ax1.fill(angles, coherences, color=self.color_scheme['accent'], alpha=0.25)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(species, color='white', fontsize=10)
            ax1.set_ylim(0, 1)
            ax1.set_title('Spectral Coherence by Species',
                         color=self.color_scheme['accent'], fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # Emergence level visualization
        ax2.set_facecolor(self.color_scheme['background'])

        # Create emergence level gauge
        emergence_normalized = min(emergence_level / 20.0, 1.0)  # Normalize to 0-1

        # Draw gauge background
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        ax2.plot(theta, r, color='white', linewidth=5, alpha=0.3)

        # Draw emergence arc
        emergence_theta = theta[:int(emergence_normalized * len(theta))]
        emergence_r = r[:len(emergence_theta)]
        ax2.plot(emergence_theta, emergence_r, color=self.color_scheme['accent'],
                linewidth=8)

        # Add needle
        needle_angle = emergence_normalized * np.pi
        ax2.plot([needle_angle, needle_angle], [0, 1],
                color='red', linewidth=4, marker='o', markersize=10)

        ax2.set_ylim(0, 1.2)
        ax2.set_xlim(0, np.pi)
        ax2.set_title(f'Emergence Level: {emergence_level:.1f}',
                     color=self.color_scheme['accent'], fontsize=12, fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, facecolor=self.color_scheme['background'],
                       edgecolor='none', dpi=300)

        return fig

    def create_photonic_network_graph(self,
                                    photonic_data: Dict,
                                    colony_data: Dict[str, ColonyAnalysis],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Create photonic network connectivity visualization
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.color_scheme['background'])
        ax.set_facecolor(self.color_scheme['background'])

        # Create network nodes for colonies
        colony_positions = {}
        for i, (species, colony) in enumerate(colony_data.items()):
            angle = i * 2 * np.pi / len(colony_data)
            x = 5 * np.cos(angle)
            y = 5 * np.sin(angle)
            colony_positions[species] = (x, y)

        # Draw colony nodes
        for species, (x, y) in colony_positions.items():
            colony = colony_data[species]
            node_size = colony.population * 20  # Size based on population

            ax.scatter(x, y, s=node_size,
                      color=self.color_scheme.get(species, '#FFFFFF'),
                      alpha=0.8, edgecolors='white', linewidth=2)
            ax.text(x, y + 1, species, ha='center', va='bottom',
                   color='white', fontsize=12, fontweight='bold')

        # Draw connections based on communication frequencies
        for species1, colony1 in colony_data.items():
            pos1 = colony_positions[species1]
            for species2, colony2 in colony_data.items():
                if species1 != species2:
                    pos2 = colony_positions[species2]

                    # Connection strength based on frequency harmony
                    freq_ratio = colony1.communication_frequency / colony2.communication_frequency
                    harmony = 1.0 / (1.0 + abs(freq_ratio - round(freq_ratio)))

                    if harmony > 0.5:  # Only draw strong connections
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                               color=self.color_scheme['secondary'],
                               alpha=harmony, linewidth=harmony * 5)

        # Add photonic elements
        coherence = photonic_data.get('avg_coherence', 0.5)
        emitters = photonic_data.get('total_emitters', 0)

        # Draw laser emitters around the network
        for i in range(min(emitters, 12)):  # Limit visual clutter
            angle = i * 2 * np.pi / 12
            x = 8 * np.cos(angle)
            y = 8 * np.sin(angle)

            ax.scatter(x, y, s=50, color='yellow', marker='*',
                      alpha=coherence, edgecolors='white', linewidth=1)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title(f'Photonic Communication Network\n'
                    f'Coherence: {coherence:.2f} | Emitters: {emitters}',
                    color=self.color_scheme['accent'], fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, facecolor=self.color_scheme['background'],
                       edgecolor='none', dpi=300)

        return fig

    def create_gls_interaction_field(self,
                                   gls_scores: Dict[str, float],
                                   spectral_signatures: Dict[str, SpectralSignature],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create GLS interaction field visualization
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.patch.set_facecolor(self.color_scheme['background'])

        # GLS Stability Scores
        ax1.set_facecolor(self.color_scheme['background'])
        species = list(gls_scores.keys())
        scores = list(gls_scores.values())
        colors = [self.color_scheme.get(s, '#FFFFFF') for s in species]

        bars = ax1.bar(species, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_title('GLS Stability Scores', color=self.color_scheme['accent'],
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Stability Score', color='white', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add score labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', color='white', fontweight='bold')

        # Alpha parameter distribution
        ax2.set_facecolor(self.color_scheme['background'])
        alphas = [sig.alpha for sig in spectral_signatures.values()]
        ax2.hist(alphas, bins=10, color=self.color_scheme['accent'], alpha=0.7,
                edgecolor='white', linewidth=2)
        ax2.set_title('Alpha Parameter Distribution', color=self.color_scheme['accent'],
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Alpha Value', color='white', fontsize=12)
        ax2.set_ylabel('Frequency', color='white', fontsize=12)
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)

        # Beta parameter vs Health correlation
        ax3.set_facecolor(self.color_scheme['background'])
        betas = [sig.beta for sig in spectral_signatures.values()]
        for i, (species, sig) in enumerate(spectral_signatures.items()):
            ax3.scatter(sig.beta, gls_scores.get(species, 0),
                       s=100, color=self.color_scheme.get(species, '#FFFFFF'),
                       alpha=0.8, edgecolors='white', linewidth=2,
                       label=species)

        ax3.set_title('Beta Parameter vs GLS Score', color=self.color_scheme['accent'],
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Beta Value', color='white', fontsize=12)
        ax3.set_ylabel('GLS Score', color='white', fontsize=12)
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3)
        ax3.legend(facecolor='black', edgecolor='white')

        # Omega resonance pattern
        ax4.set_facecolor(self.color_scheme['background'])
        omega_range = np.linspace(0, 2*np.pi, 100)
        resonance = np.sin(omega_range) ** 2

        ax4.plot(omega_range, resonance, color=self.color_scheme['accent'],
                linewidth=3, label='Resonance Pattern')

        # Mark actual omega values
        for species, sig in spectral_signatures.items():
            omega_resonance = np.sin(sig.omega) ** 2
            ax4.scatter(sig.omega, omega_resonance, s=150,
                       color=self.color_scheme.get(species, '#FFFFFF'),
                       edgecolors='white', linewidth=2, label=species)

        ax4.set_title('Omega Resonance Analysis', color=self.color_scheme['accent'],
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Omega Value', color='white', fontsize=12)
        ax4.set_ylabel('Resonance Factor', color='white', fontsize=12)
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.3)
        ax4.legend(facecolor='black', edgecolor='white')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, facecolor=self.color_scheme['background'],
                       edgecolor='none', dpi=300)

        return fig

    def create_mathematical_evolution_plot(self,
                                         historical_data: List[Dict],
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create mathematical evolution and complexity growth visualization
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.patch.set_facecolor(self.color_scheme['background'])

        if len(historical_data) < 2:
            return fig

        # Extract evolution data
        timestamps = []
        emergence_levels = []
        spectral_coherences = []
        total_populations = []
        active_equations = []

        for entry in historical_data:
            timestamps.append(len(timestamps))  # Use index as time

            analysis = entry.get('analysis', {})
            data = entry.get('data', {})

            emergence_levels.append(analysis.get('emergence_level', 0))

            system_status = data.get('system_status', {})
            if hasattr(system_status, 'spectral_coherence'):
                spectral_coherences.append(system_status.spectral_coherence)
            else:
                spectral_coherences.append(system_status.get('spectral_coherence', 0.5))

            # Calculate total population
            colonies = data.get('colonies', {})
            total_pop = sum(getattr(colony, 'population', 0) for colony in colonies.values())
            total_populations.append(total_pop)

            # Count active mathematical equations
            math_foundation = data.get('mathematical_foundation', {})
            if isinstance(math_foundation, dict):
                active_count = sum(1 for status in math_foundation.values()
                                 if status == 'ACTIVE')
            else:
                active_count = 8  # Default
            active_equations.append(active_count)

        # Emergence level evolution
        ax1.set_facecolor(self.color_scheme['background'])
        ax1.plot(timestamps, emergence_levels, color=self.color_scheme['accent'],
                linewidth=3, marker='o', markersize=6)
        ax1.set_title('Emergence Level Evolution', color=self.color_scheme['accent'],
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Steps', color='white', fontsize=12)
        ax1.set_ylabel('Emergence Level', color='white', fontsize=12)
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)

        # Spectral coherence evolution
        ax2.set_facecolor(self.color_scheme['background'])
        ax2.plot(timestamps, spectral_coherences, color=self.color_scheme['secondary'],
                linewidth=3, marker='s', markersize=6)
        ax2.set_title('Spectral Coherence Evolution', color=self.color_scheme['accent'],
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Steps', color='white', fontsize=12)
        ax2.set_ylabel('Coherence Level', color='white', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)

        # Population complexity growth
        ax3.set_facecolor(self.color_scheme['background'])
        ax3.plot(timestamps, total_populations, color='#FF851B',
                linewidth=3, marker='^', markersize=6)
        ax3.set_title('Population Complexity Growth', color=self.color_scheme['accent'],
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time Steps', color='white', fontsize=12)
        ax3.set_ylabel('Total Population', color='white', fontsize=12)
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3)

        # Mathematical foundation stability
        ax4.set_facecolor(self.color_scheme['background'])
        ax4.plot(timestamps, active_equations, color='#B10DC9',
                linewidth=3, marker='D', markersize=6)
        ax4.set_title('Mathematical Foundation Stability', color=self.color_scheme['accent'],
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time Steps', color='white', fontsize=12)
        ax4.set_ylabel('Active Equations', color='white', fontsize=12)
        ax4.set_ylim(0, 8)
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, facecolor=self.color_scheme['background'],
                       edgecolor='none', dpi=300)

        return fig

    def generate_complete_visualization_suite(self,
                                            output_dir: str = "./gls_visualizations") -> Dict[str, str]:
        """
        Generate complete suite of GLS visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Generate analysis report
        report = self.analyzer.generate_ecosystem_report()

        # Current ecosystem data
        current_data = self.habitat_model.get_complete_status()

        generated_files = {}

        # 1. 4D Habitat Projection
        positions = []  # Would need real position data
        if positions:
            fig1 = self.create_4d_habitat_projection(positions)
            file1 = os.path.join(output_dir, "4d_habitat_projection.png")
            fig1.savefig(file1, facecolor=self.color_scheme['background'], dpi=300)
            generated_files['4d_habitat'] = file1
            plt.close(fig1)

        # 2. Colony Dynamics (if historical data available)
        if len(self.analyzer.analysis_history) > 1:
            fig2 = self.create_colony_dynamics_plot(self.analyzer.analysis_history)
            file2 = os.path.join(output_dir, "colony_dynamics.png")
            fig2.savefig(file2, facecolor=self.color_scheme['background'], dpi=300)
            generated_files['colony_dynamics'] = file2
            plt.close(fig2)

        # 3. Spectral Emergence Map
        fig3 = self.create_spectral_emergence_map(
            report['spectral_coherence_analysis'],
            report['emergence_level']
        )
        file3 = os.path.join(output_dir, "spectral_emergence_map.png")
        fig3.savefig(file3, facecolor=self.color_scheme['background'], dpi=300)
        generated_files['spectral_emergence'] = file3
        plt.close(fig3)

        # 4. Photonic Network Graph
        fig4 = self.create_photonic_network_graph(
            current_data['photonic_ecosystem'],
            current_data['colonies']
        )
        file4 = os.path.join(output_dir, "photonic_network.png")
        fig4.savefig(file4, facecolor=self.color_scheme['background'], dpi=300)
        generated_files['photonic_network'] = file4
        plt.close(fig4)

        # 5. GLS Interaction Field
        spectral_sigs = {species: colony.spectral_signature
                        for species, colony in current_data['colonies'].items()}
        fig5 = self.create_gls_interaction_field(
            report['gls_stability_scores'],
            spectral_sigs
        )
        file5 = os.path.join(output_dir, "gls_interaction_field.png")
        fig5.savefig(file5, facecolor=self.color_scheme['background'], dpi=300)
        generated_files['gls_interaction'] = file5
        plt.close(fig5)

        # 6. Mathematical Evolution (if historical data available)
        if len(self.analyzer.analysis_history) > 1:
            fig6 = self.create_mathematical_evolution_plot(self.analyzer.analysis_history)
            file6 = os.path.join(output_dir, "mathematical_evolution.png")
            fig6.savefig(file6, facecolor=self.color_scheme['background'], dpi=300)
            generated_files['mathematical_evolution'] = file6
            plt.close(fig6)

        # Save analysis report
        report_file = os.path.join(output_dir, "gls_analysis_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        generated_files['analysis_report'] = report_file

        return generated_files