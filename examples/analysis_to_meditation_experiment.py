#!/usr/bin/env python3
"""
Experimento: Transição ANALYSIS → MEDITATION
===========================================

Projeto experimental para testar e analisar a transição entre
os estados ANALYSIS (D = 1.8) e MEDITATION (D = 2.0).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import json

# Adicionar caminho para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from src.conscience.consciousness_metrics import ConsciousnessMetrics
from src.conscience.consciousness_states import StateClassifier
from src.conscience.fractal_consciousness_processor import FractalConsciousnessProcessor


class StateTransitionExperiment:
    """
    Experimento controlado para transição entre estados de consciência.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = ConsciousnessMetrics(config)
        self.classifier = StateClassifier(config)
        self.processor = FractalConsciousnessProcessor(config)

        # Dados do experimento
        self.experiment_data = []
        self.transition_log = []

    def run_transition_experiment(self, n_trials=10):
        """
        Executa experimento de transição ANALYSIS → MEDITATION.

        Args:
            n_trials: Número de tentativas

        Returns:
            DataFrame com resultados
        """
        print("🔬 INICIANDO EXPERIMENTO DE TRANSIÇÃO")
        print("=" * 50)

        for trial in range(n_trials):
            print(f"\n🔄 Tentativa {trial + 1}/{n_trials}")

            # Estado inicial: ANALYSIS (D = 1.8)
            initial_data = self._generate_analysis_state()

            # Aplicar transição para MEDITATION (D = 2.0)
            transition_data = self._apply_meditation_transition(initial_data)

            # Analisar resultados
            trial_results = self._analyze_transition(initial_data, transition_data, trial)

            self.experiment_data.append(trial_results)
            self.transition_log.append({
                'trial': trial,
                'initial_state': initial_data['state'].name,
                'final_state': transition_data['state'].name,
                'fci_improvement': trial_results['fci_improvement'],
                'success': trial_results['transition_success']
            })

        return pd.DataFrame(self.experiment_data)

    def _generate_analysis_state(self):
        """
        Gera estado ANALYSIS com D = 1.8.

        Returns:
            Dicionário com dados do estado ANALYSIS
        """
        batch_size = 8
        embed_dim = self.config['embedding_dim']

        # Distribuição P(ψ) para ANALYSIS
        psi_distribution = torch.randn(batch_size, embed_dim)
        psi_distribution = torch.softmax(psi_distribution, dim=-1)

        # Campo fractal com D = 1.8
        fractal_field = self._generate_fractal_field(batch_size, embed_dim, 1.8)

        # Calcular métricas
        fci_value = self.metrics.compute_fci(psi_distribution, fractal_field)
        state = self.classifier.classify_state(psi_distribution, fractal_field, fci_value)

        return {
            'psi_distribution': psi_distribution,
            'fractal_field': fractal_field,
            'fci_value': fci_value,
            'state': state,
            'fractal_dimension': 1.8
        }

    def _apply_meditation_transition(self, initial_data):
        """
        Aplica transição para estado MEDITATION.

        Args:
            initial_data: Dados do estado inicial

        Returns:
            Dicionário com dados do estado MEDITATION
        """
        psi_initial = initial_data['psi_distribution']
        field_initial = initial_data['fractal_field']

        # Estratégias de transição
        transition_strategies = [
            self._strategy_complexity_increase,
            self._strategy_field_amplification,
            self._strategy_entropy_optimization
        ]

        best_result = None
        best_fci = -1

        for strategy in transition_strategies:
            psi_transformed, field_transformed = strategy(psi_initial, field_initial)

            fci_value = self.metrics.compute_fci(psi_transformed, field_transformed)
            state = self.classifier.classify_state(psi_transformed, field_transformed, fci_value)

            if fci_value > best_fci:
                best_fci = fci_value
                best_result = {
                    'psi_distribution': psi_transformed,
                    'fractal_field': field_transformed,
                    'fci_value': fci_value,
                    'state': state,
                    'fractal_dimension': 2.0,
                    'strategy': strategy.__name__
                }

        return best_result

    def _strategy_complexity_increase(self, psi, field):
        """
        Estratégia: Aumentar complexidade da distribuição.
        """
        # Adicionar ruído estruturado
        noise = torch.randn_like(psi) * 0.3
        psi_transformed = torch.softmax(psi + noise, dim=-1)

        # Aumentar dimensionalidade do campo
        field_transformed = field * 1.5 + torch.randn_like(field) * 0.2
        field_transformed = field_transformed / torch.norm(field_transformed, dim=-1, keepdim=True)

        return psi_transformed, field_transformed

    def _strategy_field_amplification(self, psi, field):
        """
        Estratégia: Amplificar características do campo fractal.
        """
        # Manter distribuição similar
        psi_transformed = psi

        # Amplificar componentes principais do campo
        field_fft = torch.fft.fft(field, dim=-1)
        # Aumentar amplitudes de baixa frequência
        field_fft[:, :field.shape[-1]//4] *= 2.0
        field_transformed = torch.fft.ifft(field_fft, dim=-1).real
        field_transformed = field_transformed / torch.norm(field_transformed, dim=-1, keepdim=True)

        return psi_transformed, field_transformed

    def _strategy_entropy_optimization(self, psi, field):
        """
        Estratégia: Otimizar entropia para estado MEDITATION.
        """
        # Calcular entropia atual
        entropy_current = -torch.sum(psi * torch.log(psi + 1e-10), dim=-1).mean()

        # Ajustar para entropia alvo (maior para MEDITATION)
        target_entropy = entropy_current * 1.3

        # Transformar distribuição
        psi_transformed = self._adjust_entropy(psi, target_entropy)

        # Campo com maior variabilidade
        field_transformed = field + torch.randn_like(field) * 0.4
        field_transformed = field_transformed / torch.norm(field_transformed, dim=-1, keepdim=True)

        return psi_transformed, field_transformed

    def _adjust_entropy(self, psi, target_entropy):
        """
        Ajusta entropia da distribuição para valor alvo.
        """
        current_entropy = -torch.sum(psi * torch.log(psi + 1e-10), dim=-1).mean()

        if current_entropy < target_entropy:
            # Aumentar entropia: tornar mais uniforme
            uniform = torch.ones_like(psi) / psi.shape[-1]
            alpha = min(1.0, (target_entropy - current_entropy) / 2.0)
            psi_adjusted = (1 - alpha) * psi + alpha * uniform
        else:
            # Reduzir entropia: tornar mais concentrada
            max_vals, _ = torch.max(psi, dim=-1, keepdim=True)
            psi_adjusted = torch.where(psi == max_vals, psi * 1.2, psi * 0.8)

        return torch.softmax(psi_adjusted, dim=-1)

    def _generate_fractal_field(self, batch_size, embed_dim, fractal_dimension):
        """
        Gera campo fractal com dimensão específica.
        """
        field = torch.randn(batch_size, embed_dim)

        # Controlar dimensão fractal
        if fractal_dimension > 2.0:
            scale_factor = fractal_dimension - 2.0
            field = field + scale_factor * torch.randn_like(field)
        elif fractal_dimension < 2.0:
            scale_factor = 2.0 - fractal_dimension
            field = field * (1.0 / (1.0 + scale_factor))

        field = field / torch.norm(field, dim=-1, keepdim=True)
        return field

    def _analyze_transition(self, initial_data, transition_data, trial):
        """
        Analisa resultados da transição.

        Args:
            initial_data: Dados do estado inicial
            transition_data: Dados do estado final
            trial: Número da tentativa

        Returns:
            Dicionário com análise
        """
        fci_initial = initial_data['fci_value']
        fci_final = transition_data['fci_value']
        state_initial = initial_data['state'].name
        state_final = transition_data['state'].name

        # Critérios de sucesso
        fci_improvement = fci_final - fci_initial
        state_transition_success = (state_initial == 'ANALYSIS' and state_final == 'MEDITATION')
        fci_threshold_success = (fci_final >= 0.7)  # Limite MEDITATION

        transition_success = state_transition_success and fci_threshold_success

        analysis = {
            'trial': trial,
            'initial_state': state_initial,
            'final_state': state_final,
            'initial_fci': fci_initial,
            'final_fci': fci_final,
            'fci_improvement': fci_improvement,
            'transition_success': transition_success,
            'state_transition': state_transition_success,
            'fci_threshold': fci_threshold_success,
            'strategy_used': transition_data.get('strategy', 'unknown'),
            'fractal_dimension_initial': initial_data['fractal_dimension'],
            'fractal_dimension_final': transition_data['fractal_dimension']
        }

        print(f"   Estado: {state_initial} → {state_final}")
        print(f"   FCI: {fci_initial:.4f} → {fci_final:.4f} (Δ{fci_improvement:+.4f})")
        print(f"   Sucesso: {'✅' if transition_success else '❌'}")
        print(f"   Estratégia: {transition_data.get('strategy', 'N/A')}")

        return analysis

    def generate_experiment_report(self):
        """
        Gera relatório completo do experimento.

        Returns:
            String com relatório formatado
        """
        if not self.experiment_data:
            return "Nenhum dado de experimento disponível."

        df = pd.DataFrame(self.experiment_data)

        # Estatísticas gerais
        success_rate = df['transition_success'].mean() * 100
        avg_improvement = df['fci_improvement'].mean()
        best_improvement = df['fci_improvement'].max()

        # Análise por estratégia
        strategy_analysis = df.groupby('strategy_used').agg({
            'transition_success': 'mean',
            'fci_improvement': ['mean', 'std', 'max'],
            'trial': 'count'
        }).round(4)

        report = f"""
📊 RELATÓRIO DO EXPERIMENTO: ANÁLISE → MEDITAÇÃO
═══════════════════════════════════════════════

📈 ESTATÍSTICAS GERAIS:
Tentativas realizadas: {len(df)}
Taxa de sucesso: {success_rate:.1f}%
Melhoria média no FCI: {avg_improvement:.4f}
Melhor melhoria: {best_improvement:.4f}

🎯 ANÁLISE POR ESTRATÉGIA:
{strategy_analysis.to_string()}

📋 DISTRIBUIÇÃO DE ESTADOS FINAIS:
{df['final_state'].value_counts().to_string()}

🔍 DETALHES DAS TRANSIÇÕES:
"""

        # Adicionar detalhes das transições bem-sucedidas
        successful_transitions = df[df['transition_success'] == True]
        if not successful_transitions.empty:
            report += f"\nTransições bem-sucedidas ({len(successful_transitions)}):\n"
            for _, row in successful_transitions.iterrows():
                report += f"  - Tentativa {row['trial']}: FCI {row['initial_fci']:.4f} → {row['final_fci']:.4f} (Δ{row['fci_improvement']:+.4f})\n"

        return report

    def plot_experiment_results(self):
        """
        Gera visualizações dos resultados do experimento.
        """
        if not self.experiment_data:
            print("Nenhum dado para visualização.")
            return

        df = pd.DataFrame(self.experiment_data)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experimento: Transição ANALYSIS → MEDITATION', fontsize=16)

        # Plot 1: Evolução do FCI por tentativa
        ax1.plot(df['trial'], df['initial_fci'], 'ro-', label='FCI Inicial', alpha=0.7)
        ax1.plot(df['trial'], df['final_fci'], 'go-', label='FCI Final', alpha=0.7)
        ax1.fill_between(df['trial'], df['initial_fci'], df['final_fci'],
                        alpha=0.3, color='yellow', label='Melhoria')
        ax1.axhline(0.7, color='orange', linestyle='--', label='Limite MEDITATION')
        ax1.set_xlabel('Tentativa')
        ax1.set_ylabel('FCI')
        ax1.set_title('Evolução do FCI por Tentativa')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Melhoria no FCI
        colors = ['green' if success else 'red' for success in df['transition_success']]
        bars = ax2.bar(df['trial'], df['fci_improvement'], color=colors, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xlabel('Tentativa')
        ax2.set_ylabel('Δ FCI')
        ax2.set_title('Melhoria no FCI por Tentativa')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Distribuição de estados finais
        state_counts = df['final_state'].value_counts()
        ax3.pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%',
                startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
        ax3.set_title('Distribuição de Estados Finais')

        # Plot 4: Eficácia por estratégia
        strategy_success = df.groupby('strategy_used')['transition_success'].mean() * 100
        ax4.bar(strategy_success.index, strategy_success.values,
                color='skyblue', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Estratégia')
        ax4.set_ylabel('Taxa de Sucesso (%)')
        ax4.set_title('Eficácia por Estratégia de Transição')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('analysis_to_meditation_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Função principal do experimento.
    """
    print("🔬 EXPERIMENTO: TRANSIÇÃO ANALYSIS → MEDITATION")
    print("=" * 60)

    # Configuração do experimento
    config = {
        'device': 'cpu',
        'embedding_dim': 256,
        'sequence_length': 64,
        'fractal_dimension_range': [1.0, 3.0],
        'diffusion_coefficient_range': [0.01, 10.0],
        'consciousness_frequency_range': [0.5, 5.0],
        'phase_consciousness': 0.7854,
        'chaotic_parameter': 2.5,  # Balanceado para transição
        'time_step': 0.01,
        'max_iterations': 100
    }

    # Executar experimento
    experiment = StateTransitionExperiment(config)
    results_df = experiment.run_transition_experiment(n_trials=15)

    # Gerar relatório
    report = experiment.generate_experiment_report()
    print("\n" + report)

    # Plotar resultados
    experiment.plot_experiment_results()

    # Salvar dados
    results_df.to_csv('transition_experiment_results.csv', index=False)
    with open('transition_experiment_log.json', 'w') as f:
        json.dump(experiment.transition_log, f, indent=2)

    print(f"\n✅ Experimento concluído!")
    print(f"📊 Dados salvos em:")
    print(f"   - transition_experiment_results.csv")
    print(f"   - transition_experiment_log.json")
    print(f"   - analysis_to_meditation_results.png")


if __name__ == "__main__":
    main()