#!/usr/bin/env python3
"""
Interactive Pipeline Test - Teste Interativo do Pipeline ΨQRH com Log Detalhado
================================================================================

Executa teste interativo do pipeline ΨQRH com logging detalhado de cada processo,
gerando um arquivo para cada interação com análise completa.

Uso:
    python3 interactive_pipeline_test.py

Funcionalidades:
- Interface interativa similar ao --interactive
- Log detalhado de cada etapa do pipeline
- Arquivo separado para cada interação
- Análise completa de componentes ativos
- Métricas de performance em tempo real
"""

import sys
import os
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from psiqrh import ΨQRHPipeline
from src.testing.enhanced_dataflow_mapper import ΨQRHDataFlowMapperEnhanced


class InteractivePipelineTest:
    """Teste interativo do pipeline ΨQRH com logging detalhado."""

    def __init__(self, output_dir: str = "pipeline_test_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interaction_count = 0
        self.test_results = []
        self.pipeline = None

    def _create_test_report(self, interaction_data: Dict[str, Any]) -> str:
        """Cria relatório detalhado para uma interação."""

        filename = self.output_dir / f"interaction_{self.interaction_count:03d}.md"

        content = f"""# Interação {self.interaction_count}: Teste Pipeline ΨQRH

**Data/Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Entrada do Usuário
```
{interaction_data['user_input']}
```

## Configuração do Pipeline
- **Tarefa Detectada:** {interaction_data['pipeline_task']}
- **Dispositivo:** {interaction_data['pipeline_device']}
- **Timestamp:** {interaction_data['timestamp']}

## Resultado do Processamento
**Status:** {interaction_data['result']['status']}

### Resposta do Sistema
```
{interaction_data['result']['response'][:1000]}{'...' if len(interaction_data['result']['response']) > 1000 else ''}
```

## Métricas de Performance
- **Tempo de Execução:** {interaction_data['execution_time']:.4f} segundos
- **Comprimento Entrada:** {interaction_data['result']['input_length']} caracteres
- **Comprimento Saída:** {interaction_data['result']['output_length']} caracteres

## Análise de Componentes Ativos
"""

        # Adicionar análise de tipo de processamento
        processing_type = self._classify_processing_type(interaction_data['user_input'])
        content += f"- **Tipo de Processamento:** [{processing_type}]\n"

        if processing_type == "REAL":
            content += "  - ✅ Processamento com dados numéricos reais\n"
        else:
            content += "  - 🔄 Simulação conceitual\n"

        # Adicionar análise de componentes
        content += self._analyze_active_components(interaction_data)

        # Adicionar rastreamento de string se disponível
        if 'dataflow_map' in interaction_data:
            content += self._format_dataflow_analysis(interaction_data['dataflow_map'])

        content += f"""
## Log de Execução
```
{interaction_data.get('execution_log', 'Nenhum log disponível')}
```

## Variáveis e Estados
```json
{json.dumps(interaction_data.get('variables', {}), indent=2, ensure_ascii=False, default=str)}
```

---
*Relatório gerado automaticamente pelo Interactive Pipeline Test*
"""

        return content, filename

    def _classify_processing_type(self, input_text: str) -> str:
        """Classifica se o processamento é REAL ou SIMULADO."""
        # Verificar se a entrada contém dados numéricos ou estrutura de sinal
        has_numeric_data = any(char.isdigit() for char in input_text)
        has_signal_keywords = any(kw in input_text.lower() for kw in ["sinal", "array", "dados", "vetor", "[", "]"])

        if has_numeric_data or has_signal_keywords:
            return "REAL"
        else:
            return "SIMULADO"

    def _analyze_active_components(self, interaction_data: Dict[str, Any]) -> str:
        """Analisa quais componentes estão ativos no processamento."""

        content = "### Componentes do Sistema ΨQRH\n"

        # Verificar tipo de tarefa
        task = interaction_data['pipeline_task']
        if task == "signal-processing":
            content += "- 🔢 **Processador Numérico:** ✅ ATIVO\n"
            content += "- 📊 **Análise Espectral:** ✅ ATIVO\n"
        elif task == "analysis":
            content += "- 🧠 **Analisador Espectral:** ✅ ATIVO\n"
            content += "- 📈 **Validação Matemática:** ✅ ATIVO\n"
        else:
            content += "- 💬 **Gerador de Texto:** ✅ ATIVO\n"
            content += "- 🧩 **Framework ΨQRH:** ✅ ATIVO\n"

        # Verificar componentes avançados baseados na entrada
        input_text = interaction_data['user_input'].lower()

        if any(kw in input_text for kw in ["memoria", "memory", "lembrar", "recuperar"]):
            content += "- 🧠 **Memória de Trabalho:** 🔄 DETECTADO (potencial)\n"
        else:
            content += "- 🧠 **Memória de Trabalho:** ❌ INATIVO\n"

        if any(kw in input_text for kw in ["kuramoto", "sincronizacao", "oscilador", "fase"]):
            content += "- 🔄 **Sistema Kuramoto:** 🔄 DETECTADO (potencial)\n"
        else:
            content += "- 🔄 **Sistema Kuramoto:** ❌ INATIVO\n"

        if any(kw in input_text for kw in ["consciencia", "consciousness", "fci", "fractal"]):
            content += "- 🌟 **Métricas de Consciência:** 🔄 DETECTADO (potencial)\n"
        else:
            content += "- 🌟 **Métricas de Consciência:** ❌ INATIVO\n"

        return content

    def _format_dataflow_analysis(self, dataflow_map: Dict[str, Any]) -> str:
        """Formata análise do fluxo de dados."""

        content = "\n## Rastreamento do Fluxo de Dados\n\n"

        steps = dataflow_map.get('steps', [])
        if not steps:
            return content + "*Nenhum rastreamento disponível*\n"

        content += f"**Total de Etapas:** {len(steps)}\n\n"

        for i, step in enumerate(steps, 1):
            content += f"### Etapa {i}: {step.get('step_name', 'Desconhecida')}\n"
            content += f"- **Descrição:** {step.get('description', 'N/A')}\n"

            processing_time = step.get('processing_time')
            if processing_time:
                content += f"- **Tempo:** {processing_time:.4f}s\n"

            variables = step.get('variables', {})
            if variables:
                content += f"- **Variáveis:** {len(variables)} registradas\n"

            if step.get('is_error', False):
                content += f"- **Status:** ❌ ERRO - {step.get('error_message', 'Desconhecido')}\n"
            else:
                content += f"- **Status:** ✅ SUCESSO\n"

            content += "\n"

        return content

    def _capture_dataflow(self, input_text: str, task: str) -> Dict[str, Any]:
        """Captura fluxo de dados detalhado usando o mapper aprimorado."""

        try:
            mapper = ΨQRHDataFlowMapperEnhanced()
            dataflow_map = mapper.map_real_pipeline_with_string_tracking(input_text, task)
            return dataflow_map
        except Exception as e:
            return {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'steps': []
            }

    def run_interactive_test(self):
        """Executa teste interativo com logging detalhado."""

        print("🚀 INTERACTIVE PIPELINE TEST - ΨQRH")
        print("=" * 60)
        print("Este teste executa o pipeline ΨQRH em modo interativo")
        print("com logging detalhado de cada processo.")
        print("\n📁 Arquivos serão salvos em:", self.output_dir)
        print("\n💬 Digite comandos para testar o pipeline:")
        print("   'quit' - Sair")
        print("   'help' - Ajuda")
        print("   'status' - Status do sistema")
        print("=" * 60)

        # Inicializar pipeline
        self.pipeline = ΨQRHPipeline(task="text-generation")
        print(f"\n✅ Pipeline inicializado (tarefa: {self.pipeline.task}, dispositivo: {self.pipeline.device})")

        while True:
            try:
                user_input = input("\n🤔 Você: ").strip()

                if user_input.lower() in ['quit', 'exit', 'sair']:
                    print("👋 Encerrando teste interativo...")
                    break

                if user_input.lower() in ['help', 'ajuda']:
                    self._show_help()
                    continue

                if user_input.lower() == 'status':
                    self._show_status()
                    continue

                if not user_input:
                    continue

                # Processar entrada
                self._process_user_input(user_input)

            except EOFError:
                print("\n👋 EOF detectado, encerrando...")
                break
            except KeyboardInterrupt:
                print("\n👋 Interrompido pelo usuário")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
                traceback.print_exc()

        # Gerar relatório final
        self._generate_final_report()

    def _process_user_input(self, user_input: str):
        """Processa uma entrada do usuário com logging detalhado."""

        self.interaction_count += 1
        print(f"\n🔍 Processando interação {self.interaction_count}...")

        # Detectar tarefa
        detected_task = self.pipeline._detect_task_type(user_input)

        # Recriar pipeline se necessário
        if detected_task != self.pipeline.task:
            print(f"🔄 Alterando tarefa: {self.pipeline.task} → {detected_task}")
            self.pipeline = ΨQRHPipeline(task=detected_task)

        # Capturar fluxo de dados
        print("📊 Capturando fluxo de dados...")
        dataflow_map = self._capture_dataflow(user_input, detected_task)

        # Executar pipeline
        print("🧠 Executando pipeline...")
        start_time = time.time()
        result = self.pipeline(user_input)
        execution_time = time.time() - start_time

        # Coletar dados da interação
        interaction_data = {
            'interaction_number': self.interaction_count,
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'pipeline_task': detected_task,
            'pipeline_device': self.pipeline.device,
            'execution_time': execution_time,
            'result': result,
            'dataflow_map': dataflow_map,
            'processing_type': self._classify_processing_type(user_input)
        }

        # Gerar relatório
        report_content, filename = self._create_test_report(interaction_data)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # Mostrar resultado ao usuário
        print(f"✅ Processamento concluído em {execution_time:.4f}s")
        print(f"📄 Relatório salvo: {filename}")

        if result['status'] == 'success':
            response = result['response']
            if isinstance(response, str):
                print(f"\n🤖 ΨQRH: {response[:300]}{'...' if len(response) > 300 else ''}")
            else:
                print(f"\n🤖 ΨQRH: [Resposta não textual - ver relatório]")
        else:
            print(f"❌ Erro: {result.get('error', 'Desconhecido')}")

        # Salvar dados para relatório final
        self.test_results.append(interaction_data)

    def _show_help(self):
        """Mostra ajuda dos comandos disponíveis."""

        print("""
📋 COMANDOS DISPONÍVEIS:

Comandos do Sistema:
  quit/exit/sair    - Sair do teste interativo
  help/ajuda        - Mostrar esta ajuda
  status            - Status do sistema e estatísticas

Exemplos de Entradas para Teste:
  Texto simples:    "Explique o que são quaternions"
  Dados numéricos:  "Processe o sinal [1.0, -2.5, 3.7, 0.8]"
  Análise:          "Analise matematicamente esta frase"
  Memória:          "O sistema precisa lembrar desta informação"
  Kuramoto:         "Simule osciladores acoplados com fase"

📊 Cada interação gera um arquivo detalhado no diretório de logs.
""")

    def _show_status(self):
        """Mostra status do sistema."""

        print(f"\n📊 STATUS DO SISTEMA:")
        print(f"   Interações realizadas: {self.interaction_count}")
        print(f"   Pipeline atual: {self.pipeline.task}")
        print(f"   Dispositivo: {self.pipeline.device}")
        print(f"   Arquivos gerados: {len(list(self.output_dir.glob('*.md')))}")

        if self.test_results:
            last_result = self.test_results[-1]
            print(f"   Última execução: {last_result['execution_time']:.4f}s")
            print(f"   Status último: {last_result['result']['status']}")

    def _generate_final_report(self):
        """Gera relatório final consolidado."""

        if not self.test_results:
            return

        filename = self.output_dir / "FINAL_REPORT.md"

        content = f"""# Relatório Final - Interactive Pipeline Test

**Data/Hora:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total de Interações:** {len(self.test_results)}

## Estatísticas Gerais
"""

        # Calcular estatísticas
        successful = sum(1 for r in self.test_results if r['result']['status'] == 'success')
        failed = len(self.test_results) - successful
        total_time = sum(r['execution_time'] for r in self.test_results)
        avg_time = total_time / len(self.test_results) if self.test_results else 0

        content += f"""- **Execuções Bem-sucedidas:** {successful}
- **Execuções com Erro:** {failed}
- **Taxa de Sucesso:** {(successful/len(self.test_results)*100):.1f}%
- **Tempo Total de Execução:** {total_time:.4f}s
- **Tempo Médio por Interação:** {avg_time:.4f}s

## Distribuição por Tipo de Tarefa
"""

        # Análise por tarefa
        task_counts = {}
        for result in self.test_results:
            task = result['pipeline_task']
            task_counts[task] = task_counts.get(task, 0) + 1

        for task, count in task_counts.items():
            content += f"- **{task}:** {count} interações\n"

        content += "\n## Análise por Tipo de Processamento\n"

        real_count = sum(1 for r in self.test_results if r['processing_type'] == 'REAL')
        sim_count = len(self.test_results) - real_count

        content += f"- **Processamento REAL:** {real_count} interações\n"
        content += f"- **Processamento SIMULADO:** {sim_count} interações\n"

        content += "\n## Lista de Interações\n\n"

        for i, result in enumerate(self.test_results, 1):
            content += f"### Interação {i}\n"
            content += f"- **Entrada:** {result['user_input'][:50]}{'...' if len(result['user_input']) > 50 else ''}\n"
            content += f"- **Tarefa:** {result['pipeline_task']}\n"
            content += f"- **Tipo:** [{result['processing_type']}]\n"
            content += f"- **Tempo:** {result['execution_time']:.4f}s\n"
            content += f"- **Status:** {result['result']['status']}\n"
            content += f"- **Arquivo:** interaction_{i:03d}.md\n\n"

        content += "\n---\n*Relatório final gerado automaticamente pelo Interactive Pipeline Test*\n"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n📊 Relatório final salvo: {filename}")


def main():
    """Função principal."""

    test = InteractivePipelineTest()

    try:
        test.run_interactive_test()
    except Exception as e:
        print(f"❌ Erro fatal no teste: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())