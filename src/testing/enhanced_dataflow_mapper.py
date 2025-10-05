#!/usr/bin/env python3
"""
Enhanced ΨQRH DataFlow Mapper - Rastreador Aprimorado com String Tracking
========================================================================

Versão aprimorada do dataflow mapper que rastreia o estado real da string
de entrada em cada etapa do processamento, mostrando como ela é transformada
desde a entrada original até a saída final.

Funcionalidades:
- Rastreamento detalhado da string em cada etapa
- Exibição do estado real da string em tempo real
- Mapeamento completo entrada → saída
- Documentação de todas as transformações
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Adicionar path para importar módulos do projeto
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from psiqrh import ΨQRHPipeline

class ΨQRHDataFlowMapperEnhanced:
    """Rastreador aprimorado do fluxo de dados com tracking de string."""

    def __init__(self):
        self.dataflow_map = {}
        self.string_tracking = {
            "original_input": "",
            "final_output": "",
            "transformations": [],
            "statistics": {}
        }

    def _track_string_state(self, step_name: str, string_data: Any, description: str = ""):
        """Rastreia o estado atual da string."""

        # Converter dados para string se necessário
        if isinstance(string_data, dict):
            if 'response' in string_data:
                string_state = str(string_data['response'])
            else:
                string_state = str(string_data)
        elif hasattr(string_data, '__str__'):
            string_state = str(string_data)
        else:
            string_state = repr(string_data)

        # Calcular hash para identificar mudanças
        string_hash = hashlib.md5(string_state.encode('utf-8')).hexdigest()[:8]

        transformation = {
            "step": step_name,
            "string_state": string_state,
            "length": len(string_state),
            "hash": string_hash,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }

        self.string_tracking["transformations"].append(transformation)

        # Exibir estado atual
        print(f"📝 STRING STATE [{step_name}]: '{string_state[:100]}{'...' if len(string_state) > 100 else ''}'")
        print(f"   📏 Comprimento: {len(string_state)} | 🔗 Hash: {string_hash}")

        return string_state

    def map_real_pipeline_with_string_tracking(self, input_text: str, task: str = "text-generation") -> Dict[str, Any]:
        """
        Executa e mapeia o pipeline ΨQRH com rastreamento detalhado da string.
        """
        print(f"🗺️  RASTREANDO FLUXO DE DADOS ΨQRH COM STRING TRACKING (Tarefa: {task})")
        print("=" * 80)

        # Inicializar tracking
        self.string_tracking["original_input"] = input_text

        self.dataflow_map = {
            'pipeline_name': 'ΨQRH Enhanced String Tracker',
            'timestamp': datetime.now().isoformat(),
            'input_text': input_text,
            'steps': [],
            'string_tracking': self.string_tracking
        }

        # --- Etapa 1: Entrada de Texto ---
        print(f"\n🔍 ETAPA 1: CAPTURA DA ENTRADA")
        current_string = self._track_string_state(
            "entrada_original",
            input_text,
            "String de entrada fornecida pelo usuário"
        )

        self._track_step(
            'entrada_texto',
            'Captura e armazenamento do texto de entrada do usuário.',
            input_data=input_text,
            output_data=current_string,
            string_state=current_string,
            variables={'texto_bruto': input_text, 'comprimento_entrada': len(input_text)}
        )

        # --- Etapa 2: Pré-processamento da String ---
        print(f"\n🔍 ETAPA 2: PRÉ-PROCESSAMENTO")
        # Simular pré-processamento (trim, normalização, etc.)
        preprocessed_string = input_text.strip()
        current_string = self._track_string_state(
            "preprocessamento",
            preprocessed_string,
            "String após pré-processamento (trim, normalização)"
        )

        self._track_step(
            'preprocessamento_string',
            'Pré-processamento da string de entrada (limpeza, normalização).',
            input_data=input_text,
            output_data=preprocessed_string,
            string_state=current_string,
            variables={
                'string_original': input_text,
                'string_processada': preprocessed_string,
                'mudancas': len(input_text) != len(preprocessed_string)
            }
        )

        # --- Etapa 3: Inicialização do Pipeline ---
        print(f"\n🔍 ETAPA 3: INICIALIZAÇÃO DO PIPELINE")
        start_time = time.time()
        try:
            pipeline = ΨQRHPipeline(task=task)
            init_time = time.time() - start_time

            # String permanece a mesma, mas documenta o contexto
            current_string = self._track_string_state(
                "pipeline_inicializado",
                preprocessed_string,
                "String mantida durante inicialização do pipeline"
            )

            self._track_step(
                'inicializacao_pipeline',
                'Instanciação e configuração do ΨQRHPipeline real.',
                input_data=preprocessed_string,
                output_data=pipeline,
                string_state=current_string,
                processing_time=init_time,
                variables={
                    'task': pipeline.task,
                    'device': pipeline.device,
                    'model_type': type(pipeline.model).__name__,
                    'string_mantida': preprocessed_string
                }
            )

        except Exception as e:
            print(f"💥 ERRO na inicialização do pipeline: {e}")
            self._track_step(
                'inicializacao_pipeline',
                'Falha ao instanciar o ΨQRHPipeline.',
                input_data=preprocessed_string,
                string_state=current_string,
                is_error=True,
                error_message=str(e)
            )
            return self._finalize_tracking()

        # --- Etapa 4: Entrada no Pipeline ---
        print(f"\n🔍 ETAPA 4: ENTRADA NO PIPELINE")
        current_string = self._track_string_state(
            "entrada_pipeline",
            preprocessed_string,
            "String sendo enviada para processamento no pipeline"
        )

        self._track_step(
            'entrada_no_pipeline',
            'String sendo enviada para o método principal do pipeline.',
            input_data=preprocessed_string,
            output_data=preprocessed_string,
            string_state=current_string,
            variables={
                'input_para_pipeline': preprocessed_string,
                'pronto_para_processamento': True
            }
        )

        # --- Etapa 5: Processamento Interno ---
        print(f"\n🔍 ETAPA 5: PROCESSAMENTO INTERNO")
        start_time = time.time()
        try:
            # Capturar resultado do pipeline
            result = pipeline(preprocessed_string)
            exec_time = time.time() - start_time

            # Extrair string de resposta
            if isinstance(result, dict) and 'response' in result:
                response_string = result['response']
            else:
                response_string = str(result)

            current_string = self._track_string_state(
                "processamento_completo",
                response_string,
                "String após processamento completo pelo pipeline ΨQRH"
            )

            self._track_step(
                'processamento_interno',
                'Execução do processamento interno do pipeline (transformações ΨQRH).',
                input_data=preprocessed_string,
                output_data=result,
                string_state=current_string,
                processing_time=exec_time,
                variables={
                    'status': result.get('status') if isinstance(result, dict) else 'success',
                    'input_length': len(preprocessed_string),
                    'output_length': len(response_string),
                    'string_entrada': preprocessed_string,
                    'string_saida': response_string
                }
            )

        except Exception as e:
            print(f"💥 ERRO no processamento interno: {e}")
            self._track_step(
                'processamento_interno',
                'Falha durante o processamento interno do pipeline.',
                input_data=preprocessed_string,
                string_state=current_string,
                is_error=True,
                error_message=str(e)
            )
            return self._finalize_tracking()

        # --- Etapa 6: Pós-processamento da Saída ---
        print(f"\n🔍 ETAPA 6: PÓS-PROCESSAMENTO DA SAÍDA")
        # Simular pós-processamento da saída
        final_output = str(response_string).strip() if response_string else ""

        current_string = self._track_string_state(
            "pos_processamento",
            final_output,
            "String final após pós-processamento da saída"
        )

        self._track_step(
            'pos_processamento_saida',
            'Pós-processamento e formatação da string de saída.',
            input_data=response_string,
            output_data=final_output,
            string_state=current_string,
            variables={
                'string_bruta': response_string,
                'string_final': final_output,
                'pos_processamento_aplicado': response_string != final_output
            }
        )

        # --- Etapa 7: Resultado Final ---
        print(f"\n🔍 ETAPA 7: RESULTADO FINAL")
        self.string_tracking["final_output"] = final_output

        current_string = self._track_string_state(
            "resultado_final",
            final_output,
            "String final entregue ao usuário"
        )

        self._track_step(
            'resultado_final',
            'String final processada e pronta para entrega ao usuário.',
            input_data=final_output,
            output_data=final_output,
            string_state=current_string,
            variables={
                'texto_final': final_output,
                'comprimento_final': len(final_output),
                'transformacao_completa': True
            }
        )

        return self._finalize_tracking()

    def _finalize_tracking(self) -> Dict[str, Any]:
        """Finaliza o tracking e calcula estatísticas."""

        # Calcular estatísticas
        original = self.string_tracking["original_input"]
        final = self.string_tracking["final_output"]

        self.string_tracking["statistics"] = {
            "total_transformations": len(self.string_tracking["transformations"]),
            "input_length": len(original),
            "output_length": len(final),
            "length_diff": len(final) - len(original),
            "transformation_ratio": len(final) / len(original) if len(original) > 0 else 0
        }

        # Finalizar mapa de dados
        self.dataflow_map["string_tracking"] = self.string_tracking

        print()
        print("=" * 80)
        print("🎯 RASTREAMENTO DE STRING CONCLUÍDO")
        print(f"📊 Transformações registradas: {len(self.string_tracking['transformations'])}")
        print(f"📏 Entrada: {len(original)} → Saída: {len(final)} caracteres")

        return self.dataflow_map

    def _track_step(self, name: str, description: str, **kwargs):
        """Adiciona uma etapa ao mapa de fluxo de dados."""
        step_number = len(self.dataflow_map['steps']) + 1
        print(f"[{step_number}/7] {name}...")

        step_data = {
            'step_number': step_number,
            'step_name': name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.dataflow_map['steps'].append(step_data)

    def save_dataflow_map(self, filename: str = "ΨQRH_enhanced_dataflow_map.json"):
        """Salva o mapa de fluxo de dados em arquivo JSON."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.dataflow_map, f, indent=2, ensure_ascii=False, default=str)

        print()
        print(f"💾 Mapa de rastreamento aprimorado salvo em: {filename}")
        return filename


def main():
    """Função principal para teste do mapper aprimorado."""
    input_text = "O sistema ΨQRH demonstra eficiência superior em processamento quaternônico"

    mapper = ΨQRHDataFlowMapperEnhanced()

    try:
        # Executar mapeamento aprimorado
        dataflow_map = mapper.map_real_pipeline_with_string_tracking(input_text)

        # Salvar resultados
        output_file = mapper.save_dataflow_map()

        # Resumo final
        string_stats = dataflow_map['string_tracking']['statistics']
        print(f"📊 Transformações: {string_stats['total_transformations']}")
        print(f"📏 {string_stats['input_length']} → {string_stats['output_length']} caracteres")
        print(f"📄 Arquivo gerado: {output_file}")

    except Exception as e:
        print(f"💥 ERRO GERAL NO MAPPER APRIMORADO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()