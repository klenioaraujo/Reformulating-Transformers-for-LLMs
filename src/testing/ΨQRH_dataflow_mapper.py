#!/usr/bin/env python3
"""
ΨQRH DataFlow Mapper - Rastreador do Fluxo de Dados Real
=========================================================

Este script invoca o pipeline ΨQRH real e rastreia as transformações de dados
desde a entrada inicial até a saída final, documentando cada etapa observável.

Fluxo Rastreado:
Texto de Entrada → Inicialização do Pipeline Real → Execução do Pipeline → Resultado Final
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Adicionar path para importar módulos do projeto
# sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from psiqrh import ΨQRHPipeline

class ΨQRHDataFlowMapper:
    """Rastreador do fluxo de dados real do pipeline ΨQRH."""

    def __init__(self):
        self.dataflow_map = {}

    def map_real_pipeline(self, input_text: str, task: str = "text-generation") -> Dict[str, Any]:
        """
        Executa e mapeia o pipeline ΨQRH real.
        """
        print(f"🗺️  RASTREANDO FLUXO DE DADOS REAL DO ΨQRH (Tarefa: {task})")
        print("=" * 60)

        self.dataflow_map = {
            'pipeline_name': 'ΨQRH Real Data Flow Tracer',
            'timestamp': datetime.now().isoformat(),
            'input_text': input_text,
            'steps': []
        }

        # --- Etapa 1: Entrada de Texto ---
        self._track_step(
            'entrada_texto',
            'Captura do texto de entrada do usuário.',
            input_data=input_text,
            output_data=input_text,
            variables={'texto_bruto': input_text}
        )
        current_data = input_text

        # --- Etapa 2: Inicialização do Pipeline ---
        start_time = time.time()
        try:
            pipeline = ΨQRHPipeline(task=task)
            init_time = time.time() - start_time
            
            self._track_step(
                'inicializacao_pipeline',
                'Instanciação e configuração do ΨQRHPipeline real.',
                input_data=current_data,
                output_data=pipeline, # A saída agora é o objeto do pipeline
                processing_time=init_time,
                variables={
                    'task': pipeline.task,
                    'device': pipeline.device,
                    'model_type': type(pipeline.model).__name__
                }
            )
            current_data = (pipeline, input_text) # Passa o pipeline e o texto para a próxima etapa

        except Exception as e:
            print(f"💥 ERRO na inicialização do pipeline: {e}")
            self._track_step(
                'inicializacao_pipeline',
                'Falha ao instanciar o ΨQRHPipeline.',
                input_data=current_data,
                is_error=True,
                error_message=str(e)
            )
            return self.dataflow_map

        # --- Etapa 3: Execução do Pipeline (Core) ---
        pipeline_obj, text_input = current_data
        start_time = time.time()
        try:
            result = pipeline_obj(text_input)
            exec_time = time.time() - start_time

            self._track_step(
                'execucao_pipeline',
                'Chamada principal ao pipeline (método __call__), que executa o process_text interno.',
                input_data={'text': text_input},
                output_data=result,
                processing_time=exec_time,
                variables={
                    'status': result.get('status'),
                    'input_length': result.get('input_length'),
                    'output_length': result.get('output_length')
                }
            )
            current_data = result.get('response', '')

        except Exception as e:
            print(f"💥 ERRO na execução do pipeline: {e}")
            self._track_step(
                'execucao_pipeline',
                'Falha ao executar o método __call__ do pipeline.',
                input_data={'text': text_input},
                is_error=True,
                error_message=str(e)
            )
            return self.dataflow_map

        # --- Etapa 4: Resultado Final ---
        self._track_step(
            'resultado_final',
            'O texto final processado e retornado pelo pipeline.',
            input_data=result,
            output_data=current_data,
            variables={'texto_final': current_data}
        )

        print()
        print("=" * 60)
        print("🎯 RASTREAMENTO CONCLUÍDO")
        return self.dataflow_map

    def _track_step(self, name: str, description: str, **kwargs):
        """Adiciona uma etapa ao mapa de fluxo de dados."""
        step_number = len(self.dataflow_map['steps']) + 1
        print(f"[{step_number}/4] {name}...")

        step_data = {
            'step_number': step_number,
            'step_name': name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.dataflow_map['steps'].append(step_data)

    def save_dataflow_map(self, filename: str = "ΨQRH_real_dataflow_map.json"):
        """Salva o mapa de fluxo de dados em arquivo JSON."""
        with open(filename, 'w', encoding='utf-8') as f:
            # Usar um default handler para objetos não serializáveis
            json.dump(self.dataflow_map, f, indent=2, ensure_ascii=False, default=str)

        print()
        print(f"💾 Mapa de rastreamento salvo em: {filename}")
        return filename


def main():
    """Função principal"""
    input_text = "O sistema ΨQRH demonstra eficiência superior em processamento quaternônico"

    mapper = ΨQRHDataFlowMapper()

    try:
        # Executar mapeamento real
        dataflow_map = mapper.map_real_pipeline(input_text)

        # Salvar resultados
        output_file = mapper.save_dataflow_map()

        # Resumo final
        print(f"📊 Etapas rastreadas: {len(dataflow_map['steps'])}")
        print(f"📄 Arquivo gerado: {output_file}")

    except Exception as e:
        print(f"💥 ERRO GERAL NO MAPEADOR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
