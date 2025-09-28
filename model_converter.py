#!/usr/bin/env python3
"""
Conversor de Modelos Matemáticos Open Source para Ψcws

Baixa e converte modelos matemáticos de alta qualidade para o formato Ψcws:
- Modelos pré-treinados (BERT, GPT, etc.)
- Modelos matemáticos especializados
- Arquiteturas inovadoras
"""

import torch
import numpy as np
import json
import yaml
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import hashlib
from datetime import datetime
import os

from src.conscience.conscious_wave_modulator import ConsciousWaveModulator
from src.fractal.needle_fractal_dimension import NeedleFractalDimension

logger = logging.getLogger(__name__)

@dataclass
class ModelSource:
    """Fonte de modelo open source"""
    name: str
    url: str
    format: str  # 'huggingface', 'pytorch', 'tensorflow', 'onnx'
    license: str
    description: str
    parameters: int
    architecture: str

class OpenSourceModelDownloader:
    """Downloader de modelos open source"""

    def __init__(self, cache_dir: str = "downloaded_models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Lista de modelos matemáticos de alta qualidade
        self.mathematical_models = [
            ModelSource(
                name="RoBERTa-base",
                url="roberta-base",
                format="huggingface",
                license="MIT",
                description="Robustly optimized BERT approach",
                parameters=125000000,
                architecture="Transformer"
            ),
            ModelSource(
                name="GPT-2 Small",
                url="gpt2",
                format="huggingface",
                license="MIT",
                description="Small GPT-2 model",
                parameters=124000000,
                architecture="Transformer"
            ),
            ModelSource(
                name="DistilBERT-base",
                url="distilbert-base-uncased",
                format="huggingface",
                license="Apache-2.0",
                description="Distilled version of BERT",
                parameters=66000000,
                architecture="Transformer"
            ),
            ModelSource(
                name="ALBERT-base",
                url="albert-base-v2",
                format="huggingface",
                license="Apache-2.0",
                description="A Lite BERT with parameter sharing",
                parameters=12000000,
                architecture="Transformer"
            ),
            ModelSource(
                name="T5-small",
                url="t5-small",
                format="huggingface",
                license="Apache-2.0",
                description="Text-to-Text Transfer Transformer",
                parameters=60000000,
                architecture="Encoder-Decoder"
            )
        ]

    def download_model(self, model_source: ModelSource) -> Optional[Path]:
        """Baixa modelo da fonte especificada"""
        try:
            model_dir = self.cache_dir / model_source.name
            model_dir.mkdir(exist_ok=True)

            # Verificar se já existe
            if (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists():
                logger.info(f"Modelo {model_source.name} já existe em cache")
                return model_dir

            logger.info(f"Baixando {model_source.name}...")

            if model_source.format == "huggingface":
                return self._download_huggingface_model(model_source, model_dir)
            else:
                logger.warning(f"Formato {model_source.format} não suportado")
                return None

        except Exception as e:
            logger.error(f"Erro ao baixar {model_source.name}: {e}")
            return None

    def _download_huggingface_model(self, model_source: ModelSource, model_dir: Path) -> Path:
        """Baixa modelo do HuggingFace Hub"""
        try:
            # Tentar importar transformers
            try:
                from transformers import AutoModel, AutoTokenizer, AutoConfig
            except ImportError:
                logger.error("Biblioteca transformers não instalada. Instale com: pip install transformers")
                return model_dir

            # Baixar modelo e tokenizer
            logger.info(f"Baixando {model_source.name} do HuggingFace...")

            config = AutoConfig.from_pretrained(model_source.url)
            tokenizer = AutoTokenizer.from_pretrained(model_source.url)
            model = AutoModel.from_pretrained(model_source.url)

            # Salvar localmente
            config.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)

            logger.info(f"Modelo {model_source.name} baixado com sucesso")
            return model_dir

        except Exception as e:
            logger.error(f"Erro ao baixar do HuggingFace: {e}")
            # Criar arquivo de metadados mesmo em caso de erro
            self._create_model_metadata(model_source, model_dir, success=False, error=str(e))
            return model_dir

    def _create_model_metadata(self, model_source: ModelSource, model_dir: Path,
                             success: bool = True, error: str = None):
        """Cria arquivo de metadados do modelo"""
        metadata = {
            'name': model_source.name,
            'source_url': model_source.url,
            'format': model_source.format,
            'license': model_source.license,
            'description': model_source.description,
            'parameters': model_source.parameters,
            'architecture': model_source.architecture,
            'download_timestamp': datetime.now().isoformat(),
            'download_success': success,
            'error_message': error
        }

        with open(model_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def list_available_models(self) -> List[ModelSource]:
        """Lista modelos disponíveis"""
        return self.mathematical_models

    def get_model_info(self, model_name: str) -> Optional[ModelSource]:
        """Obtém informações de um modelo específico"""
        for model in self.mathematical_models:
            if model.name.lower() == model_name.lower():
                return model
        return None

class PsiCWSConverter:
    """Conversor para formato Ψcws"""

    def __init__(self, output_dir: str = "converted_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.wave_modulator = ConsciousWaveModulator()
        self.fractal_analyzer = NeedleFractalDimension()

    def convert_model_to_psicws(self, model_dir: Path, model_source: ModelSource) -> Optional[Path]:
        """Converte modelo para formato Ψcws"""
        try:
            # Verificar se o modelo foi baixado com sucesso
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                logger.error(f"Metadados não encontrados para {model_source.name}")
                return None

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if not metadata.get('download_success', False):
                logger.error(f"Download do modelo {model_source.name} falhou")
                return None

            logger.info(f"Convertendo {model_source.name} para Ψcws...")

            # Carregar modelo
            model, tokenizer, config = self._load_model(model_dir)
            if model is None:
                return None

            # Analisar arquitetura do modelo
            architecture_analysis = self._analyze_model_architecture(model, config)

            # Extrair embeddings/representações
            model_representations = self._extract_model_representations(model, tokenizer)

            # Calcular dimensão fractal
            fractal_dimension = self._calculate_fractal_dimension(model_representations)

            # Gerar parâmetros de onda
            wave_parameters = self._generate_wave_parameters(fractal_dimension, architecture_analysis)

            # Criar arquivo Ψcws
            psicws_file = self._create_psicws_file(
                model_source, metadata, architecture_analysis,
                fractal_dimension, wave_parameters, model_representations
            )

            logger.info(f"Conversão concluída: {psicws_file}")
            return psicws_file

        except Exception as e:
            logger.error(f"Erro na conversão de {model_source.name}: {e}")
            return None

    def _load_model(self, model_dir: Path) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        """Carrega modelo, tokenizer e configuração"""
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig

            config = AutoConfig.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

            # Tentar carregar com diferentes métodos
            try:
                model = AutoModel.from_pretrained(model_dir)
            except:
                # Fallback para carregamento básico
                model = None

            return model, tokenizer, config

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return None, None, None

    def _analyze_model_architecture(self, model: Any, config: Any) -> Dict[str, Any]:
        """Analisa arquitetura do modelo"""
        analysis = {
            'model_type': getattr(config, 'model_type', 'unknown'),
            'hidden_size': getattr(config, 'hidden_size', 0),
            'num_layers': getattr(config, 'num_hidden_layers', 0),
            'num_attention_heads': getattr(config, 'num_attention_heads', 0),
            'intermediate_size': getattr(config, 'intermediate_size', 0),
            'vocab_size': getattr(config, 'vocab_size', 0)
        }

        # Contar parâmetros
        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            analysis['total_parameters'] = total_params

        return analysis

    def _extract_model_representations(self, model: Any, tokenizer: Any) -> Dict[str, Any]:
        """Extrai representações do modelo"""
        representations = {}

        # Texto de exemplo para análise
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Mathematics is the language of the universe.",
            "Artificial intelligence transforms our world."
        ]

        try:
            if model is not None and tokenizer is not None:
                # Tokenizar textos
                encoded = tokenizer(sample_texts, return_tensors='pt', padding=True, truncation=True)

                # Extrair embeddings
                with torch.no_grad():
                    outputs = model(**encoded)

                    # Tentar diferentes atributos de saída
                    if hasattr(outputs, 'last_hidden_state'):
                        embeddings = outputs.last_hidden_state
                    elif hasattr(outputs, 'hidden_states'):
                        embeddings = outputs.hidden_states[-1]  # Última camada
                    else:
                        embeddings = None

                if embeddings is not None:
                    representations['embeddings'] = embeddings.cpu().numpy().tolist()
                    representations['embedding_shape'] = list(embeddings.shape)

        except Exception as e:
            logger.warning(f"Não foi possível extrair representações: {e}")

        return representations

    def _calculate_fractal_dimension(self, representations: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula dimensão fractal das representações"""
        fractal_results = {}

        try:
            if 'embeddings' in representations:
                embeddings = np.array(representations['embeddings'])

                # Calcular dimensão fractal usando diferentes métodos
                fractal_results['box_counting'] = self.fractal_analyzer.calculate_fractal_dimension(embeddings)

                # Análise estatística básica
                fractal_results['statistics'] = {
                    'mean': float(np.mean(embeddings)),
                    'std': float(np.std(embeddings)),
                    'min': float(np.min(embeddings)),
                    'max': float(np.max(embeddings))
                }

        except Exception as e:
            logger.warning(f"Erro no cálculo fractal: {e}")
            fractal_results['error'] = str(e)

        return fractal_results

    def _generate_wave_parameters(self, fractal_dimension: Dict[str, Any],
                                architecture_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Gera parâmetros de onda baseados na análise fractal"""

        wave_params = {}

        try:
            # Usar dimensão fractal para derivar parâmetros de onda
            if 'box_counting' in fractal_dimension:
                D = fractal_dimension['box_counting']

                # Mapeamento fractal-onda (baseado no framework ΨQRH)
                wave_params['alpha'] = 1.0 + 0.1 * (D - 1.0)  # α escala com D
                wave_params['beta'] = max(0.01, 1.0 - 0.2 * D)  # β decresce com D
                wave_params['omega'] = architecture_analysis.get('hidden_size', 768) / 1000.0

                # Parâmetros baseados na arquitetura
                wave_params['complexity_factor'] = (
                    architecture_analysis.get('num_layers', 12) *
                    architecture_analysis.get('num_attention_heads', 12) / 100.0
                )

        except Exception as e:
            logger.warning(f"Erro na geração de parâmetros de onda: {e}")
            # Valores padrão
            wave_params = {
                'alpha': 1.0,
                'beta': 0.1,
                'omega': 0.768,
                'complexity_factor': 1.0
            }

        return wave_params

    def _create_psicws_file(self, model_source: ModelSource, metadata: Dict[str, Any],
                          architecture_analysis: Dict[str, Any], fractal_dimension: Dict[str, Any],
                          wave_parameters: Dict[str, Any], representations: Dict[str, Any]) -> Path:
        """Cria arquivo Ψcws final"""

        # Nome do arquivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_source.name.replace(' ', '_').replace('/', '_')
        psicws_filename = f"{safe_name}_{timestamp}.Ψcws"
        psicws_path = self.output_dir / psicws_filename

        # Estrutura do arquivo Ψcws
        psicws_data = {
            'metadata': {
                'conversion_timestamp': datetime.now().isoformat(),
                'original_model': model_source.name,
                'source_url': model_source.url,
                'license': model_source.license,
                'converter_version': 'Ψcws Converter v1.0'
            },
            'architecture_analysis': architecture_analysis,
            'fractal_analysis': fractal_dimension,
            'wave_parameters': wave_parameters,
            'model_representations': representations,
            'quality_metrics': self._calculate_quality_metrics(architecture_analysis, fractal_dimension)
        }

        # Salvar como JSON (formato Ψcws)
        with open(psicws_path, 'w') as f:
            json.dump(psicws_data, f, indent=2)

        return psicws_path

    def _calculate_quality_metrics(self, architecture: Dict[str, Any], fractal: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula métricas de qualidade da conversão"""
        metrics = {}

        try:
            # Score baseado na completude da análise
            completeness_score = 0
            if architecture.get('total_parameters', 0) > 0:
                completeness_score += 25
            if fractal.get('box_counting') is not None:
                completeness_score += 25
            if architecture.get('model_type') != 'unknown':
                completeness_score += 25
            if architecture.get('hidden_size', 0) > 0:
                completeness_score += 25

            metrics['completeness_score'] = completeness_score
            metrics['conversion_quality'] = 'High' if completeness_score >= 75 else 'Medium' if completeness_score >= 50 else 'Low'

            # Complexidade estimada
            if 'total_parameters' in architecture:
                params = architecture['total_parameters']
                if params > 100_000_000:
                    metrics['complexity'] = 'Very High'
                elif params > 50_000_000:
                    metrics['complexity'] = 'High'
                elif params > 10_000_000:
                    metrics['complexity'] = 'Medium'
                else:
                    metrics['complexity'] = 'Low'

        except Exception as e:
            logger.warning(f"Erro no cálculo de métricas: {e}")
            metrics['error'] = str(e)

        return metrics

class ModelConversionPipeline:
    """Pipeline completo de conversão de modelos"""

    def __init__(self):
        self.downloader = OpenSourceModelDownloader()
        self.converter = PsiCWSConverter()

    def convert_selected_models(self, model_names: List[str] = None) -> Dict[str, Any]:
        """Converte modelos selecionados"""
        results = {}

        # Se não especificado, converter todos
        if model_names is None:
            models_to_convert = self.downloader.list_available_models()
        else:
            models_to_convert = []
            for name in model_names:
                model = self.downloader.get_model_info(name)
                if model:
                    models_to_convert.append(model)

        logger.info(f"Convertendo {len(models_to_convert)} modelos...")

        for model_source in models_to_convert:
            logger.info(f"Processando {model_source.name}...")

            # Download
            model_dir = self.downloader.download_model(model_source)
            if model_dir is None:
                results[model_source.name] = {'status': 'download_failed'}
                continue

            # Conversão
            psicws_file = self.converter.convert_model_to_psicws(model_dir, model_source)
            if psicws_file:
                results[model_source.name] = {
                    'status': 'conversion_successful',
                    'psicws_path': str(psicws_file),
                    'model_dir': str(model_dir)
                }
            else:
                results[model_source.name] = {
                    'status': 'conversion_failed',
                    'model_dir': str(model_dir)
                }

        return results

    def generate_conversion_report(self, conversion_results: Dict[str, Any]) -> str:
        """Gera relatório de conversão"""
        report_dir = Path("conversion_reports")
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"model_conversion_report_{timestamp}.json"

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(conversion_results),
                'successful_conversions': sum(1 for r in conversion_results.values() if r['status'] == 'conversion_successful'),
                'failed_conversions': sum(1 for r in conversion_results.values() if r['status'] != 'conversion_successful')
            },
            'detailed_results': conversion_results,
            'summary': self._generate_summary(conversion_results)
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Versão markdown
        self._generate_markdown_report(report, timestamp)

        return str(report_path)

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Gera resumo da conversão"""
        successful = [name for name, result in results.items() if result['status'] == 'conversion_successful']
        failed = [name for name, result in results.items() if result['status'] != 'conversion_successful']

        return {
            'successful_models': successful,
            'failed_models': failed,
            'success_rate': len(successful) / len(results) if results else 0,
            'total_parameters_converted': sum(
                self.downloader.get_model_info(name).parameters
                for name in successful
            ) if successful else 0
        }

    def _generate_markdown_report(self, report: Dict[str, Any], timestamp: str):
        """Gera relatório em markdown"""
        report_dir = Path("conversion_reports")
        md_path = report_dir / f"model_conversion_report_{timestamp}.md"

        with open(md_path, 'w') as f:
            f.write("# Relatório de Conversão de Modelos para Ψcws\n\n")
            f.write(f"**Data**: {report['metadata']['timestamp']}\n\n")

            f.write("## Resumo Executivo\n\n")
            f.write(f"- **Total de Modelos**: {report['metadata']['total_models']}\\n")
            f.write(f"- **Conversões Bem-Sucedidas**: {report['metadata']['successful_conversions']}\\n")
            f.write(f"- **Taxa de Sucesso**: {report['metadata']['successful_conversions']/report['metadata']['total_models']*100:.1f}%\n\n")

            f.write("## Modelos Convertidos com Sucesso\n\n")
            for model_name, result in report['detailed_results'].items():
                if result['status'] == 'conversion_successful':
                    f.write(f"- **{model_name}**: {result['psicws_path']}\\n")

            if report['summary']['failed_models']:
                f.write("\n## Modelos com Falha na Conversão\n\n")
                for model_name in report['summary']['failed_models']:
                    f.write(f"- {model_name}\\n")

def main():
    """Função principal"""
    print("Conversor de Modelos Open Source para Ψcws")
    print("=" * 50)

    # Criar pipeline
    pipeline = ModelConversionPipeline()

    # Listar modelos disponíveis
    available_models = pipeline.downloader.list_available_models()
    print("\nModelos disponíveis para conversão:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model.name} ({model.parameters:,} parâmetros) - {model.license}")

    # Converter modelos selecionados
    print("\nIniciando conversão...")

    # Converter os 3 primeiros modelos para demonstração
    models_to_convert = [model.name for model in available_models[:3]]

    conversion_results = pipeline.convert_selected_models(models_to_convert)

    # Gerar relatório
    report_path = pipeline.generate_conversion_report(conversion_results)

    print(f"\n✅ Conversão concluída!")
    print(f"📊 Relatório em: {report_path}")
    print(f"📁 Modelos convertidos em: {pipeline.converter.output_dir}")
    print(f"💾 Modelos baixados em: {pipeline.downloader.cache_dir}")

    # Estatísticas
    successful = sum(1 for r in conversion_results.values() if r['status'] == 'conversion_successful')
    total = len(conversion_results)
    print(f"📈 Estatísticas: {successful}/{total} conversões bem-sucedidas ({successful/total*100:.1f}%)")

if __name__ == "__main__":
    main()