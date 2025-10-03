import torch
import yaml
import sys
import os
from typing import Optional, Dict, Any
from .quaternion_operations import QuaternionOperations
from ..fractal.spectral_filter import SpectralFilter
from .qrh_layer import QRHLayer, QRHConfig
from .gate_controller import GateController
from .negentropy_transformer_block import NegentropyTransformerBlock

class QRHFactory:
    def __init__(self, config_path: str = "configs/qrh_config.yaml"):
        """Inicializa o factory carregando configurações do arquivo YAML"""
        self.config_path = config_path
        self.yaml_config = self._load_config()

        # Configuração da QRH Layer
        qrh_config = self.yaml_config.get('qrh_layer', {})
        self.config = QRHConfig(
            embed_dim=qrh_config.get('embed_dim', 64),
            alpha=qrh_config.get('alpha', 1.0),
            use_learned_rotation=qrh_config.get('use_learned_rotation', True)
        )

        # Configuração do Consciousness Processor
        self.consciousness_config = self.yaml_config.get('consciousness_processor', {})

        # Configuração do modelo Ψcws
        self.psicws_config = self.yaml_config.get('psicws_model', {})

        self.qrh_layer = None
        self.enhanced_processor = None  # Enhanced processor for optimized quaternion processing
        self.consciousness_processor = None  # Fractal consciousness layer for ΨQRH integration
        self.consciousness_results = None  # Store consciousness results for GLS generation

    def _load_config(self) -> dict:
        """Carrega configurações do arquivo YAML"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                print(f"⚠️  Arquivo de configuração {self.config_path} não encontrado, usando padrões")
                return {}
        except Exception as e:
            print(f"⚠️  Erro ao carregar configuração: {e}, usando padrões")
            return {}

    def process_text(self, text: str, device: str = "cpu") -> str:
        """ Processa texto através do Enhanced Pipeline: Texto → α adaptativo → Quaterniôn → FFT → Análise ΨQRH-PROMPT-ENGINE: Integração do EnhancedQRHProcessor mantendo compatibilidade """
        # Inicializar Enhanced Processor e Consciousness Layer se necessário
        if self.enhanced_processor is None or self.consciousness_processor is None:
            try:
                from .enhanced_qrh_processor import EnhancedQRHProcessor
                from ..conscience import create_consciousness_processor

                self.enhanced_processor = EnhancedQRHProcessor(embed_dim=self.config.embed_dim, device=device)

                # Usar configurações do YAML para o consciousness processor
                consciousness_config = self.consciousness_config.copy()
                consciousness_config['device'] = device  # Override device

                # Obter configuração de metrics
                metrics_config = self.yaml_config.get('consciousness_metrics', {})

                self.consciousness_processor = create_consciousness_processor(consciousness_config, metrics_config)
                print("Enhanced QRH Processor integrado com α adaptativo")
                print("Fractal Consciousness Layer integrada para análise ΨQRH")
            except ImportError as e:
                print(f"Consciousness layer not available: {e}")
        # Fallback para pipeline enhanced sem consciência
        if self.enhanced_processor is None:
            return self._process_text_original(text, device)

        # Pipeline Enhanced com Consciousness Layer
        try:
            # 1. Processamento Enhanced QRH
            enhanced_result = self.enhanced_processor.process_text(text, use_cache=True)

            # 2. Análise de Consciência Fractal (se disponível)
            consciousness_analysis = ""
            if self.consciousness_processor is not None:
                try:
                    # Converter texto para tensor compatível com consciousness processor
                    consciousness_input = self._prepare_consciousness_input(text, device)
                    # Processar através da dinâmica consciente
                    consciousness_results = self.consciousness_processor(consciousness_input)
                    # Gerar relatório de consciência - garantir que é string
                    if isinstance(consciousness_results, dict):
                        consciousness_analysis = self.consciousness_processor.get_consciousness_report(consciousness_results)
                    else:
                        consciousness_analysis = str(consciousness_results)
                except Exception as e:
                    print(f"Consciousness processing error: {e}")
                    consciousness_analysis = "Análise de consciência não disponível nesta sessão"

            # 3. Combinar análises Enhanced + Consciousness
            if 'text_analysis' in enhanced_result:
                enhanced_text = enhanced_result['text_analysis']

                # Formatar LAYER1-FRACTAL
                layer1_output = ""
                if 'layer1_fractal' in enhanced_result:
                    layer1_output = self._format_layer1_fractal(enhanced_result['layer1_fractal'])

                # Integrar análise de consciência
                if consciousness_analysis:
                    combined_analysis = f"""{enhanced_text}

{layer1_output}

ANÁLISE DE CONSCIÊNCIA FRACTAL ΨQRH: {consciousness_analysis}
✨ INTEGRAÇÃO ΨQRH-CONSCIOUSNESS: Pipeline completo: Texto → Enhanced α → Quaterniôn → Consciência Fractal → Análise ΨQRH
Estado do sistema: Enhanced Processor + Fractal Consciousness Layer ativos
"""
                    # Return both the analysis text AND the consciousness results for GLS generation
                    # Also ensure the QRHFactory has access to the consciousness processor for GLS generation
                    self.consciousness_results = consciousness_results
                    return {
                        'text_analysis': combined_analysis,
                        'consciousness_results': consciousness_results
                    }
                else:
                    return f"{enhanced_text}\n\n{layer1_output}"
            else:
                # Fallback se estrutura inesperada
                return self._process_text_original(text, device)

        except Exception as e:
            print(f"Enhanced processor error, using fallback: {e}")
            return self._process_text_original(text, device)

    def generate_gls_output(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate GLS output using the consciousness processor.

        Args:
            results: Consciousness results from processing

        Returns:
            Dictionary with GLS output codes
        """
        if self.consciousness_processor is not None:
            try:
                return self.consciousness_processor.generate_gls_output(results)
            except Exception as e:
                return {
                    'processing_code': '',
                    'p5js_code': '',
                    'status': 'error',
                    'message': f'GLS generation error: {str(e)}'
                }
        else:
            return {
                'processing_code': '',
                'p5js_code': '',
                'status': 'error',
                'message': 'Consciousness processor not available for GLS generation'
            }
    def _process_text_original(self, text: str, device: str = "cpu") -> str:
        """Pipeline original como fallback"""
        if self.qrh_layer is None:
            self.qrh_layer = QRHLayer(self.config)
        if device == "cuda":
            self.qrh_layer = self.qrh_layer.cuda()
        elif device == "mps":
            self.qrh_layer = self.qrh_layer.to(torch.device("mps"))

        # Pipeline ΨQRH Original: Texto → Espectro → Quaterniôn → Texto
        # 1. Converter texto para espectro usando SpectralFilter
        spectrum = self.qrh_layer.spectral_filter.text_to_spectrum(text, target_dim=4 * self.config.embed_dim, device=device)
        # 2. Adaptar espectro para formato do QRHLayer
        quaternion_input = self._adapt_spectrum_to_qrh(spectrum, device)
        # 3. Processar com QRHLayer (transformação quaterniônica)
        with torch.no_grad():
            quaternion_output = self.qrh_layer(quaternion_input)
        # 4. Converter saída quaterniônica de volta para espectro
        output_spectrum = self._adapt_qrh_to_spectrum(quaternion_output)
        # 5. Converter espectro para texto usando SpectralFilter
        output_text = self.qrh_layer.spectral_filter.spectrum_to_text(output_spectrum, text)
        return output_text

    def _adapt_spectrum_to_qrh(self, spectrum: torch.Tensor, device: str) -> torch.Tensor:
        """Adapta espectro complexo para formato do QRHLayer"""
        # spectrum shape: [batch, spectrum_dim] (complexo)
        # QRHLayer espera: [batch, seq_len, 4 * embed_dim] (real)
        batch_size = spectrum.shape[0]
        spectrum_dim = spectrum.shape[1]

        # Extrair parte real e imaginária
        real_part = spectrum.real
        imag_part = spectrum.imag

        # Combinar em tensor real expandido
        combined = torch.stack([real_part, imag_part], dim=-1)  # [batch, spectrum_dim, 2]

        # Calcular dimensões para reshape
        seq_len = min(32, spectrum_dim // (2 * self.config.embed_dim))
        if seq_len == 0:
            seq_len = 1
        embed_dim_4 = 4 * self.config.embed_dim

        # Redimensionar e pad se necessário
        flat = combined.flatten(start_dim=1)  # [batch, spectrum_dim * 2]
        target_size = seq_len * embed_dim_4
        if flat.shape[1] > target_size:
            flat = flat[:, :target_size]
        else:
            padding = target_size - flat.shape[1]
            flat = torch.cat([flat, torch.zeros(batch_size, padding, device=device)], dim=1)

        # Reshape para formato QRHLayer
        quaternion_tensor = flat.view(batch_size, seq_len, embed_dim_4)
        return quaternion_tensor

    def _adapt_qrh_to_spectrum(self, quaternion_output: torch.Tensor) -> torch.Tensor:
        """Converte saída do QRHLayer de volta para formato espectral"""
        # quaternion_output shape: [batch, seq_len, 4 * embed_dim]
        batch_size = quaternion_output.shape[0]
        flat = quaternion_output.flatten(start_dim=1)  # [batch, seq_len * 4 * embed_dim]

        # Dividir em metades para reconstituir partes real e imaginária
        mid_point = flat.shape[1] // 2
        real_flat = flat[:, :mid_point]
        imag_flat = flat[:, mid_point:mid_point * 2] if flat.shape[1] >= mid_point * 2 else torch.zeros_like(real_flat)

        # Reconstituir tensor complexo
        spectrum = torch.complex(real_flat, imag_flat)
        return spectrum

    def _prepare_consciousness_input(self, text: str, device: str) -> torch.Tensor:
        """ Prepara entrada de texto para o consciousness processor.
        Converte texto para tensor compatível com as dimensões esperadas pelo
        FractalConsciousnessProcessor [batch, seq_len, embed_dim].

        Args:
            text: Texto de entrada
            device: Dispositivo de computação

        Returns:
            Tensor preparado para consciousness processor [1, seq_len, embed_dim]
        """
        # Obter dimensões da configuração
        embed_dim = self.consciousness_config.get('embedding_dim', 256)
        seq_len = self.consciousness_config.get('sequence_length', 64)

        # Converter texto para sequência numérica
        char_values = []
        total_size = seq_len * embed_dim

        for char in text[:total_size]:  # Limitar ao tamanho máximo
            char_val = ord(char) / 255.0  # Normalizar para [0, 1]
            char_values.append(char_val)

        # Pad ou truncar para dimensão exata
        if len(char_values) < total_size:
            # Pad com valores baseados em características do texto
            text_hash = hash(text) % 1000 / 1000.0
            pad_value = text_hash
            char_values.extend([pad_value] * (total_size - len(char_values)))
        else:
            char_values = char_values[:total_size]

        # Criar tensor com shape correto [batch=1, seq_len, embed_dim]
        consciousness_tensor = torch.tensor(char_values, dtype=torch.float32)
        consciousness_tensor = consciousness_tensor.view(1, seq_len, embed_dim)

        # Mover para dispositivo correto
        if device == "cuda":
            consciousness_tensor = consciousness_tensor.cuda()
        elif device == "mps":
            consciousness_tensor = consciousness_tensor.to(torch.device("mps"))

        return consciousness_tensor

    def _text_to_tensor(self, text: str, device: str) -> torch.Tensor:
        """Converte texto para tensor compatível com QRHLayer"""
        # Criar tensor baseado no texto (embedding simples)
        batch_size, seq_len = 1, min(len(text), 32)
        embed_dim = 4 * self.config.embed_dim  # QRHLayer espera 4 * embed_dim

        # Encoding simples baseado em caracteres
        tensor_data = []
        for i, char in enumerate(text[:seq_len]):
            char_val = ord(char) / 1000.0  # Normalizar
            tensor_data.append([char_val] * embed_dim)

        # Pad se necessário
        while len(tensor_data) < seq_len:
            tensor_data.append([0.0] * embed_dim)

        tensor = torch.tensor([tensor_data], dtype=torch.float32)
        if device == "cuda":
            tensor = tensor.cuda()
        elif device == "mps":
            tensor = tensor.to(torch.device("mps"))
        return tensor

    def _format_layer1_fractal(self, layer1_data: dict) -> str:
        """
        Formata dados do LAYER1-FRACTAL para exibição.

        Mostra os valores espectrais quaterniônicos extraídos.
        """
        stats = layer1_data.get('statistics', {})
        values = layer1_data.get('values', {})

        output = "📊 LAYER1-FRACTAL: VALORES ESPECTRAIS QUATERNIÔNICOS\n"
        output += "=" * 60 + "\n"
        output += f"Input: '{layer1_data.get('input_text', 'N/A')}'\n"
        output += f"Alpha adaptativo: {layer1_data.get('alpha', 0.0):.4f}\n"
        output += f"Shape: {layer1_data.get('shape', [])}\n\n"

        output += "📈 ESTATÍSTICAS:\n"
        output += f"  • Magnitude média: {stats.get('magnitude_mean', 0):.6f}\n"
        output += f"  • Magnitude std: {stats.get('magnitude_std', 0):.6f}\n"
        output += f"  • Magnitude max: {stats.get('magnitude_max', 0):.6f}\n"
        output += f"  • Magnitude min: {stats.get('magnitude_min', 0):.6f}\n"
        output += f"  • Phase média: {stats.get('phase_mean', 0):.6f}\n"
        output += f"  • Real média: {stats.get('real_mean', 0):.6f}\n"
        output += f"  • Imaginário média: {stats.get('imag_mean', 0):.6f}\n"
        output += f"  • Energia total: {stats.get('energy', 0):.6f}\n"
        output += f"  • Sparsity: {stats.get('sparsity', 0):.4f}\n\n"

        output += "🔢 VALORES (primeiros 10):\n"
        for key in ['magnitude', 'phase', 'real', 'imaginary']:
            if key in values:
                vals = values[key][:10]
                output += f"  {key.upper()}: ["
                output += ", ".join(f"{v:.4f}" for v in vals)
                output += "]\n"

        return output

    def _tensor_to_text(self, tensor: torch.Tensor, original_text: str) -> str:
        """Converte tensor de saída para texto"""
        # Análise quaterniônica do tensor de saída
        tensor_stats = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'max': tensor.max().item(),
            'min': tensor.min().item()
        }
        # Gerar resposta baseada no processamento quaterniônico
        response = f"Análise ΨQRH de '{original_text}':\n\n"
        response += f"Processamento quaterniônico completado.\n"
        response += f"Estatísticas espectrais: média={tensor_stats['mean']:.3f}, "
        response += f"desvio={tensor_stats['std']:.3f}\n"
        response += f"Transformação aplicada com {tensor.shape[1]} sequências "
        response += f"e {tensor.shape[2]} dimensões quaterniônicas."
        return response

    @staticmethod
    def create_qrh_layer(config_path: str, device: Optional[str] = None) -> QRHLayer:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = QRHConfig(**config_dict['qrh_layer'])
        if device:
            config.device = device
        layer = QRHLayer(config)
        return layer.to(config.device)

def example_yaml_usage(config_path: str = "configs/qrh_config.yaml"):
    """Example: Loading config from YAML and running the layer."""
    print(f"--- Running YAML-based Usage Example from '{config_path}' ---")
    # 1. Load config from YAML file
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)['qrh_layer']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load or parse '{config_path}'. {e}")
        return

    # 2. Create QRHConfig from dictionary
    config = QRHConfig(**config_dict)

    # 3. Handle device selection
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        device = "cpu"

    # 4. Initialize layer and move to device
    layer = QRHLayer(config).to(device)

    # 5. Create dummy data and run forward pass
    x = torch.randn(2, 32, 4 * config.embed_dim, device=device)
    output = layer(x)
    print(f"Successfully ran layer configured from YAML on device: {output.device}")
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    print("----------------------------------------------------------\n")
    return output

# The main execution block for usage examples and tests
if __name__ == "__main__":
    # You can optionally pass a config file path as a command-line argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/qrh_config.yaml"

    # Test QRHFactory
    print("--- Testing QRHFactory ---")
    layer = QRHFactory.create_qrh_layer(config_file, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"QRHFactory created layer on device: {layer.config.device}")
    example_yaml_usage(config_path=config_file)

 