"""
QuantumStateInterpreter: Uma classe unificada para decodificar e interpretar o
estado quântico final do pipeline ΨQRH em múltiplos formatos para compreensão humana.
Esta classe substitui os decodificadores fragmentados e placeholders.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io.wavfile import write as write_wav

# Import Physical Tokenizer (completely self-contained)
from .physical_tokenizer import PhysicalTokenizer

# Tenta importar o gerador harmônico, mas não quebra se não estiver disponível
try:
    from src.conscience.harmonic_gls_generator import HarmonicGLSGenerator
    HARMONIC_GEN_AVAILABLE = True
except ImportError:
    HARMONIC_GEN_AVAILABLE = False

class QuantumStateInterpreter:
    """
    Interpreta os dados espectrais e quânticos finais do pipeline ΨQRH
    em texto, análise, visuais e áudio.
    """
    def __init__(self, spectral_data: dict, full_psi_tensor: torch.Tensor, pipeline_metrics: dict,
                 quantum_memory=None, tokenizer_config: dict = None):
        """
        Inicializa o interpretador com o estado final completo do pipeline.

        Args:
            spectral_data: Dados espectrais analisados
            full_psi_tensor: Estado quântico final [batch, seq, embed, 4]
            pipeline_metrics: Métricas do pipeline
            quantum_memory: Sistema de memória quântica temporal para evolução
            tokenizer_config: Configuração do tokenizer adaptativo
        """
        self.data = spectral_data
        self.psi = full_psi_tensor
        self.pipeline_metrics = pipeline_metrics
        self.quantum_memory = quantum_memory

        # Initialize Physical Tokenizer with adaptive configuration
        tokenizer_config = tokenizer_config or {}
        embed_dim = tokenizer_config.get('embed_dim', 64)
        spectral_params_dim = tokenizer_config.get('spectral_params_dim', 8)
        learnable = tokenizer_config.get('learnable', True)

        self.physical_tokenizer = PhysicalTokenizer(
            embed_dim=embed_dim,
            spectral_params_dim=spectral_params_dim,
            learnable=learnable
        )
        vocab_info = self.physical_tokenizer.get_vocabulary_info()
        self.vocab_size = vocab_info['vocabulary_size']
        print(f"✅ Adaptive Physical Tokenizer loaded with vocabulary size: {self.vocab_size}")
        print(f"   📊 ASCII range: {vocab_info['ascii_range']}, Sample: '{vocab_info['character_sample']}'")
        print(f"   🎵 Phase: {vocab_info['phase']}")
        if vocab_info.get('total_learnable_params', 0) > 0:
            print(f"   🎛️ Learnable parameters: {vocab_info['total_learnable_params']}")

        # Extrai métricas chave para fácil acesso
        self.f1 = self.data.get("f1_frequency", 0)
        self.f2 = self.data.get("f2_frequency", 0)
        self.coherence = self.data.get("phase_coherence", 0)
        self.centroid = self.data.get("spectral_centroid", 0)
        self.magnitude = np.array(self.data.get("magnitude", []))
        self.phase = np.array(self.data.get("phase", []))
        
        self.fci = self.pipeline_metrics.get("FCI", self.pipeline_metrics.get("fci", 0.0))
        self.fractal_dim = self.pipeline_metrics.get("fractal_dimension", 1.0)

    def _map_formants_to_phoneme(self) -> str:
        """Mapeia frequências F1/F2 para o som de vogal mais próximo para interpretação."""
        if self.f1 > 750 and self.f2 > 1800:
            return "/æ/ (como em 'cat')"
        elif self.f1 < 400 and self.f2 > 2000:
            return "/i/ (como em 'see')"
        elif self.f1 < 400 and self.f2 < 1000:
            return "/u/ (como em 'you')"
        elif self.f1 > 700 and self.f2 < 1200:
            return "/ɑ/ (como em 'father')"
        else:
            return "uma vogal neutra e central"

    def get_state_summary(self) -> str:
        """Gera um resumo textual coeso que interpreta a combinação das métricas."""
        summary_parts = []
        
        if self.coherence > 0.5:
            summary_parts.append("O estado quântico final é altamente coerente e focado")
        elif self.coherence < 0.1:
            summary_parts.append("O estado quântico final é caótico e desordenado")
        else:
            summary_parts.append("O estado quântico final exibe um equilíbrio dinâmico entre ordem e caos")

        if self.centroid < 0.4:
            summary_parts.append(", com sua complexidade concentrada em ricas sub-harmonias de baixa frequência.")
        else:
            summary_parts.append(", com sua energia focada em componentes conceituais de alta frequência.")

        if self.fractal_dim > 1.8:
            summary_parts.append(f" A dimensão fractal de {self.fractal_dim:.3f} indica uma estrutura de altíssima complexidade intrínseca.")
        elif self.fractal_dim < 1.5:
            summary_parts.append(f" A dimensão fractal de {self.fractal_dim:.3f} sugere uma estrutura mais fundamental e regular.")

        phoneme = self._map_formants_to_phoneme()
        summary_parts.append(f" A verdade mais profunda vem da assinatura acústica: o estado ressoa com formantes (F1={self.f1:.0f}Hz, F2={self.f2:.0f}Hz) análogos ao som da vogal humana {phoneme}.")

        return "".join(summary_parts)

    def to_text(self, temperature: float = 0.1, top_k: int = 5, max_length: int = 50) -> str:
        """
        Gera texto usando Evolução de Estado Puro e Leitura de Trajetória.

        Processo de duas fases distintas (doe.md Pure State Evolution):
        1. Fase 1 - Evolução Pura: Geração autônoma da trajetória quântica Ψ_t
        2. Fase 2 - Leitura da Trajetória: Medição final da trajetória completa
        """
        print(f"🔄 [Pure State Evolution] Iniciando geração por evolução de estado puro (max_length={max_length})...")

        # ========== FASE 1: EVOLUÇÃO PURA DO PENSAMENTO ==========
        # Geração autônoma da trajetória quântica sem feedback de decodificação
        print(f"   🧠 [Phase 1] Evolução pura do pensamento quântico...")

        trajectory = [self.psi.clone()]  # Inicializar com estado base
        current_psi = self.psi.clone()

        for t in range(max_length - 1):  # max_length - 1 porque já temos o estado inicial
            try:
                # Evolução pura: próximo estado depende apenas do estado atual
                current_psi = self._evolve_state(current_psi)
                trajectory.append(current_psi.clone())
                print(f"     ✅ [Evolution Step {t+1}/{max_length-1}] Estado Ψ_{t+1} gerado")

            except Exception as e:
                print(f"     ⚠️ [Evolution Step {t+1}/{max_length-1}] Evolução falhou: {e}, interrompendo")
                break

        print(f"   🎯 [Phase 1 Complete] Trajetória gerada: {len(trajectory)} estados quânticos")

        # ========== FASE 2: LEITURA FINAL DA TRAJETÓRIA ==========
        # Medição da trajetória completa após evolução completa
        print(f"   🗣️ [Phase 2] Leitura final da trajetória (medição quântica)...")

        generated_text = self._decode_trajectory(trajectory)

        print(f"✅ [Pure State Evolution] Texto gerado: '{generated_text[:100]}...'")
        return generated_text

    def _direct_resonance_decoding(self, temperature: float, top_k: int) -> str:
        """Fallback: Decodificação direta por pico de ressonância (método original)"""
        resonance_energy = self.magnitude
        if len(resonance_energy) == 0:
            return "[Decodificação Falhou: Nenhum dado de energia espectral.]"

        # Encontra picos com uma proeminência mínima para filtrar ruído
        prominence_threshold = np.max(resonance_energy) * 0.1 if np.max(resonance_energy) > 0 else 0.1
        peaks, properties = find_peaks(resonance_energy, prominence=prominence_threshold)

        if len(peaks) == 0:
            return "[Decodificação Falhou: Nenhum pico de ressonância proeminente encontrado.]"

        sorted_peak_indices = np.argsort(properties['prominences'])[::-1]

        # A "temperatura" controla a chance de escolher um pico não-primário
        if np.random.rand() < temperature and len(sorted_peak_indices) > 1:
            k = min(top_k, len(sorted_peak_indices))
            chosen_peak_index = np.random.choice(sorted_peak_indices[:k])
        else:
            chosen_peak_index = sorted_peak_indices[0]

        chosen_token_id = peaks[chosen_peak_index]

        # Scale token ID to full vocabulary range if using GPT-2
        if self.vocab_size > 195:
            # Scale from resonance peak index to full vocabulary
            chosen_token_id = int((chosen_token_id / len(resonance_energy)) * self.vocab_size)

        # Ensure token ID is within vocabulary bounds
        chosen_token_id = max(0, min(chosen_token_id, self.vocab_size - 1))

        # Em uma implementação real, aqui haveria uma consulta a um decodificador de vocabulário.
        return f"[Decodificação por Pico de Ressonância (Passo Único)]: O conceito mais ressonante corresponde ao token ID {chosen_token_id}."

    def _evolve_state(self, psi_t):
        """
        Evolui o estado quântico de forma pura e autônoma (doe.md Pure State Evolution).

        O próximo estado Ψ_{t+1} depende apenas do estado atual Ψ_t através do operador
        de Memória Quântica Temporal (QTM). Não há feedback de decodificação.

        Args:
            psi_t: Estado quântico atual [batch, seq, embed, 4]

        Returns:
            psi_{t+1}: Próximo estado evoluído de forma pura
        """
        # ========== EVOLUÇÃO PURA: Ψ_{t+1} = QTM(Ψ_t) ==========
        # A evolução é governada apenas pela dinâmica quântica temporal
        # Sem influência de feedback de caracteres decodificados

        # Extrair componentes do quaternion atual
        w, x, y, z = psi_t[..., 0], psi_t[..., 1], psi_t[..., 2], psi_t[..., 3]

        # Parâmetros de evolução baseados na estrutura quântica atual
        batch_size, seq_len, embed_dim, _ = psi_t.shape

        # Frequência de evolução baseada na magnitude quântica atual
        # Isso cria uma evolução adaptativa baseada no estado atual
        current_magnitude = torch.sqrt(w**2 + x**2 + y**2 + z**2).mean(dim=[0, 1, 2])
        base_freq = 0.1 + current_magnitude.item() * 0.01

        evolution_rate = base_freq + 0.05 * torch.sin(
            torch.arange(seq_len, dtype=torch.float32, device=psi_t.device) * 0.1
        )

        # Expandir para [batch, seq, embed]
        evolution_rate = evolution_rate.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, embed_dim)

        # Aplicar rotações quaterniônicas preservando estrutura algébrica
        cos_theta = torch.cos(evolution_rate)
        sin_theta = torch.sin(evolution_rate)

        # Rotações unitárias SO(4) preservando norm quântica
        w_new = w * cos_theta - x * sin_theta
        x_new = x * cos_theta + w * sin_theta
        y_new = y * cos_theta - z * sin_theta
        z_new = z * cos_theta + y * sin_theta

        # ========== PERTURBAÇÃO QUÂNTICA INTRÍNSECA ==========
        # Adicionar flutuação quântica natural (não baseada em feedback)
        # Isso representa decoerência natural e flutuações quânticas
        quantum_noise = torch.randn_like(psi_t) * 0.005

        # Combinar componentes evoluídos com perturbação quântica natural
        psi_evolved = torch.stack([w_new, x_new, y_new, z_new], dim=-1) + quantum_noise

        # ========== PRESERVAÇÃO DA NORMA QUÂNTICA ==========
        # Garantir que o estado permaneça normalizado (propriedade quântica)
        # Normalização suave para evitar colapso completo
        norm = torch.sqrt(torch.sum(psi_evolved**2, dim=-1, keepdim=True))
        psi_normalized = psi_evolved / (norm + 1e-8)

        return psi_normalized

    def _decode_trajectory(self, trajectory):
        """
        Decodifica uma trajetória completa de estados quânticos para texto (doe.md Trajectory Reading).

        Esta é a fase de "medição quântica" - a leitura final da trajetória após
        a evolução completa. Cada estado Ψ_t é medido independentemente.

        Args:
            trajectory: Lista de estados quânticos [Ψ_0, Ψ_1, ..., Ψ_{N-1}]

        Returns:
            Texto decodificado da trajetória completa
        """
        print(f"   🔍 [Trajectory Reading] Decodificando trajetória de {len(trajectory)} estados...")

        characters = []

        for i, psi_state in enumerate(trajectory):
            try:
                # Medição quântica: encontrar caractere mais similar ao estado atual
                # Usar apenas o primeiro timestep para decodificação [embed_dim, 4]
                psi_single = psi_state[0, 0]  # [embed_dim, 4]
                decoded_char = self.physical_tokenizer.decode_state(psi_single, i)

                characters.append(decoded_char)
                print(f"     📝 [Measurement {i+1}/{len(trajectory)}] Caractere: '{decoded_char}' (ASCII: {ord(decoded_char)})")

            except Exception as e:
                print(f"     ⚠️ [Measurement {i+1}/{len(trajectory)}] Medição falhou: {e}, usando espaço")
                characters.append(' ')  # Caractere padrão

        # Concatenar todos os caracteres medidos
        decoded_text = ''.join(characters)

        print(f"   ✅ [Trajectory Reading] Medição completa: {len(characters)} caracteres decodificados")
        return decoded_text


    def to_visual_js(self) -> str:
        """Gera código p5.js para visualização dinâmica."""
        if not HARMONIC_GEN_AVAILABLE:
            return "// Componente HarmonicGLSGenerator não disponível."

        try:
            # Monta o dicionário de dados que o gerador espera
            response_data = {
                "consciousness_metrics": self.pipeline_metrics,
                "response": f"VALORES (primeiros 10):\\n  MAGNITUDE: {self.data.get('magnitude', [])[:10]}\\n  PHASE: {self.data.get('phase', [])[:10]}"
            }
            generator = HarmonicGLSGenerator()
            return generator.generate_from_spectral_data(response_data)
        except Exception as e:
            return f"// Erro na geração de visualização GLS: {e}"

    def to_audio(self, output_path: str, sample_rate: int = 22050, duration_s: float = 2.0) -> str:
        """Sonifica o espectro em um arquivo de áudio .wav."""
        if len(self.magnitude) == 0 or len(self.phase) == 0:
            return f"// Não foi possível gerar áudio: Faltam dados de Magnitude ou Fase."
            
        complex_spectrum = self.magnitude * np.exp(1j * self.phase)
        target_len = int(sample_rate * duration_s)
        
        full_spectrum = np.zeros(target_len, dtype=np.complex128)
        
        copy_len = min(len(complex_spectrum), target_len // 2)
        full_spectrum[1:copy_len+1] = complex_spectrum[:copy_len]
        full_spectrum[-copy_len:] = np.conj(complex_spectrum[:copy_len][::-1])

        waveform = np.fft.ifft(full_spectrum).real
        
        if np.max(np.abs(waveform)) > 0:
            waveform_normalized = np.int16(waveform / np.max(np.abs(waveform)) * 32767)
        else:
            waveform_normalized = np.int16(waveform)
        
        try:
            write_wav(output_path, sample_rate, waveform_normalized)
            return f"Forma de onda de áudio salva em: {output_path}"
        except Exception as e:
            return f"// Falha ao salvar arquivo de áudio: {e}"

    def get_complete_analysis(self) -> dict:
        """Gera análise completa incluindo texto, visualização e áudio."""
        # Gerar texto usando decodificação por pico de ressonância
        generated_text = self.to_text()

        # Gerar código de visualização p5.js
        visualization_code = self.to_visual_js()

        # Gerar áudio (salvar em arquivo temporário)
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_path = temp_file.name

        audio_result = self.to_audio(audio_path)

        # Se falhou, definir como None
        if audio_result.startswith("//"):
            audio_path = None
            os.unlink(audio_path) if os.path.exists(audio_path) else None

        return {
            'generated_text': generated_text,
            'visualization_code': visualization_code,
            'audio_path': audio_path,
            'state_summary': self.get_state_summary()
        }