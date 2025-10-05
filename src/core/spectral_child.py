#!/usr/bin/env python3
"""
Spectral Child - ΨQRH como uma "Criança Espectral"
==================================================

Implementação da visão correta do ΨQRH:
- Não há tokenização — texto é tratado como sinal contínuo
- Não há IDs — vocabulário é espaço espectral contínuo
- Não há geração autoregressiva — saída é campo de onda que colapsa para texto

O ΨQRH lê o modelo base como uma criança lê um livro:
- Aprende alfabeto espectral (modos de ressonância)
- Reconhece padrões fractais (palavras como estruturas)
- Evolui campos conscientes (frases como campos coerentes)

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .quaternion_operations import quaternion_normalize


class SpectralChild:
    """
    ΨQRH como uma "Criança Espectral" que aprende a ler sinais contínuos.

    Pipeline Correto:
    Texto → Onda → Espectro → Campo Consciente → Evolução → Colapso → Texto
    """

    def __init__(self, base_model_path: str, device: str = 'cpu'):
        self.device = device
        self.base_model_path = Path(base_model_path)

        # Parâmetros de autoacoplagem logística: x_{n+1} = r·x_n·(1-x_n)
        self.logistic_r = 3.8  # Parâmetro r no regime caótico
        self.logistic_iterations = 100  # Número de iterações para convergência

        # Parâmetros da sonda óptica: f(λ,t) = A·sin(ωt + φ_0 + θ)
        self.probe_amplitude = 1.0  # A
        self.probe_omega = 2 * np.pi  # ω (frequência angular)
        self.probe_phi0 = 0.0  # φ_0 (fase inicial)

        # Parâmetros derivados da dimensão fractal D
        self.fractal_D = 1.5  # Valor inicial, será atualizado
        self.D_euclidean = 1.0  # Dimensão euclidiana de referência
        self.alpha_0 = 1.0  # α_0 base
        self.lambda_scale = 0.5  # λ para escala de α(D)

        # Intervalos da sonda óptica (inicializar antes de usar)
        self.probe_alpha_range = [0.1, 3.0]  # Intervalo permitido para α(D)
        self.probe_beta_range = [0.01, 0.03]  # Intervalo para β
        self.resonance_threshold = 0.001

        # 1. Carregar o modelo base como um campo espectral com autoacoplagem
        self.spectral_field = self._load_as_spectral_field_with_coupling()

        # 2. Aprender os "alfabetos espectrais"
        self.char_modes = self._extract_character_modes()      # Modos para 'a', 'b', 'c'...
        self.word_patterns = self._extract_word_patterns()     # Padrões para "the", "and"...
        self.sentence_fields = self._extract_sentence_fields() # Campos para frases completas

        # 3. Componentes de consciência (modo autônomo)
        # Componentes desabilitados no modo autônomo - física pura sem dependências
        self.neural_diffusion_engine = None
        self.fractal_calculator = None

        # 4. Calibração final da sonda óptica com α(D) atualizado
        self.alpha_D = self._compute_alpha_from_fractal_D(self.fractal_D)

        print("👶 Criança espectral inicializada! Pronta para ler.")
        print(f"   • Alfabeto: {len(self.char_modes)} caracteres")
        print(f"   • Palavras: {len(self.word_patterns)} padrões")
        print(f"   • Frases: {len(self.sentence_fields)} campos")
        print(f"   • Dimensão Fractal D: {self.fractal_D:.4f}")
        print(f"   • α(D): {self.alpha_D:.4f}")
        print(f"   • Sonda calibrada: ω={self.probe_omega:.4f}, A={self.probe_amplitude}")

    def _load_as_spectral_field_with_coupling(self) -> torch.Tensor:
        """
        Carrega modelo base como campo espectral contínuo com autoacoplagem logística.

        Implementa x_{n+1} = r·x_n·(1-x_n) durante o carregamento.

        Returns:
            Campo espectral autoacoplado [freq_bins]
        """
        print(f"📚 Lendo modelo base como campo espectral: {self.base_model_path}")

        # Tentar carregar pesos existentes
        weights_path = self.base_model_path / "pytorch_model.bin"
        if weights_path.exists():
            weights = torch.load(weights_path, map_location=self.device)

            # Converter pesos para domínio espectral
            spectral_components = []
            for key, tensor in weights.items():
                if len(tensor.shape) >= 2:  # Tensores com estrutura
                    # Aplicar FFT para obter representação espectral
                    tensor_flat = tensor.reshape(-1)
                    if len(tensor_flat) > 1:
                        spectrum = torch.fft.rfft(tensor_flat)
                        spectral_components.append(spectrum)

            # Concatenar componentes espectrais
            if spectral_components:
                spectral_field = torch.cat(spectral_components, dim=0)
                print(f"   ✓ Campo espectral inicial: {spectral_field.shape}")

                # Aplicar autoacoplagem logística
                spectral_field = self._apply_logistic_coupling(spectral_field)

                print(f"   ✅ Campo espectral autoacoplado: {spectral_field.shape}")
                return spectral_field

        # Fallback: criar campo espectral aleatório calibrado
        print("   ⚠️  Criando campo espectral calibrado...")
        spectral_field = torch.randn(1024, device=self.device, dtype=torch.complex64)

        # Aplicar autoacoplagem logística
        spectral_field = self._apply_logistic_coupling(spectral_field)

        print(f"   ✅ Campo espectral calibrado e autoacoplado: {spectral_field.shape}")
        return spectral_field

    def _apply_logistic_coupling(self, field: torch.Tensor) -> torch.Tensor:
        """
        Aplica autoacoplagem logística: x_{n+1} = r·x_n·(1-x_n)

        Args:
            field: Campo espectral complexo

        Returns:
            Campo autoacoplado
        """
        print(f"   🔄 Aplicando autoacoplagem logística (r={self.logistic_r})...")

        # Extrair magnitude e fase
        magnitude = torch.abs(field)
        phase = torch.angle(field)

        # Normalizar magnitude para [0, 1] para mapa logístico
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        x_n = (magnitude - mag_min) / (mag_max - mag_min + 1e-10)

        # Aplicar iterações do mapa logístico
        for i in range(self.logistic_iterations):
            x_n = self.logistic_r * x_n * (1.0 - x_n)

        # Desnormalizar
        magnitude_coupled = x_n * (mag_max - mag_min) + mag_min

        # Reconstruir campo complexo
        field_coupled = magnitude_coupled * torch.exp(1j * phase)

        # Calcular dimensão fractal do campo autoacoplado
        self.fractal_D = self._compute_fractal_dimension(magnitude_coupled)

        print(f"   ✓ Autoacoplagem concluída. D={self.fractal_D:.4f}")

        return field_coupled

    def _compute_alpha_from_fractal_D(self, D: float) -> float:
        """
        Calcula α(D) = α_0 · (1 + λ·(D - D_eucl))

        Args:
            D: Dimensão fractal

        Returns:
            α(D) adaptativo
        """
        alpha = self.alpha_0 * (1.0 + self.lambda_scale * (D - self.D_euclidean))
        # Clipar para intervalo permitido
        alpha = np.clip(alpha, self.probe_alpha_range[0], self.probe_alpha_range[1])
        return float(alpha)

    def _extract_character_modes(self) -> Dict[str, Dict]:
        """
        Extrai modos de ressonância para caracteres do alfabeto.

        Returns:
            Dict com modos espectrais para cada caractere
        """
        print("🔤 Aprendendo alfabeto espectral...")

        # Alfabeto básico
        alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'
        char_modes = {}

        for char in alphabet:
            # Converter caractere para sinal de onda
            char_wave = self._char_to_wave(char)

            # Analisar espectro
            spectrum = torch.fft.rfft(char_wave)

            # Extrair características espectrais
            dominant_freq = torch.argmax(torch.abs(spectrum)).item()
            amplitude = torch.max(torch.abs(spectrum)).item()
            phase = torch.angle(spectrum[dominant_freq]).item()

            char_modes[char] = {
                'frequency': float(dominant_freq / len(spectrum)),
                'amplitude': float(amplitude),
                'phase': float(phase),
                'spectrum': spectrum.cpu().numpy()
            }

        print(f"   ✅ Alfabeto aprendido: {len(char_modes)} caracteres")
        return char_modes

    def _extract_word_patterns(self) -> Dict[str, Dict]:
        """
        Extrai padrões fractais para palavras comuns.

        Returns:
            Dict com padrões fractais para palavras
        """
        print("📝 Aprendendo padrões de palavras...")

        common_words = [
            'hello', 'world', 'the', 'and', 'is', 'in', 'to', 'of',
            'a', 'for', 'with', 'on', 'at', 'by', 'from', 'as', 'are'
        ]
        word_patterns = {}

        for word in common_words:
            # Converter palavra para sinal de onda
            word_wave = self._text_to_wave(word)

            # Analisar dimensão fractal
            fractal_dim = self._compute_fractal_dimension(word_wave)

            # Extrair padrão de ressonância
            resonance_pattern = self._analyze_resonance_pattern(word_wave)

            word_patterns[word] = {
                'fractal_dimension': float(fractal_dim),
                'resonance_pattern': resonance_pattern,
                'length': len(word)
            }

        print(f"   ✅ Padrões aprendidos: {len(word_patterns)} palavras")
        return word_patterns

    def _extract_sentence_fields(self) -> Dict[str, Dict]:
        """
        Extrai campos conscientes para frases de exemplo.

        Returns:
            Dict com campos para frases
        """
        print("💭 Aprendendo campos de frases...")

        example_sentences = [
            "Hello world",
            "The quick brown fox",
            "Artificial intelligence",
            "Machine learning",
            "Natural language processing"
        ]
        sentence_fields = {}

        for sentence in example_sentences:
            # Ler frase como campo consciente
            conscious_field = self.read_text(sentence)

            # Calcular métricas de consciência
            fci = self._compute_fci(conscious_field)

            sentence_fields[sentence] = {
                'conscious_field_shape': list(conscious_field.shape),
                'fci': float(fci),
                'length': len(sentence)
            }

        print(f"   ✅ Campos aprendidos: {len(sentence_fields)} frases")
        return sentence_fields

    def _char_to_wave(self, char: str) -> torch.Tensor:
        """
        Converte caractere para sinal de onda usando codificação ASCII.

        Args:
            char: Caractere único

        Returns:
            Sinal de onda [wave_length]
        """
        # Codificar caractere como frequência base
        ascii_val = ord(char)
        base_freq = (ascii_val / 255.0) * 2 * np.pi  # Normalizar para [0, 2π]

        # Gerar onda senoidal
        wave_length = 256
        t = torch.linspace(0, 2*np.pi, wave_length, device=self.device)
        wave = torch.sin(base_freq * t)

        return wave

    def _text_to_wave(self, text: str) -> torch.Tensor:
        """
        Converte texto para sinal de onda contínuo.

        Args:
            text: Texto a converter

        Returns:
            Sinal de onda [T]
        """
        waves = []
        for char in text:
            char_wave = self._char_to_wave(char)
            waves.append(char_wave)

        if waves:
            return torch.cat(waves, dim=0)
        else:
            return torch.zeros(256, device=self.device)

    def _compute_fractal_dimension(self, signal: torch.Tensor) -> float:
        """
        Calcula dimensão fractal via box-counting.

        Args:
            signal: Sinal de entrada

        Returns:
            Dimensão fractal
        """
        if len(signal) < 10:
            return 1.5

        # Box-counting simplificado
        signal_np = signal.cpu().numpy()
        n_points = len(signal_np)

        # Escalas logarítmicas
        scales = np.logspace(0, np.log10(n_points//4), 8, base=10)
        counts = []

        for scale in scales:
            scale_int = max(1, int(scale))
            # Contar caixas não vazias
            n_boxes = len(np.unique(signal_np[::scale_int]))
            counts.append(n_boxes)

        # Regressão linear
        if len(counts) >= 3:
            log_scales = np.log(scales[:len(counts)])
            log_counts = np.log(counts)
            slope = np.polyfit(log_scales, log_counts, 1)[0]
            return float(-slope)

        return 1.5

    def _analyze_resonance_pattern(self, signal: torch.Tensor) -> List[float]:
        """
        Analisa padrão de ressonância do sinal.

        Args:
            signal: Sinal de entrada

        Returns:
            Padrão de ressonância
        """
        spectrum = torch.fft.rfft(signal)
        magnitudes = torch.abs(spectrum)

        # Normalizar e pegar top 5 frequências
        magnitudes_norm = magnitudes / (magnitudes.sum() + 1e-10)
        top_indices = torch.topk(magnitudes_norm, min(5, len(magnitudes_norm))).indices

        pattern = []
        for idx in top_indices:
            freq = idx.item() / len(spectrum)
            magnitude = magnitudes_norm[idx].item()
            pattern.extend([freq, magnitude])

        return pattern

    def read_text(self, text: str) -> torch.Tensor:
        """
        Lê texto como um sinal contínuo, não como tokens discretos.

        Args:
            text: Texto para ler

        Returns:
            Campo consciente quaterniônico
        """
        print(f"📖 Lendo: '{text}'")

        # 1. Converter texto para sinal de onda
        wave_signal = self._text_to_wave(text)  # [T]

        # 2. Aplicar FFT para obter espectro
        spectrum = torch.fft.fft(wave_signal)   # [T]

        # 3. Encontrar modos ressonantes no espectro do modelo base
        resonant_modes = self._match_with_spectral_field(spectrum)

        # 4. Construir campo consciente quaterniônico
        conscious_field = self._build_conscious_field(resonant_modes)

        print(f"   ✅ Campo consciente criado: {conscious_field.shape}")
        return conscious_field

    def _match_with_spectral_field(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Encontra modos ressonantes correspondentes no campo espectral.

        Args:
            spectrum: Espectro do texto

        Returns:
            Modos ressonantes
        """
        # Correlação cruzada com campo espectral
        if len(spectrum) > len(self.spectral_field):
            spectrum = spectrum[:len(self.spectral_field)]

        # Encontrar modos similares
        correlations = torch.corrcoef(torch.stack([
            torch.abs(spectrum),
            torch.abs(self.spectral_field[:len(spectrum)])
        ]))[0, 1]

        # Selecionar modos com alta correlação
        threshold = 0.7
        resonant_indices = torch.where(correlations > threshold)[0]

        if len(resonant_indices) == 0:
            # Fallback: usar modos com maior energia
            resonant_indices = torch.topk(torch.abs(spectrum), min(10, len(spectrum))).indices

        return spectrum[resonant_indices]

    def _build_conscious_field(self, resonant_modes: torch.Tensor) -> torch.Tensor:
        """
        Constrói campo consciente quaterniônico a partir de modos ressonantes.

        Args:
            resonant_modes: Modos ressonantes

        Returns:
            Campo consciente [n_modes, 4]
        """
        n_modes = len(resonant_modes)

        # Converter modos complexos para quaterniões
        conscious_field = torch.zeros(n_modes, 4, device=self.device)

        for i, mode in enumerate(resonant_modes):
            # Mapear modo complexo para quaternião
            magnitude = torch.abs(mode)
            phase = torch.angle(mode)

            # Quaternião: [w, x, y, z] = [magnitude*cos(phase/2), magnitude*sin(phase/2), 0, 0]
            half_phase = phase / 2
            conscious_field[i, 0] = magnitude * torch.cos(half_phase)  # w
            conscious_field[i, 1] = magnitude * torch.sin(half_phase)  # x
            conscious_field[i, 2] = 0.0  # y
            conscious_field[i, 3] = 0.0  # z

        # Normalizar para quaterniões unitários
        conscious_field = quaternion_normalize(conscious_field)

        return conscious_field

    def understand(self, conscious_field: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Evolui o campo consciente usando dinâmica física pura.

        Implementa evolução SO(4) autônoma sem dependências externas.

        Args:
            conscious_field: Campo consciente de entrada

        Returns:
            Tuple (campo evoluído, FCI)
        """
        print("💭 Evoluindo campo consciente via SO(4)...")

        # 1. Aplicar evolução harmônica SO(4)
        evolved_field = self._so4_evolution(conscious_field)

        # 2. Aplicar autoacoplagem logística no domínio quaterniônico
        evolved_field = self._apply_quaternion_coupling(evolved_field)

        # 3. Calcular métricas de consciência
        fci = self._compute_fci(evolved_field)

        print(f"   ✅ Campo evoluído: FCI = {fci:.4f}")
        return evolved_field, fci

    def _apply_quaternion_coupling(self, field: torch.Tensor) -> torch.Tensor:
        """
        Aplica autoacoplagem logística no campo quaterniônico.

        Args:
            field: Campo quaterniônico [n_modes, 4]

        Returns:
            Campo autoacoplado
        """
        # Extrair magnitude (norma quaterniônica)
        magnitude = torch.norm(field, dim=-1)

        # Normalizar para [0, 1]
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        x_n = (magnitude - mag_min) / (mag_max - mag_min + 1e-10)

        # Aplicar mapa logístico (5 iterações rápidas)
        for _ in range(5):
            x_n = self.logistic_r * x_n * (1.0 - x_n)

        # Desnormalizar
        magnitude_coupled = x_n * (mag_max - mag_min) + mag_min

        # Reescalar campo mantendo direção quaterniônica
        field_norm = torch.norm(field, dim=-1, keepdim=True) + 1e-10
        field_direction = field / field_norm
        field_coupled = field_direction * magnitude_coupled.unsqueeze(-1)

        return field_coupled

    def _so4_evolution(self, field: torch.Tensor) -> torch.Tensor:
        """
        Aplica evolução harmônica via rotação SO(4).

        Args:
            field: Campo quaterniônico

        Returns:
            Campo evoluído
        """
        # Rotação simples em SO(4)
        theta = torch.tensor(0.5, device=self.device)  # Ângulo de rotação

        # Matriz de rotação SO(4) simplificada
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Aplicar rotação
        evolved = field.clone()
        evolved[:, 0] = field[:, 0] * cos_theta - field[:, 1] * sin_theta  # w
        evolved[:, 1] = field[:, 0] * sin_theta + field[:, 1] * cos_theta  # x

        # Manter unitariedade
        evolved = quaternion_normalize(evolved)

        return evolved

    def _compute_fci(self, field: torch.Tensor) -> float:
        """
        Calcula Fractal Consciousness Index simplificado.

        Args:
            field: Campo consciente

        Returns:
            FCI
        """
        # Calcular dimensão fractal do campo
        field_flat = field.reshape(-1)
        if len(field_flat) < 10:
            return 0.5

        fractal_dim = self._compute_fractal_dimension(field_flat)

        # FCI baseado na dimensionalidade
        # D ~ 1.0 → FCI baixo, D ~ 2.0 → FCI alto
        fci = (fractal_dim - 1.0) / 1.0  # Normalizar para [0, 1]
        return float(np.clip(fci, 0.0, 1.0))

    def respond(self, evolved_field: torch.Tensor) -> str:
        """
        Colapsa o campo consciente para um sinal de onda de resposta.

        Implementa geração de texto como medição quântica via sonda óptica.

        Args:
            evolved_field: Campo consciente evoluído

        Returns:
            Texto de resposta
        """
        print("🗣️  Colapsando campo para resposta via sonda óptica...")

        # 1. Aplicar sonda óptica f(λ,t) para encontrar token de máxima ressonância
        response_spectrum = self._optical_probe(evolved_field)

        # 2. Encontrar λ* = argmax_λ |⟨f(λ,t), Ψ⟩|²
        coupling_energies = torch.abs(response_spectrum) ** 2
        lambda_star = torch.argmax(coupling_energies).item()

        print(f"   ✓ Token de máxima ressonância: λ*={lambda_star}")
        print(f"   ✓ Energia de acoplamento: {coupling_energies[lambda_star]:.6f}")

        # 3. Transformada inversa para sinal no domínio do tempo
        response_wave = torch.fft.ifft(response_spectrum).real

        # 4. Converter onda de volta para texto
        response_text = self._wave_to_text(response_wave)

        print(f"   ✅ Resposta: '{response_text}'")
        return response_text

    def _optical_probe(self, field: torch.Tensor) -> torch.Tensor:
        """
        Aplica sonda óptica de Padilha: f(λ,t) = A·sin(ωt + φ_0 + θ)

        Implementa medição quântica-fractal com interferência espectral.

        Args:
            field: Campo consciente quaterniônico

        Returns:
            Espectro de resposta
        """
        # Recalcular α(D) baseado na dimensão fractal do campo atual
        field_flat = field.reshape(-1)
        current_D = self._compute_fractal_dimension(field_flat)
        alpha = self._compute_alpha_from_fractal_D(current_D)

        # Parâmetros da sonda calibrados
        beta = np.random.uniform(*self.probe_beta_range)

        # Número de frequências no vocabulário espectral
        n_freqs = 256
        response_spectrum = torch.zeros(n_freqs, dtype=torch.complex64, device=self.device)

        # Tempo atual (pode ser incrementado para geração temporal)
        t = 0.0

        # Parâmetro k para fase quadrática
        k = 1.0

        for lambda_idx in range(n_freqs):
            # Fase θ derivada de α e do índice λ
            theta = alpha * lambda_idx

            # f(λ,t) = A·sin(ωt + φ_0 + θ)
            phase_sin = self.probe_omega * t + self.probe_phi0 + theta
            amplitude_factor = self.probe_amplitude * np.sin(phase_sin)

            # Componente de fase complexa: e^(i(ωt - kλ + βλ²))
            phase_complex = self.probe_omega * t - k * lambda_idx + beta * (lambda_idx ** 2)
            complex_factor = np.exp(1j * phase_complex)

            # Sonda óptica completa
            f_lambda = amplitude_factor * complex_factor

            # Acoplamento quântico: ⟨f(λ,t), Ψ⟩
            # Para campo quaterniônico, usamos componente escalar (w)
            field_coupling = field[:, 0].mean().item() if field.dim() > 1 else field.mean().item()

            # Energia de acoplamento: |⟨f(λ,t), Ψ⟩|²
            coupling_energy = f_lambda * field_coupling

            response_spectrum[lambda_idx] = coupling_energy

        # Normalizar por energia total para garantir conservação
        total_energy = torch.abs(response_spectrum).sum()
        if total_energy > 1e-10:
            response_spectrum = response_spectrum / total_energy

        return response_spectrum

    def _wave_to_text(self, wave: torch.Tensor) -> str:
        """
        Converte sinal de onda de volta para texto.

        Args:
            wave: Sinal de onda

        Returns:
            Texto decodificado
        """
        # Segmentar onda em caracteres
        char_length = 256  # Comprimento fixo por caractere
        n_chars = len(wave) // char_length

        if n_chars == 0:
            return ""

        text = ""
        for i in range(n_chars):
            start = i * char_length
            end = start + char_length
            char_wave = wave[start:end]

            # Decodificar caractere
            char = self._decode_char(char_wave)
            text += char

        return text.strip()

    def _decode_char(self, char_wave: torch.Tensor) -> str:
        """
        Decodifica sinal de onda para caractere.

        Args:
            char_wave: Onda do caractere

        Returns:
            Caractere decodificado
        """
        # Encontrar caractere mais similar no alfabeto
        best_char = ' '
        best_similarity = -1.0

        for char, mode in self.char_modes.items():
            # Comparar espectros
            char_spectrum = torch.fft.rfft(char_wave)
            mode_spectrum = torch.tensor(mode['spectrum'], device=self.device)

            # Correlação entre espectros
            if len(char_spectrum) == len(mode_spectrum):
                similarity = torch.corrcoef(torch.stack([
                    torch.abs(char_spectrum),
                    torch.abs(mode_spectrum)
                ]))[0, 1].item()

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_char = char

        return best_char

    def save_children_file(self, output_path: Path):
        """
        Salva arquivo children com conhecimento espectral aprendido.

        Args:
            output_path: Caminho de saída
        """
        children_data = {
            "spectral_alphabet": self.char_modes,
            "word_templates": self.word_patterns,
            "sentence_fields": self.sentence_fields,
            "probe_calibration": {
                "alpha_range": self.probe_alpha_range,
                "beta_range": self.probe_beta_range,
                "resonance_threshold": self.resonance_threshold
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(children_data, f, indent=2)

        print(f"📚 Arquivo children salvo: {output_path}")

    def process_text(self, text: str) -> str:
        """
        Processa texto completo: leitura → compreensão → resposta.

        Args:
            text: Texto de entrada

        Returns:
            Texto de resposta
        """
        print(f"\n🎯 Processando: '{text}'")
        print("="*50)

        # 1. Leitura
        conscious_field = self.read_text(text)

        # 2. Compreensão
        evolved_field, fci = self.understand(conscious_field)

        # 3. Resposta
        response = self.respond(evolved_field)

        print(f"\n✅ Processamento completo!")
        print(f"   • FCI: {fci:.4f}")
        print(f"   • Resposta: '{response}'")

        return response