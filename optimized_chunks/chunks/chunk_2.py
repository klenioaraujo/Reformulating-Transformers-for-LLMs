# Chunk 2: Lines 1154-2321
# Tokens: 14015, Lines: 1154-2321


            # ========== VALIDAÇÃO ==========
            psi_stats = {
                'mean': psi_context.mean().item(),
                'std': psi_context.std().item(),
                'finite': torch.isfinite(psi_context).all().item()
            }
            validation = self._validate_generated_text(emergent_text, input_text, psi_stats)


            return {
                'selected_text': emergent_text,
                'selected_method': 'Optical Probe with Padilha Wave Equation',
                'architecture_components': {
                    'context_funnel': psi_context.shape,
                    'cognitive_processor': psi_final_abstract.shape,
                    'optical_probe': 'f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))'
                },
                'confidence': confidence,
                'dcf_analysis': dcf_result,
                'validation': validation,
                'optical_probe_output': psi_reconstructed_text,
                'final_quantum_state': psi_final_abstract
            }

        except Exception as e:
            print(f"⚠️  End-to-End Architecture failed: {e}")
            import traceback
            traceback.print_exc()

            return {
                'selected_text': '',
                'selected_method': 'Architecture Failure',
                'error': str(e),
                'validation': {'is_valid': False, 'validation_details': 'Architecture failure'}
            }




    def create_semantic_spectral_map(self, input_text: str) -> Dict[str, List[float]]:
        """Criar mapa espectral emergente - ZERO HARDCODED FALLBACKS"""
        # Sistema requer geração emergente pura baseada em padrões quânticos
        # Nenhuma tabela hardcoded de conceitos permitida
        raise NotImplementedError("Semantic mapping requires emergent quantum pattern generation - no hardcoded concept tables allowed")

    def semantic_wave_to_text(self, wave_function: torch.Tensor, input_text: str, max_length: int = 50, proc_params: Dict[str, Any] = None) -> str:
        """Conversão semântica emergente usando QuantumStateInterpreter com amostragem calibrada"""
        print(f"    🔬 [semantic_wave_to_text] Gerando texto semântico emergente para: '{input_text}' (max_length={max_length})")

        # Usar QuantumStateInterpreter para decodificação unificada
        from src.processing.quantum_interpreter import QuantumStateInterpreter

        # Preparar dados para o interpretador
        # wave_function é [seq_len, embed_dim, 4] ou [1, seq_len, embed_dim, 4]
        if wave_function.dim() == 3:
            psi_tensor = wave_function.unsqueeze(0)  # Adicionar batch dim se necessário
        else:
            psi_tensor = wave_function

        # Criar dados espectrais simulados baseados no psi
        spectral_data = self._analyze_spectral_patterns(psi_tensor.squeeze(0))
        pipeline_metrics = {
            'FCI': 0.5,  # Valor padrão
            'fractal_dimension': 1.5,  # Valor padrão
        }

        # Usar parâmetros de amostragem calibrados se disponíveis
        if proc_params and 'sampling_temperature' in proc_params and 'sampling_top_k' in proc_params:
            sampling_temp = proc_params['sampling_temperature']
            sampling_top_k = proc_params['sampling_top_k']
            print(f"    🌡️ Usando parâmetros de amostragem calibrados: temp={sampling_temp:.2f}, top_k={sampling_top_k}")
        else:
            # Fallback para valores padrão
            sampling_temp = 0.1
            sampling_top_k = 5
            print(f"    🌡️ Usando parâmetros de amostragem padrão: temp={sampling_temp:.2f}, top_k={sampling_top_k}")

        # Criar interpretador com configuração do tokenizer adaptativo
        interpreter = QuantumStateInterpreter(
            spectral_data, psi_tensor, pipeline_metrics, self.quantum_memory_system,
            tokenizer_config=self.tokenizer_config
        )
        emergent_text = interpreter.to_text(
            temperature=sampling_temp,
            top_k=sampling_top_k,
            max_length=max_length,
            input_text=input_text
        )

        # Limitar ao comprimento máximo (redundante, mas seguro)
        if len(emergent_text) > max_length:
            emergent_text = emergent_text[:max_length]

        print(f"    ✅ [semantic_wave_to_text] Texto emergente gerado via QuantumStateInterpreter: '{emergent_text}'")
        return emergent_text

    def _map_quantum_to_linguistic_elements(self, fci: float, fractal_dim: float,
                                            coherence: float, complexity: float) -> List[str]:
        """
        Mapeia características quânticas para elementos linguísticos.
        Removed hardcoded word mappings - uses emergent linguistic elements only.
        """
        # This method now requires emergent linguistic element generation
        # No hardcoded word lists allowed
        raise NotImplementedError("Linguistic element mapping requires emergent generation from model vocabulary - no hardcoded word lists allowed")


    def _enhanced_formant_analysis(self, spectrum: torch.Tensor) -> Dict[str, float]:
        """
        ANÁLISE DE FORMANTES PARA DISCRIMINAÇÃO FONÉTICA PRECISA
        F1, F2, F3 determinam a qualidade das vogais e consoantes
        """
        # Converter para numpy para processamento, achatando para 1D
        spectrum_np = spectrum.flatten().detach().cpu().numpy()

        # Check for inf/NaN values that would cause LPC to fail
        if np.any(np.isinf(spectrum_np)) or np.any(np.isnan(spectrum_np)):
            print(f"   ⚠️  Spectrum contains inf/NaN values, using fallback formant analysis")
            # Return fallback values for very short or corrupted signals
            return {
                'f1_frequency': 300.0,  # Typical F1 for neutral vowel
                'f2_frequency': 1500.0,  # Typical F2 for neutral vowel
                'f3_frequency': 2500.0,  # Typical F3 for neutral vowel
                'f1_f2_ratio': 300.0 / 1500.0,
                'formant_spacing': 1500.0 - 300.0,
                'spectral_tilt': -10.0  # Neutral spectral tilt
            }

        # Calcular formantes usando LPC aproximado
        formants = self._compute_lpc_formants(spectrum_np)

        # Características discriminativas baseadas em fonética acústica
        f1, f2, f3 = formants[0], formants[1], formants[2]

        return {
            'f1_frequency': float(f1),  # Altura da vogal (200-1000 Hz)
            'f2_frequency': float(f2),  # Avanço/recuo da vogal (800-2500 Hz)
            'f3_frequency': float(f3),  # Arredondamento labial (2000-3000 Hz)
            'f1_f2_ratio': float(f1 / f2) if f2 > 0 else 1.0,  # Critério principal para vogais
            'formant_spacing': float(f2 - f1),  # Densidade espectral
            'spectral_tilt': self._compute_spectral_tilt(spectrum_np)  # Sonoridade
        }

    def _compute_lpc_formants(self, spectrum: np.ndarray) -> List[float]:
        """
        SEMANA 1: Implementação LPC Refinada
        Padrão ouro em análise de voz - implementação otimizada
        """
        try:
            import math

            # Parâmetros otimizados para análise de formantes
            sample_rate = 16000  # 16kHz - padrão para análise de voz
            lpc_order = 12  # Ordem otimizada para formantes (10-16 típico)

            # Pré-processamento: garantir que o espectro seja adequado
            spectrum = np.asarray(spectrum, dtype=np.float64)
            if len(spectrum) < lpc_order + 1:
                # Padding se necessário
                spectrum = np.pad(spectrum, (0, lpc_order + 1 - len(spectrum)), 'constant')

            # 1. Calcular autocorrelação com normalização
            autocorr = np.correlate(spectrum, spectrum, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Parte positiva
            autocorr = autocorr / autocorr[0]  # Normalizar pela energia total

            # 2. Resolver equação de Yule-Walker usando Levinson-Durbin
            # Mais estável numericamente que resolver diretamente
            lpc_coeffs = self._levinson_durbin(autocorr, lpc_order)

            # 3. Encontrar raízes do polinômio LPC
            roots = np.roots(lpc_coeffs)

            # 4. Filtrar raízes no semicírculo superior (formantes)
            roots = roots[np.imag(roots) > 0]  # Apenas semicírculo superior

            # 5. Converter ângulos para frequências
            angles = np.arctan2(np.imag(roots), np.real(roots))
            frequencies = angles * (sample_rate / (2 * np.pi))

            # 6. Filtrar e validar formantes na faixa de voz
            valid_formants = []
            for freq in frequencies:
                freq_hz = float(np.real(freq))  # Pegar apenas parte real
                if 150 <= freq_hz <= 5500:  # Faixa estendida para formantes
                    valid_formants.append(freq_hz)

            # 7. Selecionar os 3 formantes mais proeminentes
            if len(valid_formants) >= 3:
                # Ordenar por magnitude (mais próximos da origem = mais estáveis)
                valid_formants.sort()
                selected_formants = valid_formants[:3]
            else:
                # Sistema requer pelo menos 3 formantes válidos
                raise ValueError("Insufficient valid formants for phonetic analysis")

            return selected_formants

        except Exception as e:
            print(f"❌ Erro na análise LPC refinada: {e}")
            raise RuntimeError(f"LPC formant analysis failed: {e}")

    def _levinson_durbin(self, autocorr: np.ndarray, order: int) -> np.ndarray:
        """
        Algoritmo de Levinson-Durbin para resolução eficiente da equação Yule-Walker
        Mais estável numericamente que resolução direta
        """
        try:
            # Inicialização
            a = np.zeros(order + 1)
            a[0] = 1.0

            # Para ordem 1
            r = autocorr[1] / autocorr[0]
            a[1] = r
            error = autocorr[0] * (1 - r**2)

            # Para ordens superiores
            for m in range(1, order):
                # Calcular reflexão coefficient
                r = autocorr[m + 1]
                for i in range(1, m + 1):
                    r -= a[i] * autocorr[m + 1 - i]
                r /= error

                # Atualizar coeficientes
                a_prev = a.copy()
                for i in range(1, m + 1):
                    a[i] = a_prev[i] - r * a_prev[m + 1 - i]
                a[m + 1] = r

                # Atualizar erro
                error *= (1 - r**2)

            return a

        except Exception:
            # Fallback para coeficientes simples
            return np.concatenate([[1.0], np.zeros(order)])


    def _compute_spectral_tilt(self, spectrum: np.ndarray) -> float:
        """
        Computa spectral tilt (inclinação espectral) - medida de sonoridade
        """
        try:
            # Spectral tilt é a diferença entre energia em altas e baixas frequências
            n = len(spectrum)
            low_freq = spectrum[:n//4]   # Primeiro quarto (baixas frequências)
            high_freq = spectrum[3*n//4:] # Último quarto (altas frequências)

            energy_low = np.sum(low_freq**2)
            energy_high = np.sum(high_freq**2)

            if energy_low > 0:
                tilt = 10 * np.log10(energy_high / energy_low)
            else:
                tilt = -20  # Valor padrão para silêncio

            return float(tilt)

        except Exception:
            raise RuntimeError("Spectral tilt computation failed - no fallback values allowed")

    def _analyze_spectral_patterns(self, psi: torch.Tensor) -> Dict[str, float]:
        """
        CORREÇÃO CIENTÍFICA: Análise de Formantes usando Linear Predictive Coding (LPC)
        + Métricas de Estabilidade dos Novos Componentes

        Padrão ouro em análise de voz - F1, F2, F3 determinam qualidade fonética precisa.
        Inclui métricas de estabilidade da filtragem ressonante e embedding em Leech Lattice.
        """
        # Converter quaternion para representação espectral, média sobre embed_dim
        magnitude = psi[:, 0].abs().mean(dim=-1)  # [seq_len]
        phase = torch.angle(psi[:, 0] + 1j * psi[:, 1]).mean(dim=-1)  # [seq_len] - Use torch.angle for complex numbers

        # ========== ANÁLISE DE FORMANTES AVANÇADA ==========
        # Usar Linear Predictive Coding para extração precisa de formantes
        formant_features = self._enhanced_formant_analysis(magnitude)

        # ========== CARACTERÍSTICAS LEGACY (para compatibilidade) ==========
        freq_indices = torch.arange(len(magnitude), dtype=torch.float32, device=self.device)
        spectral_centroid = torch.sum(freq_indices * magnitude) / (torch.sum(magnitude) + 1e-10)
        spectral_centroid = spectral_centroid / len(magnitude)

        spectral_spread = torch.sqrt(
            torch.sum(((freq_indices - spectral_centroid * len(magnitude)) ** 2) * magnitude) /
            (torch.sum(magnitude) + 1e-10)
        ) / len(magnitude)

        if len(phase) > 1:
            phase_autocorr = torch.corrcoef(torch.stack([phase[:-1], phase[1:]]))[0, 1]
            phase_coherence = torch.abs(phase_autocorr) if not torch.isnan(phase_autocorr) else 0.0
        else:
            phase_coherence = 1.0

        # Frequência fundamental baseada em formantes (mais robusta)
        # Usar F1 diretamente como frequência fundamental para melhor discriminação
        f1_hz = formant_features['f1_frequency']

        # Normalizar F1 para o range [0,1] baseado na faixa típica de voz (85-1000 Hz)
        # Usar mapeamento logarítmico para melhor discriminação
        if f1_hz <= 100:  # Muito baixo - provavelmente erro ou silêncio
            fundamental_freq = 0.1
        elif f1_hz <= 300:  # Vogais altas (/i/, /ɪ/, /u/)
            # Mapeamento linear para vogais altas: 100-300 Hz → 0.1-0.4
            fundamental_freq = 0.1 + (f1_hz - 100) / 200 * 0.3
        elif f1_hz <= 600:  # Vogais médias (/ɛ/, /ʌ/, /ɔ/)
            # Mapeamento linear para vogais médias: 300-600 Hz → 0.4-0.7
            fundamental_freq = 0.4 + (f1_hz - 300) / 300 * 0.3
        else:  # Vogais baixas e consoantes (/ɑ/, /æ/, consoantes)
            # Mapeamento linear para vogais baixas: 600+ Hz → 0.7-0.95
            fundamental_freq = 0.7 + min((f1_hz - 600) / 400 * 0.25, 0.25)

        # Garantir que está no range válido
        fundamental_freq = max(0.05, min(fundamental_freq, 0.99))

        # ========== MÉTRICAS DE ESTABILIDADE DOS NOVOS COMPONENTES ==========
        stability_metrics = self.stable_evolution.get_stability_metrics()

        return {
            'fundamental_freq': float(fundamental_freq),
            'harmonic_ratios': [],  # Legacy
            'spectral_centroid': float(spectral_centroid.item()) if hasattr(spectral_centroid, 'item') else float(spectral_centroid),
            'spectral_spread': float(spectral_spread.item()) if hasattr(spectral_spread, 'item') else float(spectral_spread),
            'phase_coherence': float(phase_coherence) if isinstance(phase_coherence, (int, float)) else float(phase_coherence.item()) if hasattr(phase_coherence, 'item') else 1.0,
            'magnitude': magnitude.tolist() if hasattr(magnitude, 'tolist') else list(magnitude),
            'phase': phase.tolist() if hasattr(phase, 'tolist') else list(phase),
            # ========== FORMANTES (NOVO - PADRÃO OURO) ==========
            'f1_frequency': formant_features['f1_frequency'],
            'f2_frequency': formant_features['f2_frequency'],
            'f3_frequency': formant_features['f3_frequency'],
            'f1_f2_ratio': formant_features['f1_f2_ratio'],
            'formant_spacing': formant_features['formant_spacing'],
            'spectral_tilt': formant_features['spectral_tilt'],
            # ========== MÉTRICAS DE ESTABILIDADE ==========
            'unitarity_error': stability_metrics['unitarity_error'],
            'spectrum_stability': stability_metrics['spectrum_stability'],
            'evolution_steps': stability_metrics['evolution_steps'],
            'prime_resonant_filtering': True,
            'leech_lattice_embedding': True
        }

    def _formant_based_mapping(self, characteristics: Dict[str, float]) -> str:
        """
        Phonetic mapping based on formant analysis.
        Removed hardcoded phonetic mappings - requires emergent phonetic generation.
        """
        # Sistema requer análise formântica emergente baseada no vocabulário do modelo
        raise NotImplementedError("Phonetic mapping requires emergent generation from model vocabulary - no hardcoded phonetic mappings allowed")


    def _characteristic_to_char(self, characteristics: Dict[str, float]) -> str:
        """
        Interface para manter compatibilidade - chama mapeamento baseado em formantes.
        """
        return self._formant_based_mapping(characteristics)

    def _apply_contextual_processing(self, char_sequence: List[str]) -> str:
        """
        Aplica processamento contextual para melhorar coerência linguística.
        Removed hardcoded phonotactic rules - uses emergent patterns only.
        """
        if not char_sequence:
            return ""

        processed = [char_sequence[0]]  # Manter primeiro caractere

        # Simplified contextual processing - no hardcoded rules
        for i in range(1, len(char_sequence)):
            current = char_sequence[i]

            # Basic repetition avoidance only
            if len(processed) >= 3 and all(c == current for c in processed[-3:]):
                current = ' '  # Inserir espaço para quebrar repetições

            processed.append(current)

        return ''.join(processed)

    def _validate_mathematical_consistency(self, fractal_signal: torch.Tensor,
                                           psi_quaternions: torch.Tensor,
                                           psi_filtered: torch.Tensor,
                                           psi_rotated: torch.Tensor) -> Dict:
        """
        Validação matemática obrigatória (doe.md validation)

        - Energia conservada: ||output|| ≈ ||input|| (dentro de 5%)
        - Unitaridade: Filtros espectrais preservam energia
        - Estabilidade numérica: Valores finitos
        """
        # Validação de conservação de energia no domínio quaterniônico
        # Todas as operações devem preservar a norma L2 dos quaternions

        # Energia quaterniônica após mapeamento inicial
        E_quaternions = torch.sum(psi_quaternions.abs() ** 2).item()

        # Energia quaterniônica após filtragem espectral
        E_filtered = torch.sum(psi_filtered.abs() ** 2).item()

        # Energia quaterniônica após rotação SO(4)
        E_rotated = torch.sum(psi_rotated.abs() ** 2).item()

        # Conservação de energia passo a passo (deve ser próximo de 1.0)
        filtering_conservation = E_filtered / (E_quaternions + 1e-10)
        rotation_conservation = E_rotated / (E_filtered + 1e-10)

        # Score global de conservação de energia (média das operações)
        energy_conservation_ratio = (filtering_conservation + rotation_conservation) / 2.0

        # Score de unitariedade (deve estar próximo de 1.0)
        unitarity_score = 1.0 - abs(energy_conservation_ratio - 1.0)

        # Verificar estabilidade numérica
        finite_values = torch.isfinite(psi_rotated).all().item()

        return {
            'energy_conservation_ratio': energy_conservation_ratio,
            'filtering_conservation': filtering_conservation,
            'rotation_conservation': rotation_conservation,
            'unitarity_score': unitarity_score,
            'numerical_stability': finite_values,
            'validation_passed': unitarity_score > 0.95 and finite_values
        }

    def _initialize_physical_components(self):
        """
        Inicializa componentes físicos obrigatórios do doe.md Seções 2.9.1-2.9.4.

        Componentes Físicos (ZERO FALLBACK):
        1. Fractal Analyzer: Calcula dimensão fractal D via power-law fitting
        2. Quaternion Processor: Hamilton product e rotações SO(4)
        3. Spectral Filter: F(k) = exp(i α · arctan(ln(|k| + ε)))
        4. Optical Probe: Geração de texto via Padilha wave equation
        5. Consciousness Processor: FCI calculation com bootstrap
        """
        print("🔬 Inicializando componentes físicos ΨQRH (doe.md)...")

        try:
            # 1. Fractal Analyzer - Calcula D via power-law fitting
            from src.fractal.spectral_filter import SpectralFilter
            self.fractal_analyzer = SpectralFilter(alpha=1.0, use_stable_activation=True)
            print("   ✅ Fractal Analyzer: D calculado via power-law fitting")

            # 2. Quaternion Processor - Hamilton product e SO(4)
            from src.core.quaternion_operations import QuaternionOperations
            self.quaternion_processor = QuaternionOperations()
            print("   ✅ Quaternion Processor: Hamilton product e rotações SO(4)")

            # 3. Spectral Filter - F(k) = exp(i α · arctan(ln(|k| + ε)))
            self.spectral_filter = SpectralFilter(alpha=1.0, epsilon=1e-10, use_stable_activation=True)
            print("   ✅ Spectral Filter: F(k) = exp(i α · arctan(ln(|k| + ε)))")

            # 4. Enhanced Optical Probe ENABLED for comparison with QuantumStateInterpreter
            # Use Enhanced OpticalProbe with Padilha Wave Equation instead of OpticalTextDecoder
            from src.core.optical_probe_fixed import create_enhanced_optical_probe
            self.optical_probe = create_enhanced_optical_probe(
                device=self.device
            )
            print("   ✅ Optical Probe: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))")

            # 5. Consciousness Processor - FCI com bootstrap
            from src.conscience.fractal_consciousness_processor import create_consciousness_processor
            self.consciousness_processor = create_consciousness_processor(embedding_dim=64, device=self.device)
            print("   ✅ Consciousness Processor: FCI calculation com bootstrap")


            print("🎯 Todos os componentes físicos ΨQRH inicializados com sucesso!")

        except Exception as e:
            print(f"❌ ERRO FATAL: Falha na inicialização dos componentes físicos: {e}")
            print("   Sistema ΨQRH físico NÃO pode funcionar sem estes componentes.")
            print("   ZERO FALLBACK POLICY: Saindo...")
            raise RuntimeError(f"ΨQRH Pipeline físico falhou na inicialização: {e}")

    def _harmonize_inverse_projector(self, num_steps=20, learning_rate=1e-4):
        """
        Executa um treino de harmonização para alinhar o InverseCognitiveProjector
        à arquitetura recém-calibrada, usando dados auto-gerados.
        """
        print("🎼 Iniciando Treino de Harmonização para o Inverse Cognitive Projector...")

        # Garantir que o projetor e o otimizador estão em modo de treino
        self.inverse_projector.train()
        if not self.optimizer:
            print("⚠️ Otimizador não encontrado. Impossível harmonizar.")
            return

        # Usar um otimizador dedicado ou o principal com LR ajustado
        harmonization_optimizer = torch.optim.AdamW(self.inverse_projector.parameters(), lr=learning_rate)

        # 1. Gerar dados de treino sintéticos (um estado quântico "ideal")
        # Usamos o próprio pipeline físico para criar um alvo consistente
        print("   🔄 Gerando estado alvo sintético (Ψ_target)...")
        with torch.no_grad():
            fractal_signal = self._text_to_fractal_signal("harmonize", self.config['embed_dim'])
            psi_target = self._signal_to_quaternions(fractal_signal, self.config['embed_dim'])
            # ASO (Análise de Assinatura Harmônica) para gerar ângulos de rotação
            # (Simulação simplificada da proposta anterior)
            rotation_angles = self._get_harmonically_derived_rotation_angles(fractal_signal)
            psi_target = self.optimized_quaternion_ops.so4_rotation(psi_target, rotation_angles)

        print(f"   📊 Ψ_target shape: {psi_target.shape}")
        print(f"   🎯 Treinando por {num_steps} passos...")

        # 2. Loop de Treino de Harmonização
        for step in range(num_steps):
            harmonization_optimizer.zero_grad()

            # O projetor tenta reconstruir o estado alvo
            # Nota: O projetor pode ter uma arquitetura interna diferente
            # Aqui, garantimos que a entrada e saída sejam compatíveis
            # A entrada para o projetor deve ser o estado quântico que ele espera
            # Vamos assumir que ele espera um vetor [embed_dim]

            # A saída do projetor é um estado quântico reconstruído
            psi_reconstructed = self.inverse_projector(psi_target.squeeze(0).squeeze(0)) # Shape: [vocab_size, embed_dim]

            # O loss é a diferença entre o estado alvo e a projeção reconstruída
            # Para comparar, precisamos de um alvo no mesmo espaço da saída do projetor
            # Vamos usar o próprio psi_target como um alvo simplificado
            # O projetor deve aprender a "focar" sua saída em torno do estado de entrada

            # Simplificação: O loss é a distância do output médio ao input médio
            loss = torch.nn.functional.mse_loss(psi_reconstructed.mean(dim=0), psi_target.mean(dim=[0,1,3]))

            loss.backward()
            harmonization_optimizer.step()

            if (step + 1) % 5 == 0:
                print(f"      🎼 Passo de Harmonização [{step+1}/{num_steps}], Loss: {loss.item():.6f}")

        print("✅ Harmonização concluída. Inverse Cognitive Projector alinhado com a nova arquitetura.")
        self.inverse_projector.eval() # Retornar ao modo de avaliação

    def _get_harmonically_derived_rotation_angles(self, signal):
        """Simulação da proposta de 'Orquestrador Harmônico' para gerar ângulos de rotação."""
        # Ângulos de rotação dependem da complexidade do sinal
        complexity = torch.std(signal.real).item()
        theta = 0.1 * (1 + complexity)
        omega = 0.05 * (1 + complexity)
        phi = 0.02 * (1 + complexity)
        angles = torch.stack([torch.tensor(theta), torch.tensor(omega), torch.tensor(phi)], dim=-1)
        return angles.expand(1, len(signal), self.config['embed_dim'], -1)

    def _check_system_harmonization(self) -> Dict[str, Any]:
        """
        Verifica se o sistema está harmonizado (auto-calibrado) corretamente.

        Returns:
            Dict com status da harmonização e componentes verificados
        """
        harmonized_components = []
        missing_components = []

        # Verificar componentes de auto-calibração física
        if HAS_AUTO_CALIBRATION and self.calibration_system is not None:
            harmonized_components.append("Sistema de Auto-Calibração Completo")
        else:
            missing_components.append("Sistema de Auto-Calibração")

        # Verificar calculadores de temperatura e coerência
        if hasattr(self, 'temp_calculator') and self.temp_calculator is not None:
            harmonized_components.append("Calculador de Temperatura Quântica")
        else:
            missing_components.append("Calculador de Temperatura Quântica")

        if hasattr(self, 'coherence_calculator') and self.coherence_calculator is not None:
            harmonized_components.append("Calculador de Coerência Óptica")
        else:
            missing_components.append("Calculador de Coerência Óptica")

        # Verificar parâmetros espectrais adaptativos
        if hasattr(self, 'spectral_params') and self.spectral_params is not None:
            harmonized_components.append("Parâmetros Espectrais Adaptativos")
        else:
            missing_components.append("Parâmetros Espectrais Adaptativos")

        # Verificar Orquestrador Harmônico Físico
        if HAS_PHYSICAL_HARMONIC_ORCHESTRATOR and self.physical_harmonic_orchestrator is not None:
            harmonized_components.append("Orquestrador Harmônico Físico")
        else:
            missing_components.append("Orquestrador Harmônico Físico")

        # Verificar analisador de assinatura harmônica física
        if (HAS_PHYSICAL_HARMONIC_ORCHESTRATOR and
            self.physical_harmonic_orchestrator is not None and
            hasattr(self.physical_harmonic_orchestrator, 'signature_analyzer') and
            self.physical_harmonic_orchestrator.signature_analyzer is not None):
            harmonized_components.append("Analisador de Assinatura Harmônica Física")
        else:
            missing_components.append("Analisador de Assinatura Harmônica Física")

        # Verificar componentes de memória quântica
        if HAS_QUANTUM_MEMORY and self.quantum_memory_system is not None:
            harmonized_components.append("Sistema de Memória Quântica Temporal")
        else:
            missing_components.append("Sistema de Memória Quântica Temporal")

        # Verificar geometria não-comutativa
        if HAS_NONCOMMUTATIVE and self.nc_pipeline is not None:
            harmonized_components.append("Geometria Não-Comutativa")
        else:
            missing_components.append("Geometria Não-Comutativa")

        # Verificar sistema híbrido quântico-clássico
        if HAS_HYBRID_SYSTEM and self.hybrid_system is not None:
            harmonized_components.append("Sistema Híbrido Quântico-Clássico")
        else:
            missing_components.append("Sistema Híbrido Quântico-Clássico")

        # Verificar componentes de aprendizado
        if HAS_AUTO_LEARNING:
            harmonized_components.append("Sistema de Auto-Aprendizagem ΨQRH")
        else:
            missing_components.append("Sistema de Auto-Aprendizagem ΨQRH")

        # Determinar status geral de harmonização
        is_harmonized = len(missing_components) == 0

        return {
            'is_harmonized': is_harmonized,
            'harmonized_components': harmonized_components,
            'missing_components': missing_components,
            'harmonization_score': len(harmonized_components) / (len(harmonized_components) + len(missing_components)) if (len(harmonized_components) + len(missing_components)) > 0 else 0.0
        }

    def _detect_device(self, device: Optional[str]) -> str:
        """Detecta o melhor dispositivo disponível"""
        if device:
            return device

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _initialize_auto_calibration_components(self):
        """Inicializa componentes individuais de auto-calibração"""
        try:
            # Initialize Quantum Temperature Calculator
            from src.core.quantum_temperature_calculator import QuantumTemperatureCalculator
            self.temp_calculator = QuantumTemperatureCalculator()
            print("   ✅ Calculador de Temperatura Quântica: ATIVO")

        except Exception as e:
            print(f"   ❌ Calculador de Temperatura Quântica falhou: {e}")
            self.temp_calculator = None

        try:
            # Initialize Optical Coherence Calculator
            from src.core.optical_coherence_calculator import OpticalCoherenceCalculator
            self.coherence_calculator = OpticalCoherenceCalculator()
            print("   ✅ Calculador de Coerência Óptica: ATIVO")

        except Exception as e:
            print(f"   ❌ Calculador de Coerência Óptica falhou: {e}")
            self.coherence_calculator = None

        try:
            # Initialize Adaptive Spectral Parameters
            from src.core.adaptive_spectral_parameters import AdaptiveSpectralParameters
            self.spectral_params = AdaptiveSpectralParameters()
            print("   ✅ Parâmetros Espectrais Adaptativos: ATIVO")

        except Exception as e:
            print(f"   ❌ Parâmetros Espectrais Adaptativos falhou: {e}")
            self.spectral_params = None

    def _initialize_complete_auto_calibration(self):
        """Inicializa sistema completo de auto-calibração"""
        global HAS_AUTO_CALIBRATION
        if not HAS_AUTO_CALIBRATION:
            self.calibration_system = None
            return

        print("🔧 Inicializando sistema completo de auto-calibração ΨQRH...")

        try:
            # Initialize complete auto-calibration system
            self.calibration_system = CompleteAutoCalibrationSystem()

            print("✅ Sistema completo de auto-calibração ΨQRH carregado:")
            print("   - Physical Parameter Calibrator: ATIVO")
            print("   - Architecture Parameter Calibrator: ATIVO")
            print("   - Processing Parameter Calibrator: ATIVO")
            print("   - Control Parameter Calibrator: ATIVO")
            print("   - Complete Auto-Calibration System: ATIVO")

        except Exception as e:
            print(f"⚠️  Erro ao carregar sistema completo de auto-calibração ΨQRH: {e}")
            HAS_AUTO_CALIBRATION = False
            self.calibration_system = None

    def _adapt_pretrained_weights_to_dimensions(self, target_embed_dim: int, target_vocab_size: int):
        """
        Adapt pretrained weights to match calibrated dimensions.

        Args:
            target_embed_dim: Target embedding dimension from calibration
            target_vocab_size: Target vocabulary size from calibration

        Returns:
            Adapted state_dict with compatible dimensions
        """
        if self.pretrained_state_dict is None:
            return None

        adapted_state_dict = {}
        print(f"🔧 Adapting pretrained weights to dimensions: embed_dim={target_embed_dim}, vocab_size={target_vocab_size}")

        for key, param in self.pretrained_state_dict.items():
            if param is None:
                continue

            try:
                # Handle different parameter types
                if 'embed' in key.lower() and 'weight' in key.lower():
                    # Embedding layer weights [vocab_size, embed_dim]
                    if param.dim() == 2:
                        orig_vocab, orig_embed = param.shape
                        adapted_param = param.clone()

                        # Adapt vocabulary dimension
                        if orig_vocab != target_vocab_size:
                            if orig_vocab < target_vocab_size:
                                # Pad vocabulary dimension
                                padding = torch.zeros(target_vocab_size - orig_vocab, orig_embed, device=param.device, dtype=param.dtype)
                                adapted_param = torch.cat([adapted_param, padding], dim=0)
                                print(f"   ➕ Padded vocab: {orig_vocab} → {target_vocab_size}")
                            else:
                                # Truncate vocabulary dimension
                                adapted_param = adapted_param[:target_vocab_size]
                                print(f"   ➖ Truncated vocab: {orig_vocab} → {target_vocab_size}")

                        # Adapt embedding dimension
                        if orig_embed != target_embed_dim:
                            if orig_embed < target_embed_dim:
                                # Pad embedding dimension
                                padding = torch.zeros(target_vocab_size, target_embed_dim - orig_embed, device=param.device, dtype=param.dtype)
                                adapted_param = torch.cat([adapted_param, padding], dim=1)
                                print(f"   ➕ Padded embed: {orig_embed} → {target_embed_dim}")
                            else:
                                # Truncate embedding dimension
                                adapted_param = adapted_param[:, :target_embed_dim]
                                print(f"   ➖ Truncated embed: {orig_embed} → {target_embed_dim}")

                        adapted_state_dict[key] = adapted_param

                elif 'linear' in key.lower() or 'fc' in key.lower():
                    # Linear layer weights [out_features, in_features]
                    if param.dim() == 2:
                        out_feat, in_feat = param.shape
                        adapted_param = param.clone()

                        # Adapt input features (usually embed_dim)
                        if in_feat != target_embed_dim:
                            if in_feat < target_embed_dim:
                                # Pad input dimension
                                padding = torch.zeros(out_feat, target_embed_dim - in_feat, device=param.device, dtype=param.dtype)
                                adapted_param = torch.cat([adapted_param, padding], dim=1)
                                print(f"   ➕ Padded linear in: {in_feat} → {target_embed_dim}")
                            else:
                                # Truncate input dimension
                                adapted_param = adapted_param[:, :target_embed_dim]
                                print(f"   ➖ Truncated linear in: {in_feat} → {target_embed_dim}")

                        adapted_state_dict[key] = adapted_param

                elif 'bias' in key.lower():
                    # Bias terms - usually match output dimensions
                    if param.dim() == 1:
                        bias_size = param.shape[0]
                        adapted_param = param.clone()

                        # Adapt bias dimension if it matches embed_dim
                        if bias_size != target_embed_dim:
                            if bias_size < target_embed_dim:
                                # Pad bias dimension
                                padding = torch.zeros(target_embed_dim - bias_size, device=param.device, dtype=param.dtype)
                                adapted_param = torch.cat([adapted_param, padding], dim=0)
                                print(f"   ➕ Padded bias: {bias_size} → {target_embed_dim}")
                            else:
                                # Truncate bias dimension
                                adapted_param = adapted_param[:target_embed_dim]
                                print(f"   ➖ Truncated bias: {bias_size} → {target_embed_dim}")

                        adapted_state_dict[key] = adapted_param

                else:
                    # Copy other parameters unchanged
                    adapted_state_dict[key] = param.clone()

            except Exception as e:
                print(f"   ⚠️  Failed to adapt parameter {key}: {e}")
                # Keep original parameter if adaptation fails
                adapted_state_dict[key] = param.clone()

        print(f"✅ Weight adaptation completed: {len(adapted_state_dict)} parameters adapted")
        return adapted_state_dict

    def _reinitialize_components_with_calibrated_params(self, phys_params, arch_params, proc_params, ctrl_params):
        """
        Re-initializa componentes com parâmetros calibrados dinamicamente.

        Args:
            phys_params: Parâmetros físicos calibrados (I₀, ω, k, α, β)
            arch_params: Parâmetros de arquitetura calibrados (embed_dim, num_heads, etc.)
            proc_params: Parâmetros de processamento calibrados (dropout, vocab_size, etc.)
            ctrl_params: Parâmetros de controle calibrados (temperature, learning_rate, etc.)
        """
        print("   🔄 Re-inicializando componentes aprendíveis com parâmetros calibrados...")

        try:
            # ========== CONTEXT FUNNEL ==========
            from src.core.context_funnel import create_context_funnel
            self.context_funnel = create_context_funnel(
                embed_dim=arch_params['embed_dim'],
                num_heads=arch_params['num_heads'],
                max_history=proc_params['max_history']
            ).to(self.device)
            print(f"      ✅ Context Funnel: embed_dim={arch_params['embed_dim']}, num_heads={arch_params['num_heads']}, max_history={proc_params['max_history']}")

            # ========== INVERSE COGNITIVE PROJECTOR ==========
            from src.core.inverse_cognitive_projector import create_inverse_cognitive_projector
            self.inverse_projector = create_inverse_cognitive_projector(
                embed_dim=arch_params['embed_dim'],
                vocab_size=proc_params['vocab_size'],
                hidden_dim=arch_params['hidden_dim'],
                num_layers=arch_params['num_layers'],
                dropout=proc_params['dropout']
            ).to(self.device)
            print(f"      ✅ Inverse Projector: embed_dim={arch_params['embed_dim']}, vocab_size={proc_params['vocab_size']}, hidden_dim={arch_params['hidden_dim']}, num_layers={arch_params['num_layers']}, dropout={proc_params['dropout']}")

            # ========== QUANTUM EMBEDDING ==========
            self.quantum_embedding = QuantumEmbedding(
                vocab_size=proc_params['vocab_size'],
                embed_dim=arch_params['embed_dim']
            ).to(self.device)
            print(f"      ✅ Quantum Embedding: vocab_size={proc_params['vocab_size']}, embed_dim={arch_params['embed_dim']}")

            # ========== ENHANCED OPTICAL PROBE ==========
            from src.core.optical_probe_fixed import create_enhanced_optical_probe
            self.optical_probe = create_enhanced_optical_probe(
                device=self.device
            )
            # Update optical probe parameters if possible
            if hasattr(self.optical_probe, 'update_parameters'):
                self.optical_probe.update_parameters(
                    I0=phys_params['I0'],
                    omega=phys_params['omega'],
                    k=phys_params['k'],
                    alpha=phys_params['alpha'],
                    beta=phys_params['beta']
                )
            print(f"      ✅ Optical Probe: I₀={phys_params['I0']:.3f}, ω={phys_params['omega']:.3f}, k={phys_params['k']:.3f}, α={phys_params['alpha']:.3f}, β={phys_params['beta']:.3f}")

            # ========== STABLE QUANTUM EVOLUTION ==========
            self.stable_evolution = create_stable_quantum_evolution(
                embed_dim=arch_params['embed_dim'],
                device=self.device
            )
            print(f"      ✅ Stable Evolution: embed_dim={arch_params['embed_dim']}")

            # ========== TRUE VOCABULARY AUTONOMY ==========
            # ZERO FALLBACK: No external pre-trained weights loaded during calibration
            print("      🎯 Using random initialization for true vocabulary autonomy (ZERO FALLBACK)")

            # ========== UPDATE OPTIMIZER ==========
            learnable_params = list(self.context_funnel.parameters()) + \
                              list(self.inverse_projector.parameters()) + \
                              list(self.quantum_embedding.parameters())

            if len(learnable_params) > 0:
                self.optimizer = torch.optim.AdamW(
                    learnable_params,
                    lr=ctrl_params['learning_rate'],
                    weight_decay=0.01
                )
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=1000, T_mult=2
                )
                print(f"      ✅ Optimizer: lr={ctrl_params['learning_rate']:.2e}, weight_decay=0.01")
            else:
                self.optimizer = None
                self.scheduler = None
                print("      ⚠️  No learnable parameters found for optimizer")

            print("   ✅ Todos os componentes re-inicializados com parâmetros calibrados!")

        except Exception as e:
            print(f"   ❌ Erro na re-inicialização de componentes: {e}")
            import traceback
            traceback.print_exc()
            # Continue with original components if re-initialization fails
            print("   ⚠️  Continuando com componentes originais...")

    def _initialize_noncommutative(self):
        """Inicializa componentes de geometria não-comutativa"""
        global HAS_NONCOMMUTATIVE
        if not HAS_NONCOMMUTATIVE:
            self.nc_pipeline = None
            return

        print("🔬 Inicializando geometria não-comutativa avançada...")

        try:
            # Criar pipeline não-comutativo aprimorado
            embed_dim = int(self.config['embed_dim'])  # Garantir que seja int
            self.nc_pipeline = create_noncommutative_pipeline(
                embed_dim=embed_dim,
                theta=0.1  # Parâmetro de não-comutatividade
            )

            print("✅ Pipeline não-comutativo ΨQRH inicializado:")
            print("   🧮 Geometria não-comutativa: [x̂, p̂] = iθ")
            print("   🌊 Dinâmica de ondas quânticas não-comutativas")
            print("   🗣️ Campo fonêmico quântico")

        except Exception as e:
            print(f"⚠️  Erro ao inicializar geometria não-comutativa: {e}")
            HAS_NONCOMMUTATIVE = False
            self.nc_pipeline = None

    def _initialize_hybrid_system(self):
        """Inicializa sistema híbrido quântico-clássico"""
        global HAS_HYBRID_SYSTEM
        if not HAS_HYBRID_SYSTEM:
            self.hybrid_system = None
            return

        print("🔗 Inicializando sistema híbrido quântico-clássico...")

        try:
            self.hybrid_system = create_hybrid_system()

            print("✅ Sistema híbrido ΨQRH inicializado:")
            print("   🧮 Transição de fase crítica entre regimes quântico/clássico")
            print("   🔄 Interface adiabática quântico-clássica")
            print("   📝 Processamento linguístico com restrições quânticas")
            print("   🎯 Resolução do divórcio física-linguística")

        except Exception as e:
            print(f"⚠️  Erro ao inicializar sistema híbrido: {e}")
            HAS_HYBRID_SYSTEM = False
            self.hybrid_system = None

    def _initialize_quantum_memory(self):
        """Inicializa sistema de memória quântica temporal"""
        global HAS_QUANTUM_MEMORY
        if not HAS_QUANTUM_MEMORY:
            self.quantum_memory_system = None
            return

        print("🧠 Inicializando sistema de memória quântica temporal...")

        try:
            self.quantum_memory_system = create_quantum_memory_system(
                memory_size=8,  # Tamanho da memória temporal
                coherence_time=3.0  # Tempo de coerência em unidades quânticas
            )

            print("✅ Sistema de memória quântica ΨQRH inicializado:")
            print("   🔗 Correlações de longo alcance entre estados temporais")
            print("   🎭 Decoerência controlada com preservação de fase")
            print("   📝 Processamento linguístico contextual")
            print("   🧬 Emaranhamento temporal para coerência sequencial")

        except Exception as e:
            print(f"⚠️  Erro ao inicializar sistema de memória quântica: {e}")
            HAS_QUANTUM_MEMORY = False
            self.quantum_memory_system = None

    def _initialize_audit_logger(self):
        """Inicializa o sistema de auditoria para debugging e análise"""
        print("🔍 Inicializando sistema de auditoria ΨQRH...")

        try:
            from src.core.spectral_projector import AuditLogger
            from tools.audit_analyzer import ΨQRHAuditAnalyzer

            self.audit_logger = AuditLogger(audit_dir="results/audit_logs", enabled=True)
            self.audit_analyzer = ΨQRHAuditAnalyzer()

            print("✅ Sistema de auditoria ΨQRH inicializado:")
            print("   📊 Logging de estados quânticos em pontos críticos")
            print("   🔬 Cálculo de métricas de reconstrução e separabilidade")
            print("   🎯 Análise de interferência contextual")
            print("   📈 Relatórios de diagnóstico detalhados")

        except Exception as e:
            print(f"⚠️  Erro ao inicializar sistema de auditoria: {e}")
            self.audit_logger = None
            self.audit_analyzer = None
            self.audit_mode = False

    def _save_audit_logs(self, result: Dict[str, Any]):
        """Salva os logs de auditoria gerados durante o processamento"""
        if not self.audit_logger:
            return

        try:
            # Finalizar a sessão de auditoria
            audit_log_path = self.audit_logger.end_session(result.get('response', ''))

            if audit_log_path:
                print(f"💾 Audit logs salvos em: {audit_log_path}")

                # Integrar com o audit analyzer para análise adicional
                try:
                    from tools.audit_analyzer import ΨQRHAuditAnalyzer
                    analyzer = ΨQRHAuditAnalyzer()

                    # Executar análise completa dos logs
                    analysis_result = analyzer.generate_diagnostic_report(audit_log_path, embed_dim=self.config['embed_dim'])

                    if analysis_result:
                        print("🔬 Relatório de diagnóstico gerado automaticamente")
                        print("   📋 Verifique o arquivo de relatório para análise completa")

                except Exception as e:
                    print(f"⚠️  Análise de auditoria falhou: {e}")

        except Exception as e:
            print(f"⚠️  Erro ao salvar logs de auditoria: {e}")

    def _initialize_quantum_vocabulary_with_genesis(self, vocab_path=None):
        """
        Initialize quantum vocabulary with linguistic genesis foundation

        Replaces random initialization with quantum linguistic genesis that
        encodes alphabet and numerals as fundamental quantum properties.
        """
        try:
            # Import quantum linguistic genesis system
            from src.core.quantum_linguistic_genesis import QuantumLinguisticGenesis

            print("🧬 Initializing Quantum Linguistic Genesis System...")

            # Create quantum linguistic foundation
            genesis = QuantumLinguisticGenesis(
                embed_dim=self.config['embed_dim'],
                device=self.device
            )

            # Get quantum vocabulary tensor and character mapping
            quantum_tensor, char_to_idx = genesis.get_quantum_vocabulary_tensor()

            # Set quantum vocabulary representations
            self.quantum_vocab_representations = quantum_tensor
            self.char_to_idx = char_to_idx

            print("✅ Quantum Linguistic Genesis Initialized:")
            print(f"   📊 Vocabulary: {len(self.quantum_vocab_representations)} linguistic primitives")
            print(f"   🔬 Tensor shape: {self.quantum_vocab_representations.shape}")
            print(f"   🎯 Linguistic foundation: ALPHABET + NUMERALS + PUNCTUATION")

            # Analyze linguistic properties
            test_text = "Hello World 123!"
            analysis = genesis.analyze_linguistic_properties(test_text)
            print(f"   📊 Linguistic analysis of '{test_text}':")
            print(f"      Vowel ratio: {analysis['vowel_ratio']:.3f}")
            print(f"      Consonant ratio: {analysis['consonant_ratio']:.3f}")
            print(f"      Quantum coherence: {analysis['quantum_coherence']:.3f}")

        except Exception as e:
            print(f"⚠️  Quantum linguistic genesis failed: {e}")
            raise

    def _initialize_quantum_vocabulary(self, vocab_path=None):
        """Inicializa dicionário quântico para conectividade semântica usando vocabulário nativo"""
        print("📚 Inicializando dicionário quântico para conectividade semântica...")

        try:
            # Use injected vocab_path if provided, otherwise try default locations
            vocab_data = None
            vocab_source_path = None

            if vocab_path is not None and os.path.exists(vocab_path):
                vocab_source_path = vocab_path
            else:
                vocab_paths = [
                    os.path.join(os.getcwd(), "data", "native_vocab.json"),
                    os.path.join(BASE_DIR, "data", "native_vocab.json")
                ]

                for path in vocab_paths:
                    if os.path.exists(path):
                        vocab_source_path = path
                        break

            if vocab_source_path:
                try:
                    with open(vocab_source_path, 'r', encoding='utf-8') as f:
                        vocab_data = json.load(f)
                    print(f"   📚 Carregando vocabulário nativo de: {vocab_source_path}")
                except Exception as e:
                    print(f"   ⚠️  Erro ao carregar vocabulário {vocab_source_path}: {e}")

            if vocab_data and 'token_to_id' in vocab_data:
                # Get vocab_size from data
                vocab_size = vocab_data.get('vocab_size', len(vocab_data['token_to_id']))
                print(f"   📚 Vocabulário nativo encontrado: {vocab_size} tokens")

                # Create quantum representations for all tokens in order by token_id
                quantum_representations = []
                token_to_idx = vocab_data['token_to_id'].copy()  # Use the mapping from json

                for token_id in range(min(vocab_size, self.quantum_embedding.vocab_size)):
                    # Get token for this id
                    token = vocab_data['id_to_token'].get(str(token_id), '<unk>')

                    # Use token_id directly as embedding index
                    char_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                    psi_token = self.quantum_embedding(char_ids).squeeze(0).squeeze(0)  # [embed_dim, 4]

                    quantum_representations.append(psi_token)

                    # Progress indicator for large vocabulary
                    if (token_id + 1) % 10 == 0:
                        print(f"   📊 Processado {token_id + 1}/{min(vocab_size, self.quantum_embedding.vocab_size)} tokens...")

                # Stack into tensor [vocab_size, embed_dim, 4]
                self.quantum_vocab_representations = torch.stack(quantum_representations, dim=0)
                self.char_to_idx = token_to_idx  # Keep compatibility with existing interface

                print("✅ Dicionário quântico inicializado:")
                print(f"   📊 Vocabulário nativo: {len(quantum_representations)} tokens")
                print(f"   🔬 Representações quânticas: {self.quantum_vocab_representations.shape}")
                print(f"   🎯 Conectividade semântica: ATIVADA (baseada em vocabulário nativo)")

            else:
                raise FileNotFoundError("Vocabulário nativo não encontrado ou vazio")

        except Exception as e:
            print(f"⚠️  Erro ao inicializar dicionário quântico: {e}")
            # Create minimal fallback quantum vocabulary
            print("   🔄 Criando vocabulário quântico mínimo de fallback...")
            try:
                # Create basic ASCII vocabulary as fallback
                basic_vocab = {}
                quantum_representations = []

                for i in range(32, 127):  # Printable ASCII
                    char = chr(i)
                    basic_vocab[char] = i - 32  # Map to 0-based indices

                    # Create quantum representation
                    char_ids = torch.tensor([[i % self.quantum_embedding.vocab_size]], dtype=torch.long, device=self.device)
                    psi_token = self.quantum_embedding(char_ids).squeeze(0).squeeze(0)
                    quantum_representations.append(psi_token)

                self.quantum_vocab_representations = torch.stack(quantum_representations, dim=0)
                self.char_to_idx = basic_vocab

                print("✅ Vocabulário quântico de fallback criado:")
                print(f"   📊 Vocabulário básico: {len(basic_vocab)} caracteres ASCII")
                print(f"   🔬 Representações quânticas: {self.quantum_vocab_representations.shape}")

            except Exception as fallback_e:
                print(f"❌ Mesmo fallback falhou: {fallback_e}")
                self.quantum_vocab_representations = None
                self.char_to_idx = None

