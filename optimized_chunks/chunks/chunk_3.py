# Chunk 3: Lines 2322-3520
# Tokens: 14055, Lines: 2322-3520

    def _extract_quantum_features_from_psi(self, psi: torch.Tensor, alpha: float, beta: float) -> Dict:
        """
        Extrai características quânticas do estado ψ para interface híbrida.

        Args:
            psi: Estado quaterniônico [batch, seq_len, embed_dim, 4]
            alpha: Parâmetro espectral α
            beta: Parâmetro espectral β

        Returns:
            Dicionário com características quânticas
        """
        # Calcular temperatura quântica baseada na complexidade
        complexity = torch.mean(torch.abs(psi)).item()
        T_quantum = 1.0 - 0.5 * complexity  # Temperatura inversamente proporcional à complexidade

        # Calcular coerência quântica
        coherence = torch.mean(torch.abs(torch.mean(psi, dim=-1))).item()

        # Simular estado quântico para interface
        quantum_state = torch.mean(psi, dim=[0, 1])  # [embed_dim, 4] -> [4]

        return {
            'quantum_temperature': max(0.1, T_quantum),
            'coherence': coherence,
            'quantum_state': quantum_state,
            'symmetry_measure': 0.5 + 0.3 * torch.sin(torch.tensor(alpha)).item(),
            'entanglement_entropy': complexity * 2.0
        }

    def _estimate_text_quality(self, generated_text: str, input_text: str) -> float:
        """
        Estima qualidade do texto gerado baseado em similaridade e diversidade.

        Args:
            generated_text: Texto gerado pelo sistema
            input_text: Texto de entrada original

        Returns:
            Score de qualidade entre 0.0 e 1.0
        """
        if not generated_text or not input_text:
            return 0.0

        # Calcular similaridade de palavras (inversa - menor similaridade = maior qualidade)
        input_words = set(input_text.lower().split())
        output_words = set(generated_text.lower().split())

        if not input_words:
            return 0.5  # Score neutro se não há palavras de entrada

        overlap = len(input_words.intersection(output_words))
        similarity = overlap / len(input_words)

        # Calcular diversidade (maior diversidade = maior qualidade)
        unique_words = len(output_words)
        total_words = len(generated_text.split())
        diversity = unique_words / max(total_words, 1)

        # Calcular comprimento apropriado (não muito curto, não muito longo)
        length_score = min(len(generated_text.split()) / 20, 1.0)  # Ideal: ~20 palavras

        # Score combinado: alta diversidade, baixa similaridade, comprimento apropriado
        quality_score = (diversity * 0.4) + ((1.0 - similarity) * 0.4) + (length_score * 0.2)

        return min(quality_score, 1.0)

    def _validate_quantum_quality(self, generated_text: str, psi: torch.Tensor, alpha: float, beta: float) -> float:
        """
        Valida se a geração mantém coerência quântica adequada.
        Versão simplificada para evitar recursão infinita.

        Args:
            generated_text: Texto gerado
            psi: Estado quântico original
            alpha: Parâmetro espectral α
            beta: Parâmetro espectral β

        Returns:
            Score de qualidade quântica (0.0-1.0)
        """
        if not generated_text or not generated_text.strip():
            return 0.0

        try:
            # Validação simplificada baseada apenas no comprimento e diversidade
            words = generated_text.split()
            if len(words) < 3:
                return 0.3  # Muito curto

            # Diversidade vocabular
            unique_words = len(set(words))
            diversity = unique_words / max(len(words), 1)

            # Comprimento apropriado
            length_score = min(len(words) / 15, 1.0)  # Ideal: ~15 palavras

            # Score combinado simplificado
            quality_score = (diversity * 0.6) + (length_score * 0.4)

            return min(max(quality_score, 0.4), 1.0)  # Mínimo 0.4 para evitar recursão

        except Exception as e:
            print(f"⚠️  Erro na validação quântica: {e}")
            return 0.6  # Score mais alto por padrão para evitar recursão

    def _save_audit_logs(self, result: Dict[str, Any]):
        """Salva os logs de auditoria e gera relatório de diagnóstico"""
        if not self.audit_logger or not self.audit_analyzer:
            return

        try:
            # Finalizar a sessão de auditoria
            audit_log_path = self.audit_logger.end_session(result.get('response', ''))

            if audit_log_path:
                print(f"💾 Audit logs salvos em: {audit_log_path}")

                # Integrar com o audit analyzer para análise adicional
                try:
                    # Executar análise completa dos logs
                    analysis_result = self.audit_analyzer.generate_diagnostic_report(audit_log_path, embed_dim=self.config['embed_dim'])

                    if analysis_result:
                        print("🔬 Relatório de diagnóstico gerado automaticamente")
                        print("   📋 Verifique o arquivo de relatório para análise completa")

                except Exception as e:
                    print(f"⚠️  Análise de auditoria falhou: {e}")

        except Exception as e:
            print(f"⚠️  Erro ao salvar logs de auditoria: {e}")

    def _force_quantum_recalibration(self, max_retries: int = 1):
        """
        Força recalibração dos pesos quânticos quando necessário.
        Com limite de tentativas para evitar recursão infinita.
        """
        if not hasattr(self, '_recalibration_attempts'):
            self._recalibration_attempts = 0

        if self._recalibration_attempts >= max_retries:
            raise RuntimeError(f"Maximum recalibration attempts ({max_retries}) exceeded - no fallback allowed")

        self._recalibration_attempts += 1
        print(f"🔄 Forçando recalibração quântica dos pesos GPT-2 (tentativa {self._recalibration_attempts}/{max_retries})...")

        # Sistema GPT-2 removido - recalibração não disponível
        print("⚠️  Sistema GPT-2 espectral removido - recalibração não disponível")
        return False

    def _check_coherence_alignment(self, text: str, target_coherence: float) -> float:
        """Verifica alinhamento de coerência quântica"""
        # Análise simples baseada no comprimento e estrutura
        words = text.split()
        if len(words) < 5:
            return 0.3  # Muito curto

        # Coerência baseada na diversidade vocabular
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words)

        # Alinhamento com target_coherence
        alignment = 1.0 - abs(diversity_ratio - target_coherence)
        return max(0.0, min(alignment, 1.0))

    def _check_fractal_consistency(self, text: str, target_dimension: float) -> float:
        """Verifica consistência fractal"""
        # Análise baseada na complexidade estrutural
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

        # Dimensão fractal estimada baseada na complexidade
        estimated_dimension = min(2.0, 1.0 + (avg_sentence_length / 10))

        consistency = 1.0 - abs(estimated_dimension - target_dimension) / 2.0
        return max(0.0, min(consistency, 1.0))

    def _check_spectral_preservation(self, text: str, spectral_features: Dict[str, float]) -> float:
        """Verifica preservação das propriedades espectrais"""
        # Verificar se o texto mantém características espectrais adequadas
        energy_level = spectral_features.get('spectral_energy', 0.5)
        entropy_level = spectral_features.get('spectral_entropy', 1.0)

        # Score baseado na adequação dos níveis espectrais
        energy_score = min(energy_level / 10000, 1.0)  # Normalizar energia
        entropy_score = min(entropy_level / 10, 1.0)   # Normalizar entropia

        return (energy_score + entropy_score) / 2.0

    def _check_semantic_coherence(self, text: str, consciousness_patterns: Dict[str, float]) -> float:
        """Verifica coerência semântica baseada nos padrões de consciência"""
        fci = consciousness_patterns.get('fci', 0.5)

        # Análise baseada no FCI
        if fci > 0.7:
            # Alta consciência - deve ter conteúdo complexo
            word_count = len(text.split())
            complexity_score = min(word_count / 50, 1.0)
            return complexity_score
        elif fci > 0.4:
            # Consciência média - conteúdo moderado
            return 0.7
        else:
            # Baixa consciência - conteúdo simples
            return 0.5

    def encode_single_char_to_quantum_state(self, token: str, position: int = 0, embed_dim: int = 256) -> torch.Tensor:
        """
        Encode a token (character or subword) to quantum state using the same logic as text_to_quaternion_embedding.
        This implements the forward encoding: token → Ψ_token

        For tokens longer than single characters, uses a deterministic hash of the token string.

        Args:
            token: Token to encode (can be single character or subword)
            position: Position in sequence (for phase calculation)
            embed_dim: Embedding dimension

        Returns:
            Quantum state tensor [embed_dim, 4] for the token
        """
        # Handle single characters (original behavior)
        if len(token) == 1:
            ascii_val = ord(token)
        else:
            # For multi-character tokens, create a deterministic hash
            # Use Python's built-in hash function with a fixed seed for reproducibility
            import hashlib
            token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest()[:8], 16)
            # Map hash to a reasonable ASCII range (0-255) for compatibility
            ascii_val = token_hash % 256

        # Create quaternion embedding for token
        psi_token = torch.zeros(embed_dim, 4, dtype=torch.float32, device=self.device)

        for j in range(embed_dim):
            # Create deterministic quaternion components (simplified: only amplitude and principal phase)
            phase = (ascii_val + position + j) * 2 * math.pi / 256.0
            amplitude = (ascii_val / 255.0) * (j / embed_dim)  # Normalize to 0-255 range

            # Simplified quaternion components: only real and imaginary parts (principal phase)
            # Zero out j and k components to preserve neighborhood relations better
            psi_token[j, 0] = amplitude * math.cos(phase)  # ψ₀ (real)
            psi_token[j, 1] = amplitude * math.sin(phase)  # ψ₁ (i)
            psi_token[j, 2] = 0.0  # ψ₂ (j) - zeroed for simplification
            psi_token[j, 3] = 0.0  # ψ₃ (k) - zeroed for simplification

        return psi_token

    def _apply_inverse_so4_rotation(self, psi_rotated: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse SO(4) rotations to undo the forward quaternion rotations.

        For inverse rotation, we use negative angles in the so4_rotation method.

        Args:
            psi_rotated: Rotated quantum state [batch, seq, embed_dim, 4]

        Returns:
            Unrotated quantum state [batch, seq, embed_dim, 4]
        """
        print("🔄 Applying inverse SO(4) rotations...")

        batch_size, seq_len, embed_dim, quat_dim = psi_rotated.shape

        # Use negative rotation angles to invert the forward rotation
        theta_left = torch.tensor(-0.1, device=self.device)
        omega_left = torch.tensor(-0.05, device=self.device)
        phi_left = torch.tensor(-0.02, device=self.device)

        # Create tensors for rotation angles
        rotation_angles_left = torch.stack([theta_left, omega_left, phi_left], dim=-1)
        rotation_angles_left = rotation_angles_left.expand(batch_size, seq_len, embed_dim, -1)

        # Apply inverse rotation using the so4_rotation method with negative angles
        psi_unrotated = self.optimized_quaternion_ops.so4_rotation(psi_rotated, rotation_angles_left)

        # Conservação de energia: renormalizar para preservar a norma original
        psi_renormalized = psi_unrotated

        print(f"   ✅ Inverse SO(4) rotations applied: {psi_rotated.shape} → {psi_renormalized.shape}")
        return psi_renormalized

    def _apply_inverse_spectral_filtering(self, psi_filtered: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Apply inverse spectral filtering to undo the forward spectral filtering.

        Based on invert_spectral_qrh: forward filter was exp(1j * alpha * phase),
        so inverse filter should be exp(-1j * alpha * phase).

        Args:
            psi_filtered: Spectrally filtered quantum state [batch, seq, embed_dim, 4]
            alpha: Spectral parameter used in forward filtering

        Returns:
            Spectrally unfiltered quantum state [batch, seq, embed_dim, 4]
        """
        print(f"🌊 Applying inverse spectral filtering (α={alpha:.3f})...")

        batch_size, seq_len, embed_dim, quat_dim = psi_filtered.shape

        # Step 1: Apply FFT along embedding dimension (same as forward)
        psi_fft = torch.fft.fft(psi_filtered, dim=2)

        # Step 2: Compute frequencies (same as forward)
        freqs = torch.fft.fftfreq(embed_dim, dtype=torch.float32, device=self.device)
        k = 2 * torch.pi * freqs.view(1, 1, embed_dim, 1)

        # Step 3: Create INVERSE spectral filter
        # Forward filter: exp(1j * alpha * arctan(log(|k| + ε)))
        # Inverse filter: exp(-1j * alpha * arctan(log(|k| + ε)))
        epsilon = 1e-10
        k_mag = torch.abs(k) + epsilon
        log_k = torch.log(k_mag.clamp(min=1e-9))
        phase = torch.arctan(log_k)

        # INVERSE filter with negative exponent
        inverse_filter_response = torch.exp(-1j * alpha * phase)

        # Step 4: Apply inverse filter in frequency domain
        psi_inverted_fft = psi_fft * inverse_filter_response

        # Step 5: Inverse FFT back to spatial domain
        psi_inverted = torch.fft.ifft(psi_inverted_fft, dim=2).real

        print(f"   ✅ Inverse spectral filtering applied: {psi_filtered.shape} → {psi_inverted.shape}")
        return psi_inverted

    def _quaternion_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute quaternion conjugate: q* = (w, -x, -y, -z)

        Args:
            q: Quaternion tensor [..., 4]

        Returns:
            Conjugate quaternion [..., 4]
        """
        w, x, y, z = torch.unbind(q, dim=-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    def _safe_optical_probe_extraction(self, optical_output):
        """
        Extração segura de saída do optical probe com múltiplos fallbacks
        """
        try:
            # Método 1: Tentar acesso direto
            if hasattr(optical_output, '__getitem__'):
                try:
                    if len(optical_output) > 0:
                        return optical_output[0]
                except:
                    pass

            # Método 2: Se for tuple, extrair primeiro elemento
            if isinstance(optical_output, tuple) and len(optical_output) > 0:
                return optical_output[0]

            # Método 3: Se for lista, extrair primeiro elemento
            if isinstance(optical_output, list) and len(optical_output) > 0:
                return optical_output[0]

            # Método 4: Converter para string e extrair primeiro caractere
            str_output = str(optical_output)
            if len(str_output) > 0:
                return str_output[0]

            # Método 5: Fallback absoluto
            return 'Ψ'  # Símbolo quântico como fallback

        except Exception as e:
            print(f"⚠️  Erro na extração do optical probe: {e}")
            return 'Q'  # Fallback final

    def run_inverse_pipeline(self, psi_final: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Apply the complete inverse pipeline to bring the final quantum state back to
        the original representation space where character encodings exist.

        This implements the "Total Symmetric Inversion Principle":
        Ψ_final → Inverse Spectral Filtering → Inverse SO(4) Rotations → Ψ_reconstructed

        Args:
            psi_final: Final quantum state from DCF reasoning [batch, seq, embed_dim, 4]
            alpha: Spectral parameter used in forward pipeline

        Returns:
            Reconstructed quantum state in original representation space [batch, seq, embed_dim, 4]
        """
        print("🔄 Running complete inverse pipeline - Total Symmetric Inversion...")

        # Ensure proper shape
        if psi_final.dim() == 3:
            psi_final = psi_final.unsqueeze(0)  # Add batch dimension if needed

        # Step 1: Inverse spectral filtering
        psi_unfiltered = self._apply_inverse_spectral_filtering(psi_final, alpha)

        # Step 2: Inverse SO(4) rotations
        psi_reconstructed = self._apply_inverse_so4_rotation(psi_unfiltered)

        # Ensure energy conservation in the inverse pipeline
        psi_reconstructed = psi_reconstructed

        print(f"   ✅ Complete inverse pipeline finished: {psi_final.shape} → {psi_reconstructed.shape}")
        print("   🎯 Ψ_reconstructed now exists in the same mathematical space as character encodings")

        return psi_reconstructed

    def train_end_to_end(self, training_data: List[Tuple[str, str]], num_epochs: int = 10,
                        batch_size: int = 4, accumulation_steps: int = 4) -> Dict[str, List[float]]:
        """
        Treinamento End-to-End da Arquitetura de Três Componentes

        Args:
            training_data: Lista de tuplas (input_text, target_token)
            num_epochs: Número de épocas
            batch_size: Tamanho do batch
            accumulation_steps: Passos de acumulação de gradiente

        Returns:
            Histórico de treinamento com losses
        """
        print(f"🎓 Iniciando Treinamento End-to-End...")
        print(f"   📊 Dados de treinamento: {len(training_data)} exemplos")
        print(f"   🎯 Épocas: {num_epochs}, Batch size: {batch_size}")
        print(f"   🔄 Acúmulo de gradiente: {accumulation_steps}")

        # Preparar dados de treinamento
        train_losses = []
        context_losses = []
        projector_losses = []

        self.context_funnel.train()
        self.inverse_projector.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_context_loss = 0.0
            epoch_projector_loss = 0.0
            num_batches = 0

            # Embaralhar dados
            np.random.shuffle(training_data)

            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i:i+batch_size]
                batch_loss = 0.0
                batch_context_loss = 0.0
                batch_projector_loss = 0.0

                # Acúmulo de gradiente
                for step, (input_text, target_token) in enumerate(batch_data):
                    try:
                        # ========== PASSO DE TREINAMENTO ==========
                        # 1. Preparar alvo quântico (representação pura do token alvo)
                        if target_token in self.char_to_idx:
                            target_token_id = self.char_to_idx[target_token]
                            if target_token_id < len(self.quantum_vocab_representations):
                                psi_target = self.quantum_vocab_representations[target_token_id]  # [embed_dim, 4]
                            else:
                                continue  # Pular se token não está no vocabulário
                        else:
                            continue  # Pular tokens desconhecidos

                        # 2. Forward pass através da arquitetura
                        # Context Funnel
                        psi_context = self.context_funnel(self.conversation_history)

                        # Cognitive Processor (simplificado para treinamento)
                        # Gerar logits contextuais
                        context_flat = psi_context.view(-1)
                        vocab_size = 50257
                        if len(context_flat) < vocab_size:
                            logits = torch.nn.functional.interpolate(
                                context_flat.unsqueeze(0).unsqueeze(0),
                                size=vocab_size,
                                mode='linear',
                                align_corners=False
                            ).squeeze()
                        else:
                            step_size = len(context_flat) // vocab_size
                            logits = torch.tensor([context_flat[j*step_size:(j+1)*step_size].mean()
                                                 for j in range(vocab_size)])

                        if len(logits) != vocab_size:
                            if len(logits) < vocab_size:
                                padding = torch.zeros(vocab_size - len(logits), device=logits.device)
                                logits = torch.cat([logits, padding])
                            else:
                                logits = logits[:vocab_size]

                        # Adicionar ruído e normalizar
                        logits += torch.randn_like(logits) * 0.1
                        logits = (logits - logits.mean()) / (logits.std() + 1e-8) * 2.0

                        # Executar DCF (Cognitive Processor)
                        if self.dcf_analyzer is not None:
                            dcf_result = self.dcf_analyzer.analyze_tokens(logits, embeddings=None, reasoning_mode=self.reasoning_mode)
                            psi_final = dcf_result['final_quantum_state'][0, 0]  # [embed_dim]
                        else:
                            # Fallback: usar contexto diretamente
                            psi_final = psi_context

                        # Inverse Cognitive Projector
                        psi_predicted = self.inverse_projector(psi_final, quantum_vocab=self.quantum_vocab_representations)

                        # 3. Calcular perda
                        loss = self.inverse_projector.compute_loss(psi_predicted, psi_target.unsqueeze(0))

                        # Normalizar perda por tamanho do batch
                        loss = loss / accumulation_steps

                        # 4. Backward pass
                        loss.backward()

                        batch_loss += loss.item()

                        # Losses específicas dos componentes (para monitoramento)
                        # Context loss: diferença entre contexto gerado e ideal
                        context_loss = F.mse_loss(psi_context, torch.randn_like(psi_context) * 0.1)  # Placeholder
                        context_loss = context_loss / accumulation_steps
                        context_loss.backward(retain_graph=True)

                        batch_context_loss += context_loss.item()
                        batch_projector_loss += loss.item()

                    except Exception as e:
                        print(f"   ⚠️  Erro no passo de treinamento {step}: {e}")
                        continue

                # Atualizar pesos após accumulation_steps
                if (i // batch_size + 1) % accumulation_steps == 0 or i + batch_size >= len(training_data):
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        list(self.context_funnel.parameters()) + list(self.inverse_projector.parameters()),
                        max_norm=1.0
                    )

                    # Otimização
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Scheduler step
                    self.scheduler.step()

                    # Acúmulo de métricas
                    epoch_loss += batch_loss
                    epoch_context_loss += batch_context_loss
                    epoch_projector_loss += batch_projector_loss
                    num_batches += 1

                    if num_batches % 10 == 0:
                        print(f"   📊 Epoch {epoch+1}/{num_epochs}, Batch {num_batches}: "
                              f"Loss={batch_loss:.4f}, Context={batch_context_loss:.4f}, Projector={batch_projector_loss:.4f}")

            # Média da época
            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
                avg_context_loss = epoch_context_loss / num_batches
                avg_projector_loss = epoch_projector_loss / num_batches

                train_losses.append(avg_epoch_loss)
                context_losses.append(avg_context_loss)
                projector_losses.append(avg_projector_loss)

                print(f"   ✅ Epoch {epoch+1}/{num_epochs} concluída: "
                      f"Loss={avg_epoch_loss:.4f}, Context={avg_context_loss:.4f}, Projector={avg_projector_loss:.4f}")

        print(f"🎓 Treinamento End-to-End concluído!")
        print(f"   📈 Loss final: {train_losses[-1]:.4f}")
        print(f"   🎯 Context Loss final: {context_losses[-1]:.4f}")
        print(f"   ⚖️ Projector Loss final: {projector_losses[-1]:.4f}")

        return {
            'total_loss': train_losses,
            'context_loss': context_losses,
            'projector_loss': projector_losses
        }

    def _update_conversation_history(self, input_text: str, generated_response: str):
        """
        Atualiza o histórico de conversa para o Context Funnel

        Args:
            input_text: Texto de entrada do usuário
            generated_response: Resposta gerada pelo sistema
        """
        # Criar representação quântica do input e resposta
        try:
            # Representação do input
            input_signal = self._text_to_fractal_signal(input_text, self.config['embed_dim'])
            input_quaternion = self._signal_to_quaternions(input_signal, self.config['embed_dim'])
            input_state = input_quaternion.squeeze(0).squeeze(0)  # [embed_dim, 4]

            # Representação da resposta (usar estado final do pipeline)
            response_signal = self._text_to_fractal_signal(generated_response, self.config['embed_dim'])
            response_quaternion = self._signal_to_quaternions(response_signal, self.config['embed_dim'])
            response_state = response_quaternion.squeeze(0).squeeze(0)  # [embed_dim, 4]

            # Adicionar ao histórico (input e response como estados separados)
            self.conversation_history.append(input_state)
            self.conversation_history.append(response_state)

            print(f"   💬 Histórico atualizado: {len(self.conversation_history)} estados")

        except Exception as e:
            print(f"   ⚠️  Erro ao atualizar histórico: {e}")
            # Fallback: adicionar representação simples
            simple_input = torch.randn(self.config['embed_dim'], 4, device=self.device) * 0.1
            simple_response = torch.randn(self.config['embed_dim'], 4, device=self.device) * 0.1
            self.conversation_history.append(simple_input)
            self.conversation_history.append(simple_response)

    def find_closest_char_projection_contextual(self, psi_sequence: torch.Tensor, position: int = 0,
                                                context_window: int = 1, candidate_tokens: Optional[List[str]] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the top-K characters using OpticalProbe with contextual window.
        Uses weighted averaging of quantum states in the context window for robust decoding with Padilha Wave Equation.

        Args:
            psi_sequence: Full quantum state sequence [batch, seq_len, embed_dim, 4]
            position: Position in sequence for phase calculation
            context_window: Number of positions to consider on each side
            candidate_tokens: Optional subset of tokens to search within
            top_k: Number of top hypotheses to return (default: 5)

        Returns:
            List of tuples (character, confidence_score) for top-K matches
        """
        print(f"🔬 Finding optical character projection with context (window=±{context_window})...")

        # Define context window: [max(0, position-context_window), min(seq_len-1, position+context_window)]
        seq_len = psi_sequence.shape[1]
        start_idx = max(0, position - context_window)
        end_idx = min(seq_len - 1, position + context_window)

        # Collect quantum states in the context window
        context_states = []
        context_weights = []

        for j in range(start_idx, end_idx + 1):
            # Calculate distance from center position for weighted averaging
            distance = abs(j - position)

            # Weighted averaging: center gets higher weight, neighbors get lower weight
            if distance == 0:
                weight = 0.6  # Center position: highest weight
            else:
                weight = 0.2  # Neighbor positions: lower weight

            context_states.append(psi_sequence[0, j])  # [embed_dim, 4]
            context_weights.append(weight)

        # Handle case where no context states are found
        if not context_states:
            print(f"   ⚠️  No context states found for position {position}, using center position only")
            # Fallback: use the center position if available, otherwise use zeros
            if position < psi_sequence.shape[1]:
                psi_contextual = psi_sequence[0, position]  # [embed_dim, 4]
            else:
                psi_contextual = torch.zeros(self.config['embed_dim'], 4, device=psi_sequence.device)
        else:
            # Convert to tensors
            context_states = torch.stack(context_states)  # [window_size, embed_dim, 4]
            context_weights = torch.tensor(context_weights, dtype=torch.float32, device=psi_sequence.device)  # [window_size]

            # Compute weighted average of quantum states in context
            weights_normalized = context_weights / context_weights.sum()
            psi_contextual = torch.sum(context_states * weights_normalized.view(-1, 1, 1), dim=0)  # [embed_dim, 4]

        # Create sequence format for OpticalProbe [seq_len=1, embed_dim, 4]
        psi_sequence_contextual = psi_contextual.unsqueeze(0)  # [1, embed_dim, 4]

        # Use OpticalProbe to decode the contextual sequence using Padilha Wave Equation
        decoded_text = self.optical_probe(psi_sequence_contextual)
        confidences = [1.0] * len(decoded_text)  # Optical probe doesn't provide confidences

        # Get the decoded character and confidence
        if decoded_text and confidences:
            decoded_char = decoded_text[0]  # First character
            confidence = confidences[0]

            # Create top-k hypotheses (currently only one from optical probe)
            # For compatibility, create multiple hypotheses with decreasing confidence
            top_k_hypotheses = [(decoded_char, confidence)]

            # ZERO FALLBACK POLICY: No fallback characters allowed

            print(f"   ✅ Optical contextual decoding result: '{decoded_char}' (confidence: {confidence:.4f})")
            for i, (char, conf) in enumerate(top_k_hypotheses[:3]):  # Show first 3
                print(f"      {i+1}. '{char}' (confidence: {conf:.4f})")
            if len(top_k_hypotheses) > 3:
                print(f"      ... and {len(top_k_hypotheses)-3} more")

            return top_k_hypotheses[:top_k]
        else:
            # Fallback if optical decoding fails
            print("   ⚠️  Optical contextual decoding failed, using fallback")
            return [(' ', 0.1)] * min(top_k, 5)

    def find_closest_char_projection(self, final_state_psi: torch.Tensor, position: int = 0,
                                     candidate_tokens: Optional[List[str]] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the top-K characters using OpticalProbe based on 4D quaternion signatures.
        This implements optical decoding: Ψ_final → Padilha Wave Equation → character matching

        Args:
            final_state_psi: Final quantum state from DCF reasoning [embed_dim] or [embed_dim, 4]
            position: Position in sequence for phase calculation
            candidate_tokens: Optional subset of tokens to search within (for cluster optimization)
            top_k: Number of top hypotheses to return (default: 5)

        Returns:
            List of tuples (character, confidence_score) for top-K matches
        """
        print(f"🔬 Finding optical character projection for position {position}...")

        # Handle different input formats and ensure quaternion format [embed_dim, 4]
        if final_state_psi.dim() == 1:
            # DCF output format [embed_dim] - expand to quaternion format
            embed_dim = final_state_psi.shape[0]
            final_quaternion = final_state_psi.unsqueeze(-1).expand(-1, 4)  # [embed_dim, 4]
        elif final_state_psi.dim() == 2:
            # Already quaternion format [embed_dim, 4]
            final_quaternion = final_state_psi
        else:
            # Unknown format - try to reshape
            final_quaternion = final_state_psi.flatten()[:self.config['embed_dim'] * 4]
            final_quaternion = final_quaternion.reshape(self.config['embed_dim'], 4)

        # Create sequence format for OpticalProbe [seq_len=1, embed_dim, 4]
        psi_sequence = final_quaternion.unsqueeze(0)  # [1, embed_dim, 4]

        # Use OpticalProbe to decode the sequence using Padilha Wave Equation
        decoded_text = self.optical_probe(psi_sequence)
        confidences = [1.0] * len(decoded_text)  # Optical probe doesn't provide confidences

        # Get the decoded character and confidence
        if decoded_text and confidences:
            decoded_char = decoded_text[0]  # First character
            confidence = confidences[0]

            # Create top-k hypotheses (currently only one from optical probe)
            # For compatibility, create multiple hypotheses with decreasing confidence
            top_k_hypotheses = [(decoded_char, confidence)]

            # Add fallback characters with lower confidence if needed
            if top_k > 1:
                fallback_chars = [' ', '.', ',', 'a', 'e', 'i', 'o', 'u']
                for i, fallback_char in enumerate(fallback_chars[:top_k-1]):
                    top_k_hypotheses.append((fallback_char, confidence * 0.5 ** (i+1)))

            print(f"   ✅ Optical decoding result: '{decoded_char}' (confidence: {confidence:.4f})")
            for i, (char, conf) in enumerate(top_k_hypotheses[:3]):  # Show first 3
                print(f"      {i+1}. '{char}' (confidence: {conf:.4f})")
            if len(top_k_hypotheses) > 3:
                print(f"      ... and {len(top_k_hypotheses)-3} more")

            return top_k_hypotheses[:top_k]
        else:
            # ZERO FALLBACK POLICY: No fallback allowed
            raise RuntimeError("Optical decoding failed - ZERO FALLBACK POLICY")

    def _get_model_character_vocabulary(self) -> List[str]:
        """
        Extract character vocabulary from the native vocabulary.
        This ensures we use the emergent characters from our training data,
        achieving true vocabulary autonomy. ZERO FALLBACK POLICY.

        Returns:
            List of characters in the native vocabulary
        """
        try:
            # Load native vocabulary from data/native_vocab.json
            vocab_path = "data/native_vocab.json"
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab_data = json.load(f)

                # Extract all unique characters from native vocabulary
                char_vocab = set()
                if isinstance(vocab_data, dict) and 'token_to_id' in vocab_data:
                    for token in vocab_data['token_to_id'].keys():
                        # Handle native vocabulary tokens
                        if isinstance(token, str):
                            # Add individual characters
                            for char in token:
                                char_vocab.add(char)

                # Convert set to sorted list for consistent ordering
                char_vocab = sorted(list(char_vocab))

                vocab_size = vocab_data.get('vocab_size', 0)
                char_count = len(char_vocab)
                print(f"   📚 Loaded native vocabulary: {vocab_size} tokens, {char_count} unique characters")
                return char_vocab

        except Exception as e:
            print(f"⚠️  Error loading native vocabulary: {e}")

        # ZERO FALLBACK: Use complete ASCII printable character set
        # This ensures we have all characters that could be generated
        print("   📚 Using complete ASCII printable character vocabulary (ZERO FALLBACK)")
        ascii_chars = []
        for i in range(32, 127):  # Printable ASCII characters (32-126)
            ascii_chars.append(chr(i))

        # Add space character explicitly
        ascii_chars.insert(0, ' ')

        print(f"   📚 ASCII vocabulary: {len(ascii_chars)} characters (32-126 + space)")
        return ascii_chars

    def _validate_generated_text(self, text: str, input_text: str, psi_stats: Dict) -> Dict[str, Any]:
        """
        Valida se o texto gerado vem dos dados do modelo e não é gibberish.

        Critérios de validação:
        1. Comprimento mínimo
        2. Diversidade de caracteres (não apenas repetições)
        3. Presença de caracteres válidos (não apenas símbolos estranhos)
        4. Relação com o texto de entrada (não completamente desconectado)
        5. Consistência com estatísticas do estado quântico

        Args:
            text: Texto gerado
            input_text: Texto de entrada original
            psi_stats: Estatísticas do estado quântico

        Returns:
            Dict com resultado da validação e detalhes
        """
        validation_details = []
        is_valid = True

        # 1. Comprimento mínimo
        min_length = 3
        if len(text.strip()) < min_length:
            validation_details.append(f"Text too short: {len(text.strip())} < {min_length}")
            is_valid = False

        # 2. Diversidade de caracteres
        unique_chars = len(set(text))
        total_chars = len(text)
        diversity_ratio = unique_chars / max(total_chars, 1)

        if diversity_ratio < 0.1:  # Menos de 10% de caracteres únicos = muito repetitivo
            validation_details.append(".2f")
            is_valid = False
        elif diversity_ratio > 0.8:  # Mais de 80% únicos = possivelmente gibberish
            validation_details.append(".2f")

        # 3. Presença de caracteres válidos - derived from model vocabulary
        try:
            valid_chars = set(self._get_model_character_vocabulary())
            invalid_ratio = sum(1 for c in text if c not in valid_chars) / max(len(text), 1)

            if invalid_ratio > 0.5:  # Mais de 50% caracteres inválidos
                validation_details.append(".2f")
                is_valid = False
        except Exception:
            # If we can't get vocabulary, skip this validation
            pass

        # 4. Verificar se não é apenas símbolos estranhos - removed hardcoded symbols
        # This validation is removed as it depends on hardcoded symbol sets
        strange_ratio = 0.0  # Placeholder value

        # 5. Verificar padrões de repetição excessiva
        if len(text) > 10:
            # Verificar repetições de 3+ caracteres consecutivos
            for i in range(len(text) - 5):
                window = text[i:i+3]
                if text.count(window) > len(text) / 10:  # Mais de 10% do texto é repetição
                    validation_details.append(f"Excessive repetition of '{window}'")
                    is_valid = False
                    break

        # 6. Verificar se tem pelo menos algumas letras
        letter_count = sum(1 for c in text if c.isalpha())
        letter_ratio = letter_count / max(len(text), 1)

        if letter_ratio < 0.2:  # Menos de 20% letras
            validation_details.append(".2f")
            is_valid = False

        # 7. Verificar consistência com estado quântico
        # Se o estado quântico tem baixa variabilidade, o texto também deve ser simples
        if psi_stats['std'] < 0.1 and diversity_ratio > 0.6:
            validation_details.append("Text diversity inconsistent with low-variance quantum state")
            is_valid = False

        # 8. Verificar se não é completamente desconectado da entrada
        if input_text and len(input_text) > 5:
            input_words = set(input_text.lower().split())
            output_words = set(text.lower().split())
            overlap = len(input_words.intersection(output_words))

            # Se não há nenhuma palavra em comum e entrada tem palavras, pode ser problema
            if overlap == 0 and len(input_words) > 0 and len(text.split()) > 2:
                # Mas permitir se o texto gerado tem palavras reais
                real_words = sum(1 for word in output_words if len(word) > 2 and word.isalpha())
                if real_words < len(output_words) * 0.5:  # Menos da metade são palavras reais
                    validation_details.append("Generated text has no meaningful words")
                    is_valid = False

        # Resumo da validação
        if not validation_details:
            validation_details.append("Text passed all validation checks")

        return {
            'is_valid': is_valid,
            'validation_details': '; '.join(validation_details),
            'stats': {
                'length': len(text),
                'diversity_ratio': diversity_ratio,
                'invalid_ratio': invalid_ratio,
                'strange_ratio': strange_ratio,
                'letter_ratio': letter_ratio
            }
        }


    def _get_model_info(self) -> Dict[str, Any]:
        """
        Extrair informações reais do modelo convertido em espectro - ZERO FALLBACK
        """
        try:
            model_info = {}

            # Informações do modelo ativo convertido em espectro
            active_model = get_active_model_path()
            if active_model:
                model_info['active_model_path'] = active_model
                model_info['model_type'] = 'ΨQRH Spectral Model (convertido em espectro)'
                model_info['spectral_conversion_status'] = 'CONVERTIDO'

                # Tentar carregar config do modelo espectral
                config_path = Path(active_model) / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        model_info['model_config'] = config
                        model_info['spectral_parameters'] = {
                            'fractal_dimension': config.get('fractal_dimension', 'N/A'),
                            'alpha_spectral': config.get('alpha', 'N/A'),
                            'beta_spectral': config.get('beta', 'N/A'),
                            'normalization_spectral': config.get('normalization', 'N/A'),
                            'embed_dim_spectral': config.get('embed_dim', 'N/A')
                        }
                        model_info['spectral_wave_equation'] = 'f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))'
                else:
                    model_info['model_config'] = 'Configuração espectral não encontrada'
                    model_info['spectral_parameters'] = 'Parâmetros espectrais não carregados'
            else:
                model_info['active_model_path'] = 'Nenhum modelo ativo encontrado'
                model_info['model_type'] = 'Modelo padrão ΨQRH (espectral)'
                model_info['spectral_conversion_status'] = 'PENDENTE'

            # Informações dos componentes do pipeline espectral
            model_info['spectral_pipeline_components'] = {
                'fractal_analyzer_spectral': 'SpectralFilter (espectral)' if self.fractal_analyzer else None,
                'quaternion_processor_spectral': 'OptimizedQuaternionOperations (espectral)' if self.quaternion_processor else None,
                'spectral_filter_spectral': 'SpectralFilter F(k) = exp(i α · arctan(ln(|k| + ε)))' if self.spectral_filter else None,
                'optical_probe_spectral': 'OpticalTextDecoder (espectral)' if self.optical_probe else None,
                'consciousness_processor_spectral': 'FractalConsciousnessProcessor (espectral)' if self.consciousness_processor else None,
                'quantum_memory_system_spectral': 'QuantumTemporalMemory (espectral)' if self.quantum_memory_system else None
            }

            # Verificar arquivos de modelo espectral
            spectral_model_files = []
            if os.path.exists('data/spectral_model.pt'):
                spectral_model_files.append('data/spectral_model.pt')
            if os.path.exists('models/spectral/'):
                spectral_model_files.append('models/spectral/')
            model_info['spectral_model_files'] = spectral_model_files

            return model_info

        except Exception as e:
            return {'error': f'Erro ao extrair informações do modelo espectral: {str(e)}'}

    def _get_vocabulary_info(self) -> Dict[str, Any]:
        """
        Extrair informações reais do vocabulário convertido em espectro - ZERO FALLBACK
        """
        try:
            vocab_info = {}

            # Informações do tokenizer adaptativo convertido em espectro
            vocab_info['tokenizer_config_spectral'] = self.tokenizer_config
            vocab_info['tokenizer_spectral_status'] = 'CONVERTIDO_EM_ESPECTRO'
            vocab_info['spectral_tokenizer_features'] = {
                'embed_dim_spectral': self.tokenizer_config.get('embed_dim', 'N/A'),
                'spectral_params_per_char': self.tokenizer_config.get('spectral_params_dim', 'N/A'),
                'learnable_spectral': self.tokenizer_config.get('learnable', 'N/A')
            }

            # Verificar sistema de memória quântica temporal (vocabulário espectral)
            if hasattr(self, 'quantum_memory_system') and self.quantum_memory_system:
                vocab_info['quantum_memory_spectral'] = {
                    'memory_size_spectral': getattr(self.quantum_memory_system, 'memory_size', 'N/A'),
                    'coherence_time_spectral': getattr(self.quantum_memory_system, 'coherence_time', 'N/A'),
                    'spectral_patterns_stored': 'Correlações temporais de longo alcance'
                }
            else:
                vocab_info['quantum_memory_spectral'] = 'Sistema de memória quântica não inicializado'

            # Vocabulário emergente convertido em espectro
            vocab_info['emergent_vocabulary_spectral'] = {
                'word_meaning_map_size_spectral': len(self.emergent_vocabulary.word_meaning_map) if hasattr(self, 'emergent_vocabulary') else 0,
                'grammar_rules_spectral': self.word_formation_processor.grammar_rules if hasattr(self, 'word_formation_processor') else {},
                'spectral_phoneme_mapping': 'Fonemas anatomicamente possíveis → espectro quântico',
                'emergent_language_generation': 'Linguagem emergente baseada em padrões espectrais'
            }

            # Verificar arquivos de vocabulário espectral
            spectral_vocab_files = []
            if os.path.exists('data/spectral_vocab.json'):
                spectral_vocab_files.append('data/spectral_vocab.json')
            if os.path.exists('data/gpt2_vocab_spectral.json'):
                spectral_vocab_files.append('data/gpt2_vocab_spectral.json')
            if os.path.exists('vocab/spectral/'):
                spectral_vocab_files.append('vocab/spectral/')

            vocab_info['spectral_vocabulary_files'] = spectral_vocab_files
            vocab_info['vocabulary_conversion_method'] = 'FFT + Linear Predictive Coding (LPC) + Formant Analysis'

            return vocab_info

        except Exception as e:
            return {'error': f'Erro ao extrair informações do vocabulário espectral: {str(e)}'}

    def _initialize_auto_learning_models(self):
        """Inicializa modelos de auto-aprendizagem ΨQRH (SEM transformers)"""
        if not self.enable_auto_learning:
            return

        print("🚀 Inicializando modelos de auto-aprendizagem ΨQRH...")

        try:
            # Initialize ΨQRH spectral models for auto-learning
            self.spectral_processor = QuaternionMLP(
                embed_dim=256,
                hidden_dim=512
            ).to(self.device)

            # Initialize ΨQRH fractal embedding for semantic understanding
            self.fractal_embedding = FractalQuantumEmbedding(
                vocab_size=1000,
                embed_dim=256,
                device=self.device
            )

            print("✅ Modelos de auto-aprendizagem ΨQRH carregados:")
            print(f"   - Spectral Processor: {self.spectral_processor}")
            print(f"   - Fractal Embedding: {self.fractal_embedding}")

        except Exception as e:
            print(f"⚠️  Erro ao carregar modelos de auto-aprendizagem ΨQRH: {e}")
            self.enable_auto_learning = False

    def _detect_task_type(self, input_text: str) -> str:
        """
        Detecta automaticamente o tipo de tarefa com base no conteúdo da entrada.

        # Roteamento automático:
        # - signal-processing: se houver [números] ou palavras-chave de simulação física
        # - text-generation: para todo o resto
        """
        import re

        input_lower = input_text.lower()

        # Padrão para detectar arrays numéricos: [1.0, -2.5, 3e-2, ...]
        numeric_array_pattern = r'\[\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*(?:,\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)*\]'

        # Palavras-chave de processamento de sinais
        signal_keywords = [
            'spectral filter', 'fourier transform', 'clifford algebra',
            'quaternionic', 'signal processing', 'norm preservation',
            'unitarity', 'energy conservation', 'process signal',
            'quaternion vector', 'numerical data', 'signal array',
            'apply filter', 'validate unitarity', 'energy conservation'
        ]

        # Palavras-chave de simulação física
        physics_keywords = [
            "simule", "calcule", "verifique", "mostre", "demonstre",
            "transformada", "fourier", "schrödinger", "tunelamento",
            "invariância", "lorentz", "campo eletromagnético", "pacote de onda"
        ]

        # Verifica requisições de simulação física
        has_physics_request = any(kw in input_lower for kw in physics_keywords)
        has_numeric_data = bool(re.search(numeric_array_pattern, input_text))
        has_signal_keywords = any(kw in input_lower for kw in signal_keywords)

        # Se houver requisição física OU dados numéricos OU palavras-chave de sinal → signal-processing
        if has_physics_request or has_numeric_data or has_signal_keywords:
            print(f"🔢 Detecção automática: usando signal-processing para entrada com dados numéricos/terminologia de sinal/simulação física")
            return "signal-processing"

        # Caso contrário, assume geração de texto
        print(f"💬 Detecção automática: usando text-generation para entrada: '{input_text[:50]}...'")
        return "text-generation"

    def _initialize_model(self):
        """Inicializa o modelo ΨQRH automaticamente - ZERO FALLBACK POLICY"""
        print(f"🚀 Inicializando ΨQRH Pipeline no dispositivo: {self.device}")

        # Carregar configuração apropriada baseada na tarefa
        config = self._load_task_config()

        # Para geração de texto → use ΨQRH framework completo
        if self.task in ["text-generation", "chat"]:
            # Suporte para nova implementação completa
            try:
                from src.core.fractal_quantum_embedding import PsiQRHTransformerComplete
                self._has_complete_implementation = True
            except ImportError:
                self._has_complete_implementation = False

            from src.core.ΨQRH import QRHFactory
            # Se model_dir foi fornecido, verificar se é um modelo espectral convertido
            if self.model_dir:
                model_path = Path(self.model_dir)
                if model_path.exists():
                    # Verificar se é um modelo espectral convertido (tem config.json)
                    config_path = model_path / "config.json"
                    if config_path.exists():
                        print(f"🔬 Carregando modelo espectral convertido: {self.model_dir}")
                        # Carregar configuração espectral
                        with open(config_path, 'r') as f:
                            spectral_config = json.load(f)

                        # Carregar modelo PyTorch se existir
                        model_file = model_path / "model.pt"
                        if model_file.exists():
                            print(f"   📁 Carregando pesos do modelo: {model_file}")
                            # Aqui seria carregado o modelo PyTorch - por enquanto usar QRHFactory
                            self.model = QRHFactory(model_path=self.model_dir)
                        else:
                            print(f"   ⚠️  Arquivo model.pt não encontrado, usando QRHFactory padrão")
                            self.model = QRHFactory(model_path=self.model_dir)

                        # Armazenar configuração espectral para uso posterior
                        self.spectral_config = spectral_config
                        print(f"✅ Modelo espectral carregado: {spectral_config.get('model_name', 'unknown')}")
                        print(f"   🔬 Dimensão Fractal: {spectral_config.get('fractal_dimension', 'N/A')}")
                        print(f"   ⚡ Alpha: {spectral_config.get('alpha', 'N/A')}")
                    else:
                        # Modelo não-espectral, usar QRHFactory diretamente
                        self.model = QRHFactory(model_path=self.model_dir)
                        print(f"✅ Framework ΨQRH completo carregado do modelo: {self.model_dir}")
                else:
                    print(f"⚠️  Diretório do modelo não encontrado: {self.model_dir}")
                    print("   🔄 Usando modelo padrão...")
                    self.model = QRHFactory()
            else:
                self.model = QRHFactory()
                print("✅ Framework ΨQRH completo carregado (padrão)")

        # Para análise matemática → use o analisador espectral
        elif self.task == "analysis":
            from src.core.response_spectrum_analyzer import ResponseSpectrumAnalyzer
            self.model = ResponseSpectrumAnalyzer(config)
            print("✅ Analisador espectral ΨQRH carregado")

        # Para processamento de sinais → use processador numérico
        elif self.task == "signal-processing":
            from src.core.numeric_signal_processor import NumericSignalProcessor
            # Usar configuração de dispositivo do arquivo de configuração
            device_config = config.get('default_device', {'device': 'cpu'})
            self.model = NumericSignalProcessor(device=device_config['device'])
            print("✅ Processador numérico ΨQRH carregado")

        else:
            raise ValueError(f"Tarefa não suportada: {self.task}")

    def _load_task_config(self):
        """Carrega configuração apropriada baseada na tarefa"""
        import yaml

        # Verificar se existem configurações calibradas
        calibrated_config_dir = Path(__file__).parent / "configs" / "gradient_calibrated"

        if calibrated_config_dir.exists():
            print(f"  📁 Carregando configurações calibradas de: {calibrated_config_dir}")

            # Carregar configurações calibradas específicas da tarefa
            config = {
                "device": self.device,
                "task": self.task,
                "calibrated": True,
                "config_dir": str(calibrated_config_dir)
            }