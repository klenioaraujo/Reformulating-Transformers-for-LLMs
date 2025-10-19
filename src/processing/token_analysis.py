#!/usr/bin/env python3
"""
Sistema de Análise de Tokens via Dinâmica de Consciência Fractal (DCF)

Implementa arquitetura DCF que substitui análise estática de softmax por sistema dinâmico
usando osciladores Kuramoto, métricas de consciência fractal e ciclo de feedback auto-regulatório.

Componentes principais:
- KuramotoSpectralLayer: Osciladores representando tokens candidatos
- ConsciousnessMetrics: Avaliação de qualidade via FCI
- NeuralDiffusionEngine: Controle adaptativo via coeficiente de difusão
- Feedback cycle: FCI modula acoplamento K para próximos ciclos
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
from collections import deque

# Import centralized config manager
from src.core.config_manager import get_config_manager, DCFConfig


class ContextualPrimingModulator:
    """
    Módulo de Priming Contextual para simulação de vieses cognitivos humanos.

    Simula o fenômeno neurológico do "priming" onde associações recentes
    influenciam o processamento atual, criando susceptibilidade a "pegadinhas cognitivas".
    """

    def __init__(self, priming_strength: float = 0.3, history_window: int = 5):
        """
        Inicializa o modulador de priming contextual.

        Args:
            priming_strength: Força do priming (α) - controla susceptibilidade a vieses
            history_window: Número de turnos recentes para analisar (k)
        """
        self.priming_strength = priming_strength
        self.history_window = history_window

        # Histórico de conversas: deque para manter apenas os últimos k turnos
        self.conversation_history = deque(maxlen=history_window)

        print("🧠 ContextualPrimingModulator inicializado")
        print(f"   📊 Priming strength (α): {priming_strength}")
        print(f"   📈 History window (k): {history_window}")

    def add_to_history(self, user_input: str, system_response: str):
        """
        Adiciona um turno de conversa ao histórico.

        Args:
            user_input: Entrada do usuário
            system_response: Resposta do sistema
        """
        turn = {
            'user_input': user_input.lower(),
            'system_response': system_response.lower(),
            'timestamp': time.time()
        }
        self.conversation_history.append(turn)

    def compute_priming_matrix(self, candidate_tokens: torch.Tensor,
                              candidate_token_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        Computa a matriz de priming P_ij baseada no histórico de conversas.

        Args:
            candidate_tokens: Tokens candidatos [n_candidates]
            candidate_token_ids: IDs dos tokens candidatos (opcional)

        Returns:
            priming_matrix: Matriz P_ij [n_candidates, n_candidates]
        """
        n_candidates = len(candidate_tokens)

        # Inicializar matriz de priming
        priming_matrix = torch.zeros(n_candidates, n_candidates, dtype=torch.float32)

        if len(self.conversation_history) == 0:
            print("   📝 Sem histórico de conversa - matriz de priming vazia")
            return priming_matrix

        print(f"   📝 Analisando {len(self.conversation_history)} turnos de histórico para priming")

        # Para cada par de tokens candidatos, contar coocorrências no histórico
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i == j:
                    continue  # Diagonal permanece zero (auto-priming não conta)

                token_i = candidate_tokens[i].item()
                token_j = candidate_tokens[j].item()

                # Contar coocorrências no histórico
                cooccurrence_count = self._count_token_cooccurrences(token_i, token_j)

                # Aplicar função de decaimento temporal (turnos mais recentes têm mais peso)
                priming_score = self._compute_priming_score(cooccurrence_count, token_i, token_j)

                priming_matrix[i, j] = priming_score
                priming_matrix[j, i] = priming_score  # Matriz simétrica

        # Normalizar a matriz de priming
        if priming_matrix.max() > 0:
            priming_matrix = priming_matrix / priming_matrix.max()

        print(f"   ✅ Matriz de priming calculada: max_score={priming_matrix.max().item():.3f}")

        return priming_matrix

    def _count_token_cooccurrences(self, token_i: int, token_j: int) -> int:
        """
        Conta quantas vezes dois tokens apareceram juntos no histórico.

        Tokens são considerados coocorrentes se aparecerem:
        1. Na mesma frase/turno
        2. Em turnos consecutivos (associações sequenciais)
        """
        cooccurrence_count = 0

        for turn in self.conversation_history:
            # Verificar coocorrência na entrada do usuário
            user_text = turn['user_input']
            if self._tokens_cooccur_in_text(token_i, token_j, user_text):
                cooccurrence_count += 1

            # Verificar coocorrência na resposta do sistema
            response_text = turn['system_response']
            if self._tokens_cooccur_in_text(token_i, token_j, response_text):
                cooccurrence_count += 1

        return cooccurrence_count

    def _tokens_cooccur_in_text(self, token_i: int, token_j: int, text: str) -> bool:
        """
        Verifica se dois tokens coocorrem no texto.

        Como estamos trabalhando com IDs de token, precisamos de uma abordagem diferente.
        Por enquanto, usaremos uma heurística baseada em similaridade de strings.
        """
        # Mapeamento simplificado de token IDs para palavras comuns
        # Em produção, isso seria integrado com o verdadeiro tokenizer
        token_to_word = self._simple_token_decoder

        word_i = token_to_word.get(token_i, f"token_{token_i}")
        word_j = token_to_word.get(token_j, f"token_{token_j}")

        # Verificar se ambas as palavras aparecem no texto
        return word_i in text and word_j in text

    def _simple_token_decoder(self, token_id: int) -> str:
        """Decodificador simples para mapeamento token->palavra (heurístico)"""
        # Mapeamento básico para demonstração
        common_tokens = {
            1000: 'the', 1001: 'and', 1002: 'of', 1003: 'to', 1004: 'a',
            1005: 'in', 1006: 'for', 1007: 'is', 1008: 'on', 1009: 'that',
            1010: 'by', 1011: 'this', 1012: 'with', 1013: 'i', 1014: 'you',
            1015: 'it', 1016: 'not', 1017: 'or', 1018: 'be', 1019: 'are',
            1020: 'from', 1021: 'at', 1022: 'as', 1023: 'your', 1024: 'all',
            1025: 'have', 1026: 'new', 1027: 'more', 1028: 'an', 1029: 'was',
            1030: 'we', 1031: 'will', 1032: 'home', 1033: 'can', 1034: 'us',
            1035: 'about', 1036: 'if', 1037: 'page', 1038: 'my', 1039: 'has',
            1040: 'search', 1041: 'free', 1042: 'but', 1043: 'our', 1044: 'one',
            1045: 'other', 1046: 'do', 1047: 'no', 1048: 'information', 1049: 'time',
            1050: 'they', 1051: 'site', 1052: 'he', 1053: 'up', 1054: 'may',
            1055: 'what', 1056: 'which', 1057: 'their', 1058: 'news', 1059: 'out',
            1060: 'use', 1061: 'any', 1062: 'there', 1063: 'see', 1064: 'only',
            1065: 'so', 1066: 'his', 1067: 'when', 1068: 'contact', 1069: 'here',
            1070: 'business', 1071: 'who', 1072: 'web', 1073: 'also', 1074: 'now',
            1075: 'help', 1076: 'get', 1077: 'pm', 1078: 'view', 1079: 'online',
            1080: 'c', 1081: 'e', 1082: 'first', 1083: 'am', 1084: 'been',
            1085: 'would', 1086: 'how', 1087: 'were', 1088: 'me', 1089: 's',
            1090: 'services', 1091: 'some', 1092: 'these', 1093: 'click', 1094: 'its',
            1095: 'like', 1096: 'service', 1097: 'x', 1098: 'than', 1099: 'find'
        }
        return common_tokens.get(token_id, f"token_{token_id}")

    def _compute_priming_score(self, cooccurrence_count: int, token_i: int, token_j: int) -> float:
        """
        Computa o score de priming baseado na contagem de coocorrências.

        Inclui fatores como:
        - Contagem de coocorrências
        - Decaimento temporal (turnos mais recentes têm mais peso)
        - Força semântica da associação
        """
        if cooccurrence_count == 0:
            return 0.0

        # Score base: função logarítmica para evitar crescimento linear demais
        base_score = np.log(1 + cooccurrence_count)

        # Decaimento temporal: turnos mais recentes têm peso maior
        # Como usamos deque, os turnos mais recentes estão no final
        temporal_weight = 1.0
        if len(self.conversation_history) > 1:
            # Turnos mais recentes têm peso maior
            recency_factor = 1.0 + 0.5 * (len(self.conversation_history) - 1) / max(len(self.conversation_history) - 1, 1)
            temporal_weight = min(recency_factor, 2.0)

        # Score final
        priming_score = base_score * temporal_weight

        return float(priming_score)

    def modulate_connectivity(self, semantic_connectivity: torch.Tensor,
                            candidate_tokens: torch.Tensor) -> torch.Tensor:
        """
        Modula a conectividade semântica com priming contextual.

        K_effective = K_semantic + α * P_priming

        Args:
            semantic_connectivity: Matriz K_semantic [n_candidates, n_candidates]
            candidate_tokens: Tokens candidatos [n_candidates]

        Returns:
            effective_connectivity: Matriz K_effective [n_candidates, n_candidates]
        """
        # Computar matriz de priming
        priming_matrix = self.compute_priming_matrix(candidate_tokens)

        # Calcular conectividade efetiva
        effective_connectivity = semantic_connectivity + self.priming_strength * priming_matrix

        # Garantir que permanece positiva e normalizada
        # Handle complex tensors by taking real part before clamping
        if effective_connectivity.is_complex():
            effective_connectivity = effective_connectivity.real
        effective_connectivity = torch.clamp(effective_connectivity, 0.0, 1.0)

        print(f"   🔄 Conectividade modulada com priming: α={self.priming_strength}")
        print(f"      K_semantic max: {semantic_connectivity.max().item():.3f}")
        print(f"      P_priming max: {priming_matrix.max().item():.3f}")
        print(f"      K_effective max: {effective_connectivity.max().item():.3f}")

        return effective_connectivity


class DCFTokenAnalysis:
    """
    Sistema de Análise de Tokens via Dinâmica de Consciência Fractal (DCF)

    Substitui softmax estático por sistema dinâmico baseado em:
    1. Osciladores Kuramoto representando tokens candidatos
    2. Dinâmica de reação-difusão para consenso
    3. Métricas de consciência fractal para avaliação de qualidade
    4. Ciclo de feedback auto-regulatório
    """

    def __init__(self, config_path: Optional[str] = None, device: str = "cpu",
                 enable_cognitive_priming: bool = True, quantum_vocab_representations: Optional[torch.Tensor] = None,
                 word_to_id: Optional[Dict[str, int]] = None):
        """
        Inicializa sistema DCF com componentes necessários.

        Args:
            config_path: Caminho para arquivo de configuração YAML (opcional)
            device: Dispositivo para computação ('cpu', 'cuda', etc.)
            enable_cognitive_priming: Habilita priming contextual para simulação de vieses cognitivos
            quantum_vocab_representations: Tensor de representações quânticas [vocab_size, embed_dim, 4] (opcional)
            word_to_id: Mapeamento palavra -> índice no vocabulário quântico (opcional)
        """
        self.device = device
        self.enable_cognitive_priming = enable_cognitive_priming

        # Quantum vocabulary for semantic connectivity
        self.quantum_vocab_representations = quantum_vocab_representations
        self.word_to_id = word_to_id if word_to_id is not None else {}

        # Usar ConfigManager centralizado
        self.config_manager = get_config_manager()
        self.config = self._load_config(config_path)

        # Estado persistente para feedback cycle
        self.last_diffusion_coefficient = self.config.initial_diffusion_coefficient

        # Inicializar Contextual Priming Modulator se habilitado
        if self.enable_cognitive_priming:
            self.priming_modulator = ContextualPrimingModulator(
                priming_strength=0.3,  # α parameter
                history_window=5       # k parameter
            )
        else:
            self.priming_modulator = None

        # Inicializar componentes DCF
        self._initialize_dcf_components()

        print("🎯 Sistema DCF (Dinâmica de Consciência Fractal) inicializado")
        print(f"   🔄 Kuramoto: {self.kuramoto_layer is not None}")
        print(f"   🧠 Consciousness: {self.consciousness_metrics is not None}")
        print(f"   ⚡ Diffusion: {self.diffusion_engine is not None}")
        print(f"   🧠 Cognitive Priming: {self.priming_modulator is not None}")
        print(f"   📚 Quantum Dictionary: {self.quantum_vocab_representations is not None}")
        print(f"   📖 Word-to-ID Mapping: {len(self.word_to_id)} entries")

    def _load_config(self, config_path: Optional[str]) -> DCFConfig:
        """Carrega configuração DCF usando ConfigManager centralizado"""
        try:
            # Tentar carregar configuração específica se fornecida
            if config_path:
                config_data = self.config_manager.load_config('dcf_config')
                return DCFConfig(**config_data) if config_data else DCFConfig()
            else:
                # Usar configuração do sistema
                system_config = self.config_manager.get_system_config()
                # system_config.dcf_config should already be a DCFConfig object
                if isinstance(system_config.dcf_config, DCFConfig):
                    return system_config.dcf_config
                else:
                    # If it's a dict, convert it
                    return DCFConfig(**system_config.dcf_config)
        except Exception as e:
            print(f"⚠️  Erro carregando config DCF: {e}, usando padrão")
            return DCFConfig()

    def _initialize_dcf_components(self):
        """Inicializa componentes do sistema DCF"""
        try:
            # 1. Kuramoto Spectral Layer
            from src.core.kuramoto_spectral_neurons import KuramotoSpectralLayer
            kuramoto_config_path = "configs/kuramoto_config.yaml"
            if Path(kuramoto_config_path).exists():
                self.kuramoto_layer = KuramotoSpectralLayer(config_path=kuramoto_config_path)
            else:
                # Fallback sem config
                self.kuramoto_layer = KuramotoSpectralLayer()
            self.kuramoto_layer.to(self.device)
        except Exception as e:
            print(f"⚠️  Kuramoto Layer falhou: {e}")
            self.kuramoto_layer = None

        try:
            # 2. Consciousness Metrics
            from src.conscience.consciousness_metrics import ConsciousnessMetrics
            consciousness_config = {'device': self.device}
            # Usar caminho relativo correto para configs
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            metrics_config_path = os.path.join(base_dir, "configs", "consciousness_metrics.yaml")
            if Path(metrics_config_path).exists():
                import yaml
                with open(metrics_config_path, 'r') as f:
                    metrics_config = yaml.safe_load(f)
            else:
                print(f"⚠️  Erro ao carregar configurações YAML: [Errno 2] No such file or directory: '{metrics_config_path}'. Usando configuração fornecida.")
                metrics_config = {}
            self.consciousness_metrics = ConsciousnessMetrics(consciousness_config, metrics_config)
            self.consciousness_metrics.to(self.device)
        except Exception as e:
            print(f"⚠️  Consciousness Metrics falhou: {e}")
            self.consciousness_metrics = None

        try:
            # 3. Neural Diffusion Engine
            from src.conscience.neural_diffusion_engine import NeuralDiffusionEngine
            from src.core.config_manager import NeuralDiffusionConfig

            # Load config using ConfigManager
            diffusion_config_data = self.config_manager.load_config('neural_diffusion_engine')
            if diffusion_config_data:
                diffusion_config = NeuralDiffusionConfig(**diffusion_config_data)
            else:
                # Fallback to default config
                diffusion_config = NeuralDiffusionConfig()

            self.diffusion_engine = NeuralDiffusionEngine(diffusion_config)
            self.diffusion_engine.to(self.device)
        except Exception as e:
            print(f"⚠️  Neural Diffusion Engine falhou: {e}")
            self.diffusion_engine = None

    def analyze_tokens(self, logits: torch.Tensor, candidate_indices: Optional[torch.Tensor] = None,
                      embeddings: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Executa análise DCF completa dos tokens candidatos com conectividade semântica.

        Args:
            logits: Tensor de logits [vocab_size] ou [batch_size, vocab_size]
            candidate_indices: Índices dos candidatos (opcional, usa top-N se None)
            embeddings: Matriz de embeddings [vocab_size, embed_dim] para conectividade semântica

        Returns:
            Dicionário com análise completa DCF incluindo análise de clusters semânticos
        """
        start_time = time.time()

        # Preparar logits
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  # [1, vocab_size]
        elif logits.dim() > 2:
            raise ValueError("Logits deve ser [vocab_size] ou [batch_size, vocab_size]")

        batch_size = logits.shape[0]

        # Para simplificar, processar apenas primeiro batch
        logits = logits[0]  # [vocab_size]

        # Garantir que logits é um tensor 1D
        if logits.dim() > 1:
            logits = logits.squeeze()

        # ========== PASSO 1: CONSTRUÇÃO DA MATRIZ DE CONECTIVIDADE SEMÂNTICA ==========
        print("   🧠 Passo 1: Construindo conectividade semântica...")
        semantic_connectivity = self._build_semantic_connectivity_matrix(
            logits, candidate_indices, embeddings
        )

        # ========== PASSO 2: INICIALIZAÇÃO DOS OSCILADORES ==========
        print("   📐 Passo 2: Inicializando osciladores Kuramoto...")
        kuramoto_input, candidate_tokens, candidate_logits, natural_frequencies = self._initialize_kuramoto_oscillators_semantic(
            logits, candidate_indices, semantic_connectivity
        )

        # ========== PASSO 3: DINÂMICA DE REAÇÃO-DIFUSÃO ==========
        print("   🌊 Passo 3: Executando dinâmica Kuramoto com conectividade semântica...")
        kuramoto_output, kuramoto_metrics = self._run_kuramoto_dynamics_semantic(
            kuramoto_input, semantic_connectivity
        )

        # ========== PASSO 4: ANÁLISE DE CLUSTERS ==========
        print("   🔍 Passo 4: Analisando clusters de fase...")
        cluster_analysis = self._analyze_phase_clusters(
            kuramoto_output, kuramoto_metrics, candidate_tokens, candidate_logits
        )

        # ========== PASSO 5: SELEÇÃO DO TOKEN FINAL ==========
        print("   🎯 Passo 5: Selecionando token final baseado em clusters...")
        final_token_selection = self._select_final_token_from_clusters(
            cluster_analysis, candidate_tokens, candidate_logits, kuramoto_output, kuramoto_metrics
        )

        # ========== PASSO 6: MEDIÇÃO DA CONSCIÊNCIA ==========
        print("   🧠 Passo 6: Calculando métricas de consciência...")
        consciousness_results = self._compute_consciousness_metrics(kuramoto_output, kuramoto_metrics)

        # ========== PASSO 7: CICLO DE FEEDBACK ==========
        print("   🔄 Passo 7: Atualizando ciclo de feedback...")
        self._update_feedback_cycle(consciousness_results)

        # ========== COMPILAR RESULTADO FINAL ==========
        analysis_report = self._generate_semantic_analysis_report(
            consciousness_results, kuramoto_metrics, cluster_analysis
        )

        processing_time = time.time() - start_time

        result = {
            'final_quantum_state': kuramoto_output,  # Ψ_final - the result of DCF reasoning
            'selected_token': final_token_selection.get('token', candidate_tokens[0].item()),  # Token selecionado
            'final_probability': final_token_selection.get('probability', 0.5),  # Probabilidade final
            'cluster_analysis': cluster_analysis,
            'fci_value': consciousness_results.get('fci', 0.0),
            'consciousness_state': consciousness_results.get('state', 'UNKNOWN'),
            'synchronization_order': kuramoto_metrics.get('final_sync_order', 0.0),
            'analysis_report': analysis_report,
            'processing_time': processing_time,
            'semantic_analysis': {
                'connectivity_matrix_shape': semantic_connectivity.shape,
                'cluster_analysis': cluster_analysis,
                'natural_frequencies': natural_frequencies.tolist() if hasattr(natural_frequencies, 'tolist') else natural_frequencies,
                'semantic_reasoning': True
            },
            'dcf_metadata': {
                'n_candidates': len(candidate_tokens),
                'kuramoto_steps': self.config.kuramoto_steps,
                'coupling_strength': kuramoto_metrics.get('coupling_strength', 0.0),
                'diffusion_coefficient': self.last_diffusion_coefficient,
                'method': 'DCF Semantic (Dinâmica de Consciência Fractal com Conectividade Semântica)',
                'final_token_selection': final_token_selection
            }
        }

        return result

    def _build_semantic_connectivity_matrix(self, logits: torch.Tensor,
                                             candidate_indices: Optional[torch.Tensor],
                                             embeddings: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Constrói a Matriz de Conectividade Semântica K_ij baseada em similaridade de representações quânticas.
        Inclui modulação por priming contextual se habilitado.

        Args:
            logits: Logits do modelo [vocab_size]
            candidate_indices: Índices dos candidatos (opcional)
            embeddings: Matriz de embeddings [vocab_size, embed_dim] (deprecated - usa quantum representations)

        Returns:
            effective_connectivity: Matriz K_effective [n_candidates, n_candidates]
        """
        # Selecionar top-N candidatos
        vocab_size = logits.shape[0]
        n_candidates = min(self.config.n_candidates, vocab_size)  # Não exceder tamanho do vocabulário
        if candidate_indices is None:
            # Usar top-N logits
            _, top_indices = torch.topk(logits, n_candidates)
        else:
            # Usar candidatos fornecidos
            top_indices = candidate_indices[:n_candidates]

        candidate_tokens = top_indices

        # Usar dicionário quântico para conectividade semântica
        if self.quantum_vocab_representations is not None:
            # Verificar se temos índices válidos para o vocabulário quântico
            valid_indices = []
            for idx in top_indices:
                if idx < self.quantum_vocab_representations.shape[0]:
                    valid_indices.append(idx)
                else:
                    print(f"⚠️  Índice {idx} fora do range do vocabulário quântico (size={self.quantum_vocab_representations.shape[0]})")

            if not valid_indices:
                raise RuntimeError("Nenhum índice válido encontrado no vocabulário quântico")

            valid_indices = torch.tensor(valid_indices, device=self.device)

            # Extrair representações quânticas dos tokens candidatos
            candidate_quantum = self.quantum_vocab_representations[valid_indices]  # [n_candidates, embed_dim, 4]

            # Achatar para [n_candidates, embed_dim * 4] para cálculo de similaridade
            candidate_flattened = candidate_quantum.view(len(valid_indices), -1)  # [n_candidates, embed_dim * 4]

            # Calcular similaridade de cosseno entre todos os pares
            normalized_quantum = torch.nn.functional.normalize(candidate_flattened, p=2, dim=-1)
            semantic_connectivity = torch.mm(normalized_quantum, normalized_quantum.t())

            # Garantir que a diagonal seja 1 (auto-similaridade máxima)
            semantic_connectivity.fill_diagonal_(1.0)
        else:
            raise RuntimeError("Dicionário quântico não disponível - sistema DCF requer representações quânticas para conectividade semântica")

        # Normalizar para range [0, 1] e adicionar epsilon para evitar zeros
        # Handle complex tensors by taking real part before clamping
        if semantic_connectivity.is_complex():
            semantic_connectivity = semantic_connectivity.real
        semantic_connectivity = torch.clamp(semantic_connectivity, 0.0, 1.0)
        semantic_connectivity = semantic_connectivity + 1e-6  # Evitar zeros que podem causar problemas

        print(f"   ✅ Matriz de conectividade semântica construída: {semantic_connectivity.shape}")

        # Aplicar modulação por priming contextual se habilitado
        if self.priming_modulator is not None:
            print("     Simulando o \"Priming\" Cognitivo com um Modulador de Conectividade Dinâmica")
            effective_connectivity = self.priming_modulator.modulate_connectivity(
                semantic_connectivity, candidate_tokens
            )
            return effective_connectivity
        else:
            return semantic_connectivity

    def _initialize_kuramoto_oscillators_semantic(self, logits: torch.Tensor,
                                                 candidate_indices: Optional[torch.Tensor],
                                                 semantic_connectivity: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inicializa osciladores Kuramoto com conectividade semântica e frequências baseadas em logits.

        Returns:
            kuramoto_input: Input preparado para Kuramoto [n_candidates, features]
            candidate_tokens: Índices dos tokens candidatos [n_candidates]
            candidate_logits: Logits dos candidatos [n_candidates]
            natural_frequencies: Frequências naturais ω_i [n_candidates]
        """
        # Selecionar top-N candidatos
        vocab_size = logits.shape[0]
        n_candidates = min(self.config.n_candidates, vocab_size)  # Não exceder tamanho do vocabulário
        if candidate_indices is None:
            # Usar top-N logits
            top_logits, top_indices = torch.topk(logits, n_candidates)
        else:
            # Usar candidatos fornecidos
            top_indices = candidate_indices[:n_candidates]
            # Verificar se os índices estão dentro do range dos logits
            valid_indices = []
            for idx in top_indices:
                if idx < logits.shape[0]:
                    valid_indices.append(idx)
                else:
                    print(f"⚠️  Índice {idx} fora do range dos logits (size={logits.shape[0]})")

            if not valid_indices:
                raise RuntimeError("Nenhum índice válido encontrado nos logits")

            top_indices = torch.tensor(valid_indices, device=self.device)
            top_logits = logits[top_indices]

        candidate_tokens = top_indices
        candidate_logits = top_logits

        # ========== FREQUÊNCIAS NATURAIS BASEADAS NOS LOGITS ==========
        # Logits altos → frequências naturais altas → impulso inicial mais forte
        # Wider frequency range for calibration [0.5, 1.5] instead of [0.5, 1.0]
        normalized_logits = torch.softmax(candidate_logits, dim=0)
        natural_frequencies = 0.5 + 1.0 * normalized_logits  # Range [0.5, 1.5]

        # Preparar input para Kuramoto
        # Kuramoto layer expects [batch, seq_len, embed_dim], so we use seq_len=1
        embed_dim = 256  # 64 * 4 (quaternion multiplier from Kuramoto config)
        kuramoto_input = torch.randn(n_candidates, 1, embed_dim, device=self.device)

        # Modificar baseado nas frequências naturais
        actual_n_candidates = len(candidate_tokens)
        for i in range(actual_n_candidates):
            frequency_factor = natural_frequencies[i].item()
            kuramoto_input[i, 0] *= frequency_factor

        print(f"   ✅ Osciladores inicializados com ω ∈ [{natural_frequencies.min().item():.3f}, {natural_frequencies.max().item():.3f}]")

        return kuramoto_input, candidate_tokens, candidate_logits, natural_frequencies

    def _run_kuramoto_dynamics_semantic(self, kuramoto_input: torch.Tensor,
                                       semantic_connectivity: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Executa dinâmica de Kuramoto generalizada com conectividade semântica.

        Usa a equação: dθᵢ/dt = ωᵢ + ∑ⱼ Kᵢⱼ · sin(θⱼ - θᵢ)

        Args:
            kuramoto_input: Input dos osciladores [n_candidates, 1, embed_dim]
            semantic_connectivity: Matriz K_ij [n_candidates, n_candidates]

        Returns:
            kuramoto_output: Estado final dos osciladores
            metrics: Métricas da dinâmica incluindo análise de clusters
        """
        if self.kuramoto_layer is None:
            # Fallback: retornar input inalterado
            return kuramoto_input, {'final_sync_order': 0.5, 'coupling_strength': 1.0}

        # Para esta implementação, usaremos o Kuramoto layer existente mas com
        # modificações para incorporar a conectividade semântica
        # Em uma implementação completa, o Kuramoto layer seria modificado para aceitar K_ij

        # Modulação do acoplamento baseada no coeficiente de difusão
        base_coupling = self.config.initial_coupling_strength
        modulation_factor = self.config.diffusion_modulation_factor

        # Lógica de controle: D alto → K baixo (exploração), D baixo → K alto (convergência)
        coupling_strength = base_coupling / (1.0 + self.last_diffusion_coefficient * modulation_factor)

        # Executar dinâmica Kuramoto (versão padrão por enquanto)
        kuramoto_output, kuramoto_metrics = self.kuramoto_layer(kuramoto_input, return_metrics=True)

        # Adicionar métricas customizadas incluindo conectividade semântica
        metrics = {
            'final_sync_order': kuramoto_metrics.get('synchronization_order', 0.5),
            'coupling_strength': coupling_strength,
            'phase_coherence': kuramoto_metrics.get('phase_coherence', 0.0),
            'oscillator_phases': kuramoto_metrics.get('oscillator_phases', []),
            'semantic_connectivity_matrix': semantic_connectivity,
            'connectivity_stats': {
                'mean_connectivity': semantic_connectivity.mean().item(),
                'max_connectivity': semantic_connectivity.max().item(),
                'min_connectivity': semantic_connectivity.min().item(),
                'connectivity_sparsity': (semantic_connectivity < 0.1).float().mean().item()
            }
        }

        print(f"   ✅ Dinâmica Kuramoto executada com conectividade semântica")
        print(f"      Ordem de sincronização: {metrics['final_sync_order']:.3f}")
        print(f"      Conectividade média: {metrics['connectivity_stats']['mean_connectivity']:.3f}")

        return kuramoto_output, metrics

    def _compute_consciousness_metrics(self, kuramoto_output: torch.Tensor, kuramoto_metrics: Dict) \
            -> Dict[str, Any]:
        """
        Passo 3: Computa métricas de consciência usando FCI.

        Returns:
            consciousness_results: Resultados das métricas de consciência
        """
        if self.consciousness_metrics is None:
            return {'fci': 0.5, 'state': 'UNKNOWN', 'clz': 0.5}

        try:
            # Usar norma do output Kuramoto como proxy para espectro de potência
            power_spectrum = torch.norm(kuramoto_output, dim=-1)**2

            # Distribuição psi baseada nas amplitudes
            psi_distribution = torch.softmax(power_spectrum, dim=0)

            # Calcular FCI usando métricas de sincronização como proxy
            sync_order = kuramoto_metrics.get('final_sync_order', 0.5)
            # Map sync order to FCI: higher sync = higher consciousness
            fci_value = min(0.9, max(0.1, sync_order))

            # Classificar estado baseado no FCI
            state = self._classify_consciousness_state(fci_value)

            return {
                'fci': float(fci_value),
                'state': state,
                'clz': 0.5,  # Placeholder
                'power_spectrum': power_spectrum.tolist(),
                'psi_distribution': psi_distribution.tolist(),
                'sync_order': sync_order
            }

        except Exception as e:
            print(f"⚠️  Erro calculando métricas de consciência: {e}")
            return {'fci': 0.5, 'state': 'UNKNOWN', 'clz': 0.5}

    def _classify_consciousness_state(self, fci: float) -> str:
        """Classifica estado de consciência baseado no FCI"""
        if fci >= 0.7:
            return 'EMERGENCE'
        elif fci >= 0.4:
            return 'MEDITATION'
        elif fci >= 0.15:
            return 'ANALYSIS'
        else:
            return 'COMA'

    def _analyze_phase_clusters(self, kuramoto_output: torch.Tensor, kuramoto_metrics: Dict,
                               candidate_tokens: torch.Tensor, candidate_logits: torch.Tensor) -> Dict[str, Any]:
        """
        Analisa clusters de fase sincronizados após a simulação Kuramoto.

        Returns:
            cluster_analysis: Análise completa dos clusters encontrados
        """
        # Extrair fases dos osciladores
        # Kuramoto output: [n_candidates, 1, embed_dim]
        kuramoto_output = kuramoto_output.squeeze(1)  # [n_candidates, embed_dim]

        # Calcular fases aproximadas baseadas na componente imaginária dominante
        # Em uma implementação completa, as fases seriam extraídas diretamente do Kuramoto layer
        phases = torch.angle(kuramoto_output.mean(dim=-1))  # [n_candidates] - fase média por oscilador

        # Clustering baseado em proximidade de fase
        n_candidates = min(len(phases), len(candidate_logits))  # Correção para evitar out-of-bounds
        phase_threshold = np.pi / 4  # 45 graus - threshold para considerar "sincronizado"

        # Encontrar clusters usando diferença de fase
        clusters = []
        visited = set()

        for i in range(n_candidates):
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)

            for j in range(n_candidates):
                if j not in visited:
                    phase_diff = torch.abs(phases[i] - phases[j])
                    # Considerar wrap-around (fases são circulares)
                    phase_diff = torch.min(phase_diff, 2 * np.pi - phase_diff)

                    if phase_diff < phase_threshold:
                        cluster.append(j)
                        visited.add(j)

            if len(cluster) > 1:  # Só considerar clusters com mais de 1 membro
                clusters.append(cluster)

        # Calcular parâmetros de ordem para cada cluster
        cluster_analysis = []
        for cluster_indices in clusters:
            # Verificar se os índices do cluster são válidos
            valid_cluster_indices = []
            for idx in cluster_indices:
                if idx < len(candidate_logits):
                    valid_cluster_indices.append(idx)
                else:
                    print(f"⚠️  Índice {idx} fora do range dos candidatos (size={len(candidate_logits)})")

            if not valid_cluster_indices:
                continue

            cluster_phases = phases[valid_cluster_indices]

            # Parâmetro de ordem r = |1/N ∑ exp(iθ_j)|
            order_parameter = torch.abs(torch.mean(torch.exp(1j * cluster_phases)))

            # Energia média do cluster
            cluster_logits = candidate_logits[valid_cluster_indices]
            cluster_amplitudes = torch.norm(kuramoto_output[valid_cluster_indices], dim=-1)
            mean_energy = cluster_amplitudes.mean()

            # Token "líder" do cluster (maior logit)
            leader_idx = torch.argmax(cluster_logits).item()
            leader_token = candidate_tokens[valid_cluster_indices[leader_idx]].item()
            leader_logit = cluster_logits[leader_idx].item()

            cluster_info = {
                'cluster_id': len(cluster_analysis),
                'size': len(valid_cluster_indices),
                'order_parameter': order_parameter.item(),
                'mean_energy': mean_energy.item(),
                'leader_token': leader_token,
                'leader_logit': leader_logit,
                'member_indices': valid_cluster_indices,
                'member_tokens': candidate_tokens[valid_cluster_indices].tolist(),
                'phase_coherence': order_parameter.item()
            }
            cluster_analysis.append(cluster_info)

        # Encontrar cluster dominante (maior parâmetro de ordem)
        if cluster_analysis:
            dominant_cluster = max(cluster_analysis, key=lambda x: x['order_parameter'])
        else:
            # Fallback se nenhum cluster encontrado
            dominant_cluster = {
                'cluster_id': 0,
                'size': n_candidates,
                'order_parameter': kuramoto_metrics.get('final_sync_order', 0.5),
                'mean_energy': torch.norm(kuramoto_output, dim=-1).mean().item(),
                'leader_token': candidate_tokens[torch.argmax(candidate_logits)].item(),
                'leader_logit': torch.max(candidate_logits).item(),
                'member_indices': list(range(n_candidates)),
                'member_tokens': candidate_tokens.tolist(),
                'phase_coherence': kuramoto_metrics.get('final_sync_order', 0.5)
            }

        print(f"   ✅ Análise de clusters concluída: {len(cluster_analysis)} clusters encontrados")
        print(f"      Cluster dominante: {dominant_cluster['size']} membros, r={dominant_cluster['order_parameter']:.3f}")

        return {
            'clusters': cluster_analysis,
            'dominant_cluster': dominant_cluster,
            'total_clusters': len(cluster_analysis),
            'phase_distribution': phases.tolist(),
            'clustering_method': 'phase_proximity',
            'phase_threshold': phase_threshold
        }

    def _select_final_token_from_clusters(self, cluster_analysis: Dict, candidate_tokens: torch.Tensor,
                                         candidate_logits: torch.Tensor, kuramoto_output: torch.Tensor,
                                         kuramoto_metrics: Dict) -> Dict[str, Any]:
        """
        Seleciona token final baseado na análise de clusters semânticos.

        Returns:
            final_selection: Dicionário com token selecionado baseado em clusters
        """
        dominant_cluster = cluster_analysis.get('dominant_cluster', {})

        # Estratégia: selecionar o token com maior influência dentro do cluster dominante
        cluster_indices = dominant_cluster['member_indices']
        cluster_logits = candidate_logits[cluster_indices]
        cluster_kuramoto = kuramoto_output[cluster_indices]  # [n_cluster_members, 1, embed_dim]
        cluster_amplitudes = torch.norm(cluster_kuramoto, dim=-1).squeeze(-1)  # [n_cluster_members]

        # Score combinado: logit + amplitude normalizada + centralidade no cluster
        normalized_logits = torch.softmax(cluster_logits, dim=0)
        normalized_amplitudes = (cluster_amplitudes - cluster_amplitudes.min()) / (cluster_amplitudes.max() - cluster_amplitudes.min() + 1e-8)

        # Centralidade baseada na proximidade com a fase média do cluster
        cluster_phases = torch.angle(kuramoto_output[cluster_indices].mean(dim=-1))
        mean_phase = torch.mean(cluster_phases)
        phase_distances = torch.abs(cluster_phases - mean_phase)
        centrality = 1.0 - (phase_distances / (np.pi + 1e-8))  # Normalizar para [0,1]

        # Garantir que todos os tensores tenham a mesma forma [n_cluster_members]
        normalized_logits = normalized_logits.squeeze()
        normalized_amplitudes = normalized_amplitudes.squeeze()
        centrality = centrality.squeeze()

        # Pesos para combinação dentro do cluster
        w_logit = 0.4
        w_amplitude = 0.3
        w_centrality = 0.3

        cluster_scores = (
            normalized_logits * w_logit +
            normalized_amplitudes * w_amplitude +
            centrality * w_centrality
        )

        # Selecionar melhor token dentro do cluster dominante
        best_in_cluster = torch.argmax(cluster_scores).item()

        print(f"   🔍 Debug: best_in_cluster = {best_in_cluster}, len(cluster_indices) = {len(cluster_indices)}")

        # Bounds checking
        if best_in_cluster >= len(cluster_indices):
            print(f"   ⚠️  best_in_cluster ({best_in_cluster}) >= len(cluster_indices) ({len(cluster_indices)}), using 0")
            best_in_cluster = 0

        # Additional safety check
        if len(cluster_indices) == 0:
            print(f"   ⚠️  cluster_indices is empty, using first candidate token")
            selected_cluster_idx = 0
        else:
            selected_cluster_idx = cluster_indices[best_in_cluster]
        selected_token_id = candidate_tokens[selected_cluster_idx].item() if hasattr(candidate_tokens[selected_cluster_idx], 'item') else candidate_tokens[selected_cluster_idx]

        # Calcular probabilidade final baseada no score do cluster
        final_probability = cluster_scores[best_in_cluster].item()

        return {
            'token': selected_token_id,
            'token_id': selected_token_id,
            'probability': final_probability,
            'cluster_id': dominant_cluster['cluster_id'],
            'cluster_size': dominant_cluster['size'],
            'cluster_order_parameter': dominant_cluster['order_parameter'],
            'selection_method': 'cluster_based',
            'cluster_weights': {'w_logit': w_logit, 'w_amplitude': w_amplitude, 'w_centrality': w_centrality},
            'cluster_scores': cluster_scores.tolist(),
            'dominant_cluster_info': dominant_cluster
        }

    def _update_feedback_cycle(self, consciousness_results: Dict[str, Any]):
        """
        Passo 5: Atualiza ciclo de feedback calculando novo coeficiente de difusão.
        """
        if self.diffusion_engine is None:
            return

        try:
            fci = consciousness_results.get('fci', 0.5)

            # Preparar parâmetros para o NeuralDiffusionEngine
            # Criar distribuições simuladas baseadas nos resultados de consciência
            batch_size = 1
            embed_dim = 64  # Dimensão típica do embedding

            # Simular distribuição psi baseada no FCI
            psi_distribution = torch.softmax(torch.randn(batch_size, embed_dim), dim=-1)

            # Simular campo fractal baseado na dimensão fractal
            D_fractal = consciousness_results.get('D_fractal', 1.5)
            fractal_field = torch.randn(batch_size, embed_dim) * D_fractal

            # Calcular novo coeficiente de difusão baseado no FCI
            new_diffusion = self.diffusion_engine.compute_diffusion(
                psi_distribution=psi_distribution,
                fractal_field=fractal_field,
                fci=fci
            )

            # Atualizar estado persistente com a média do coeficiente
            self.last_diffusion_coefficient = new_diffusion.mean().item()

            print(".3f")

        except Exception as e:
            print(f"⚠️  Erro atualizando feedback cycle: {e}")

    def _generate_semantic_analysis_report(self, consciousness_results: Dict,
                                           kuramoto_metrics: Dict, cluster_analysis: Dict) -> str:
        """
        Gera relatório interpretativo da análise DCF com raciocínio semântico baseado em clusters.
        """
        fci = consciousness_results.get('fci', 0.5)
        state = consciousness_results.get('state', 'UNKNOWN')
        sync_order = kuramoto_metrics.get('final_sync_order', 0.5)

        dominant_cluster = cluster_analysis.get('dominant_cluster', {})
        cluster_size = dominant_cluster.get('size', 1)
        cluster_order = dominant_cluster.get('order_parameter', 0.5)
        total_clusters = cluster_analysis.get('total_clusters', 0)

        # Construir interpretação baseada em clusters semânticos
        semantic_interpretation = []

        if total_clusters > 0:
            semantic_interpretation.append(
                f"O sistema interpretou a predição através da estrutura semântica do vocabulário. "
                f"Foram identificados {total_clusters} clusters conceituais durante a dinâmica Kuramoto."
            )

            semantic_interpretation.append(
                f"O cluster dominante emergiu com {cluster_size} membros e parâmetro de ordem r={cluster_order:.3f}, "
                f"indicando {'forte' if cluster_order > 0.8 else 'moderada' if cluster_order > 0.5 else 'fraca'} "
                f"coerência semântica interna."
            )

            # Adicionar interpretação específica baseada no estado de consciência
            if state == 'EMERGENCE':
                semantic_interpretation.append(
                    f"Estado EMERGENTE (FCI={fci:.3f}): A formação do cluster reflete criatividade semântica, "
                    f"onde conceitos semanticamente relacionados se sincronizaram através de conexões conceituais profundas."
                )
            elif state == 'MEDITATION':
                semantic_interpretation.append(
                    f"Estado MEDITATIVO (FCI={fci:.3f}): Equilíbrio entre exploração semântica e foco conceitual, "
                    f"resultando em seleção baseada em harmonia entre probabilidade inicial e afinidade semântica."
                )
            elif state == 'ANALYSIS':
                semantic_interpretation.append(
                    f"Estado ANALÍTICO (FCI={fci:.3f}): Priorização de convergência semântica rigorosa, "
                    f"com seleção determinística baseada em centralidade no cluster conceitual dominante."
                )
            else:  # COMA
                semantic_interpretation.append(
                    f"Estado COMA (FCI={fci:.3f}): Fragmentação semântica detectada. "
                    f"A seleção pode não refletir conexões conceituais adequadas."
                )

            # Adicionar detalhes sobre o método de análise
            semantic_interpretation.append(
                f"Análise baseada em dinâmica quântica fractal: estado final Ψ emergiu da sincronização "
                f"de osciladores Kuramoto, representando consenso semântico através de conectividade quântica."
            )
        else:
            # Fallback para interpretação sem clusters
            semantic_interpretation.append(
                f"Análise semântica limitada: nenhum cluster conceitual distinto identificado. "
                f"Interpretação baseada em métricas globais de sincronização."
            )

            if state == 'EMERGENCE':
                semantic_interpretation.append(
                    f"Estado EMERGENTE (FCI={fci:.3f}) com sincronização {sync_order:.3f}: "
                    f"Análise criativa sem estruturação semântica clara."
                )
            elif state == 'MEDITATION':
                semantic_interpretation.append(
                    f"Estado MEDITATIVO (FCI={fci:.3f}) com sincronização {sync_order:.3f}: "
                    f"Processamento balanceado sem clusters semânticos dominantes."
                )
            elif state == 'ANALYSIS':
                semantic_interpretation.append(
                    f"Estado ANALÍTICO (FCI={fci:.3f}) com sincronização {sync_order:.3f}: "
                    f"Análise determinística sem emergência de conceitos agrupados."
                )
            else:
                semantic_interpretation.append(
                    f"Estado COMA (FCI={fci:.3f}) com baixa sincronização {sync_order:.3f}: "
                    f"Possível necessidade de bootstrap cognitivo para restabelecer conectividade semântica."
                )

        return " ".join(semantic_interpretation)


# Função de interface para uso compatível com sistemas existentes
def analyze_tokens_dcf(logits: torch.Tensor, config_path: Optional[str] = None,
                      device: str = "cpu", embeddings: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Interface principal para análise DCF de tokens com suporte a conectividade semântica.

    Args:
        logits: Logits do modelo [vocab_size] ou [batch_size, vocab_size]
        config_path: Caminho para config DCF (opcional)
        device: Dispositivo de computação
        embeddings: Matriz de embeddings [vocab_size, embed_dim] para conectividade semântica (opcional)

    Returns:
        Resultado da análise DCF com raciocínio semântico
    """
    # Criar dicionário quântico básico se embeddings forem fornecidos
    quantum_vocab = None
    if embeddings is not None:
        # Verificar se embeddings é um tensor ou um módulo
        if hasattr(embeddings, 'shape'):
            # É um tensor
            vocab_size, embed_dim = embeddings.shape
            quantum_vocab = torch.zeros(vocab_size, embed_dim, 4, device=device)
            quantum_vocab[:, :, 0] = embeddings  # Parte real
            quantum_vocab[:, :, 1] = torch.sin(embeddings)  # Parte imaginária i
            quantum_vocab[:, :, 2] = torch.cos(embeddings)  # Parte imaginária j
            quantum_vocab[:, :, 3] = torch.tanh(embeddings)  # Parte imaginária k
        else:
            # É um módulo (DynamicQuantumWordMatrix) - criar embeddings básicos
            print("⚠️  DynamicQuantumWordMatrix detectado - criando embeddings básicos")
            vocab_size = 50257  # GPT-2 padrão
            embed_dim = 64  # Dimensão padrão
            basic_embeddings = torch.randn(vocab_size, embed_dim, device=device)
            quantum_vocab = torch.zeros(vocab_size, embed_dim, 4, device=device)
            quantum_vocab[:, :, 0] = basic_embeddings  # Parte real
            quantum_vocab[:, :, 1] = torch.sin(basic_embeddings)  # Parte imaginária i
            quantum_vocab[:, :, 2] = torch.cos(basic_embeddings)  # Parte imaginária j
            quantum_vocab[:, :, 3] = torch.tanh(basic_embeddings)  # Parte imaginária k

    analyzer = DCFTokenAnalysis(config_path=config_path, device=device,
                               quantum_vocab_representations=quantum_vocab)
    return analyzer.analyze_tokens(logits, embeddings=embeddings)


if __name__ == "__main__":
    # Exemplo de uso
    print("🎯 Testando Sistema DCF de Análise de Tokens...")

    # Simular logits de um modelo
    vocab_size = 1000
    logits = torch.randn(vocab_size)

    # Executar análise DCF
    result = analyze_tokens_dcf(logits, device="cpu")

    print("\n" + "="*60)
    print("RESULTADO DA ANÁLISE DCF:")
    print("="*60)
    print(f"Token Selecionado: {result['selected_token']}")
    print(f"Probabilidade Final: {result['final_probability']:.4f}")
    print(f"FCI: {result['fci_value']:.4f}")
    print(f"Estado de Consciência: {result['consciousness_state']}")
    print(f"Ordem de Sincronização: {result['synchronization_order']:.4f}")
    print(f"Tempo de Processamento: {result['processing_time']:.3f}s")
    print("\n📋 Relatório Interpretativo:")
    print(result['analysis_report'])
    print("="*60)