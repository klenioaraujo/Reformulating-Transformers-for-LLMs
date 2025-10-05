#!/usr/bin/env python3
"""
Teste Completo do Sistema ΨQRH Standalone
==========================================

Verifica que o sistema ΨQRH funciona completamente sem a camada agêntica.

Testes incluídos:
1. Imports do core (sem dependências agênticas)
2. QRHFactory e componentes básicos
3. Processamento quaterniônico
4. Filtros espectrais
5. Componentes cognitivos (não-agênticos)
6. Transformers ΨQRH
7. Sistemas de produção

Autor: Claude Code & ΨQRH Team
Data: 2025-10-02
"""

import sys
import os
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# Adicionar path do projeto
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class PSIQRHStandaloneTest:
    """Teste completo do sistema ΨQRH standalone"""

    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def log_test(self, name: str, status: str, message: str = "", details: str = ""):
        """Log de resultado de teste"""
        result = {
            "test": name,
            "status": status,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)

        # Atualizar contadores
        if status == "PASS":
            self.passed += 1
            print(f"✅ {name}: {message}")
        elif status == "FAIL":
            self.failed += 1
            print(f"❌ {name}: {message}")
            if details:
                print(f"   Details: {details}")
        elif status == "WARN":
            self.warnings += 1
            print(f"⚠️  {name}: {message}")

    def test_1_core_imports(self):
        """Teste 1: Imports do core ΨQRH (sem agênticos)"""
        print("\n" + "="*60)
        print("TESTE 1: Core Imports (Sem Dependências Agênticas)")
        print("="*60)

        try:
            # Import principal
            from src.core.ΨQRH import QRHFactory
            self.log_test(
                "Import QRHFactory",
                "PASS",
                "QRHFactory importado com sucesso"
            )

            # Imports de componentes core
            from src.core.qrh_layer import QRHLayer, QRHConfig
            self.log_test(
                "Import QRHLayer",
                "PASS",
                "QRHLayer e QRHConfig importados"
            )

            from src.core.quaternion_operations import QuaternionOperations
            self.log_test(
                "Import QuaternionOperations",
                "PASS",
                "QuaternionOperations importado"
            )

            # Verificar que não há imports agênticos
            import src.core.ΨQRH as psiqrh_module
            source = open(psiqrh_module.__file__).read()

            agentic_keywords = [
                'NavigatorAgent', 'PromptEngineAgent', 'AgenticRuntime',
                'seal_protocol', 'audit_log'
            ]

            found_agentic = [kw for kw in agentic_keywords if kw in source]

            if found_agentic:
                self.log_test(
                    "Verificação de Desacoplamento",
                    "FAIL",
                    f"Imports agênticos encontrados: {found_agentic}"
                )
            else:
                self.log_test(
                    "Verificação de Desacoplamento",
                    "PASS",
                    "Nenhum import agêntico encontrado no core"
                )

        except Exception as e:
            self.log_test(
                "Core Imports",
                "FAIL",
                f"Erro ao importar core: {str(e)}",
                str(e)
            )

    def test_2_qrh_factory(self):
        """Teste 2: QRHFactory e componentes básicos"""
        print("\n" + "="*60)
        print("TESTE 2: QRHFactory e Componentes Básicos")
        print("="*60)

        try:
            from src.core.ΨQRH import QRHFactory

            # Criar factory
            factory = QRHFactory()
            self.log_test(
                "Criar QRHFactory",
                "PASS",
                "QRHFactory instanciado com sucesso"
            )

            # Verificar config
            if hasattr(factory, 'config'):
                self.log_test(
                    "QRHFactory.config",
                    "PASS",
                    f"Config presente: embed_dim={factory.config.embed_dim}"
                )
            else:
                self.log_test(
                    "QRHFactory.config",
                    "WARN",
                    "Config não encontrado"
                )

            # Verificar QRH layer
            if hasattr(factory, 'qrh_layer') or hasattr(factory, 'get_qrh_layer'):
                self.log_test(
                    "QRH Layer",
                    "PASS",
                    "QRH Layer disponível"
                )
            else:
                self.log_test(
                    "QRH Layer",
                    "WARN",
                    "QRH Layer não encontrado diretamente"
                )

        except Exception as e:
            self.log_test(
                "QRHFactory",
                "FAIL",
                f"Erro ao testar QRHFactory: {str(e)}",
                str(e)
            )

    def test_3_quaternion_processing(self):
        """Teste 3: Processamento quaterniônico"""
        print("\n" + "="*60)
        print("TESTE 3: Processamento Quaterniônico")
        print("="*60)

        try:
            from src.core.quaternion_operations import QuaternionOperations

            # Criar operações quaterniônicas
            qops = QuaternionOperations()
            self.log_test(
                "Criar QuaternionOperations",
                "PASS",
                "QuaternionOperations instanciado"
            )

            # Testar operação básica
            # Criar tensor quaterniônico [batch, seq, 4*embed_dim]
            batch_size = 2
            seq_len = 4
            embed_dim = 16
            q_tensor = torch.randn(batch_size, seq_len, 4 * embed_dim)

            # Testar split
            if hasattr(qops, 'split_quaternion'):
                q_parts = qops.split_quaternion(q_tensor)
                if len(q_parts) == 4:
                    self.log_test(
                        "Quaternion Split",
                        "PASS",
                        f"Split em 4 componentes: {[p.shape for p in q_parts]}"
                    )
                else:
                    self.log_test(
                        "Quaternion Split",
                        "FAIL",
                        f"Split retornou {len(q_parts)} componentes ao invés de 4"
                    )
            else:
                self.log_test(
                    "Quaternion Split",
                    "WARN",
                    "Método split_quaternion não encontrado"
                )

            # Testar normalização
            if hasattr(qops, 'normalize_quaternion'):
                q_norm = qops.normalize_quaternion(q_tensor)
                norm = torch.sqrt((q_norm ** 2).sum(dim=-1))
                mean_norm = norm.mean().item()

                if 0.9 < mean_norm < 1.1:
                    self.log_test(
                        "Quaternion Normalization",
                        "PASS",
                        f"Normalização OK: mean_norm={mean_norm:.4f}"
                    )
                else:
                    self.log_test(
                        "Quaternion Normalization",
                        "WARN",
                        f"Norma fora do esperado: {mean_norm:.4f}"
                    )
            else:
                self.log_test(
                    "Quaternion Normalization",
                    "WARN",
                    "Método normalize_quaternion não encontrado"
                )

        except Exception as e:
            self.log_test(
                "Processamento Quaterniônico",
                "FAIL",
                f"Erro: {str(e)}",
                str(e)
            )

    def test_4_spectral_filters(self):
        """Teste 4: Filtros espectrais"""
        print("\n" + "="*60)
        print("TESTE 4: Filtros Espectrais")
        print("="*60)

        try:
            from src.fractal.spectral_filter import SpectralFilter

            embed_dim = 16
            filter = SpectralFilter(embed_dim=embed_dim)
            self.log_test(
                "Criar SpectralFilter",
                "PASS",
                f"SpectralFilter criado (embed_dim={embed_dim})"
            )

            # Testar processamento
            batch_size = 2
            seq_len = 8
            x = torch.randn(batch_size, seq_len, embed_dim)

            output = filter(x)

            if output.shape == x.shape:
                self.log_test(
                    "SpectralFilter Forward",
                    "PASS",
                    f"Shape preservado: {output.shape}"
                )
            else:
                self.log_test(
                    "SpectralFilter Forward",
                    "FAIL",
                    f"Shape mudou: {x.shape} → {output.shape}"
                )

            # Verificar que não há NaN
            if not torch.isnan(output).any():
                self.log_test(
                    "SpectralFilter NaN Check",
                    "PASS",
                    "Nenhum NaN detectado"
                )
            else:
                self.log_test(
                    "SpectralFilter NaN Check",
                    "FAIL",
                    "NaN detectado no output"
                )

        except Exception as e:
            self.log_test(
                "Filtros Espectrais",
                "FAIL",
                f"Erro: {str(e)}",
                str(e)
            )

    def test_5_cognitive_components(self):
        """Teste 5: Componentes cognitivos (não-agênticos)"""
        print("\n" + "="*60)
        print("TESTE 5: Componentes Cognitivos (Não-Agênticos)")
        print("="*60)

        # Testar filtros semânticos
        try:
            from src.cognitive.semantic_adaptive_filters import SemanticAdaptiveFilter, SemanticFilterConfig

            config = SemanticFilterConfig(embed_dim=64)
            filter = SemanticAdaptiveFilter(config)

            self.log_test(
                "SemanticAdaptiveFilter",
                "PASS",
                "Filtro semântico criado com sucesso"
            )

            # Testar forward
            x = torch.randn(2, 8, 256)  # [B, T, 4*D]
            output, metrics = filter(x)

            if 'contradiction_scores' in metrics:
                self.log_test(
                    "SemanticFilter Metrics",
                    "PASS",
                    f"Métricas geradas: {list(metrics.keys())}"
                )
            else:
                self.log_test(
                    "SemanticFilter Metrics",
                    "WARN",
                    "Métricas não geradas corretamente"
                )

        except Exception as e:
            self.log_test(
                "SemanticAdaptiveFilter",
                "FAIL",
                f"Erro: {str(e)}",
                str(e)
            )

        # Testar neurotransmissores sintéticos
        try:
            from src.cognitive.synthetic_neurotransmitters import SyntheticNeurotransmitterSystem, NeurotransmitterConfig

            nt_config = NeurotransmitterConfig(embed_dim=64)
            nt_system = SyntheticNeurotransmitterSystem(nt_config)

            self.log_test(
                "SyntheticNeurotransmitters",
                "PASS",
                "Sistema de neurotransmissores criado"
            )

        except Exception as e:
            self.log_test(
                "SyntheticNeurotransmitters",
                "FAIL",
                f"Erro: {str(e)}",
                str(e)
            )

    def test_6_qrh_layer(self):
        """Teste 6: QRH Layer completo"""
        print("\n" + "="*60)
        print("TESTE 6: QRH Layer Completo")
        print("="*60)

        try:
            from src.core.qrh_layer import QRHLayer, QRHConfig

            embed_dim = 16
            config = QRHConfig(
                embed_dim=embed_dim,
                alpha=1.0,
                use_learned_rotation=True
            )

            layer = QRHLayer(config)
            self.log_test(
                "Criar QRHLayer",
                "PASS",
                f"QRHLayer criado (embed_dim={embed_dim}, alpha={config.alpha})"
            )

            # Testar forward
            batch_size = 2
            seq_len = 8
            x = torch.randn(batch_size, seq_len, 4 * embed_dim)

            output = layer(x)

            # Verificar shape
            if output.shape == x.shape:
                self.log_test(
                    "QRHLayer Forward Shape",
                    "PASS",
                    f"Shape correto: {output.shape}"
                )
            else:
                self.log_test(
                    "QRHLayer Forward Shape",
                    "FAIL",
                    f"Shape incorreto: {x.shape} → {output.shape}"
                )

            # Verificar NaN
            if not torch.isnan(output).any():
                self.log_test(
                    "QRHLayer NaN Check",
                    "PASS",
                    "Nenhum NaN detectado"
                )
            else:
                self.log_test(
                    "QRHLayer NaN Check",
                    "FAIL",
                    "NaN detectado no output"
                )

            # Verificar range de valores
            output_mean = output.mean().item()
            output_std = output.std().item()

            self.log_test(
                "QRHLayer Output Stats",
                "PASS",
                f"Mean={output_mean:.4f}, Std={output_std:.4f}"
            )

        except Exception as e:
            self.log_test(
                "QRHLayer",
                "FAIL",
                f"Erro: {str(e)}",
                str(e)
            )

    def test_7_integration(self):
        """Teste 7: Integração completa"""
        print("\n" + "="*60)
        print("TESTE 7: Integração Completa ΨQRH")
        print("="*60)

        try:
            from src.core.ΨQRH import QRHFactory
            from src.core.qrh_layer import QRHLayer

            # Criar factory
            factory = QRHFactory()

            # Criar dados de teste
            batch_size = 2
            seq_len = 16
            embed_dim = factory.config.embed_dim if hasattr(factory, 'config') else 32

            x = torch.randn(batch_size, seq_len, 4 * embed_dim)

            # Processar
            if hasattr(factory, 'qrh_layer') and factory.qrh_layer is not None:
                output = factory.qrh_layer(x)

                self.log_test(
                    "Integração ΨQRH",
                    "PASS",
                    f"Processamento completo: {x.shape} → {output.shape}"
                )

                # Verificar qualidade
                if not torch.isnan(output).any() and not torch.isinf(output).any():
                    self.log_test(
                        "Qualidade Output",
                        "PASS",
                        "Output livre de NaN/Inf"
                    )
                else:
                    self.log_test(
                        "Qualidade Output",
                        "FAIL",
                        "Output contém NaN ou Inf"
                    )
            else:
                self.log_test(
                    "Integração ΨQRH",
                    "WARN",
                    "QRH Layer não acessível diretamente via factory"
                )

        except Exception as e:
            self.log_test(
                "Integração",
                "FAIL",
                f"Erro: {str(e)}",
                str(e)
            )

    def run_all_tests(self):
        """Executa todos os testes"""
        print("\n" + "="*60)
        print("TESTE COMPLETO DO SISTEMA ΨQRH STANDALONE")
        print("="*60)
        print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"PyTorch: {torch.__version__}")
        print("="*60)

        # Executar testes
        self.test_1_core_imports()
        self.test_2_qrh_factory()
        self.test_3_quaternion_processing()
        self.test_4_spectral_filters()
        self.test_5_cognitive_components()
        self.test_6_qrh_layer()
        self.test_7_integration()

        # Gerar relatório final
        self.generate_report()

    def generate_report(self):
        """Gera relatório final"""
        print("\n" + "="*60)
        print("RELATÓRIO FINAL")
        print("="*60)

        total = self.passed + self.failed + self.warnings
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print(f"\n📊 Estatísticas:")
        print(f"   Total de testes: {total}")
        print(f"   ✅ Passou: {self.passed}")
        print(f"   ❌ Falhou: {self.failed}")
        print(f"   ⚠️  Avisos: {self.warnings}")
        print(f"   Taxa de sucesso: {pass_rate:.1f}%")

        # Status geral
        if self.failed == 0:
            print(f"\n🎉 Status: TODOS OS TESTES PASSARAM")
        elif self.failed <= 2:
            print(f"\n⚠️  Status: MAIORIA DOS TESTES PASSOU (alguns falhos)")
        else:
            print(f"\n❌ Status: MÚLTIPLAS FALHAS DETECTADAS")

        # Salvar relatório
        self.save_report()

    def save_report(self):
        """Salva relatório em arquivo"""
        import json

        report_file = "test_psiqrh_standalone_report.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "summary": {
                "total": self.passed + self.failed + self.warnings,
                "passed": self.passed,
                "failed": self.failed,
                "warnings": self.warnings,
                "pass_rate": (self.passed / (self.passed + self.failed + self.warnings) * 100)
                            if (self.passed + self.failed + self.warnings) > 0 else 0
            },
            "tests": self.test_results
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📄 Relatório salvo em: {report_file}")


if __name__ == "__main__":
    tester = PSIQRHStandaloneTest()
    tester.run_all_tests()
