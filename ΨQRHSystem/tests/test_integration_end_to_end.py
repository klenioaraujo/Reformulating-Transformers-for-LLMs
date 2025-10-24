#!/usr/bin/env python3
"""
Teste de Integração End-to-End para o Sistema ΨQRH

Este teste verifica se o sistema completo funciona corretamente
após todas as correções implementadas, atendendo aos critérios finais.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, project_root)


class TestEndToEndIntegration:
    """Teste de integração end-to-end do sistema ΨQRH"""

    def test_cli_execution_no_errors(self):
        """Testa se o comando CLI executa sem erros"""
        try:
            # Executar comando CLI
            result = subprocess.run(
                [sys.executable, "interfaces/CLI.py", "Qual a cor do céu?"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            # Verificar código de saída
            assert result.returncode == 0, f"CLI falhou com código {result.returncode}"

            # Verificar que não há erros na saída
            assert "❌ Erro" not in result.stdout, "Encontrado erro na saída"
            assert "Traceback" not in result.stdout, "Encontrado traceback na saída"

            print("✅ CLI executou sem erros")

        except subprocess.TimeoutExpired:
            assert False, "CLI excedeu timeout de 60 segundos"
        except Exception as e:
            assert False, f"Falha na execução CLI: {e}"

    def test_response_semantic_relevance(self):
        """Testa se a resposta é semanticamente relevante"""
        try:
            result = subprocess.run(
                [sys.executable, "interfaces/CLI.py", "Qual a cor do céu?"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = result.stdout

            # Verificar que contém resposta sobre cor do céu
            assert "céu" in output.lower() or "sky" in output.lower(), "Resposta não menciona céu"
            assert "azul" in output.lower() or "blue" in output.lower(), "Resposta não menciona azul"

            # Verificar que não é resposta genérica
            assert "quantum processing with unknown state" not in output.lower(), "Resposta ainda é genérica"

            print("✅ Resposta semanticamente relevante")

        except Exception as e:
            assert False, f"Falha na verificação semântica: {e}"

    def test_energy_validation_logical(self):
        """Testa se validações de energia fazem sentido"""
        try:
            result = subprocess.run(
                [sys.executable, "interfaces/CLI.py", "Qual a cor do céu?"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = result.stdout

            # Verificar que energia é reportada como violada (comportamento correto)
            assert "VIOLADA" in output, "Energia deve ser reportada como violada"

            # Verificar que validações passaram
            assert "APROVADAS" in output, "Validações devem estar aprovadas"

            print("✅ Validações de energia fazem sentido")

        except Exception as e:
            assert False, f"Falha na verificação de validações: {e}"

    def test_modern_components_usage(self):
        """Testa se componentes modernos são utilizados"""
        try:
            result = subprocess.run(
                [sys.executable, "interfaces/CLI.py", "Qual a cor do céu?"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = result.stdout

            # Verificar uso de componentes modernos
            modern_components = [
                "FractalConsciousnessProcessor",
                "Sistema DCF"
            ]

            for component in modern_components:
                assert component in output, f"Componente moderno {component} não foi utilizado"

            # PiAutoCalibration é usado internamente, verificar indiretamente
            assert "π-calibration" in output.lower() or "pi-calibration" in output.lower(), "PiAutoCalibration não foi utilizado"

            print("✅ Componentes modernos são utilizados")

        except Exception as e:
            assert False, f"Falha na verificação de componentes: {e}"

    def test_final_criteria_verification(self):
        """Verificação completa dos critérios finais"""
        print("\n🔬 VERIFICAÇÃO FINAL DOS CRITÉRIOS ΨQRH")
        print("=" * 50)

        # 1. Sem Erros
        print("1. Sem Erros...")
        self.test_cli_execution_no_errors()
        print("   ✅ PASSOU")

        # 2. Resposta Relevante
        print("2. Resposta Relevante...")
        self.test_response_semantic_relevance()
        print("   ✅ PASSOU")

        # 3. Validações Lógicas
        print("3. Validações Lógicas...")
        self.test_energy_validation_logical()
        print("   ✅ PASSOU")

        # 4. Logs Claros
        print("4. Logs Claros...")
        self.test_modern_components_usage()
        print("   ✅ PASSOU")

        print("=" * 50)
        print("🎉 SISTEMA ΨQRH TOTALMENTE CORRIGIDO E FUNCIONAL!")
        print("✅ Todos os critérios finais foram atendidos")


def run_end_to_end_test():
    """Executa teste end-to-end"""
    test = TestEndToEndIntegration()

    try:
        test.test_final_criteria_verification()
        return True
    except Exception as e:
        print(f"❌ FALHA NO TESTE END-TO-END: {e}")
        return False


if __name__ == '__main__':
    success = run_end_to_end_test()
    sys.exit(0 if success else 1)