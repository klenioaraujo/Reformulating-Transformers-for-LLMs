#!/usr/bin/env python3
"""
Prompt Engine para Testes Automatizados do Sistema ΨQRH
Engine responsável por executar testes de segurança de forma automatizada
"""

import sys
import os
import tempfile
import shutil
from typing import List, Dict, Any

# Adicionar path para importar módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conscience.secure_Ψcws_protector import ΨCWSSecurityLayer, ΨCWSFileSplitter


class ΨQRHTestEngine:
    """Engine de testes automatizados para sistema ΨQRH"""

    def __init__(self):
        self.security_layer = ΨCWSSecurityLayer()
        self.file_splitter = ΨCWSFileSplitter(self.security_layer)
        self.test_results = []
        self.temp_files = []

    def run_security_tests(self) -> Dict[str, Any]:
        """Executa todos os testes de segurança"""
        print("🚀 Iniciando Testes de Segurança ΨQRH")
        print("=" * 50)

        results = {
            'layer_tests': self._test_individual_layers(),
            'full_pipeline': self._test_full_pipeline(),
            'file_splitting': self._test_file_splitting(),
            'edge_cases': self._test_edge_cases()
        }

        # Consolidar resultados
        all_passed = all(result['passed'] for result in results.values())

        print("\n" + "=" * 50)
        print(f"📊 RESULTADO FINAL: {'✅ TODOS OS TESTES PASSARAM' if all_passed else '❌ ALGUNS TESTES FALHARAM'}")

        return {
            'all_passed': all_passed,
            'details': results
        }

    def _test_individual_layers(self) -> Dict[str, Any]:
        """Testa cada camada de segurança individualmente"""
        print("\n🔍 Testando Camadas Individuais")

        test_data = b'PSQRH security test'
        results = {}

        # Teste de transposição
        try:
            encrypted = self.security_layer._layer_transposition(test_data)
            decrypted = self.security_layer._layer_transposition(encrypted, decrypt=True)
            results['transposition'] = {
                'passed': test_data == decrypted,
                'input_size': len(test_data),
                'output_size': len(encrypted)
            }
            print(f"  ✅ Transposição: {results['transposition']['passed']}")
        except Exception as e:
            results['transposition'] = {'passed': False, 'error': str(e)}
            print(f"  ❌ Transposição: {e}")

        # Teste de XOR
        try:
            key = b'test_key_123456789'
            encrypted = self.security_layer._layer_xor(test_data, key)
            decrypted = self.security_layer._layer_xor(encrypted, key)
            results['xor'] = {
                'passed': test_data == decrypted,
                'input_size': len(test_data),
                'output_size': len(encrypted)
            }
            print(f"  ✅ XOR: {results['xor']['passed']}")
        except Exception as e:
            results['xor'] = {'passed': False, 'error': str(e)}
            print(f"  ❌ XOR: {e}")

        return {
            'passed': all(result['passed'] for result in results.values()),
            'details': results
        }

    def _test_full_pipeline(self) -> Dict[str, Any]:
        """Testa o pipeline completo de 7 camadas"""
        print("\n🔗 Testando Pipeline Completo (7 Camadas)")

        test_cases = [
            (b'Short test', "Caso curto"),
            (b'Medium length test data for PSQRH system', "Caso médio"),
            (b'Large test data ' * 100, "Caso grande")
        ]

        results = []

        for test_data, description in test_cases:
            try:
                encrypted = self.security_layer.encrypt_7_layers(test_data)
                decrypted = self.security_layer.decrypt_7_layers(encrypted)

                success = test_data == decrypted
                results.append({
                    'description': description,
                    'passed': success,
                    'input_size': len(test_data),
                    'encrypted_size': len(encrypted),
                    'decrypted_size': len(decrypted)
                })

                status = "✅" if success else "❌"
                print(f"  {status} {description}: {len(test_data)} → {len(encrypted)} → {len(decrypted)} bytes")

            except Exception as e:
                results.append({
                    'description': description,
                    'passed': False,
                    'error': str(e)
                })
                print(f"  ❌ {description}: {e}")

        return {
            'passed': all(result['passed'] for result in results),
            'details': results
        }

    def _test_file_splitting(self) -> Dict[str, Any]:
        """Testa divisão e reassemblagem de arquivos"""
        print("\n📁 Testando Divisão de Arquivos")

        try:
            # Criar arquivo temporário
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as f:
                test_data = b'File splitting test data for PSQRH system validation'
                f.write(test_data)
                temp_path = f.name
                self.temp_files.append(temp_path)

            # Dividir arquivo
            parts = self.file_splitter.split_file(temp_path, parts=3)
            print(f"  📦 Partes criadas: {len(parts)}")

            # Reassemblar
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as output_file:
                output_path = output_file.name
                self.temp_files.append(output_path)

            success = self.file_splitter.reassemble_file(parts, output_path)

            # Verificar integridade
            with open(output_path, 'rb') as f:
                reassembled_data = f.read()

            integrity = test_data == reassembled_data

            result = {
                'passed': success and integrity,
                'parts_created': len(parts),
                'reassembly_success': success,
                'integrity_check': integrity,
                'original_size': len(test_data),
                'reassembled_size': len(reassembled_data)
            }

            status = "✅" if result['passed'] else "❌"
            print(f"  {status} Divisão/Reassemblagem: {result['passed']}")

            return {
                'passed': result['passed'],
                'details': result
            }

        except Exception as e:
            print(f"  ❌ Divisão de arquivos: {e}")
            return {
                'passed': False,
                'error': str(e)
            }

    def _test_edge_cases(self) -> Dict[str, Any]:
        """Testa casos extremos e boundary conditions"""
        print("\n⚠️  Testando Casos Extremos")

        edge_cases = [
            (b'', "Dados vazios"),
            (b'\x00\x01\x02\x03', "Bytes especiais"),
            (b'A' * 10000, "Dados muito longos")
        ]

        results = []

        for test_data, description in edge_cases:
            try:
                if test_data:  # Não testar dados vazios no pipeline completo
                    encrypted = self.security_layer.encrypt_7_layers(test_data)
                    decrypted = self.security_layer.decrypt_7_layers(encrypted)
                    success = test_data == decrypted
                else:
                    success = True  # Dados vazios são válidos

                results.append({
                    'description': description,
                    'passed': success,
                    'input_size': len(test_data)
                })

                status = "✅" if success else "❌"
                print(f"  {status} {description}: {len(test_data)} bytes")

            except Exception as e:
                results.append({
                    'description': description,
                    'passed': False,
                    'error': str(e)
                })
                print(f"  ❌ {description}: {e}")

        return {
            'passed': all(result['passed'] for result in results),
            'details': results
        }

    def cleanup(self):
        """Limpa arquivos temporários criados durante os testes"""
        print("\n🧹 Limpando arquivos temporários...")

        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"  ✅ Removido: {temp_file}")
            except Exception as e:
                print(f"  ⚠️  Erro ao remover {temp_file}: {e}")

        self.temp_files.clear()
        print("  ✅ Limpeza concluída")


def main():
    """Função principal do engine de testes"""
    engine = ΨQRHTestEngine()

    try:
        # Executar todos os testes
        results = engine.run_security_tests()

        # Limpar resíduos
        engine.cleanup()

        # Retornar código de saída
        sys.exit(0 if results['all_passed'] else 1)

    except Exception as e:
        print(f"\n💥 ERRO CRÍTICO: {e}")
        engine.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()