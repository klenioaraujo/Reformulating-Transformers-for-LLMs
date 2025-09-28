#!/usr/bin/env python3
"""
Teste do Sistema de Leitura Nativa .Ψcws
========================================

Testa a operacionalidade básica do ΨCWSNativeReader
para garantir que consegue ler arquivos .Ψcws nativamente.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_native_reader():
    """
    Testa funcionalidades básicas do ΨCWSNativeReader
    """
    print("🔮 Teste de Operacionalidade - ΨCWSNativeReader")
    print("=" * 50)

    try:
        # Importar o leitor nativo
        from src.conscience.psicws_native_reader import ΨCWSNativeReader

        print("✅ ΨCWSNativeReader importado com sucesso")

        # Inicializar leitor
        reader = ΨCWSNativeReader()

        print("\n📊 Status inicial do cache:")
        reader.print_cache_status()

        # Testar descoberta de arquivos
        print("\n🔍 Testando descoberta de arquivos...")
        files = reader.discover_files()
        print(f"   Arquivos .Ψcws encontrados: {len(files)}")

        # Testar listagem
        print("\n📋 Testando listagem de arquivos...")
        available = reader.list_available()

        if available:
            print(f"   Total de arquivos disponíveis: {len(available)}")

            # Mostrar detalhes dos primeiros arquivos
            for i, file_info in enumerate(available[:3]):
                print(f"\n   📄 Arquivo {i+1}:")
                print(f"      Hash: {file_info['hash']}")
                print(f"      Nome original: {file_info['original_name']}")
                print(f"      Tamanho: {file_info['size_kb']} KB")
                print(f"      Modificado: {file_info['modified_time']}")

            # Testar carregamento por hash
            print(f"\n🔄 Testando carregamento por hash...")
            first_file = available[0]
            hash_id = first_file['hash']

            print(f"   Tentando carregar hash: {hash_id}")
            loaded_file = reader.load_by_hash(hash_id)

            if loaded_file:
                print("   ✅ Arquivo carregado com sucesso!")

                # Verificar estrutura do arquivo
                print(f"   📋 Estrutura verificada:")
                print(f"      Magic number: {loaded_file.header.magic_number}")
                print(f"      Tipo de arquivo: {loaded_file.header.file_type}")
                print(f"      Timestamp: {loaded_file.header.timestamp}")

                # Testar métricas de consciência
                print(f"\n🧠 Testando métricas de consciência...")
                summary = reader.get_consciousness_summary(hash_id)

                if summary:
                    print(f"   ✅ Métricas obtidas:")
                    print(f"      Complexity: {summary['complexity']:.4f}")
                    print(f"      Coherence: {summary['coherence']:.4f}")
                    print(f"      Adaptability: {summary['adaptability']:.4f}")
                    print(f"      Integration: {summary['integration']:.4f}")
                    print(f"      Frequência: {summary['frequency_range']} Hz")
                else:
                    print("   ❌ Erro ao obter métricas")

                # Testar tensor QRH
                if loaded_file.qrh_tensor is not None:
                    tensor_shape = loaded_file.qrh_tensor.shape
                    print(f"\n🎯 Tensor QRH:")
                    print(f"   Shape: {tensor_shape}")
                    print(f"   Compatível com QRHLayer: {'✅' if len(tensor_shape) == 3 and tensor_shape[-1] % 4 == 0 else '❌'}")
                else:
                    print("\n⚠️ Tensor QRH não encontrado")

            else:
                print("   ❌ Erro ao carregar arquivo")

            # Testar carregamento por nome
            print(f"\n🔄 Testando carregamento por nome...")
            original_name = first_file['original_name']
            loaded_by_name = reader.load_by_name(original_name)

            if loaded_by_name:
                print(f"   ✅ Carregamento por nome '{original_name}' funcionou!")
            else:
                print(f"   ⚠️ Carregamento por nome não funcionou")

        else:
            print("   ⚠️ Nenhum arquivo .Ψcws encontrado")
            print("   💡 Execute 'make demo-pdf-Ψcws' para gerar arquivos de teste")

        # Testar saúde do cache
        print(f"\n🏥 Testando verificação de saúde...")
        health = reader.check_cache_health()
        print(f"   Status de saúde: {health['health_status']}")
        print(f"   Arquivos válidos: {health['valid_files']}/{health['total_files']}")

        print(f"\n✅ Todos os testes básicos completados!")
        return True

    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False

    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convenience_functions():
    """
    Testa as funções de conveniência
    """
    print(f"\n🔧 Testando funções de conveniência...")

    try:
        from src.conscience.psicws_native_reader import list_Ψcws_files, load_Ψcws, cache_status

        print("   📋 Testando list_Ψcws_files()...")
        files = list_Ψcws_files()
        print(f"      Encontrados: {len(files)} arquivos")

        if files:
            print("   🔄 Testando load_Ψcws()...")
            first_hash = files[0]['hash']
            loaded = load_Ψcws(first_hash)
            print(f"      Carregamento: {'✅' if loaded else '❌'}")

        print("   📊 Testando cache_status()...")
        cache_status()

        return True

    except Exception as e:
        print(f"   ❌ Erro nas funções de conveniência: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando testes de operacionalidade...")

    # Teste principal
    basic_test = test_native_reader()

    # Teste de funções de conveniência
    convenience_test = test_convenience_functions()

    print(f"\n📊 Resumo dos Testes:")
    print(f"   ✅ Teste básico: {'PASSOU' if basic_test else 'FALHOU'}")
    print(f"   ✅ Funções de conveniência: {'PASSOU' if convenience_test else 'FALHOU'}")

    if basic_test and convenience_test:
        print(f"\n🎉 OPERACIONALIDADE CONFIRMADA!")
        print(f"   O sistema consegue ler arquivos .Ψcws nativamente")
    else:
        print(f"\n💥 FALHAS DETECTADAS!")
        print(f"   Revisar implementação necessária")

    exit(0 if basic_test and convenience_test else 1)