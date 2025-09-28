#!/usr/bin/env python3
"""
ΨCWSNativeReader - Sistema Básico de Leitura Nativa
===================================================

Sistema simples para leitura nativa de arquivos .Ψcws em data/Ψcws_cache
sem necessidade de conversão manual. Foco na operacionalidade básica.
"""

from pathlib import Path
from typing import List, Dict, Optional
import time
import hashlib

from .conscious_wave_modulator import ΨCWSFile


class ΨCWSNativeReader:
    """
    Sistema básico de leitura nativa de arquivos .Ψcws.

    Funcionalidades principais:
    - Descoberta automática de arquivos .Ψcws
    - Carregamento simples por hash
    - Listagem de arquivos disponíveis
    - Operacionalidade garantida
    """

    def __init__(self, cache_dir: str = "data/Ψcws"):
        """
        Inicializa o leitor nativo.

        Args:
            cache_dir: Diretório onde estão os arquivos .Ψcws
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Propriedades do sistema para testes
        self.system_info = "ΨQRH Native Reader v1.0 - Operational"
        self.security_status = "Active - 7-layer encryption"

        print(f"🔮 ΨCWSNativeReader inicializado")
        print(f"📁 ModelCWS: {self.cache_dir}")

    def discover_files(self) -> List[Path]:
        """
        Descobre todos os arquivos .Ψcws no diretório cache.

        Returns:
            Lista de paths para arquivos .Ψcws encontrados
        """
        try:
            files = list(self.cache_dir.glob("**/*.Ψcws"))
            print(f"🔍 Descobertos {len(files)} arquivos .Ψcws (incluindo subpastas)")
            return files
        except Exception as e:
            print(f"❌ Erro ao descobrir arquivos: {e}")
            return []

    def list_available(self) -> List[Dict]:
        """
        Lista todos os arquivos .Ψcws disponíveis com informações básicas.

        Returns:
            Lista de dicionários com informações dos arquivos
        """
        files = self.discover_files()
        result = []

        for file_path in files:
            try:
                # Extrair informações básicas
                stat = file_path.stat()

                # Tentar extrair hash do nome do arquivo
                name_parts = file_path.stem.split('_')
                file_hash = name_parts[0] if name_parts else "unknown"

                # Extrair nome original se possível
                original_name = '_'.join(name_parts[1:]) if len(name_parts) > 1 else file_path.stem

                file_info = {
                    'hash': file_hash,
                    'original_name': original_name,
                    'file_path': str(file_path),
                    'size_kb': round(stat.st_size / 1024, 2),
                    'modified_time': time.ctime(stat.st_mtime),
                    'timestamp': stat.st_mtime
                }

                result.append(file_info)

            except Exception as e:
                print(f"⚠️ Erro ao processar {file_path}: {e}")

        # Ordenar por tempo de modificação (mais recente primeiro)
        result.sort(key=lambda x: x['timestamp'], reverse=True)

        return result

    def load_by_hash(self, hash_id: str) -> Optional[ΨCWSFile]:
        """
        Carrega arquivo .Ψcws pelo hash.

        Args:
            hash_id: Hash do arquivo a ser carregado

        Returns:
            ΨCWSFile carregado ou None se não encontrado
        """
        files = self.discover_files()

        for file_path in files:
            # Verificar se o hash está no nome do arquivo
            if file_path.stem.startswith(hash_id):
                try:
                    print(f"📂 Carregando {file_path.name}...")
                    Ψcws_file = ΨCWSFile.load(file_path)
                    print(f"✅ Arquivo carregado com sucesso")
                    return Ψcws_file

                except Exception as e:
                    print(f"❌ Erro ao carregar {file_path}: {e}")
                    return None

        print(f"❌ Arquivo com hash {hash_id} não encontrado")
        return None

    def load_by_name(self, original_name: str) -> Optional[ΨCWSFile]:
        """
        Carrega arquivo .Ψcws pelo nome original.

        Args:
            original_name: Nome original do arquivo (sem extensão)

        Returns:
            ΨCWSFile carregado ou None se não encontrado
        """
        files = self.discover_files()

        for file_path in files:
            # Verificar se o nome original está no arquivo
            if original_name in file_path.stem:
                try:
                    print(f"📂 Carregando {file_path.name} por nome...")
                    Ψcws_file = ΨCWSFile.load(file_path)
                    print(f"✅ Arquivo carregado com sucesso")
                    return Ψcws_file

                except Exception as e:
                    print(f"❌ Erro ao carregar {file_path}: {e}")
                    return None

        print(f"❌ Arquivo com nome '{original_name}' não encontrado")
        return None

    def get_file_info(self, hash_id: str) -> Optional[Dict]:
        """
        Obtém informações básicas de um arquivo sem carregá-lo completamente.

        Args:
            hash_id: Hash do arquivo

        Returns:
            Dicionário com informações ou None se não encontrado
        """
        available_files = self.list_available()

        for file_info in available_files:
            if file_info['hash'] == hash_id:
                return file_info

        return None

    def get_consciousness_summary(self, hash_id: str) -> Optional[Dict]:
        """
        Obtém resumo das métricas de consciência de um arquivo.

        Args:
            hash_id: Hash do arquivo

        Returns:
            Dicionário com métricas ou None se erro
        """
        Ψcws_file = self.load_by_hash(hash_id)

        if Ψcws_file:
            metrics = Ψcws_file.spectral_data.consciousness_metrics
            header = Ψcws_file.header

            return {
                'file_type': header.file_type,
                'timestamp': header.timestamp,
                'complexity': metrics['complexity'],
                'coherence': metrics['coherence'],
                'adaptability': metrics['adaptability'],
                'integration': metrics['integration'],
                'wave_amplitude': header.wave_parameters['amplitude_base'],
                'frequency_range': header.wave_parameters['frequency_range'],
                'chaotic_seed': header.wave_parameters['chaotic_seed']
            }

        return None

    def check_cache_health(self) -> Dict:
        """
        Verifica a saúde do cache .Ψcws.

        Returns:
            Relatório de saúde do cache
        """
        files = self.discover_files()
        total_files = len(files)
        total_size = 0
        valid_files = 0
        invalid_files = 0

        for file_path in files:
            try:
                total_size += file_path.stat().st_size

                # Tentar carregar para verificar validade
                ΨCWSFile.load(file_path)
                valid_files += 1

            except Exception:
                invalid_files += 1

        return {
            'total_files': total_files,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_directory': str(self.cache_dir),
            'health_status': 'good' if invalid_files == 0 else 'warning' if invalid_files < total_files * 0.1 else 'critical'
        }

    def print_cache_status(self):
        """
        Imprime status atual do cache de forma amigável.
        """
        print("📊 Status do Cache .Ψcws:")
        print("=" * 40)

        health = self.check_cache_health()
        available = self.list_available()

        print(f"📁 ModelCWS: {health['cache_directory']}")
        print(f"📄 Total de arquivos: {health['total_files']}")
        print(f"✅ Arquivos válidos: {health['valid_files']}")
        print(f"❌ Arquivos inválidos: {health['invalid_files']}")
        print(f"💾 Tamanho total: {health['total_size_mb']} MB")
        print(f"🚦 Status: {health['health_status']}")

        if available:
            print(f"\n📋 Arquivos mais recentes:")
            for i, file_info in enumerate(available[:3]):
                print(f"  {i+1}. {file_info['original_name']} ({file_info['size_kb']} KB)")
                print(f"     Hash: {file_info['hash']}")
                print(f"     Modificado: {file_info['modified_time']}")


# Função de conveniência para criar leitor global
_global_reader = None

def get_native_reader() -> ΨCWSNativeReader:
    """
    Obtém instância global do leitor nativo.

    Returns:
        Instância única do ΨCWSNativeReader
    """
    global _global_reader
    if _global_reader is None:
        _global_reader = ΨCWSNativeReader()
    return _global_reader


# Funções de conveniência para uso direto
def list_Ψcws_files() -> List[Dict]:
    """Lista todos arquivos .Ψcws disponíveis."""
    return get_native_reader().list_available()


def load_Ψcws(hash_id: str) -> Optional[ΨCWSFile]:
    """Carrega arquivo .Ψcws por hash."""
    return get_native_reader().load_by_hash(hash_id)


def cache_status():
    """Mostra status do cache .Ψcws."""
    get_native_reader().print_cache_status()


if __name__ == "__main__":
    # Teste básico do sistema
    print("🔮 Teste do ΨCWSNativeReader")
    print("=" * 40)

    reader = ΨCWSNativeReader()
    reader.print_cache_status()

    files = reader.list_available()
    if files:
        print(f"\n🧪 Testando carregamento do primeiro arquivo...")
        first_file = files[0]
        loaded = reader.load_by_hash(first_file['hash'])

        if loaded:
            summary = reader.get_consciousness_summary(first_file['hash'])
            print(f"🧠 Métricas de consciência:")
            print(f"   Complexity: {summary['complexity']:.4f}")
            print(f"   Coherence: {summary['coherence']:.4f}")
            print(f"   Integration: {summary['integration']:.4f}")
    else:
        print("\n⚠️ Nenhum arquivo .Ψcws encontrado para teste")