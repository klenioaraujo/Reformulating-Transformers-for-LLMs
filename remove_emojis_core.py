#!/usr/bin/env python3
"""
Script para remover todos os emojis do core do sistema ΨQRH
===========================================================

Remove emojis dos arquivos core para manter o sistema limpo
sem elementos gráficos desnecessários.
"""

import os
import re
from pathlib import Path

def remove_emojis_from_file(file_path: Path) -> bool:
    """Remove emojis de um arquivo específico"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern para emojis comuns encontrados
        emoji_patterns = [
            r'🔮', r'⚡', r'📊', r'🧠', r'❌', r'✅', r'🎯', r'🚀',
            r'📋', r'🔧', r'🔬', r'💾', r'📈', r'🏃', r'🔍', r'🎉',
            r'⚠️', r'📂', r'💬', r'🤔', r'🤖', r'👋', r'📝', r'📁',
            r'⏱️', r'🎪', r'🔥', r'💡', r'🌟', r'🎊'
        ]

        # Remove cada emoji
        for emoji in emoji_patterns:
            content = re.sub(emoji + r'\s*', '', content)

        # Limpa múltiplos espaços resultantes
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'^\s+', '', content, flags=re.MULTILINE)

        # Se houve mudanças, salva o arquivo
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return False

    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return False

def remove_emojis_from_core():
    """Remove emojis de todos os arquivos core"""

    core_dirs = [
        "src/core",
        "src/prompt_engine"
    ]

    project_root = Path(__file__).parent

    files_modified = 0
    total_files = 0

    for core_dir in core_dirs:
        core_path = project_root / core_dir

        if not core_path.exists():
            print(f"Diretório não encontrado: {core_path}")
            continue

        # Processa arquivos Python
        for py_file in core_path.glob("*.py"):
            total_files += 1
            if remove_emojis_from_file(py_file):
                files_modified += 1
                print(f"Emojis removidos de: {py_file.name}")
            else:
                print(f"Nenhum emoji encontrado em: {py_file.name}")

    print(f"\nResumo:")
    print(f"Arquivos processados: {total_files}")
    print(f"Arquivos modificados: {files_modified}")
    print(f"Arquivos sem emojis: {total_files - files_modified}")

if __name__ == "__main__":
    print("Removendo emojis do core do sistema ΨQRH...")
    print("=" * 50)
    remove_emojis_from_core()
    print("\nRemoção de emojis concluída!")