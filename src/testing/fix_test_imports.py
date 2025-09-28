#!/usr/bin/env python3
"""
Script para corrigir imports nos arquivos de teste
Salvo em data/test_logs/ seguindo política de isolamento
"""
import os
import re

def fix_imports_in_file(filepath):
    """Corrige imports em um arquivo de teste"""
    changes = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Mapeamento de imports antigos -> novos
        import_mappings = {
            r'from qrh_layer import': r'from src.core.qrh_layer import',
            r'import qrh_layer': r'import src.core.qrh_layer as qrh_layer',
            r'from optimized_components import': r'from src.core.optimized_components import',
            r'import optimized_components': r'import src.core.optimized_components as optimized_components',
            r'from production_system import': r'from src.core.production_system import',
            r'import production_system': r'import src.core.production_system as production_system',
            r'from ΨQRH import': r'from src.core.ΨQRH import',
            r'from spectral_filter import': r'from src.fractal.spectral_filter import',
            r'from agentic_runtime import': r'from src.cognitive.agentic_runtime import',
            r'from navigator_agent import': r'from src.cognitive.navigator_agent import',
            r'from semantic_adaptive_filters import': r'from src.cognitive.semantic_adaptive_filters import',
            r'from negentropy_transformer_block import': r'from src.core.negentropy_transformer_block import',
            r'from quaternion_operations import': r'from src.core.quaternion_operations import',
            r'from enhanced_qrh_layer import': r'from src.core.enhanced_qrh_layer import',
        }

        for old_import, new_import in import_mappings.items():
            new_content = re.sub(old_import, new_import, content)
            if new_content != content:
                changes.append(f"{old_import} -> {new_import}")
                content = new_content

        # Se houve mudanças, salva o arquivo
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return changes

        return []

    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")
        return []

def main():
    """Processa todos os arquivos de teste"""
    # Get the current working directory and find tests folder
    current_dir = os.getcwd()
    tests_dir = os.path.join(current_dir, 'tests')
    total_files = 0
    files_changed = 0
    all_changes = {}

    for filename in os.listdir(tests_dir):
        if filename.endswith('.py') and filename.startswith('test_'):
            filepath = os.path.join(tests_dir, filename)
            total_files += 1

            changes = fix_imports_in_file(filepath)
            if changes:
                files_changed += 1
                all_changes[filename] = changes
                print(f"✅ {filename}: {len(changes)} imports corrigidos")
            else:
                print(f"⏭️  {filename}: nenhum import para corrigir")

    print(f"\n📊 Resumo:")
    print(f"   Arquivos processados: {total_files}")
    print(f"   Arquivos alterados: {files_changed}")

    # Salva relatório detalhado
    report_path = os.path.join(os.path.dirname(__file__), 'import_fixes_report.txt')
    with open(report_path, 'w') as f:
        f.write("Relatório de Correção de Imports\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Arquivos processados: {total_files}\n")
        f.write(f"Arquivos alterados: {files_changed}\n\n")

        for filename, changes in all_changes.items():
            f.write(f"{filename}:\n")
            for change in changes:
                f.write(f"  - {change}\n")
            f.write("\n")

    print(f"📄 Relatório salvo em: {report_path}")

if __name__ == "__main__":
    main()