#!/usr/bin/env python3
"""
Script interativo para seleção de modelo ΨQRH
Mostra apenas modelos certificados por padrão
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.model_manager import ModelManager


def select_model_interactive():
    """Menu interativo para seleção de modelo."""
    manager = ModelManager()
    registry = manager.load_registry()

    # Filtrar apenas certificados
    certified = [m for m in registry['models'] if m['certification'] == 'certified']
    uncertified = [m for m in registry['models'] if m['certification'] != 'certified']

    active_model = registry.get('active_model')

    if not certified and not uncertified:
        print("❌ Nenhum modelo encontrado")
        print("💡 Execute: make new-model SOURCE=gpt2 NAME=gpt2_chat")
        return None

    print("=" * 80)
    print("🔬 SELEÇÃO DE MODELO ΨQRH")
    print("=" * 80)

    # Mostrar modelos certificados
    if certified:
        print("\n✅ MODELOS CERTIFICADOS (Recomendados):")
        print("-" * 80)
        for i, model in enumerate(certified, 1):
            status = "[ATIVO]" if model['name'] == active_model else "      "
            print(f"  {i}. {status} {model['name']}")
        print("-" * 80)

    # Mostrar não certificados (opcional)
    if uncertified:
        print(f"\n⚠️  Modelos não certificados disponíveis: {len(uncertified)}")
        print("    (Digite 'todos' para ver lista completa)")

    print("\n" + "=" * 80)

    # Prompt de seleção
    while True:
        choice = input("\n👉 Escolha (número, nome ou Enter para ativo): ").strip()

        # Enter = ativo
        if not choice:
            if active_model:
                print(f"✅ Usando modelo ativo: {active_model}")
                return active_model
            else:
                print("❌ Nenhum modelo ativo definido")
                continue

        # "todos" = mostrar não certificados
        if choice.lower() == 'todos':
            print("\n⚠️  MODELOS NÃO CERTIFICADOS:")
            print("-" * 80)
            for i, model in enumerate(uncertified, 1):
                status = "[ATIVO]" if model['name'] == active_model else "      "
                cert = model['certification'].upper()
                print(f"  {len(certified) + i}. {status} {model['name']} [{cert}]")
            print("-" * 80)
            continue

        # Número
        if choice.isdigit():
            idx = int(choice) - 1
            all_models = certified + uncertified

            if 0 <= idx < len(all_models):
                selected = all_models[idx]
                print(f"✅ Selecionado: {selected['name']} [{selected['certification']}]")
                return selected['name']
            else:
                print(f"❌ Número inválido. Escolha entre 1 e {len(all_models)}")
                continue

        # Nome
        # Procurar em todos os modelos
        matches = [m for m in registry['models'] if m['name'].lower() == choice.lower()]

        if matches:
            selected = matches[0]
            print(f"✅ Selecionado: {selected['name']} [{selected['certification']}]")
            return selected['name']

        # Busca parcial
        partial = [m for m in registry['models'] if choice.lower() in m['name'].lower()]

        if len(partial) == 1:
            selected = partial[0]
            print(f"✅ Encontrado: {selected['name']} [{selected['certification']}]")
            return selected['name']
        elif len(partial) > 1:
            print(f"⚠️  Múltiplos modelos encontrados com '{choice}':")
            for m in partial[:5]:
                print(f"    - {m['name']}")
            print("    Digite o nome completo ou número")
            continue

        print(f"❌ Modelo '{choice}' não encontrado")
        print("💡 Digite o número, nome completo ou 'todos' para ver lista completa")


if __name__ == "__main__":
    selected = select_model_interactive()
    if selected:
        print(selected)
    else:
        sys.exit(1)
