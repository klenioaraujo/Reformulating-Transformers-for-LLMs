#!/usr/bin/env python3
"""
Test Script for Secure Data Pipeline

Tests the complete secure data pipeline from asset creation to training validation.
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.create_secure_asset import SecureAssetCreator
from scripts.secure_asset_validator import SecureAssetValidator
from scripts.secure_training_integration import SecureTrainingSystem


def create_test_data():
    """Create test data file"""
    test_content = """
This is a test document for ΨQRH secure data pipeline.
It contains sensitive information that should be protected.

Security levels:
- Personal: Basic protection
- Enterprise: Enhanced security with audit logging
- Government: Maximum security with classification

The ΨQRH framework ensures data confidentiality and integrity
through spectral transformations and digital certification.
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        return f.name


def test_personal_level():
    """Test personal security level"""
    print("\n🧪 Testando Nível Personal")
    print("=" * 40)

    # Create test data
    test_file = create_test_data()
    print(f"📄 Arquivo de teste criado: {test_file}")

    # Create secure asset
    creator = SecureAssetCreator()
    try:
        asset_path, manifest_path, cert_path = creator.create_secure_asset(
            source_file=test_file,
            asset_name="test-personal",
            security_level="personal",
            key=None,  # Personal uses default key
            author="Test User",
            description="Test asset for personal security level",
            classification=""
        )
        print("✅ Ativo personal criado com sucesso")

        # Validate asset
        validator = SecureAssetValidator()
        if validator.validate_asset("test-personal"):
            print("✅ Ativo personal validado com sucesso")
        else:
            print("❌ Falha na validação do ativo personal")
            return False

    except Exception as e:
        print(f"❌ Erro ao criar ativo personal: {e}")
        return False
    finally:
        # Cleanup
        os.unlink(test_file)

    return True


def test_enterprise_level():
    """Test enterprise security level"""
    print("\n🏢 Testando Nível Enterprise")
    print("=" * 40)

    # Create test data
    test_file = create_test_data()
    print(f"📄 Arquivo de teste criado: {test_file}")

    # Create secure asset
    creator = SecureAssetCreator()
    try:
        asset_path, manifest_path, cert_path = creator.create_secure_asset(
            source_file=test_file,
            asset_name="test-enterprise",
            security_level="enterprise",
            key="ENTERPRISE_SECRET_KEY_123",
            author="Enterprise User",
            description="Test asset for enterprise security level",
            classification="Internal Use Only"
        )
        print("✅ Ativo enterprise criado com sucesso")

        # Validate asset
        validator = SecureAssetValidator()
        if validator.validate_asset("test-enterprise", "ENTERPRISE_SECRET_KEY_123"):
            print("✅ Ativo enterprise validado com sucesso")
        else:
            print("❌ Falha na validação do ativo enterprise")
            return False

        # Test validation without key (should fail)
        if validator.validate_asset("test-enterprise"):
            print("❌ Validação deveria falhar sem chave")
            return False
        else:
            print("✅ Validação falhou corretamente sem chave")

    except Exception as e:
        print(f"❌ Erro ao criar ativo enterprise: {e}")
        return False
    finally:
        # Cleanup
        os.unlink(test_file)

    return True


def test_training_integration():
    """Test training system integration"""
    print("\n🎓 Testando Integração com Sistema de Treinamento")
    print("=" * 50)

    training_system = SecureTrainingSystem()

    # Test with personal asset
    if training_system.validate_training_asset("test-personal"):
        print("✅ Integração com ativo personal funcionando")
    else:
        print("❌ Falha na integração com ativo personal")
        return False

    # Test with enterprise asset
    if training_system.validate_training_asset("test-enterprise", "ENTERPRISE_SECRET_KEY_123"):
        print("✅ Integração com ativo enterprise funcionando")
    else:
        print("❌ Falha na integração com ativo enterprise")
        return False

    # Test enterprise asset without key (should fail)
    try:
        training_system.get_secure_training_data("test-enterprise")
        print("❌ Deveria falhar sem chave")
        return False
    except ValueError:
        print("✅ Falha correta sem chave")

    return True


def test_manifest_and_certification():
    """Test manifest and certification files"""
    print("\n📋 Testando Manifestos e Certificações")
    print("=" * 45)

    validator = SecureAssetValidator()

    # List assets
    assets = validator.list_secure_assets()
    print(f"📦 Ativos encontrados: {len(assets)}")

    for asset in assets:
        print(f"  • {asset['name']} ({asset['security_level']})")

    # Check audit log
    audit_log_path = Path('data/audit_log.jsonl')
    if audit_log_path.exists():
        with open(audit_log_path, 'r') as f:
            audit_entries = [json.loads(line) for line in f if line.strip()]
        print(f"📊 Entradas de auditoria: {len(audit_entries)}")
    else:
        print("📊 Log de auditoria não encontrado (esperado para personal)")

    return True


def main():
    """Run all tests"""
    print("🚀 Teste Completo do Pipeline de Dados Seguros")
    print("=" * 55)

    tests = [
        ("Nível Personal", test_personal_level),
        ("Nível Enterprise", test_enterprise_level),
        ("Integração Treinamento", test_training_integration),
        ("Manifestos e Certificações", test_manifest_and_certification)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 55)
    print("📊 RESUMO DOS TESTES")
    print("=" * 55)

    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
        if success:
            passed += 1

    print(f"\n🎯 Resultado: {passed}/{len(results)} testes passaram")

    if passed == len(results):
        print("\n🎉 Pipeline de dados seguros funcionando perfeitamente!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} testes falharam")
        return 1


if __name__ == "__main__":
    sys.exit(main())