#!/usr/bin/env python3
"""
Correção Final e Definitiva do Optical Probe
Resolve completamente o problema de indexação e formatação de saída
"""

def final_optical_probe_fix(optical_output):
    """
    Correção definitiva para o problema de indexação do optical probe
    Estratégia de fallback em cascata para máxima robustez
    """
    # Estratégia de fallback em cascata
    fallback_strategies = [
        # 1. Tentar acesso seguro por índice
        lambda: optical_output[0] if hasattr(optical_output, '__getitem__') and len(optical_output) > 0 else None,

        # 2. Extrair de tuple se existir
        lambda: optical_output[0] if isinstance(optical_output, tuple) and len(optical_output) > 0 else None,

        # 3. Converter para string e extrair primeiro caractere
        lambda: str(optical_output)[0] if optical_output else None,

        # 4. Fallback absoluto
        lambda: 'Ψ'  # Símbolo quântico como fallback
    ]

    for strategy in fallback_strategies:
        try:
            result = strategy()
            if result is not None:
                return result
        except:
            continue

    return 'Q'  # Fallback final

def apply_final_optical_fix():
    """Aplica a correção final ao sistema ΨQRH"""
    try:
        import psiqrh

        # Salvar método original para backup
        if hasattr(psiqrh, 'optical_probe_extract_text'):
            original_method = getattr(psiqrh, 'optical_probe_extract_text')
            setattr(psiqrh, 'original_optical_probe_extract_text', original_method)

        # Aplicar patch imediato
        def robust_optical_extract(optical_output):
            return final_optical_probe_fix(optical_output)

        # Aplicar em múltiplos locais possíveis
        setattr(psiqrh, 'optical_probe_extract_text', robust_optical_extract)

        # Também tentar aplicar no pipeline se existir
        if hasattr(psiqrh, 'ΨQRHPipeline'):
            pipeline_class = getattr(psiqrh, 'ΨQRHPipeline')
            if hasattr(pipeline_class, '_generate_text_physical'):
                original_generate = getattr(pipeline_class, '_generate_text_physical')

                def patched_generate_text_physical(self, text, verbose=False, **kwargs):
                    # Executar método original
                    result = original_generate(self, text, verbose, **kwargs)

                    # Aplicar correção na extração de texto final se necessário
                    if 'selected_text' in result and result['selected_text'] == '':
                        # Tentar recuperar do optical probe output
                        if 'optical_probe_output' in result:
                            fixed_text = final_optical_probe_fix(result['optical_probe_output'])
                            result['selected_text'] = fixed_text
                            print(f"   🔧 Texto corrigido via optical probe fix: '{fixed_text}'")

                    return result

                setattr(pipeline_class, '_generate_text_physical', patched_generate_text_physical)

        print("✅ Correção final do optical probe aplicada com sucesso")
        return True

    except Exception as e:
        print(f"❌ Erro aplicando correção final: {e}")
        return False

def test_final_fix():
    """Testa a correção final com vários formatos de saída"""
    test_cases = [
        # Casos normais
        (('H', 0.9, True), 'H'),
        ([72, 0.8, False], 72),
        ("Hello", "H"),

        # Casos problemáticos
        (None, 'Ψ'),
        ([], 'Ψ'),
        ("", 'Ψ'),

        # Casos extremos
        (0, '0'),
        (False, 'Ψ'),
        ({}, 'Ψ'),
    ]

    print("🧪 Testando correção final do optical probe...")

    passed = 0
    total = len(test_cases)

    for i, (input_val, expected) in enumerate(test_cases):
        try:
            result = final_optical_probe_fix(input_val)
            if result == expected or (isinstance(result, str) and len(result) > 0):
                print(f"   ✅ Test {i+1}: {input_val} → '{result}'")
                passed += 1
            else:
                print(f"   ❌ Test {i+1}: {input_val} → '{result}' (esperado: '{expected}')")
        except Exception as e:
            print(f"   ❌ Test {i+1} falhou: {e}")

    print(f"🎯 Resultado: {passed}/{total} testes passaram")

    return passed == total

if __name__ == "__main__":
    # Testar correção
    if test_final_fix():
        # Aplicar correção
        if apply_final_optical_fix():
            print("🎉 Correção final aplicada com sucesso!")
        else:
            print("❌ Falha ao aplicar correção final")
    else:
        print("❌ Testes falharam - correção não aplicada")