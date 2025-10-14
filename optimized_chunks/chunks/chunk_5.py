# Chunk 5: Lines 4774-5191
# Tokens: 4503, Lines: 4774-5191


    test_cases = [
        "O que são quaternions?",
        "Explique a transformada de Fourier",
        "Como funciona o framework ΨQRH?"
    ]

    # Use default tokenizer config if not provided
    if tokenizer_config is None:
        tokenizer_config = {
            'embed_dim': 64,
            'spectral_params_dim': 8,
            'learnable': True
        }

    pipeline = ΨQRHPipeline(task="text-generation", model_dir=model_dir, enable_auto_calibration=enable_auto_calibration, tokenizer_config=tokenizer_config, audit_mode=audit_mode)

    for i, test_text in enumerate(test_cases, 1):
        print(f"\n--- Teste {i}/{len(test_cases)} ---")
        print(f"Entrada: {test_text}")

        result = pipeline(test_text)

        if result['status'] == 'success':
            print(f"✅ Sucesso! ({result['output_length']} caracteres)")
            if result.get('auto_learning_enhanced', False):
                print(f"🤖 Auto-learning: ENHANCED")
            if verbose:
                print(f"Resposta: {result['response'][:200]}...")
        else:
            print(f"❌ Erro: {result.get('error', 'Desconhecido')}")

    print("\n🎯 Teste concluído!")
    return 0

def run_test_echo(model_dir: Optional[str] = None, audit_mode: bool = False, reasoning_mode: str = 'geometric') -> int:
    """Executa teste de eco rápido (uma entrada/saída)"""
    print("🎤 Executando teste de eco no modelo ativo...")

    # Exibir informações do modelo
    model_info = get_model_info(model_dir)
    print(f"📁 Modelo: {model_info['name']}")
    print(f"✅ Status: {'CERTIFICADO' if model_info['certification'] == 'certified' else 'NÃO CERTIFICADO'}")

    # Criar pipeline
    pipeline = ΨQRHPipeline(task="text-generation", model_dir=model_dir, audit_mode=audit_mode, reasoning_mode=reasoning_mode)

    # Teste de eco simples
    test_input = "Olá, este é um teste de eco rápido do ΨQRH."
    print(f"\n📤 Entrada: {test_input}")

    result = pipeline(test_input)

    if result['status'] == 'success':
        response = result['response']
        if isinstance(response, dict) and 'text_analysis' in response:
            response = response['text_analysis']
        print(f"📥 Resposta: {response}")
        print(f"✅ Teste de eco concluído com sucesso!")
    else:
        print(f"❌ Erro no teste de eco: {result.get('error', 'Desconhecido')}")

    return 0

def run_physics_tests() -> int:
    """Executa testes de validação física do ΨQRH"""
    print("🔬 Executando testes de validação física...")
    print("📋 Testes incluídos:")
    print("   1. Fractal Embedding Physics (quaternion unitarity)")
    print("   2. Spectral Attention Physics (energy conservation)")
    print("   3. SO(4) Evolution Physics (unitary transformations)")
    print("   4. Optical Probe Physics (resonance detection)")
    print("   5. Complete ΨQRH Transformer (end-to-end pipeline)")
    print()

    import subprocess
    result = subprocess.run(
        ['python3', 'examples/test_complete_psiqrh.py'],
        capture_output=False
    )

    if result.returncode == 0:
        print("\n✅ Todos os testes físicos passaram!")
    else:
        print(f"\n❌ Testes físicos falharam (código: {result.returncode})")

    return result.returncode

def run_interactive_mode(task: str, device: Optional[str], verbose: bool = False, model_dir: Optional[str] = None, enable_auto_calibration: bool = True, audit_mode: bool = False) -> int:
    """Modo interativo de chat com auto-aprendizagem e salvamento estruturado"""
    import yaml
    from datetime import datetime

    # Criar diretório de sessão interativa
    session_dir = Path("results") / "interactive_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = session_dir / f"session_{session_timestamp}.json"

    # Inicializar arquivo de sessão
    session_data = {
        'session_start': session_timestamp,
        'task': task,
        'device': device,
        'model_dir': model_dir,
        'enable_auto_calibration': enable_auto_calibration,
        'conversations': []
    }

    # Exibir cabeçalho informativo
    display_model_header(model_dir)

    if enable_auto_calibration and HAS_AUTO_CALIBRATION:
        print("🤖 Auto-calibração: ATIVADA (ΨQRH Spectral + Fractal)")
    else:
        print("🤖 Auto-calibração: DESATIVADA")

    print(f"💾 Sessão interativa será salva em: {session_file}")

    # Criar pipeline inicial com task padrão
    pipeline = ΨQRHPipeline(task=task, device=device, model_dir=model_dir, enable_auto_calibration=enable_auto_calibration, tokenizer_config=tokenizer_config, audit_mode=audit_mode, reasoning_mode=reasoning_mode)

    while True:
        try:
            user_input = input("\n🤔 Você: ").strip()

            if user_input.lower() in ['quit', 'exit', 'sair']:
                print("👋 Até logo!")
                break

            if user_input.lower() in ['help', 'ajuda']:
                print("""
Comandos disponíveis:
  quit/exit/sair - Sair do modo interativo
  help/ajuda - Mostrar esta ajuda
  [qualquer texto] - Processar com ΨQRH
                """)
                continue

            if not user_input:
                continue

            # Usar pipeline existente, apenas atualizar task se necessário
            current_task = pipeline.task
            detected_task = pipeline._detect_task_type(user_input)

            # Recriar pipeline apenas se a tarefa mudou
            if detected_task != current_task:
                pipeline = ΨQRHPipeline(task=detected_task, device=device, model_dir=model_dir, enable_auto_calibration=enable_auto_calibration, tokenizer_config=tokenizer_config, audit_mode=audit_mode, reasoning_mode=reasoning_mode)
                print(f"🔄 Tarefa detectada: {detected_task} (anterior: {current_task})")

            print(f"🧠 ΨQRH processando... (Tarefa: {pipeline.task})")
            result = pipeline(user_input)

            # Preparar dados da conversa para salvar
            conversation_entry = {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'user_input': user_input,
                'detected_task': pipeline.task,
                'result': result
            }

            if result['status'] == 'success':
                response = result['response']

                # Handle both string and dictionary responses
                if isinstance(response, dict) and 'text_analysis' in response:
                    print(f"🤖 ΨQRH: {response['text_analysis']}")

                    # Generate GLS output if consciousness results are available
                    if 'consciousness_results' in response:
                        try:
                            # Import GLS generator
                            from src.conscience.gls_output_generator import GLSOutputGenerator
                            gls_generator = GLSOutputGenerator()

                            # Generate both Processing and p5.js code
                            processing_code = gls_generator.generate_processing_code(response['consciousness_results'])
                            p5js_code = gls_generator.generate_p5js_code(response['consciousness_results'])

                            print("\n🎨 GLS VISUALIZATION CODE GENERATED:")
                            print("=" * 50)
                            print("📱 Processing Code (copy to Processing IDE):")
                            print(processing_code[:500] + "..." if len(processing_code) > 500 else processing_code)
                            print("\n🌐 p5.js Code (copy to HTML file):")
                            print(p5js_code[:500] + "..." if len(p5js_code) > 500 else p5js_code)
                            print("=" * 50)

                            # Salvar códigos GLS também
                            conversation_entry['gls_codes'] = {
                                'processing': processing_code,
                                'p5js': p5js_code
                            }
                        except Exception as e:
                            print(f"⚠️  GLS output generation failed: {e}")
                            conversation_entry['gls_error'] = str(e)
                else:
                    print(f"🤖 ΨQRH: {response}")

                if result.get('auto_learning_enhanced', False):
                    print("🤖 [Auto-learning enhancement applied]")

                if verbose:
                    print(f"📊 Metadados: {result['device']}, {result['output_length']} chars")
            else:
                print(f"❌ Erro: {result.get('error', 'Desconhecido')}")

            # Adicionar conversa à sessão e salvar
            session_data['conversations'].append(conversation_entry)
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

        except EOFError:
            print("\n👋 EOF detectado, encerrando modo interativo")
            break
        except KeyboardInterrupt:
            print("\n👋 Interrompido pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")

    # Finalizar sessão
    session_data['session_end'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_data['total_conversations'] = len(session_data['conversations'])

    with open(session_file, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Sessão completa salva em: {session_file}")
    print(f"📊 Total de conversas: {session_data['total_conversations']}")

    return 0

def process_single_text(text: str, task: str, device: Optional[str], verbose: bool = False, model_dir: Optional[str] = None, enable_auto_calibration: bool = True, tokenizer_config: Optional[Dict[str, Any]] = None, json_output: bool = False, audit_mode: bool = False, selected_model: str = 'gpt2', reasoning_mode: str = 'geometric') -> int:
    """Processa um único texto com auto-aprendizagem e salva saídas estruturadas"""
    import yaml
    from datetime import datetime
    import os

    # Usar detecção automática de tarefa baseada no conteúdo do texto
    pipeline = ΨQRHPipeline(task=task, device=device, input_text=text, model_dir=model_dir, enable_auto_calibration=enable_auto_calibration, tokenizer_config=tokenizer_config, audit_mode=audit_mode, reasoning_mode=reasoning_mode)

    # Integrar seleção de modelo na configuração do pipeline
    if hasattr(pipeline, 'config') and selected_model != 'gpt2':
        pipeline.config['selected_model'] = selected_model
        print(f"🤖 Modelo selecionado: {selected_model}")
    else:
        pipeline.config['selected_model'] = 'gpt2'  # Default fallback

    print(f"🧠 Processando: {text}")
    print(f"📋 Tarefa detectada: {pipeline.task}")
    if enable_auto_calibration:
        print(f"🤖 Auto-calibração: ATIVADA")
    result = pipeline(text)

    # Criar diretório de resultados se não existir
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Timestamp para identificação única
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"psiqrh_result_{timestamp}"

    if result['status'] == 'success':
        # ========== SALVAR RESULTADOS ESTRUTURADOS ==========
        def tensor_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [tensor_to_serializable(item) for item in obj]
            else:
                return obj

        json_result = {
            'timestamp': timestamp,
            'input_text': text,
            'task': result['task'],
            'device': result['device'],
            'status': result['status'],
            'response': result['response'],
            'input_length': result['input_length'],
            'output_length': result['output_length'],
            'processing_time': result.get('processing_time', 0),
            'selected_method': result.get('selected_method', 'N/A'),
            'auto_calibration_applied': result.get('auto_calibration_applied', False),
            'physical_metrics': result.get('physical_metrics', {}),
            'mathematical_validation': result.get('mathematical_validation', {}),
            'pipeline_steps': result.get('pipeline_steps', []),
            'dcf_analysis': result.get('dcf_analysis', {}),
            'spectral_analysis': result.get('spectral_analysis', {}),
            'dcf_validation': result.get('dcf_validation', {}),
            'dcf_metadata': result.get('dcf_metadata', {}),
            'semantic_analysis': result.get('semantic_analysis', {})
        }

        # Convert any tensors in the result to serializable format
        serializable_json_result = tensor_to_serializable(json_result)

        if json_output:
            # JSON-only output mode
            print(json.dumps(serializable_json_result, ensure_ascii=False))
            return 0

        # Default console output mode
        print(f"\n✅ Resultado ({result['device']}):")
        if result.get('auto_calibration_applied', False):
            print("🤖 [Auto-calibration applied]")

        print(f"\n💾 Salvando resultados estruturados...")

        json_file = results_dir / f"{base_filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_json_result, f, indent=2, ensure_ascii=False)
        print(f"   📄 Resultado JSON salvo: {json_file}")

        # 2. Análise DCF detalhada em YAML
        dcf_analysis = {}

        # Método comparison
        if 'method_comparison' in result and result.get('method_comparison'):
            dcf_analysis['method_comparison'] = result['method_comparison']

        # Análise DCF específica
        if 'dcf_analysis' in result:
            dcf_analysis['dcf_analysis'] = result['dcf_analysis']

        # Análise espectral completa
        if 'spectral_analysis' in result and result['spectral_analysis']:
            dcf_analysis['spectral_analysis'] = result['spectral_analysis']

        # Análise quântica
        if 'quantum_interpretation' in result:
            dcf_analysis['quantum_interpretation'] = result['quantum_interpretation']

        if dcf_analysis:
            yaml_file = results_dir / f"{base_filename}_dcf.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(dcf_analysis, f, default_flow_style=False, indent=2, allow_unicode=True)
            print(f"   📋 Análise DCF YAML salva: {yaml_file}")

        # 3. Métricas físicas em arquivo separado
        if 'physical_metrics' in result:
            metrics_file = results_dir / f"{base_filename}_metrics.json"
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'input_text': text[:100] + '...' if len(text) > 100 else text,
                    'physical_metrics': result['physical_metrics'],
                    'mathematical_validation': result.get('mathematical_validation', {}),
                    'dcf_metadata': result.get('dcf_metadata', {})
                }, f, indent=2, ensure_ascii=False)
            print(f"   📊 Métricas físicas salvas: {metrics_file}")

        # ========== EXIBIÇÃO RESUMIDA NO CONSOLE ==========
        print("\n🎯 SISTEMA DCF - RESUMO DA ANÁLISE:")
        print("=" * 60)

        # Mostrar informações resumidas
        if 'method_comparison' in result and result.get('method_comparison'):
            comparison = result['method_comparison']
            print(f"🔍 Métodos comparados: {len(comparison)} métodos")

            # Mostrar método selecionado
            selected_method = result.get('selected_method', 'N/A')
            print(f"🎯 Método selecionado: {selected_method}")

        # Mostrar informações do DCF se disponíveis
        if 'dcf_analysis' in result:
            dcf_data = result['dcf_analysis']
            print(f"🧠 FCI: {dcf_data.get('fci_value', dcf_data.get('fci', 0)):.4f}")
            print(f"🎭 Estado: {dcf_data.get('consciousness_state', 'N/A')}")
            print(f"🔄 Sincronização: {dcf_data.get('synchronization_order', 0):.4f}")

        print(f"📝 Resposta: {result['response'][:100]}{'...' if len(result['response']) > 100 else ''}")
        print(f"💾 Arquivos salvos em: {results_dir}/")
        print(f"   • {base_filename}.json (resultado principal)")
        print(f"   • {base_filename}_dcf.yaml (análise DCF)")
        print(f"   • {base_filename}_metrics.json (métricas físicas)")
        print("=" * 60)

        # Nota sobre saída JSON
        print(f"\n💡 Para saída JSON limpa: python3 psiqrh.py \"{text}\" --json")
        print("=" * 60)

        if verbose:
            print(f"\n📊 Metadados:")
            print(f"  - Tarefa: {result['task']}")
            print(f"  - Dispositivo: {result['device']}")
            print(f"  - Entrada: {result['input_length']} caracteres")
            print(f"  - Saída: {result['output_length']} caracteres")
            print(f"  - Auto-calibration: {'APPLIED' if result.get('auto_calibration_applied', False) else 'BASELINE'}")

    else:
        print(f"❌ Erro: {result.get('error', 'Desconhecido')}")

        # Salvar erro também
        error_result = {
            'timestamp': timestamp,
            'input_text': text,
            'status': 'error',
            'error': result.get('error', 'Desconhecido'),
            'task': task,
            'device': device
        }

        error_file = results_dir / f"{base_filename}_error.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=2, ensure_ascii=False)
        print(f"💾 Erro salvo em: {error_file}")

        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())