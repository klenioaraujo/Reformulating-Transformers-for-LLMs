#!/usr/bin/env python3
"""
Autonomous Spectral Calibrator - Sistema de Auto-Acoplamento Espectral
=====================================================================

Implementa sistema de auto-acoplamento espectral dinâmico que integra:
1. Calibração FCI com dados ΨTWS
2. Conversão de embedding com modulação semântica
3. Auto-acoplamento espectral para diversificação de tokens

Baseado no padrão: Da Calibração à Conversão Física
"""

@dataclass
class SemanticCategory:
    """Categoria semântica para modulação de embedding"""
    name: str
    target_fci: float
    alpha_modulation: float
    description: str


class AutonomousSpectralCalibrator:
    """
    Sistema de auto-acoplamento espectral dinâmico

    Integra calibração FCI com conversão de embedding e auto-acoplamento
    para gerar tokens diversos via ressonância física.
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Caminho para configuração de calibração
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carregar configuração de calibração
        if config_path:
            self.calibration_config = self._load_calibration_config(config_path)
        else:
            self.calibration_config = self._load_default_calibration()

        # Inicializar categorias semânticas
        self.semantic_categories = self._initialize_semantic_categories()

        # Parâmetros de auto-acoplamento
        self.alpha_range = (0.1, 3.0)
        self.beta_range = (0.5, 1.5)
        self.coupling_strength = 1.0

        print("🚀 Sistema de Auto-Acoplamento Espectral Inicializado")
        print(f"📊 Categorias semânticas: {len(self.semantic_categories)}")
        print(f"🔧 Configuração: {self.calibration_config.get('state_thresholds', {})}")

    def _load_calibration_config(self, config_path: str) -> Dict:
        """Carrega configuração de calibração FCI"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuração de calibração carregada: {config_path}")
            return config
        except Exception as e:
            print(f"⚠️  Erro ao carregar configuração: {e}")
            return self._load_default_calibration()

    def _load_default_calibration(self) -> Dict:
        """Carrega configuração padrão de calibração"""
        return {
            'state_thresholds': {
                'emergence': {'min_fci': 0.644},
                'meditation': {'min_fci': 0.636},
                'analysis': {'min_fci': 0.620}
            }
        }

    def _initialize_semantic_categories(self) -> Dict[str, SemanticCategory]:
        """Inicializa categorias semânticas baseadas na calibração"""
        thresholds = self.calibration_config['state_thresholds']

        return {
            'creative': SemanticCategory(
                name='creative',
                target_fci=thresholds['emergence']['min_fci'],
                alpha_modulation=1.2,  # α mais alto para criatividade
                description='Estados criativos e emergentes'
            ),
            'analytical': SemanticCategory(
                name='analytical',
                target_fci=thresholds['analysis']['min_fci'],
                alpha_modulation=0.8,  # α mais baixo para análise
                description='Estados analíticos e focados'
            ),
            'meditative': SemanticCategory(
                name='meditative',
                target_fci=thresholds['meditation']['min_fci'],
                alpha_modulation=1.0,  # α neutro para meditação
                description='Estados meditativos e introspectivos'
            ),
            'neutral': SemanticCategory(
                name='neutral',
                target_fci=0.63,  # Valor intermediário
                alpha_modulation=1.0,
                description='Estados neutros e balanceados'
            )
        }


def update_config_with_params(base_configs: Dict, param_values: Dict) -> Dict:
    """
    Atualiza configs com novos valores de parâmetros.
    Preserva estrutura original.
    """
    updated_configs = copy.deepcopy(base_configs)

    for param_name, value in param_values.items():
        # Parse: config_name.path.to.param
        parts = param_name.split('.')
        config_name = parts[0]
        param_path = parts[1:]

        if config_name in updated_configs:
            # Navegar até o parâmetro
            current = updated_configs[config_name]
            for key in param_path[:-1]:
                current = current[key]

            # Atualizar valor
            current[param_path[-1]] = float(value)

    return updated_configs


def run_spectral_pipeline(
    stimulus_text: str,
    configs: Dict,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, str]:
    """
    Executa pipeline REAL de processamento espectral.
    SEM MOCKS - apenas processamento espectral real.
    """
    # 1. NumericSignalProcessor: Texto → Espectro
    processor = NumericSignalProcessor(device=device)

    # Converter texto para array numérico via ASCII encoding
    char_values = [ord(c) / 127.0 for c in stimulus_text]
    numeric_array = np.array(char_values, dtype=np.float32)

    # Processar numericamente
    signal_result = processor.process_text(str(char_values))

    # 2. Criar representação quaterniônica do sinal
    # Expandir para dimensão quaterniônica (4 componentes)
    signal_tensor = torch.tensor(numeric_array, device=device)

    # Pad para múltiplo de 4
    pad_size = (4 - len(signal_tensor) % 4) % 4
    if pad_size > 0:
        signal_tensor = torch.cat([
            signal_tensor,
            torch.zeros(pad_size, device=device)
        ])

    # Reshape para quaternions
    seq_len = len(signal_tensor) // 4
    signal_quat = signal_tensor.view(1, seq_len, 4)  # [batch=1, seq_len, 4]

    # Expandir para embed_dim * 4
    embed_dim = 64  # Da config
    signal_expanded = torch.nn.functional.interpolate(
        signal_quat.permute(0, 2, 1),  # [1, 4, seq_len]
        size=embed_dim,
        mode='linear',
        align_corners=False
    ).permute(0, 2, 1)  # [1, embed_dim, 4]

    signal_expanded = signal_expanded.view(1, embed_dim, 4)
    signal_expanded = signal_expanded.reshape(1, -1, 4 * 64)  # [1, seq_len, 256]

    # 3. ConsciousWorkingMemory com Equação de Padilha
    # Criar config temporário
    config_path_cwm = Path(__file__).parent.parent / "configs" / "working_memory_config.yaml"
    memory = ConsciousWorkingMemory(config_path=str(config_path_cwm))
    memory.to(device)

    # Estado de consciência (valores espectrais extraídos do sinal)
    spectrum = torch.fft.fft(signal_tensor)
    spectral_energy = torch.abs(spectrum).sum().item()
    spectral_entropy = -torch.sum(
        torch.abs(spectrum) * torch.log(torch.abs(spectrum) + 1e-10)
    ).item()

    consciousness_state = {
        'entropy': min(max(spectral_entropy / 10.0, 0.0), 1.0),
        'fractal_dimension': 2.0 + 0.5 * (spectral_energy / 100.0),
        'fci': min(max(1.0 - spectral_entropy / 20.0, 0.0), 1.0)
    }

    # Processar através da memória
    memory_output, _ = memory(signal_expanded, consciousness_state)

    # 4. KuramotoSpectralLayer - Sincronização espectral
    config_path_kuramoto = Path(__file__).parent.parent / "configs" / "kuramoto_config.yaml"
    kuramoto = KuramotoSpectralLayer(config_path=str(config_path_kuramoto))
    kuramoto.to(device)

    kuramoto_output, _ = kuramoto(memory_output)

    # 5. NegentropyTransformerBlock
    transformer_block = NegentropyTransformerBlock(
        d_model=256,  # 4 * 64
        nhead=8,
        dim_feedforward=1024,
        dropout=0.1,
        qrh_embed_dim=64
    )
    transformer_block.to(device)

    final_output = transformer_block(kuramoto_output)

    # 6. Reconstrução via IFFT
    # Extrair espectro da saída
    output_flat = final_output.view(-1)

    # Transformada inversa para reconstruir sinal
    reconstructed_spectrum = output_flat[:len(spectrum)]
    reconstructed_signal = torch.fft.ifft(reconstructed_spectrum).real

    # Converter de volta para texto
    reconstructed_chars = []
    for val in reconstructed_signal:
        char_code = int(torch.clamp(val * 127.0, 0, 127).item())
        try:
            reconstructed_chars.append(chr(char_code))
        except ValueError:
            reconstructed_chars.append('?')

    reconstructed_text = ''.join(reconstructed_chars[:len(stimulus_text)])

    return final_output, reconstructed_text


def compute_spectral_similarity(text1: str, text2: str, device: str = 'cpu') -> float:
    """
    Calcula similaridade espectral entre dois textos.
    Métrica REAL baseada em FFT, sem hardcoding.
    """
    # Converter para espectros
    def text_to_spectrum(text):
        char_vals = torch.tensor([ord(c) / 127.0 for c in text], device=device)
        return torch.fft.fft(char_vals)

    spec1 = text_to_spectrum(text1)
    spec2_full = text_to_spectrum(text2)

    # Alinhar comprimentos
    min_len = min(len(spec1), len(spec2_full))
    spec2 = spec2_full[:min_len]
    spec1 = spec1[:min_len]

    # Similaridade espectral = correlação de magnitude
    mag1 = torch.abs(spec1)
    mag2 = torch.abs(spec2)

    # Normalizar
    mag1_norm = mag1 / (torch.norm(mag1) + 1e-10)
    mag2_norm = mag2 / (torch.norm(mag2) + 1e-10)

    # Produto escalar
    similarity = torch.dot(mag1_norm, mag2_norm).item()

    # Mapear para [0, 1]
    similarity = (similarity + 1.0) / 2.0

    return similarity


def autonomous_calibration(
    config_paths: List[Path],
    stimulus: str,
    output_dir: Path,
    device: str = 'cpu'
):
    """
    Calibração autônoma usando apenas processamento espectral real.
    """
    # Extrair espaço de parâmetros
    search_space, base_configs = extract_tunable_parameters(config_paths)

    if not search_space:
        print("⚠️  Nenhum parâmetro ajustável encontrado!")
        return

    param_names = list(search_space.keys())
    param_combinations = list(product(*[search_space[name] for name in param_names]))

    print(f"\n🔬 Iniciando calibração autônoma...")
    print(f"  • Combinações a testar: {len(param_combinations)}")
    print(f"  • Estímulo: '{stimulus}'")
    print(f"  • Device: {device}")

    best_fitness = -1.0
    best_params = None
    best_reconstruction = ""

    for i, values in enumerate(param_combinations):
        current_params = dict(zip(param_names, values))

        # Atualizar configs
        updated_configs = update_config_with_params(base_configs, current_params)

        # Salvar configs temporários
        temp_config_dir = Path("/tmp/psiqrh_calibration")
        temp_config_dir.mkdir(exist_ok=True)

        for config_name, config_data in updated_configs.items():
            temp_path = temp_config_dir / f"{config_name}.yaml"
            with open(temp_path, 'w') as f:
                yaml.dump(config_data, f)

        # Executar pipeline real
        try:
            output_tensor, reconstructed = run_spectral_pipeline(
                stimulus, updated_configs, device
            )

            # Fitness = similaridade espectral
            fitness = compute_spectral_similarity(stimulus, reconstructed, device)

        except Exception as e:
            print(f"\n⚠️  Erro na combinação {i+1}: {e}")
            fitness = 0.0
            reconstructed = ""

        # Progress
        progress = "█" * int((i + 1) / len(param_combinations) * 50)
        print(f"  [{progress:<50}] {i+1}/{len(param_combinations)} | Fitness: {fitness:.4f}", end='\r')

        if fitness > best_fitness:
            best_fitness = fitness
            best_params = current_params
            best_reconstruction = reconstructed

            if fitness > 0.99:
                print("\n✨ Calibração perfeita encontrada!")
                break

    print("\n\n" + "="*70)
    print(f"✅ Calibração Autônoma Completa!")
    print(f"  • Fitness máximo: {best_fitness:.4f}")
    print(f"  • Estímulo original: '{stimulus}'")
    print(f"  • Eco reconstruído: '{best_reconstruction}'")
    print(f"  • Parâmetros ótimos:")
    for name, value in best_params.items():
        print(f"    └─ {name}: {value:.6f}")
    print("="*70)

    # Salvar configurações calibradas
    save_calibrated_configs(base_configs, best_params, output_dir)

    print(f"\n💾 Configs salvas em: {output_dir}")


def save_calibrated_configs(
    base_configs: Dict,
    optimal_params: Dict,
    output_dir: Path
):
    """
    Salva configurações calibradas preservando estrutura YAML original.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Atualizar com parâmetros ótimos
    calibrated_configs = update_config_with_params(base_configs, optimal_params)

    # Salvar cada config
    for config_name, config_data in calibrated_configs.items():
        output_path = output_dir / f"{config_name}_calibrated.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ {output_path}")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    # Configs para calibrar
    config_files = [
        repo_root / "configs" / "kuramoto_config.yaml",
        repo_root / "configs" / "working_memory_config.yaml",
    ]

    # Estímulo de teste
    stimulus = "Hello World"

    # Diretório de saída
    output_dir = repo_root / "configs" / "calibrated"

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Executar calibração
    autonomous_calibration(config_files, stimulus, output_dir, device)


if __name__ == "__main__":
    main()
