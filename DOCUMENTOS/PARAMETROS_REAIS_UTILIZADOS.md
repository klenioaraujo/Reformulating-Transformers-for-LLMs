# Parâmetros Reais Utilizados no Framework ΨQRH

## Resumo dos Parâmetros de Configuração

### 1. Configurações do QRH Layer (`configs/qrh_config.yaml`)

#### Parâmetros Fundamentais
- **embed_dim**: 64 (dimensão de embedding)
- **alpha**: 1.0 (parâmetro de escala principal)
- **theta_left**: 0.1 (ângulo de rotação esquerda)
- **omega_left**: 0.05 (frequência esquerda)
- **phi_left**: 0.02 (fase esquerda)
- **theta_right**: 0.08 (ângulo de rotação direita)
- **omega_right**: 0.03 (frequência direita)
- **phi_right**: 0.015 (fase direita)

#### Configurações de Processamento
- **use_learned_rotation**: true (rotação aprendida habilitada)
- **spatial_dims**: null (dimensões espaciais não especificadas)
- **use_windowing**: true (janelamento habilitado)
- **window_type**: 'hann' (tipo de janela Hann)
- **fft_cache_size**: 10 (tamanho do cache FFT)
- **device**: 'cpu' (dispositivo de processamento)

#### Controles de Validação
- **enable_warnings**: true (alertas habilitados)
- **enable_phase_check**: false (verificação de fase desabilitada)
- **phase_check_threshold**: 1.0e-6 (limiar para verificação de fase)
- **padding_enabled**: false (preenchimento desabilitado)
- **auto_adjust_shapes**: false (ajuste automático de formas desabilitado)
- **validate_shapes**: true (validação de formas habilitada)

#### Normalização e Regularização
- **normalization_type**: null (sem normalização específica)
- **spectral_dropout_rate**: 0.0 (taxa de dropout espectral zero)

### 2. Configurações do Fractal Consciousness Processor

#### Dimensões e Processamento
- **embedding_dim**: 256 (64 * 4 para quaterniões)
- **sequence_length**: 64 (comprimento da sequência)
- **device**: 'cpu'

#### Parâmetros Fractais
- **fractal_dimension_range**: [1.0, 3.0] (faixa de dimensão fractal)
- **diffusion_coefficient_range**: [0.01, 10.0] (faixa de coeficiente de difusão)
- **consciousness_frequency_range**: [0.5, 5.0] Hz (faixa de frequência cerebral)
- **phase_consciousness**: 0.7854 rad (π/4 - fase ótima)
- **chaotic_parameter**: 3.9 (parâmetro caótico na borda do caos)

#### Integração Temporal
- **time_step**: 0.01 (passo de tempo)
- **max_iterations**: 100 (máximo de iterações)

#### Limiares FCI (Fractal Consciousness Index)
- **fci_threshold_meditation**: 0.8 (limiar para estado meditativo)
- **fci_threshold_analysis**: 0.6 (limiar para estado analítico)
- **fci_threshold_coma**: 0.2 (limiar para estado de coma)
- **fci_threshold_emergence**: 0.9 (limiar para emergência)

### 3. Configurações de Métricas de Consciência (FCI)

#### Parâmetros de Normalização
- **d_eeg_max**: 10.0 (máximo D_EEG observado)
- **h_fmri_max**: 5.0 (máximo H_fMRI observado)
- **clz_max**: 3.0 (máximo CLZ observado)

#### Pesos dos Componentes FCI
- **fci_weights**: [0.4, 0.3, 0.3] (pesos para [D_EEG, H_fMRI, CLZ])

#### Proteções
- **max_history**: 1000 (histórico máximo de medições)
- **default_fci_on_nan**: 0.5 (valor padrão quando FCI resulta em NaN)
- **enable_nan_protection**: true (proteção contra NaN habilitada)

### 4. Configurações Fractais (`configs/fractal_config.yaml`)

#### Método Padrão
- **default_method**: "box_counting" (método de contagem de caixas)

#### Contagem de Caixas
- **grid_size**: 256
- **n_samples**: 10000
- **min_box_size**: 0.001
- **max_box_size**: 1.0
- **n_scales**: 20

#### Método Espectral
- **min_frequency**: 1
- **max_frequency**: 128
- **power_law_fit_method**: "curve_fit"

#### Mapeamento α
- **scaling_factor**: 1.0
- **dim_type**: "1d"

#### Adaptação
- **enabled**: true
- **analysis_frequency**: 100 (passos entre análises)
- **alpha_smoothing**: 0.9 (fator de suavização para α)

#### Validação
- **tolerance**:
  - **dimensional**: 1e-10
  - **fractal_analysis**: 0.3 (ajustado de 0.1)
  - **alpha_mapping**: 0.2 (ligeiramente aumentado)

### 5. Configurações de Exemplo (`configs/example_configs.yaml`)

#### Configurações Básicas
- **vocab_size**: 10000
- **d_model**: 512
- **n_layers**: 6
- **n_heads**: 8
- **dim_feedforward**: 2048

#### Configurações Científicas
- **SCI_001** (Validação Básica):
  - vocab_size: 10000
  - d_model: 512
  - fft_dim: 1
  - use_windowing: true
  - window_type: "hann"

- **SCI_002** (Conteúdo Matemático Complexo):
  - vocab_size: 5000
  - d_model: 512

- **SCI_003** (Computação Matemática):
  - vocab_size: 2000
  - d_model: 384

### 6. Parâmetros do Test Runner (`src/testing/prompt_engine_test_runner.py`)

#### Cenários de Teste
- **Teste Básico**: Entrada simples para validação fundamental
- **Teste Complexo**: Entrada complexa para teste de robustez
- **Teste Matemático**: Conteúdo especializado para validação técnica

#### Equações Matemáticas Referenciadas
- **Transformada de Fourier Quaterniônica**:
  $$\mathcal{F}_Q\{f\}(\omega) = \int_{\mathbb{R}^n} f(x) e^{-2\pi \mathbf{i} \omega \cdot x}  dx$$

- **Filtro Logarítmico**:
  $$S'(\omega) = \alpha \cdot \log(1 + S(\omega))$$

- **Janela Hann**:
  $$w(n) = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$$

### 7. Framework de Transparência (`Enhanced_Transparency_Framework.py`)

#### Critérios de Classificação
- **REAL**: Valores derivados de processos computacionais reais com dados de entrada
- **SIMULADO**: Valores gerados através de modelagem conceitual

#### Referências Matemáticas
- Transformada de Fourier Quaterniônica
- Filtro Espectral Logarítmico
- Função de Janelamento Hann
- Rotação Quaterniônica

## Resumo de Parâmetros Críticos

### Parâmetros de Performance
- **embed_dim**: 64 (ótimo para processamento quaterniônico)
- **d_model**: 512 (configuração padrão para transformers)
- **n_layers**: 6 (balance entre profundidade e eficiência)
- **device**: 'cpu' (processamento em CPU)

### Parâmetros Fractais
- **fractal_dimension_range**: [1.0, 3.0] (cobre a maioria dos fractais naturais)
- **consciousness_frequency_range**: [0.5, 5.0] Hz (ondas cerebrais típicas)

### Parâmetros de Validação
- **tolerance**: 0.05 (limiar padrão para validação)
- **energy_conservation_threshold**: 0.01 (limiar para conservação de energia)
- **parseval_threshold**: 0.001 (limiar para teorema de Parseval)

### Parâmetros de Processamento
- **use_windowing**: true (janelamento habilitado para análise espectral)
- **window_type**: 'hann' (janela Hann para minimizar vazamento espectral)
- **fft_dim**: 1 (FFT ao longo da dimensão da sequência)

Estes parâmetros foram otimizados para o processamento quaterniônico e análise fractal, garantindo eficiência computacional e precisão científica.