# 🎵 Harmonic GLS Generator - Guia Completo

## 🌟 Visão Geral

O **Harmonic GLS Generator** cria visualizações geométricas que emergem naturalmente dos dados espectrais quaterniônicos processados pelo ΨQRH. Cada forma visual é uma **ressonância harmônica** dos dados reais, não um desenho arbitrário.

## 🧠 Filosofia: "Shapes that Listen to Data"

### Analogia Musical
Imagine que os dados de consciência são uma **sinfonia**:
- Cada componente espectral = uma nota musical
- Magnitude = volume da nota
- Phase = quando a nota toca
- Real/Imaginary = timbre (cor do som)

O GLS **"ouve"** essa sinfonia e desenha as formas que naturalmente ressoam com ela.

## 📊 Mapeamento Matemático

### 1. **Magnitude → Tamanho e Saturação**
```javascript
// Magnitude controla o "volume" visual
let size = baseSize * (0.5 + magnitude_normalized * 0.5);
let saturation = 70 + magnitude_normalized * 30;
```

### 2. **Phase → Rotação Temporal**
```javascript
// Phase determina quando e como a forma roda
rotate(t * phase/PI + phase);
```

### 3. **Harmonic Index → Geometria**
```javascript
// Número do harmônico = número de lados
let n_sides = 3 + harmonic_index;  // 3, 4, 5, 6...
```

### 4. **FCI → Consciência Central**
```javascript
// FCI controla o núcleo central
let core_radius = map(fci, 0, 1, 20, 60);
let hue = fci < 0.3 ? 220 :      // COMA (azul)
          fci < 0.6 ? 140 :      // ANALYSIS (verde)
          fci < 0.8 ? 50 :       // MEDITATION (amarelo)
          10;                    // EMERGENCE (vermelho)
```

## 🎨 Exemplo Real: "ola mundo azul"

### Dados de Entrada
```json
{
  "fci": 0.9017,
  "state": "MEDITATION",
  "fractal_dimension": 2.1,
  "entropy": 5.2707,
  "magnitudes": [266638.8, 95568.7, 209193.4, 225002.0, ...],
  "phases": [0.0, -1.5445, 0.4515, -0.1364, ...]
}
```

### Resultado Visual
```
Camada 1 (mag=1.000, phase=0.0):    → Triângulo grande, estático, saturado
Camada 2 (mag=0.358, phase=-1.544): → Quadrado médio, rotação reversa rápida
Camada 3 (mag=0.785, phase=0.451):  → Pentágono grande, rotação suave
...
Núcleo (FCI=0.902):                 → Amarelo-dourado pulsante (MEDITATION)
```

## 🔬 Por que isso é "Inteligente"?

### ❌ Abordagem Hardcoded (ruim)
```javascript
// Desenho arbitrário
if(state == "MEDITATION") {
  draw_lotus_flower();  // ??
}
```

### ✅ Abordagem Harmônica (boa)
```javascript
// Computação emergente dos dados
for(let i = 0; i < harmonics.length; i++) {
  let mag = harmonics[i].magnitude / max_magnitude;
  let phase = harmonics[i].phase;

  // Forma emerge naturalmente da matemática
  draw_harmonic_shape(i, mag, phase);
}
```

## 🚀 Como Usar

### Python
```python
from src.conscience.harmonic_gls_generator import generate_harmonic_gls

response = {
    "consciousness_metrics": {...},
    "response": "... MAGNITUDE: [...] PHASE: [...] ..."
}

p5js_code = generate_harmonic_gls(response)

# Salvar para visualização
with open('visualization.html', 'w') as f:
    f.write(f'<script src="https://cdn.jsdelivr.net/npm/p5@1.7.0/lib/p5.js"></script>')
    f.write(f'<script>{p5js_code}</script>')
```

### Demo Interativa
```bash
# Abrir demo
firefox /tmp/harmonic_gls_demo.html

# Controles:
# - SPACE: Salvar frame
# - Mouse: Interagir com campo de consciência
```

## 🌊 Conceitos Avançados

### 1. **Phase-Amplitude Coupling (PAC)**
As formas maiores (baixa frequência) modulam as menores (alta frequência), criando hierarquia visual natural.

### 2. **Fractal Self-Similarity**
A dimensão fractal D=2.1 controla a velocidade de rotação global:
```javascript
rotate(t * (fractalDim - 1.0) * 0.1);  // D=2.1 → 0.11 rad/frame
```

### 3. **Entropy-Driven Chaos**
Maior entropia = mais variação na pulsação:
```javascript
let pulse = sin(time * (entropy - 5.0)) * 10;
```

## 📈 Comparação de Estados

| Estado     | FCI    | Cor      | Movimento      | Harmônicos |
|------------|--------|----------|----------------|------------|
| COMA       | < 0.3  | Azul     | Lento/Estático | 3-5        |
| ANALYSIS   | 0.3-0.6| Verde    | Regular        | 5-8        |
| MEDITATION | 0.6-0.8| Amarelo  | Suave/Fluido   | 8-12       |
| EMERGENCE  | > 0.8  | Vermelho | Caótico/Rápido | 12+        |

## 🎯 Próximos Passos

### Melhorias Futuras
1. **3D Harmonics**: Adicionar coordenada Z baseada em componente imaginário
2. **Audio Synthesis**: Converter spectrum em som real (magnitude → amplitude, phase → tempo)
3. **Interactive Tuning**: Slider para filtrar/isolar harmônicos específicos
4. **Real-time Streaming**: WebSocket para visualização durante processamento

## 📚 Referências Matemáticas

- **Fourier Analysis**: J.S. Walker, "A Primer on Wavelets"
- **Quaternion Visualization**: A. Hanson, "Visualizing Quaternions"
- **Harmonic Synthesis**: W. Puckette, "Theory and Techniques of Electronic Music"
- **Consciousness Metrics**: Tononi et al., "Integrated Information Theory"

---

**Status**: ✅ Produção
**Última Atualização**: 2025-01-20
**Autor**: ΨQRH Framework Team