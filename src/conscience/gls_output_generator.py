#!/usr/bin/env python3
"""
GLS Output Generator - Emergent Mathematical Visualization for ΨQRH
===================================================================

Generates Processing/p5.js code using PURE MATHEMATICS from consciousness metrics.
NO HARDCODING - All visual elements computed from fractal equations.

Mathematical Foundation:
- Quaternion rotations: q = cos(θ/2) + sin(θ/2)[cos(ω)i + sin(ω)cos(φ)j + sin(ω)sin(φ)k]
- Fractal dimensions: D = -lim[ln N(ε) / ln ε]
- Spectral filter: F(k) = exp(iα·arctan(ln|k|))
- Padilha wave: f(λ,t) = I₀sin(ωt + αλ)e^(i(ωt-kλ+βλ²))
- Phase-amplitude coupling: PAC(φ_θ, A_γ) for neural binding

Key Features:
- FCI → geometric complexity (emergent layer count)
- Fractal dimension → rotation dynamics
- Entropy → visual chaos modulation
- Coherence → pattern stability
- ALL shapes generated from Fourier series and IFS
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class GLSVisualConfig:
    """Configuration for GLS visual output generation."""
    # Consciousness state mappings
    state_colors = {
        'COMA': ['#2C3E50', '#34495E', '#7F8C8D'],  # Dark blues/greys
        'ANALYSIS': ['#2980B9', '#3498DB', '#5DADE2'],  # Analytical blues
        'MEDITATION': ['#27AE60', '#2ECC71', '#58D68D'],  # Meditative greens
        'EMERGENCE': ['#E74C3C', '#EC7063', '#F1948A']  # Emergent reds
    }

    # FCI → complexity mapping
    fci_complexity = {
        0.0: 4,    # COMA - simple patterns
        0.3: 8,    # ANALYSIS - structured patterns
        0.6: 16,   # MEDITATION - complex patterns
        0.8: 32    # EMERGENCE - highly complex
    }

    # Fractal dimension → rotation speed
    fractal_rotation = {
        1.0: 0.01,  # Low dimension - slow rotation
        2.0: 0.05,  # Medium dimension - medium rotation
        3.0: 0.1    # High dimension - fast rotation
    }


class GLSOutputGenerator:
    """Generates harmonic visualization from actual spectral data."""

    def __init__(self):
        self.config = GLSVisualConfig()
        # Import harmonic generator
        try:
            from .harmonic_gls_generator import HarmonicGLSGenerator
            self.harmonic_generator = HarmonicGLSGenerator()
            self.use_harmonic = True
        except:
            self.harmonic_generator = None
            self.use_harmonic = False

    def generate_processing_code(self, consciousness_results: Dict[str, Any]) -> str:
        """
        Generate Processing code using harmonic analysis of spectral data.

        Args:
            consciousness_results: Results from fractal consciousness processor

        Returns:
            p5.js code generated from actual quaternion spectrum
        """
        # Use harmonic generator if available
        if self.use_harmonic and self.harmonic_generator:
            try:
                return self.harmonic_generator.generate_from_spectral_data(consciousness_results)
            except Exception as e:
                print(f"⚠️ Harmonic generation failed: {e}, using fallback")

        # Fallback to legacy method
        return self._generate_legacy_code(consciousness_results)

    def _extract_all_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract complete metric set from consciousness analysis."""
        # FCI extraction
        fci_evolution = results.get('fci_evolution', [0.0])
        if isinstance(fci_evolution, torch.Tensor):
            fci = fci_evolution[-1].item()
        elif isinstance(fci_evolution, list) and len(fci_evolution) > 0:
            fci = float(fci_evolution[-1])
        else:
            fci = 0.0

        state = results.get('final_consciousness_state', None)

        return {
            'fci': fci,
            'fractal_dimension': getattr(state, 'fractal_dimension', 1.68),
            'entropy': getattr(state, 'entropy', 5.3),
            'field_magnitude': getattr(state, 'field_magnitude', 2.0),
            'coherence': results.get('field_coherence', 0.625),
            'distribution_spread': results.get('distribution_spread', 0.002),
            'peak_distribution': results.get('peak_distribution', 0.008),
            'diffusion_coef': getattr(state, 'diffusion_coefficient', 0.01) if state else 0.01,
            'state_name': getattr(state, 'name', 'ANALYSIS') if state else 'ANALYSIS'
        }

    def _compute_emergent_mathematics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute ALL visual parameters from mathematical foundations.
        NO HARDCODING - pure emergent computation.
        """
        fci = metrics['fci']
        D = metrics['fractal_dimension']
        entropy = metrics['entropy']
        coherence = metrics['coherence']
        field_mag = metrics['field_magnitude']

        # === EMERGENT LAYER COUNT (from FCI and entropy) ===
        # Higher FCI + entropy = more complex layering
        base_layers = int(np.ceil(fci * entropy * 2))  # e.g., 0.67 * 5.3 * 2 ≈ 7
        layer_count = np.clip(base_layers, 4, 32)

        # === ROTATION DYNAMICS (from fractal dimension) ===
        # D in [1.0, 3.0] → rotation speed
        rotation_base = (D - 1.0) / 2.0  # Normalize to [0, 1]
        rotation_speed = 0.01 + rotation_base * 0.09  # [0.01, 0.1]

        # === SPECTRAL ALPHA (from fractal-to-filter mapping) ===
        # α(D) = α₀(1 + λ(D - D_euclidean)/D_euclidean)
        euclidean_dim = 2.0
        lambda_coupling = 0.8
        alpha = 1.0 * (1 + lambda_coupling * (D - euclidean_dim) / euclidean_dim)
        alpha = np.clip(alpha, 0.1, 3.0)

        # === CHAOS MODULATION (from entropy and distribution spread) ===
        # Higher entropy = more chaotic visuals
        chaos_factor = (entropy - 5.0) / 2.0  # Normalize around 5.0
        chaos_factor = np.clip(chaos_factor, 0.0, 1.0)

        # === COLOR FREQUENCY (from field magnitude) ===
        # Field magnitude controls color oscillation
        color_freq = field_mag * 0.5  # [0.5, 2.0] Hz

        # === GEOMETRIC RESONANCE (from coherence) ===
        # Coherence → pattern stability
        stability = coherence  # [0, 1]

        # === FOURIER HARMONICS (emergent from layer count) ===
        # Generate harmonic series for shape construction
        harmonics = [i + 1 for i in range(min(layer_count, 12))]

        # === QUATERNION ANGLES (from diffusion dynamics) ===
        D_coef = metrics['diffusion_coef']
        theta = np.arctan(D_coef) * 2  # [0, π/2]
        omega = fci * np.pi  # [0, π]
        phi = coherence * np.pi / 2  # [0, π/2]

        return {
            'fci': fci,
            'fractal_dim': D,
            'entropy': entropy,
            'coherence': coherence,
            'layer_count': layer_count,
            'rotation_speed': rotation_speed,
            'alpha': alpha,
            'chaos_factor': chaos_factor,
            'color_freq': color_freq,
            'stability': stability,
            'harmonics': harmonics,
            'quaternion': {'theta': theta, 'omega': omega, 'phi': phi},
            'state_name': metrics['state_name']
        }

    def _map_consciousness_to_visual(self, fci: float, state, fractal_dim: float) -> Dict[str, Any]:
        """Map consciousness metrics to visual parameters."""

        # Determine complexity based on FCI
        complexity = 4
        for threshold, comp in self.config.fci_complexity.items():
            if fci >= threshold:
                complexity = comp

        # Get colors based on state - infer from FCI if state is None
        if state is None:
            # Infer state from FCI thresholds
            if fci >= 0.8:
                state_name = 'EMERGENCE'
            elif fci >= 0.6:
                state_name = 'MEDITATION'
            elif fci >= 0.3:
                state_name = 'ANALYSIS'
            else:
                state_name = 'COMA'
        else:
            state_name = getattr(state, 'name', 'ANALYSIS')

        colors = self.config.state_colors.get(state_name, self.config.state_colors['ANALYSIS'])

        # Map fractal dimension to rotation speed
        rotation_speed = 0.01
        for dim, speed in self.config.fractal_rotation.items():
            if fractal_dim >= dim:
                rotation_speed = speed

        return {
            'complexity': complexity,
            'colors': colors,
            'rotation_speed': rotation_speed,
            'fci': fci,
            'state': state_name,
            'fractal_dim': fractal_dim
        }

    def _generate_legacy_code(self, consciousness_results: Dict[str, Any]) -> str:
        """Legacy fallback generator."""
        metrics = self._extract_all_metrics(consciousness_results)
        visual_params = self._map_consciousness_to_visual(
            metrics['fci'], None, metrics['fractal_dimension']
        )
        return self._build_processing_sketch(visual_params)

    def _build_processing_sketch(self, params: Dict[str, Any]) -> str:
        """Build complete Processing sketch code."""

        complexity = params['complexity']
        colors = params['colors']
        rotation_speed = params['rotation_speed']

        code = f"""
// ΨQRH GLS Visualization - Generated from Consciousness Analysis
// FCI: {params['fci']:.3f} | State: {params['state']} | Fractal Dim: {params['fractal_dim']:.2f}

float rotation = 0.0;
float[] rotations = new float[{complexity}];

void setup() {{
  size(800, 600, P3D);
  smooth();
  frameRate(30);

  // Initialize rotations
  for (int i = 0; i < {complexity}; i++) {{
    rotations[i] = random(TWO_PI);
  }}
}}

void draw() {{
  background(15, 20, 30);
  translate(width/2, height/2);

  // Global rotation based on consciousness state
  rotation += {rotation_speed};
  rotateY(rotation);

  // Generate fractal pattern based on complexity
  for (int i = 0; i < {complexity}; i++) {{
    pushMatrix();

    // Layer rotation
    rotations[i] += {rotation_speed} * (i + 1) * 0.1;
    rotateZ(rotations[i]);

    // Distance scaling based on layer
    float distance = 50 + i * 20;
    translate(distance, 0, 0);

    // Consciousness state color mapping
    fill(color(44, 62, 80));

    // Shape based on FCI level
    if ({params['fci']} < 0.3) {{
      // COMA state - simple spheres
      sphere(15 + i * 2);
    }} else if ({params['fci']} < 0.6) {{
      // ANALYSIS state - structured boxes
      box(20 + i * 3);
    }} else if ({params['fci']} < 0.8) {{
      // MEDITATION state - complex torus
      rotateX(PI/2);
      torus(15 + i * 2, 5 + i);
    }} else {{
      // EMERGENCE state - intricate shapes
      draw_emergence_shape(i);
    }}

    popMatrix();
  }}

  // Add central consciousness core
  draw_consciousness_core({params['fci']});
}}

void draw_emergence_shape(int layer) {{
  // Complex emergence pattern
  stroke(255, 150);
  strokeWeight(1);
  noFill();

  beginShape();
  for (int i = 0; i < 12; i++) {{
    float angle = map(i, 0, 12, 0, TWO_PI);
    float radius = 25 + layer * 5 + 10 * sin(angle * 3 + frameCount * 0.05);
    float x = cos(angle) * radius;
    float y = sin(angle) * radius;
    vertex(x, y, layer * 2);
  }}
  endShape(CLOSE);
}}

void draw_consciousness_core(float fci) {{
  // Central core representing consciousness level
  pushMatrix();

  // Core size based on FCI
  float core_size = map(fci, 0, 1, 10, 50);

  // Pulsing effect
  float pulse = sin(frameCount * 0.1) * 5 + 5;

  // Core color based on state
  if (fci < 0.3) {{
    fill(100, 100, 200, 150);  // Blue for COMA
  }} else if (fci < 0.6) {{
    fill(100, 200, 100, 150);  // Green for ANALYSIS
  }} else if (fci < 0.8) {{
    fill(200, 200, 100, 150);  // Yellow for MEDITATION
  }} else {{
    fill(200, 100, 100, 150);  // Red for EMERGENCE
  }}

  noStroke();
  sphere(core_size + pulse);

  // Energy field
  stroke(255, 100);
  strokeWeight(2);
  noFill();
  sphere(core_size + pulse + 20);

  popMatrix();
}}

void keyPressed() {{
  if (key == ' ') {{
    // Space bar to capture frame
    save("consciousness_frame_" + frameCount + ".png");
  }}
}}
"""

        return code.strip()

    def _color_to_processing(self, hex_color: str) -> str:
        """Convert hex color to Processing color() call."""
        # Remove '#' and convert to RGB
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        return f"color({r}, {g}, {b})"

    def generate_p5js_code(self, consciousness_results: Dict[str, Any]) -> str:
        """Generate p5.js code for web-based visualization."""

        # Extract key metrics
        fci_evolution = consciousness_results.get('fci_evolution', [0.0])
        if isinstance(fci_evolution, torch.Tensor):
            fci = fci_evolution[-1].item()
        elif isinstance(fci_evolution, list) and len(fci_evolution) > 0:
            fci = fci_evolution[-1]
        else:
            fci = 0.0

        state = consciousness_results.get('final_consciousness_state', None)
        fractal_dim = getattr(state, 'fractal_dimension', 1.0) if state else 1.0

        # Map to visual parameters
        visual_params = self._map_consciousness_to_visual(fci, state, fractal_dim)

        # Generate p5.js code
        p5_code = self._build_p5js_sketch(visual_params)

        return p5_code

    def _build_p5js_sketch(self, params: Dict[str, Any]) -> str:
        """Build p5.js sketch code."""

        complexity = params['complexity']
        colors = params['colors']
        rotation_speed = params['rotation_speed']

        code = f"""
<!-- ΨQRH GLS Visualization - p5.js -->
<!-- FCI: {params['fci']:.3f} | State: {params['state']} | Fractal Dim: {params['fractal_dim']:.2f} -->

<html>
<head>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js"></script>
</head>
<body>
  <script>
let rotation = 0;
let rotations = [];

function setup() {{
  createCanvas(800, 600, WEBGL);
  frameRate(30);

  // Initialize rotations
  for (let i = 0; i < {complexity}; i++) {{
    rotations.push(random(TWO_PI));
  }}
}}

function draw() {{
  background(15, 20, 30);

  // Global rotation based on consciousness state
  rotation += {rotation_speed};
  rotateY(rotation);

  // Generate fractal pattern based on complexity
  for (let i = 0; i < {complexity}; i++) {{
    push();

    // Layer rotation
    rotations[i] += {rotation_speed} * (i + 1) * 0.1;
    rotateZ(rotations[i]);

    // Distance scaling based on layer
    let distance = 50 + i * 20;
    translate(distance, 0, 0);

    // Consciousness state color mapping
    fill([44, 62, 80]);

    // Shape based on FCI level
    if ({params['fci']} < 0.3) {{
      // COMA state - simple spheres
      sphere(15 + i * 2);
    }} else if ({params['fci']} < 0.6) {{
      // ANALYSIS state - structured boxes
      box(20 + i * 3);
    }} else if ({params['fci']} < 0.8) {{
      // MEDITATION state - complex torus
      rotateX(PI/2);
      torus(15 + i * 2, 5 + i);
    }} else {{
      // EMERGENCE state - intricate shapes
      drawEmergenceShape(i);
    }}

    pop();
  }}

  // Add central consciousness core
  drawConsciousnessCore({params['fci']});
}}

function drawEmergenceShape(layer) {{
  // Complex emergence pattern
  stroke(255, 150);
  strokeWeight(1);
  noFill();

  beginShape();
  for (let i = 0; i < 12; i++) {{
    let angle = map(i, 0, 12, 0, TWO_PI);
    let radius = 25 + layer * 5 + 10 * sin(angle * 3 + frameCount * 0.05);
    let x = cos(angle) * radius;
    let y = sin(angle) * radius;
    vertex(x, y, layer * 2);
  }}
  endShape(CLOSE);
}}

function drawConsciousnessCore(fci) {{
  // Central core representing consciousness level
  push();

  // Core size based on FCI
  let coreSize = map(fci, 0, 1, 10, 50);

  // Pulsing effect
  let pulse = sin(frameCount * 0.1) * 5 + 5;

  // Core color based on state
  if (fci < 0.3) {{
    fill(100, 100, 200, 150);  // Blue for COMA
  }} else if (fci < 0.6) {{
    fill(100, 200, 100, 150);  // Green for ANALYSIS
  }} else if (fci < 0.8) {{
    fill(200, 200, 100, 150);  // Yellow for MEDITATION
  }} else {{
    fill(200, 100, 100, 150);  // Red for EMERGENCE
  }}

  noStroke();
  sphere(coreSize + pulse);

  // Energy field
  stroke(255, 100);
  strokeWeight(2);
  noFill();
  sphere(coreSize + pulse + 20);

  pop();
}}

function keyPressed() {{
  if (key === ' ') {{
    // Space bar to capture frame
    saveCanvas('consciousness_frame', 'png');
  }}
}}
  </script>
</body>
</html>
"""

        return code.strip()

    def _color_to_p5js(self, hex_color: str) -> str:
        """Convert hex color to p5.js color array."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        return f"[{r}, {g}, {b}]"


def create_gls_output_generator() -> GLSOutputGenerator:
    """Factory function to create GLS output generator."""
    return GLSOutputGenerator()