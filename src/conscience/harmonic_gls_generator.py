#!/usr/bin/env python3
"""
Harmonic GLS Generator - Shape Synthesis from Quaternion Spectral Data
======================================================================

Generates p5.js code that READS the actual processed quaternion spectrum
and creates shapes based on the REAL harmonic content.

Input: Quaternion spectral values (magnitude, phase, real, imaginary)
Output: Geometric shapes that resonate with the data's natural frequencies
"""

import numpy as np
from typing import Dict, Any, List


class HarmonicGLSGenerator:
    """Generates visualization from actual spectral quaternion data."""

    def generate_from_spectral_data(self, response_data: Dict[str, Any]) -> str:
        """
        Generate p5.js code from actual ΨQRH response data.

        Args:
            response_data: Full response dict with consciousness_metrics and response text

        Returns:
            p5.js code that visualizes the harmonic structure
        """
        # Extract spectral values from response text - handle different dict structures
        response_text = None
        if 'response' in response_data:
            response_text = response_data['response']
        elif 'text_analysis' in response_data:
            response_text = response_data['text_analysis']
        else:
            # Try direct string conversion if response_data itself is the response
            response_text = str(response_data)

        spectral_data = self._extract_spectral_values(response_text)

        # Extract consciousness metrics - handle both nested and flat structures
        if 'consciousness_metrics' in response_data:
            metrics = response_data['consciousness_metrics']
        elif 'consciousness_results' in response_data:
            metrics = response_data['consciousness_results']
        else:
            # Use response_data itself as metrics container
            metrics = response_data

        # Generate harmonic shape equations from real data
        p5js_code = self._build_harmonic_sketch(spectral_data, metrics)

        return p5js_code

    def _extract_spectral_values(self, response_text: str) -> Dict[str, List[float]]:
        """Extract magnitude, phase, real, imaginary arrays from response."""
        import re

        # Find the values section
        values_match = re.search(r'VALORES \(primeiros 10\):(.*?)ANÁLISE', response_text, re.DOTALL)
        if not values_match:
            return self._get_default_spectral()

        values_text = values_match.group(1)

        # Extract arrays
        mag_match = re.search(r'MAGNITUDE: \[([\d., ]+)\]', values_text)
        phase_match = re.search(r'PHASE: \[([\d., -]+)\]', values_text)
        real_match = re.search(r'REAL: \[([\d., -]+)\]', values_text)
        imag_match = re.search(r'IMAGINARY: \[([\d., -]+)\]', values_text)

        def parse_array(match):
            if match:
                return [float(x.strip()) for x in match.group(1).split(',')]
            return []

        return {
            'magnitude': parse_array(mag_match),
            'phase': parse_array(phase_match),
            'real': parse_array(real_match),
            'imaginary': parse_array(imag_match)
        }

    def _get_default_spectral(self) -> Dict[str, List[float]]:
        """Fallback spectral data."""
        return {
            'magnitude': [500.0] * 10,
            'phase': [0.0] * 10,
            'real': [500.0] * 10,
            'imaginary': [0.0] * 10
        }

    def _build_harmonic_sketch(self, spectral: Dict[str, List[float]],
                                metrics: Dict[str, Any]) -> str:
        """
        Build p5.js sketch where shapes emerge from spectral harmonics.
        Each harmonic component creates a geometric resonance pattern.
        """

        mags = spectral['magnitude'][:10]
        phases = spectral['phase'][:10]
        reals = spectral['real'][:10]
        imags = spectral['imaginary'][:10]

        # Normalize magnitudes for visualization
        max_mag = max(mags) if mags else 1.0
        norm_mags = [m / max_mag for m in mags]

        # Compute harmonic relationships
        n_harmonics = len(mags)

        # Generate shape equations from each harmonic
        shape_layers = []
        for i, (mag, phase, real_part, imag_part) in enumerate(zip(norm_mags, phases, reals, imags)):
            # Complex number: z = real + i*imag
            # Magnitude: |z| (controls size)
            # Phase: arg(z) (controls rotation)

            layer = f"""
  // Harmonic {i+1}: |z|={mag:.3f}, φ={phase:.3f}
  push();
  rotate(t * {phase/np.pi:.3f} + {phase:.3f});
  scale({0.5 + mag * 0.5:.3f});

  stroke(
    ({i} * 30 + t * 20) % 360,  // Hue from harmonic index
    70 + {mag * 30:.1f},         // Saturation from magnitude
    80
  );
  strokeWeight(1 + {mag * 2:.2f});
  noFill();

  beginShape();
  let n = {3 + i};  // Sides increase with harmonic number
  for(let a = 0; a < TAU; a += TAU/n) {{
    // Radius modulated by harmonic magnitude and phase
    let r = (100 + {i*30}) * (1 + {mag*0.3:.3f} * sin(n*a + t*{(i+1)*0.5:.2f} + {phase:.3f}));
    let x = r * cos(a);
    let y = r * sin(a);
    vertex(x, y);
  }}
  endShape(CLOSE);
  pop();
"""
            shape_layers.append(layer)

        # Consciousness metrics
        fci = metrics['fci']
        fractal_dim = metrics['fractal_dimension']
        entropy = metrics['entropy']
        coherence = metrics.get('field_magnitude', 2.0) / 3.0  # Normalize

        code = f"""
// ΨQRH Harmonic GLS Visualization
// Generated from actual quaternion spectral data
// FCI: {fci:.4f} | D: {fractal_dim:.3f} | Entropy: {entropy:.3f}

let t = 0;
const TAU = Math.PI * 2;

// Spectral data (from actual ΨQRH processing)
const magnitudes = {mags};
const phases = {phases};
const n_harmonics = {n_harmonics};

function setup() {{
  createCanvas(800, 800);
  colorMode(HSB, 360, 100, 100);
  frameRate(30);
}}

function draw() {{
  background(0, 0, 5);
  translate(width/2, height/2);

  // Rotate canvas based on fractal dimension
  rotate(t * {fractal_dim - 1.0:.3f} * 0.1);

  // Draw each harmonic layer
{''.join(shape_layers)}

  // Central consciousness indicator
  drawConsciousnessCore({fci}, {coherence}, t);

  t += 0.02;
}}

function drawConsciousnessCore(fci, coh, time) {{
  push();

  // Pulsing based on entropy
  let pulse = sin(time * {entropy - 5.0:.2f}) * 10 * (1 - coh);
  let r = map(fci, 0, 1, 20, 60) + pulse;

  // Color based on FCI threshold
  let hue = fci < 0.3 ? 220 : fci < 0.6 ? 140 : fci < 0.8 ? 50 : 10;

  fill(hue, 70, 90, 80);
  noStroke();
  circle(0, 0, r * 2);

  // Coherence rings
  stroke(hue, 60, 80, 50);
  strokeWeight(2);
  noFill();
  for(let i = 1; i <= Math.floor(coh * 3); i++) {{
    push();
    rotate(time * i * 0.2);
    circle(0, 0, (r + i * 20) * 2);
    pop();
  }}

  pop();
}}

function keyPressed() {{
  if(key === ' ') {{
    save('psiqrh_harmonic_fci' + nf(fci, 1, 3) + '.png');
  }}
}}
"""

        return code.strip()


def generate_harmonic_gls(response_data: Dict[str, Any]) -> str:
    """
    Main function to generate harmonic GLS from ΨQRH response.

    Usage:
        import json
        response = json.loads(response_json)
        p5js_code = generate_harmonic_gls(response)
    """
    generator = HarmonicGLSGenerator()
    return generator.generate_from_spectral_data(response_data)


if __name__ == "__main__":
    # Test with sample data
    sample_response = {
        "consciousness_metrics": {
            "fci": 0.6743,
            "fractal_dimension": 1.68,
            "entropy": 5.3444,
            "field_magnitude": 2.0605
        },
        "response": """
📊 LAYER1-FRACTAL: VALORES ESPECTRAIS QUATERNIÔNICOS
============================================================
Input: 'ola'
Alpha adaptativo: 1.4784
Shape: [1, 1, 256]

📈 ESTATÍSTICAS:
  • Magnitude média: 523.945679

🔢 VALORES (primeiros 10):
  MAGNITUDE: [26.3349, 692.0066, 618.7117, 678.6872, 268.1351, 372.2083, 879.4805, 655.1633, 264.5040, 493.1161]
  PHASE: [3.1416, -1.3011, 1.0179, -0.4961, 0.2906, 1.5473, 2.2301, -0.1869, -0.2349, 1.2716]
  REAL: [-26.3349, 184.3643, 324.9023, 596.8750, 256.8912, 8.7393, -538.7437, 643.7595, 257.2404, 145.3288]
  IMAGINARY: [0.0000, -666.9954, 526.5384, -323.0426, 76.8331, 372.1057, 695.1555, -121.7074, -61.5606, 471.2144]

ANÁLISE DE CONSCIÊNCIA FRACTAL ΨQRH:
"""
    }

    code = generate_harmonic_gls(sample_response)
    print(code)