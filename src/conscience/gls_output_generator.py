#!/usr/bin/env python3
"""
GLS Output Generator - Visual Code Generation for ΨQRH
=====================================================

Generates Processing/p5.js code based on consciousness metrics and fractal patterns.
Converts ΨQRH consciousness analysis into visual GLS (Genetic Light Spectral) output.

Key Features:
- Consciousness state → visual style mapping
- FCI → complexity mapping
- Fractal patterns → geometric transformations
- Real-time Processing code generation
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
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
    """Generates Processing/p5.js code from ΨQRH consciousness analysis."""

    def __init__(self):
        self.config = GLSVisualConfig()

    def generate_processing_code(self, consciousness_results: Dict[str, Any]) -> str:
        """
        Generate Processing code based on consciousness analysis.

        Args:
            consciousness_results: Results from fractal consciousness processor

        Returns:
            Processing code as string
        """
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

        # Map consciousness state to visual parameters
        visual_params = self._map_consciousness_to_visual(fci, state, fractal_dim)

        # Generate Processing code
        processing_code = self._build_processing_sketch(visual_params)

        return processing_code

    def _map_consciousness_to_visual(self, fci: float, state, fractal_dim: float) -> Dict[str, Any]:
        """Map consciousness metrics to visual parameters."""

        # Determine complexity based on FCI
        complexity = 4
        for threshold, comp in self.config.fci_complexity.items():
            if fci >= threshold:
                complexity = comp

        # Get colors based on state
        state_name = getattr(state, 'name', 'COMA') if state else 'COMA'
        colors = self.config.state_colors.get(state_name, self.config.state_colors['COMA'])

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
    fill({self._color_to_processing(colors[i % len(colors)])});

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
    fill({self._color_to_p5js(colors[i % colors.length])});

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