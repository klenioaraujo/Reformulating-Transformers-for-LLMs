# Make the insect specimens easily importable

from .base_specimen import PsiQRHBase, FractalGLS
from .chrysopidae import Chrysopidae_PsiQRH, Chrysopidae, ChrysopidaeDNA
from .tettigoniidae import Tettigoniidae_PsiQRH
from .camponotus import Camponotus_PsiQRH
from .apis_mellifera import ApisMellifera_PsiQRH
from .araneae import Araneae_PsiQRH
from .dna import AraneaeDNA
from .communication import PadilhaWave

# GLS Framework
from .gls_framework import (
    gls_stability_score,
    dna_to_alpha_mapping,
    enhanced_dna_to_alpha_mapping,
    gls_health_report,
    population_health_analysis,
    GLSRealtimeVisualizer,
    GLSBrowserVisualizer,
    launch_gls_browser_monitor,
    launch_gls_realtime_monitor,
    test_gls_equations
)

# GLS Testing
from .gls_tests import (
    GLSFrameworkTestSuite,
    run_framework_tests,
    gls_similarity,
    run_gls_evolutionary_simulation
)

__all__ = [
    # Base classes
    "PsiQRHBase",
    "FractalGLS",

    # Specimens
    "Chrysopidae_PsiQRH",
    "Chrysopidae",
    "Tettigoniidae_PsiQRH",
    "Camponotus_PsiQRH",
    "ApisMellifera_PsiQRH",
    "Araneae_PsiQRH",

    # DNA and Communication
    "ChrysopidaeDNA",
    "AraneaeDNA",
    "PadilhaWave",

    # GLS Framework Functions
    "gls_stability_score",
    "dna_to_alpha_mapping",
    "enhanced_dna_to_alpha_mapping",
    "gls_health_report",
    "population_health_analysis",

    # Visualization
    "GLSRealtimeVisualizer",
    "GLSBrowserVisualizer",
    "launch_gls_browser_monitor",
    "launch_gls_realtime_monitor",
    "test_gls_equations",

    # Testing
    "GLSFrameworkTestSuite",
    "run_framework_tests",
    "gls_similarity",
    "run_gls_evolutionary_simulation"
]
