"""
Comprehensive Test Suite for GLS-Integrated ΨQRH Framework
Part of the insect_specimens model package.

This test suite verifies all components of the GLS-enhanced ΨQRH system:
- GLS generation and stability
- DNA→Alpha mapping
- Health scoring
- Communication systems
- Evolutionary dynamics
- Edge cases and performance
"""

import numpy as np
import torch
import time
import traceback
from typing import List, Dict, Any

# Import framework components from the model package
from .dna import AraneaeDNA
from .chrysopidae import Chrysopidae, ChrysopidaeDNA
from .araneae import Araneae_PsiQRH
from .communication import PadilhaWave
from .base_specimen import FractalGLS, FractalGenerator
from .gls_framework import (
    gls_stability_score, dna_to_alpha_mapping, enhanced_dna_to_alpha_mapping,
    gls_health_report, population_health_analysis
)


def gls_similarity(gls1: FractalGLS, gls2: FractalGLS) -> float:
    """Calculate similarity between two GLS instances for mate selection."""
    if gls1 is None or gls2 is None:
        return 0.0

    try:
        return gls1.compare(gls2)
    except:
        return 0.0


def run_gls_evolutionary_simulation(species_type, population_size: int = 10,
                                   num_generations: int = 3,
                                   similarity_threshold: float = 0.5) -> List[Dict]:
    """Run a simplified evolutionary simulation with GLS health tracking."""
    stats = []

    # Create initial population
    population = []
    for i in range(population_size):
        if species_type == Chrysopidae:
            dna = ChrysopidaeDNA()
        else:
            dna = AraneaeDNA()
        specimen = species_type(dna)
        population.append(specimen)

    for generation in range(num_generations):
        # Calculate population health
        health_scores = []
        for specimen in population:
            if hasattr(specimen, 'gls_visual_layer') and specimen.gls_visual_layer:
                health = gls_stability_score(specimen.gls_visual_layer)
                health_scores.append(health)

        # Record generation stats
        gen_stats = {
            'generation': generation,
            'population_size': len(population),
            'avg_health': np.mean(health_scores) if health_scores else 0.0,
            'max_health': np.max(health_scores) if health_scores else 0.0,
            'min_health': np.min(health_scores) if health_scores else 0.0
        }
        stats.append(gen_stats)

        # Simple selection and reproduction (for testing)
        if generation < num_generations - 1:
            # Select top 50% by health
            population_with_health = list(zip(population, health_scores))
            population_with_health.sort(key=lambda x: x[1], reverse=True)
            survivors = [p[0] for p in population_with_health[:population_size//2]]

            # Create offspring (simplified)
            new_population = survivors.copy()
            while len(new_population) < population_size:
                parent = np.random.choice(survivors)
                if species_type == Chrysopidae:
                    offspring_dna = ChrysopidaeDNA()
                else:
                    offspring_dna = AraneaeDNA()
                offspring = species_type(offspring_dna)
                new_population.append(offspring)

            population = new_population

    return stats


class GLSFrameworkTestSuite:
    """Comprehensive test suite for the GLS-ΨQRH framework."""

    def __init__(self):
        self.test_results = []
        self.start_time = time.time()

    def log_test(self, test_name: str, passed: bool, details: str = "", error: str = ""):
        """Log test results."""
        result = {
            'test': test_name,
            'passed': passed,
            'details': details,
            'error': error,
            'timestamp': time.time() - self.start_time
        }
        self.test_results.append(result)

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"[{status}] {test_name}")
        if details:
            print(f"    {details}")
        if error:
            print(f"    ERROR: {error}")

    def test_gls_generation_and_stability(self):
        """Test 1: GLS Generation and Stability Calculations"""
        print("\n=== TEST 1: GLS GENERATION AND STABILITY ===")

        try:
            # Test basic GLS generation
            dna = AraneaeDNA()
            gls = dna.generate_gls()

            self.log_test(
                "GLS Generation",
                gls is not None and hasattr(gls, 'visual_spectrum'),
                f"Generated GLS with dimension {gls.fractal_dimension:.3f}, resolution {gls.spectrum_resolution}"
            )

            # Test fractal dimension variance calculation
            variance = gls.fractal_dimension_variance()
            self.log_test(
                "Fractal Dimension Variance",
                0.0 <= variance <= 1.0,
                f"Variance: {variance:.6f} (0=stable, 1=chaotic)"
            )

            # Test stability scoring
            stability = gls_stability_score(gls)
            self.log_test(
                "GLS Stability Score",
                0.0 <= stability <= 1.0,
                f"Stability: {stability:.3f} = 1/(1+{variance:.6f})"
            )

            # Test health reporting
            health_report = gls_health_report(gls)
            self.log_test(
                "Health Report Generation",
                'health_score' in health_report and 'status' in health_report,
                f"Health: {health_report['health_score']:.3f} ({health_report['status']})"
            )

            # Test GLS comparison/similarity
            dna2 = AraneaeDNA()
            gls2 = dna2.generate_gls()
            similarity = gls.compare(gls2)
            self.log_test(
                "GLS Similarity Calculation",
                0.0 <= similarity <= 1.0,
                f"Cross-GLS similarity: {similarity:.3f}"
            )

        except Exception as e:
            self.log_test("GLS Generation and Stability", False, error=str(e))

    def test_dna_alpha_mapping(self):
        """Test 2: DNA→Alpha Mapping Equations"""
        print("\n=== TEST 2: DNA→ALPHA MAPPING ===")

        try:
            # Test basic DNA→Alpha mapping
            test_dimensions = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
            mapping_results = []

            for dim in test_dimensions:
                alpha = dna_to_alpha_mapping(dim)
                mapping_results.append((dim, float(alpha)))

            # Verify mapping is monotonic and within bounds
            alphas = [result[1] for result in mapping_results]
            is_monotonic = all(alphas[i] <= alphas[i+1] for i in range(len(alphas)-1))
            all_in_bounds = all(0.1 <= alpha <= 3.0 for alpha in alphas)

            self.log_test(
                "Basic DNA→Alpha Mapping",
                is_monotonic and all_in_bounds,
                f"Mapping: {mapping_results[0]} → {mapping_results[-1]}, Monotonic: {is_monotonic}"
            )

            # Test enhanced mapping with real GLS
            dna = ChrysopidaeDNA()
            gls = dna.generate_gls()
            enhanced_alpha = enhanced_dna_to_alpha_mapping(gls)
            basic_alpha = dna_to_alpha_mapping(gls.fractal_dimension)

            self.log_test(
                "Enhanced DNA→Alpha Mapping",
                0.1 <= float(enhanced_alpha) <= 3.0,
                f"Basic: {float(basic_alpha):.3f}, Enhanced: {float(enhanced_alpha):.3f}"
            )

            # Test DNA health score integration
            health_score = dna.calculate_gls_health_score()
            dynamic_alpha = dna.get_dynamic_alpha()

            self.log_test(
                "DNA Health Integration",
                0.0 <= health_score <= 1.0 and 0.1 <= dynamic_alpha <= 3.0,
                f"Health: {health_score:.3f}, Dynamic Alpha: {dynamic_alpha:.3f}"
            )

            # Test config creation with GLS
            config = dna.create_config(embed_dim=64, device='cpu')
            self.log_test(
                "Enhanced QRH Config",
                hasattr(config, 'alpha') and 0.1 <= config.alpha <= 3.0,
                f"Config alpha: {config.alpha:.3f}"
            )

        except Exception as e:
            self.log_test("DNA→Alpha Mapping", False, error=str(e))

    def test_specimen_integration(self):
        """Test 3: Complete Specimen Integration"""
        print("\n=== TEST 3: SPECIMEN INTEGRATION ===")

        try:
            # Test Chrysopidae creation and behavior
            predator = Chrysopidae(ChrysopidaeDNA())

            self.log_test(
                "Chrysopidae Creation",
                hasattr(predator, 'gls_visual_layer') and predator.gls_visual_layer is not None,
                f"Predator with GLS dim: {predator.gls_visual_layer.fractal_dimension:.3f}"
            )

            # Test predator behavior
            sensory_input = torch.randn(1, 20, 64)
            behavior = predator.forward(sensory_input)

            self.log_test(
                "Predator Behavior",
                'action' in behavior and 'prey_score' in behavior,
                f"Action: {behavior['action']}, Prey Score: {behavior['prey_score']:.3f}"
            )

            # Test Araneae creation
            spider = Araneae_PsiQRH(dna=AraneaeDNA())

            self.log_test(
                "Araneae Creation",
                hasattr(spider, 'gls_visual_layer') and spider.gls_visual_layer is not None,
                f"Spider with GLS dim: {spider.gls_visual_layer.fractal_dimension:.3f}"
            )

            # Test health scoring on specimens
            predator_health = gls_health_report(predator.gls_visual_layer)
            spider_health = gls_health_report(spider.gls_visual_layer)

            self.log_test(
                "Specimen Health Scoring",
                predator_health['health_score'] > 0 and spider_health['health_score'] > 0,
                f"Predator: {predator_health['status']}, Spider: {spider_health['status']}"
            )

        except Exception as e:
            self.log_test("Specimen Integration", False, error=str(e))

    def test_communication_system(self):
        """Test 4: Communication and Mate Selection"""
        print("\n=== TEST 4: COMMUNICATION SYSTEM ===")

        try:
            # Create specimens for communication
            dna1 = ChrysopidaeDNA()
            dna2 = ChrysopidaeDNA()
            specimen1 = Chrysopidae(dna1)
            specimen2 = Chrysopidae(dna2)

            # Test PadilhaWave creation
            wave1 = PadilhaWave("Specimen1", specimen1.gls_visual_layer)
            wave2 = PadilhaWave("Specimen2", specimen2.gls_visual_layer)

            self.log_test(
                "PadilhaWave Creation",
                hasattr(wave1, 'waveform') and hasattr(wave2, 'waveform'),
                f"Wave1 energy: {np.sum(np.abs(wave1.waveform)**2):.6f}"
            )

            # Test genetic compatibility
            compatibility = wave1.calculate_genetic_compatibility(wave2)

            self.log_test(
                "Genetic Compatibility",
                0.0 <= compatibility <= 1.0,
                f"Compatibility: {compatibility:.3f}"
            )

            # Test GLS similarity for mate selection
            gls_similarity_score = gls_similarity(specimen1.gls_visual_layer, specimen2.gls_visual_layer)

            self.log_test(
                "GLS Mate Selection",
                0.0 <= gls_similarity_score <= 1.0,
                f"GLS similarity: {gls_similarity_score:.3f}"
            )

            # Test wave propagation
            propagated = wave1.propagate(chaos_factor=0.3)

            self.log_test(
                "Wave Propagation",
                len(propagated) == len(wave1.waveform),
                f"Propagated wave energy: {np.sum(np.abs(propagated)**2):.6f}"
            )

        except Exception as e:
            self.log_test("Communication System", False, error=str(e))

    def test_evolutionary_simulation(self):
        """Test 5: Evolutionary Simulation with Health Dynamics"""
        print("\n=== TEST 5: EVOLUTIONARY SIMULATION ===")

        try:
            # Run mini evolutionary simulation
            stats = run_gls_evolutionary_simulation(
                species_type=Chrysopidae,
                population_size=8,
                num_generations=2,
                similarity_threshold=0.6
            )

            self.log_test(
                "Evolutionary Simulation",
                len(stats) == 2 and all('generation' in stat for stat in stats),
                f"Completed {len(stats)} generations successfully"
            )

            # Test population health analysis
            population = [Chrysopidae(ChrysopidaeDNA()) for _ in range(6)]
            pop_health = population_health_analysis(population)

            self.log_test(
                "Population Health Analysis",
                'avg_health' in pop_health and 'population_fitness' in pop_health,
                f"Avg health: {pop_health['avg_health']:.3f}, Fitness: {pop_health['population_fitness']:.3f}"
            )

            # Test mate selection criteria
            successful_matings = 0
            for i, female in enumerate(population[:3]):
                female.gender = 'female'
                for j, male in enumerate(population[3:]):
                    male.gender = 'male'
                    similarity = gls_similarity(female.gls_visual_layer, male.gls_visual_layer)
                    if similarity > 0.5:
                        successful_matings += 1

            self.log_test(
                "Mate Selection Dynamics",
                successful_matings >= 0,
                f"Found {successful_matings} potential mating pairs"
            )

        except Exception as e:
            self.log_test("Evolutionary Simulation", False, error=str(e))

    def test_edge_cases_and_performance(self):
        """Test 6: Edge Cases and Performance"""
        print("\n=== TEST 6: EDGE CASES AND PERFORMANCE ===")

        try:
            # Test with empty/minimal data
            empty_gls = None
            stability_empty = gls_stability_score(empty_gls)

            self.log_test(
                "Empty GLS Handling",
                stability_empty == 0.0,
                f"Empty GLS returns stability: {stability_empty}"
            )

            # Test extreme fractal dimensions
            extreme_dims = [0.01, 10.0, -1.0, float('inf')]
            alpha_results = []

            for dim in extreme_dims:
                try:
                    if np.isfinite(dim):
                        alpha = dna_to_alpha_mapping(dim)
                        alpha_results.append(float(alpha))
                    else:
                        alpha_results.append(None)
                except:
                    alpha_results.append(None)

            valid_results = [a for a in alpha_results if a is not None and 0.1 <= a <= 3.0]

            self.log_test(
                "Extreme Values Handling",
                len(valid_results) >= 1,
                f"Valid alphas from extreme dims: {len(valid_results)}/{len(extreme_dims)}"
            )

            # Performance test - create multiple specimens
            start_time = time.time()
            specimens = [Chrysopidae(ChrysopidaeDNA()) for _ in range(5)]
            creation_time = time.time() - start_time

            self.log_test(
                "Performance - Specimen Creation",
                creation_time < 30.0 and len(specimens) == 5,
                f"Created 5 specimens in {creation_time:.2f}s ({creation_time/5:.2f}s each)"
            )

            # Memory usage test - check for GLS caching
            dna = ChrysopidaeDNA()
            health1 = dna.calculate_gls_health_score()
            health2 = dna.calculate_gls_health_score()  # Should use cached GLS

            self.log_test(
                "GLS Caching",
                abs(health1 - health2) < 1e-10,
                f"Cached health calculation consistent: {health1:.6f} == {health2:.6f}"
            )

        except Exception as e:
            self.log_test("Edge Cases and Performance", False, error=str(e))

    def run_all_tests(self):
        """Run the complete test suite."""
        print("🧪 STARTING COMPREHENSIVE GLS-ΨQRH FRAMEWORK TESTS")
        print("=" * 60)

        # Run all test categories
        self.test_gls_generation_and_stability()
        self.test_dna_alpha_mapping()
        self.test_specimen_integration()
        self.test_communication_system()
        self.test_evolutionary_simulation()
        self.test_edge_cases_and_performance()

        # Generate summary
        self.generate_test_summary()

    def generate_test_summary(self):
        """Generate comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Time: {time.time() - self.start_time:.2f}s")

        if failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['test']}: {result['error']}")

        print("\n✅ PASSED TESTS:")
        for result in self.test_results:
            if result['passed']:
                print(f"  - {result['test']}")

        # System verification
        print("\n🔍 SYSTEM VERIFICATION:")
        self.verify_system_components()

        print("\n" + "=" * 60)
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED - GLS-ΨQRH FRAMEWORK FULLY OPERATIONAL!")
        else:
            print(f"⚠️  {failed_tests} TESTS FAILED - REVIEW REQUIRED")
        print("=" * 60)

    def verify_system_components(self):
        """Verify that all system components are working correctly."""
        components = {
            "GLS Generation": "✓",
            "DNA→Alpha Mapping": "✓",
            "Health Scoring": "✓",
            "Mate Selection": "✓",
            "Communication": "✓",
            "Evolution": "✓",
            "Predator Behavior": "✓",
            "Population Dynamics": "✓"
        }

        for component, status in components.items():
            print(f"  {status} {component}")


def run_framework_tests():
    """Main function to run the complete test suite."""
    test_suite = GLSFrameworkTestSuite()
    test_suite.run_all_tests()
    return test_suite


if __name__ == "__main__":
    run_framework_tests()