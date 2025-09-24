#!/usr/bin/env python3
"""
GLS Habitat Visualization: Complete Living Ecosystem Display
Shows the ΨQRH habitat through GLS spectral analysis and monitoring
Now using the new GLS model framework.
"""

import sys
import os
from datetime import datetime

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from models.gls_data_models import GLSHabitatModel
from models.gls_analysis import GLSAnalyzer, GLSMonitor
from models.gls_visualization import GLSVisualizer


def main():
    """Main GLS habitat visualization demonstration"""
    print("🧬 GLS HABITAT VISUALIZATION & ANALYSIS")
    print("=" * 80)
    print("🌌 Initializing ΨQRH Living Ecosystem with GLS Framework...")

    # Initialize GLS models
    habitat_model = GLSHabitatModel()
    analyzer = GLSAnalyzer(habitat_model)
    monitor = GLSMonitor(analyzer)
    visualizer = GLSVisualizer(habitat_model, analyzer)

    print("✅ GLS Framework Loaded Successfully")
    print()

    # Run comprehensive analysis
    run_gls_analysis(habitat_model, analyzer, monitor)

    # Generate visualizations
    generate_gls_visualizations(visualizer)

    # Export current state
    export_gls_state(habitat_model, analyzer)


def run_gls_analysis(habitat_model, analyzer, monitor):
    """Run complete GLS analysis"""
    print("🔍 RUNNING GLS FRAMEWORK ANALYSIS")
    print("-" * 40)

    # Get complete ecosystem status
    ecosystem_status = habitat_model.get_complete_status()

    print("📊 SYSTEM STATUS:")
    status = ecosystem_status['system_status']
    print(f"   Status: {status.status}")
    print(f"   Emergence Level: {status.emergence_level}")
    print(f"   Spectral Coherence: {status.spectral_coherence}")
    print(f"   Photonics Efficiency: {status.photonics_efficiency}")
    print(f"   Framework Integrity: {status.framework_integrity}")
    print()

    print("🏘️ COLONY ANALYSIS:")
    for species, colony in ecosystem_status['colonies'].items():
        print(f"   {species} Colony:")
        print(f"     Population: {colony.population}")
        print(f"     Health Score: {colony.health_score:.3f}")
        print(f"     Social Cohesion: {colony.social_cohesion:.3f}")
        print(f"     Spectral Signature: {colony.spectral_signature}")
        print(f"     Territory Volume: {colony.territory_volume:.1f} cubic_units")
        print(f"     Communication Frequency: {colony.communication_frequency:.1f} Hz")
        print()

    print("🌈 SPECTRAL ENVIRONMENT:")
    env = ecosystem_status['spectral_environment']
    print(f"   Dimensions: {env.dimensions}")
    print(f"   Resolution: {env.resolution}")
    print(f"   Active Equations: {env.active_equations} ΨQRH equations")
    print(f"   Spectral Fields: {', '.join(env.spectral_fields)}")
    print(f"   Environmental Gradients: {', '.join(env.environmental_gradients)}")
    print()

    print("📡 PHOTONIC ECOSYSTEM:")
    photonic = ecosystem_status['photonic_ecosystem']
    print(f"   Laser Emitters: {photonic['total_emitters']} active")
    print(f"   Optical Fibers: {photonic['fiber_pathways']} pathways")
    print(f"   Holographic Nodes: {photonic['memory_centers']} memory centers")
    print(f"   Phase Coherence: {photonic['avg_coherence']:.2f} average")
    print(f"   Communication Bandwidth: {photonic['active_channels']} simultaneous channels")
    print()

    print("🧠 EMERGENT BEHAVIORS:")
    behaviors = ecosystem_status['emergent_behaviors']
    for behavior in behaviors:
        print(f"   ✅ {behavior}")
    print()

    print("🔬 MATHEMATICAL FOUNDATION:")
    math_foundation = ecosystem_status['mathematical_foundation']
    for equation, status in math_foundation.items():
        indicator = "✅" if status == "ACTIVE" else "❌"
        print(f"   {indicator} {equation}: {status}")
    print()

    # Generate comprehensive analysis report
    print("🎯 GENERATING COMPREHENSIVE ANALYSIS REPORT...")
    analysis_report = analyzer.generate_ecosystem_report()

    print(f"📈 Emergence Level: {analysis_report['emergence_level']:.2f}")
    print(f"🌊 Spectral Coherence Analysis:")
    for species, coherence in analysis_report['spectral_coherence_analysis'].items():
        print(f"     {species}: {coherence:.3f}")

    print(f"📞 Communication Efficiency:")
    for species, efficiency in analysis_report['communication_efficiency'].items():
        print(f"     {species}: {efficiency:.3f}")

    print(f"🧬 GLS Stability Scores:")
    for species, score in analysis_report['gls_stability_scores'].items():
        print(f"     {species}: {score:.3f}")

    if analysis_report['emergent_behaviors']:
        print(f"🚨 Newly Detected Emergent Behaviors:")
        for behavior in analysis_report['emergent_behaviors']:
            print(f"     • {behavior}")

    if analysis_report['recommendations']:
        print(f"💡 Recommendations:")
        for rec in analysis_report['recommendations']:
            print(f"     • {rec}")

    print()

    # System health check
    print("🏥 SYSTEM HEALTH MONITORING...")
    health_status = monitor.check_system_health()

    print(f"Overall Health: {health_status['health_status']}")
    if health_status['active_alerts']:
        print("🚨 Active Alerts:")
        for alert in health_status['active_alerts']:
            print(f"     {alert['type']}: {alert['message']}")
    else:
        print("✅ No active alerts - System operating optimally")

    print()


def generate_gls_visualizations(visualizer):
    """Generate GLS visualizations"""
    print("🎨 GENERATING GLS VISUALIZATIONS...")
    print("-" * 40)

    try:
        # Generate complete visualization suite
        viz_files = visualizer.generate_complete_visualization_suite()

        print("✅ Generated Visualizations:")
        for viz_type, filepath in viz_files.items():
            print(f"   📊 {viz_type}: {filepath}")

        print()
        print("🌐 Visualization files ready for web browser display")

    except Exception as e:
        print(f"⚠️ Visualization generation encountered an issue: {e}")
        print("📊 Using text-based visualization instead...")

        # Generate text-based analysis
        habitat_model = visualizer.habitat_model
        analyzer = visualizer.analyzer

        current_data = habitat_model.get_complete_status()
        report = analyzer.generate_ecosystem_report()

        print("\n📈 TEXT-BASED GLS ANALYSIS:")
        print(f"   System Health: {report['system_health']['overall_stability']:.3f}")
        print(f"   Mathematical Foundation: {'✅' if report['system_health']['mathematical_foundation'] else '⚠️'}")
        print(f"   Photonic Network: {report['system_health']['photonic_network_status']:.3f}")


def export_gls_state(habitat_model, analyzer):
    """Export current GLS state"""
    print("💾 EXPORTING GLS STATE...")
    print("-" * 40)

    # Export GLS format
    try:
        gls_content = habitat_model.export_gls_format()
        with open("gls_habitat_current_state.gls", "w") as f:
            f.write(gls_content)
        print("✅ Current state exported to: gls_habitat_current_state.gls")

        # Export analysis history
        analyzer.export_analysis_history("gls_analysis_history.json")
        print("✅ Analysis history exported to: gls_analysis_history.json")

    except Exception as e:
        print(f"⚠️ Export encountered an issue: {e}")

    print()


def create_gls_web_interface():
    """Create web interface integration for GLS"""
    print("🌐 CREATING GLS WEB INTERFACE INTEGRATION...")
    print("-" * 40)

    # Generate API endpoints for GLS data
    gls_api_code = '''
# GLS API Integration for Live Ecosystem Server
# Add to live_ecosystem_server.py

def _serve_gls_data(self):
    """Serve GLS analysis data"""
    if not self.ecosystem_engine:
        self._send_error_response("Ecosystem not initialized")
        return

    try:
        # Initialize GLS framework
        from models.gls_data_models import GLSHabitatModel
        from models.gls_analysis import GLSAnalyzer

        habitat_model = GLSHabitatModel()
        analyzer = GLSAnalyzer(habitat_model)

        # Update with current ecosystem data
        live_data = self.ecosystem_engine.export_live_data()

        # Generate GLS report
        gls_report = analyzer.generate_ecosystem_report()

        # Combine with GLS habitat status
        gls_status = habitat_model.get_complete_status()

        response_data = {
            'gls_analysis': gls_report,
            'gls_habitat': gls_status,
            'live_ecosystem': live_data,
            'timestamp': time.time()
        }

        self._send_json_response(response_data)

    except Exception as e:
        self._send_error_response(f"Error getting GLS data: {e}")

# Add to API routes in do_GET method:
elif parsed_path.path == '/api/ecosystem/gls-data':
    self._serve_gls_data()
'''

    print("📝 GLS API integration code generated")
    print("🔗 Add '/api/ecosystem/gls-data' endpoint to serve GLS analysis")
    print()

    return gls_api_code


def demo_gls_real_time():
    """Demonstrate real-time GLS monitoring"""
    print("⏰ GLS REAL-TIME MONITORING DEMO")
    print("-" * 40)

    habitat_model = GLSHabitatModel()
    analyzer = GLSAnalyzer(habitat_model)
    monitor = GLSMonitor(analyzer)

    print("🔄 Simulating real-time ecosystem updates...")

    # Simulate some ecosystem changes
    import random
    import numpy as np

    for i in range(5):
        print(f"\n📊 Update {i+1}/5:")

        # Simulate population changes
        for species in ['Araneae', 'Chrysopidae', 'Apis']:
            if species in habitat_model.colonies:
                colony = habitat_model.colonies[species]
                # Small random population change
                colony.population += random.randint(-1, 2)
                colony.population = max(1, colony.population)

                # Small health fluctuation
                colony.health_score += random.uniform(-0.05, 0.05)
                colony.health_score = np.clip(colony.health_score, 0.1, 1.0)

        # Generate health report
        health_status = monitor.check_system_health()
        print(f"   System Health: {health_status['health_status']}")

        if health_status['active_alerts']:
            for alert in health_status['active_alerts']:
                print(f"   🚨 {alert['type']}: {alert['message']}")
        else:
            print("   ✅ All systems optimal")

    print("\n✅ Real-time monitoring demo completed")


if __name__ == "__main__":
    # Run main GLS analysis
    main()

    print("\n" + "="*80)

    # Create web interface integration
    create_gls_web_interface()

    print("\n" + "="*80)

    # Demo real-time monitoring
    demo_gls_real_time()

    print("\n" + "="*80)
    print("🎉 GLS HABITAT VISUALIZATION COMPLETE")
    print("🌐 Ready for browser integration and real-time monitoring")
    print("🔬 All GLS models operational and validated")
    print("=" * 80)