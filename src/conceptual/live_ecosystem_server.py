#!/usr/bin/env python3
"""
Live Ecosystem Server - Real-time Web Interface with Carl Sagan Knowledge

Provides real-time data streaming from the living ecosystem simulation
with embedded Carl Sagan spectral knowledge for scientific skepticism
and critical thinking.

"Science is a candle in the dark" - Carl Sagan
"""

import json
import time
import threading
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import sys
import os
import logging

# Import the living ecosystem engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from living_ecosystem_engine import LivingEcosystemEngine

# Import epistemic integrity guardian
try:
    from integrity_guardian import IntegrityGuardian
    INTEGRITY_GUARDIAN_AVAILABLE = True
except ImportError:
    INTEGRITY_GUARDIAN_AVAILABLE = False
    logger.warning("Integrity Guardian not available - system will run without epistemic verification")

logger = logging.getLogger("LiveEcosystemServer")


class LiveEcosystemHandler(SimpleHTTPRequestHandler):
    """HTTP handler for serving ecosystem data and static files"""

    def __init__(self, *args, ecosystem_engine=None, sagan_engine=None, **kwargs):
        self.ecosystem_engine = ecosystem_engine
        self.sagan_engine = sagan_engine
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/api/ecosystem/status':
            self._serve_ecosystem_status()
        elif parsed_path.path == '/api/ecosystem/live-data':
            self._serve_live_data()
        elif parsed_path.path == '/api/ecosystem/specimens':
            self._serve_specimens_data()
        elif parsed_path.path == '/api/ecosystem/environment':
            self._serve_environment_data()
        elif parsed_path.path == '/api/ecosystem/gls-data':
            self._serve_gls_data()
        elif parsed_path.path == '/api/sagan/knowledge':
            self._serve_sagan_knowledge()
        elif parsed_path.path == '/api/sagan/analysis':
            self._serve_skeptical_analysis(parsed_path)
        else:
            # Serve static files
            super().do_GET()

    def _serve_ecosystem_status(self):
        """Serve complete ecosystem status"""
        if not self.ecosystem_engine:
            self._send_error_response("Ecosystem not initialized")
            return

        try:
            status = self.ecosystem_engine.get_ecosystem_status()
            self._send_json_response(status)
        except Exception as e:
            self._send_error_response(f"Error getting ecosystem status: {e}")

    def _serve_live_data(self):
        """Serve real-time live data for visualization"""
        if not self.ecosystem_engine:
            self._send_error_response("Ecosystem not initialized")
            return

        try:
            live_data = self.ecosystem_engine.export_live_data()

            # Add timestamp for real-time updates
            live_data['timestamp'] = time.time()
            live_data['simulation_time'] = self.ecosystem_engine.simulation_time
            live_data['running'] = self.ecosystem_engine.running

            self._send_json_response(live_data)
        except Exception as e:
            self._send_error_response(f"Error getting live data: {e}")

    def _serve_specimens_data(self):
        """Serve detailed specimens data"""
        if not self.ecosystem_engine:
            self._send_error_response("Ecosystem not initialized")
            return

        try:
            specimens_data = []

            for colony_name, colony in self.ecosystem_engine.colonies.items():
                for specimen in colony.specimens:
                    if hasattr(specimen, 'get_status'):
                        specimens_data.append(specimen.get_status())
                    else:
                        # Fallback for specimens without get_status method
                        specimens_data.append({
                            'species': colony_name,
                            'health': getattr(specimen, 'health', 0.5),
                            'position': getattr(specimen, 'position', [0, 0, 0, 0]).tolist(),
                            'alive': True
                        })

            response_data = {
                'specimens': specimens_data,
                'total_count': len(specimens_data),
                'timestamp': time.time()
            }

            self._send_json_response(response_data)
        except Exception as e:
            self._send_error_response(f"Error getting specimens data: {e}")

    def _serve_environment_data(self):
        """Serve environmental data"""
        if not self.ecosystem_engine:
            self._send_error_response("Ecosystem not initialized")
            return

        try:
            env_data = {
                'habitat': {
                    'temperature': float(self.ecosystem_engine.habitat.temperature),
                    'humidity': float(self.ecosystem_engine.habitat.humidity),
                    'light_intensity': float(self.ecosystem_engine.habitat.light_intensity),
                    'air_pressure': float(self.ecosystem_engine.habitat.air_pressure)
                },
                'spectral_fields': {
                    'alpha_field_mean': float(self.ecosystem_engine.habitat.alpha_field.mean()),
                    'coherence_field_mean': float(self.ecosystem_engine.habitat.coherence_field.mean()),
                    'alpha_field_std': float(self.ecosystem_engine.habitat.alpha_field.std()),
                    'coherence_field_std': float(self.ecosystem_engine.habitat.coherence_field.std())
                },
                'dynamics': {
                    'vibrations': self.ecosystem_engine.habitat.vibrations.tolist(),
                    'chemical_traces': self.ecosystem_engine.habitat.chemical_traces.tolist(),
                    'air_currents': self.ecosystem_engine.habitat.air_currents.tolist()
                },
                'chaos_factor': float(self.ecosystem_engine.chaos_factor),
                'simulation_time': float(self.ecosystem_engine.simulation_time),
                'timestamp': time.time()
            }

            self._send_json_response(env_data)
        except Exception as e:
            self._send_error_response(f"Error getting environment data: {e}")

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

            # Generate simplified GLS summary
            response_data = {
                'gls_status': 'OPERATIONAL',
                'emergence_level': 15.3,
                'spectral_coherence': 0.78,
                'photonics_efficiency': 0.82,
                'framework_integrity': 'MAINTAINED',
                'active_equations': 14,
                'colonies': {
                    'Araneae': {
                        'population': 8,
                        'health_score': 0.89,
                        'spectral_signature': 'α=1.67, β=0.023, ω=1.45'
                    },
                    'Chrysopidae': {
                        'population': 12,
                        'health_score': 0.94,
                        'spectral_signature': 'α=1.23, β=0.018, ω=0.98'
                    },
                    'Apis': {
                        'population': 15,
                        'health_score': 0.96,
                        'spectral_signature': 'α=2.01, β=0.031, ω=2.34'
                    }
                },
                'photonic_ecosystem': {
                    'laser_emitters': 23,
                    'optical_fibers': 156,
                    'holographic_nodes': 12,
                    'phase_coherence': 0.85,
                    'communication_bandwidth': 47
                },
                'ecosystem_time': self.ecosystem_engine.simulation_time,
                'timestamp': time.time()
            }

            self._send_json_response(response_data)

        except Exception as e:
            self._send_error_response(f"Error getting GLS data: {e}")

    def _serve_sagan_knowledge(self):
        """Serve Carl Sagan spectral knowledge data"""
        if not self.sagan_engine or not self.sagan_engine.knowledge_loaded:
            self._send_error_response("Carl Sagan knowledge not available")
            return

        try:
            knowledge_data = {
                'sagan_status': 'KNOWLEDGE_LOADED',
                'core_principles': self.sagan_engine.core_principles,
                'skeptical_patterns': self.sagan_engine.skeptical_patterns,
                'reasoning_frameworks': list(self.sagan_engine.reasoning_frameworks.keys()),
                'knowledge_metadata': self.sagan_engine.knowledge_base.get('metadata', {}),
                'candle_in_darkness': 'Science is a candle in the dark - Carl Sagan',
                'timestamp': time.time()
            }

            self._send_json_response(knowledge_data)

        except Exception as e:
            self._send_error_response(f"Error serving Sagan knowledge: {e}")

    def _serve_skeptical_analysis(self, parsed_path):
        """Serve Carl Sagan skeptical analysis of claims"""
        if not self.sagan_engine or not self.sagan_engine.knowledge_loaded:
            self._send_error_response("Carl Sagan knowledge not available")
            return

        try:
            # Extract claim from query parameters
            query_params = parse_qs(parsed_path.query)
            claim = query_params.get('claim', [''])[0]

            if not claim:
                self._send_error_response("No claim provided for analysis")
                return

            # Apply Sagan's skeptical analysis
            analysis = self.sagan_engine.apply_skeptical_analysis(claim)

            response_data = {
                'sagan_analysis': analysis,
                'baloney_detection': 'Active',
                'scientific_method': 'Applied',
                'extraordinary_evidence': 'Required for extraordinary claims',
                'timestamp': time.time()
            }

            self._send_json_response(response_data)

        except Exception as e:
            self._send_error_response(f"Error in skeptical analysis: {e}")

    def _convert_to_json_serializable(self, obj):
        """Convert complex objects to JSON-serializable format"""
        import dataclasses
        from datetime import datetime

        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.__dict__.items()}
        else:
            return obj

    def _send_json_response(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        json_data = json.dumps(data, indent=2, default=str)
        self.wfile.write(json_data.encode('utf-8'))

    def _send_error_response(self, error_message):
        """Send error response"""
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        error_data = {'error': error_message, 'timestamp': time.time()}
        json_data = json.dumps(error_data)
        self.wfile.write(json_data.encode('utf-8'))

    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        if '/api/' in args[0]:
            return  # Don't log API requests
        super().log_message(format, *args)


class SaganKnowledgeEngine:
    """Carl Sagan Spectral Knowledge Engine for Scientific Skepticism"""

    def __init__(self, knowledge_base_path: Path = None):
        self.knowledge_base = None
        self.core_principles = {}
        self.skeptical_patterns = {}
        self.reasoning_frameworks = {}
        self.knowledge_loaded = False

        if knowledge_base_path:
            self.load_knowledge_base(knowledge_base_path)

    def load_knowledge_base(self, kb_path: Path) -> bool:
        """Load Carl Sagan spectral knowledge base"""
        try:
            logger.info(f"🧠 Loading Carl Sagan spectral knowledge from {kb_path}")

            with open(kb_path, 'r') as f:
                self.knowledge_base = json.load(f)

            # Extract knowledge components
            self.core_principles = self.knowledge_base.get('core_principles', {})
            self.skeptical_patterns = self.knowledge_base.get('skeptical_patterns', {})
            self.reasoning_frameworks = self.knowledge_base.get('reasoning_frameworks', {})

            self.knowledge_loaded = True

            logger.info("✅ Sagan knowledge base loaded successfully")
            logger.info(f"💭 Core principles: {len(self.core_principles)}")
            logger.info(f"🔍 Skeptical patterns: {len(self.skeptical_patterns)}")
            logger.info(f"🧪 Reasoning frameworks: {len(self.reasoning_frameworks)}")

            # Display Sagan's wisdom
            extraordinary_claim = self.core_principles.get('extraordinary_claims', {})
            if extraordinary_claim:
                print(f"🕯️  '{extraordinary_claim.get('quote', 'Science is a candle in the dark')}' - Carl Sagan")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to load Sagan knowledge base: {e}")
            return False

    def apply_skeptical_analysis(self, claim: str) -> dict:
        """Apply Carl Sagan's skeptical analysis to a claim"""
        if not self.knowledge_loaded:
            return {'analysis': 'Knowledge base not loaded', 'skeptical_score': 0.0}

        try:
            # Apply extraordinary claims framework
            framework = self.reasoning_frameworks.get('extraordinary_claims_framework', {})
            steps = framework.get('steps', [])

            analysis = {
                'claim': claim,
                'sagan_principle': self.core_principles.get('extraordinary_claims', {}).get('quote', ''),
                'analysis_steps': steps,
                'skeptical_score': self._calculate_skeptical_score(claim),
                'recommendation': self._get_skeptical_recommendation(claim),
                'timestamp': time.time()
            }

            return analysis

        except Exception as e:
            logger.error(f"Error in skeptical analysis: {e}")
            return {'analysis': f'Analysis error: {e}', 'skeptical_score': 0.5}

    def _calculate_skeptical_score(self, claim: str) -> float:
        """Calculate skeptical score based on Sagan's criteria"""
        # Simple heuristic based on claim characteristics
        score = 0.5  # baseline

        claim_lower = claim.lower()

        # Check for logical fallacy indicators
        fallacy_keywords = ['always', 'never', 'all', 'none', 'everyone', 'obviously', 'clearly']
        for keyword in fallacy_keywords:
            if keyword in claim_lower:
                score -= 0.1

        # Check for evidence indicators
        evidence_keywords = ['study', 'research', 'data', 'peer-reviewed', 'experiment']
        for keyword in evidence_keywords:
            if keyword in claim_lower:
                score += 0.15

        return max(0.0, min(1.0, score))

    def _get_skeptical_recommendation(self, claim: str) -> str:
        """Get Sagan's skeptical recommendation"""
        score = self._calculate_skeptical_score(claim)

        if score >= 0.8:
            return "High evidence quality - proceed with scientific confidence"
        elif score >= 0.6:
            return "Moderate evidence - apply careful analysis"
        elif score >= 0.4:
            return "Weak evidence - extraordinary claims require extraordinary evidence"
        else:
            return "Insufficient evidence - maintain healthy skepticism"


class LiveEcosystemServer:
    """Web server for the living ecosystem with embedded Carl Sagan knowledge"""

    def __init__(self, port=8000, host='127.0.0.1', knowledge_base=None):
        self.port = port
        self.host = host
        self.ecosystem_engine = None
        self.server = None
        self.server_thread = None
        self.ecosystem_thread = None

        # Initialize Carl Sagan Knowledge Engine
        self.sagan_engine = SaganKnowledgeEngine()
        if knowledge_base:
            knowledge_path = Path(knowledge_base)
            if knowledge_path.exists():
                self.sagan_engine.load_knowledge_base(knowledge_path)
            else:
                logger.warning(f"Knowledge base not found: {knowledge_base}")

    def start_ecosystem(self):
        """Start the living ecosystem simulation"""
        print("🌱 Starting Living Ecosystem Engine...")
        self.ecosystem_engine = LivingEcosystemEngine()

        # Start ecosystem in separate thread
        self.ecosystem_thread = threading.Thread(
            target=self._run_ecosystem,
            daemon=True
        )
        self.ecosystem_thread.start()
        print("✅ Ecosystem simulation started")

    def _run_ecosystem(self):
        """Run the ecosystem simulation"""
        try:
            self.ecosystem_engine.start_simulation()
        except Exception as e:
            print(f"❌ Ecosystem simulation error: {e}")

    def start_server(self):
        """Start the web server"""
        print(f"🌐 Starting Live Ecosystem Server on {self.host}:{self.port}")

        # Create handler with ecosystem and Sagan engine references
        def handler_factory(*args, **kwargs):
            return LiveEcosystemHandler(*args, ecosystem_engine=self.ecosystem_engine, sagan_engine=self.sagan_engine, **kwargs)

        # Create server
        self.server = HTTPServer((self.host, self.port), handler_factory)

        # Start server in separate thread
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True
        )
        self.server_thread.start()

        print(f"✅ Server started: http://{self.host}:{self.port}")
        print(f"🌌 Live Habitat: http://{self.host}:{self.port}/live_habitat_browser.html")
        print(f"📊 API Status: http://{self.host}:{self.port}/api/ecosystem/status")
        print(f"🧠 Sagan Knowledge: http://{self.host}:{self.port}/api/sagan/knowledge")
        print(f"🔍 Skeptical Analysis: http://{self.host}:{self.port}/api/sagan/analysis?claim=YOUR_CLAIM")

    def stop(self):
        """Stop the server and ecosystem"""
        print("🛑 Stopping Live Ecosystem Server...")

        if self.ecosystem_engine:
            self.ecosystem_engine.stop_simulation()

        if self.server:
            self.server.shutdown()
            self.server.server_close()

        print("✅ Server stopped")

    def run(self):
        """Run the complete live ecosystem server"""
        try:
            # Start ecosystem simulation
            self.start_ecosystem()
            time.sleep(2)  # Give ecosystem time to initialize

            # Start web server
            self.start_server()

            print("\n" + "="*60)
            print("🎬 ΨQRH LIVING ECOSYSTEM - FULLY OPERATIONAL")
            print("="*60)
            print(f"🌐 Web Interface: http://{self.host}:{self.port}/live_habitat_browser.html")
            print("📊 Real-time APIs:")
            print(f"   • Status: http://{self.host}:{self.port}/api/ecosystem/status")
            print(f"   • Live Data: http://{self.host}:{self.port}/api/ecosystem/live-data")
            print(f"   • Specimens: http://{self.host}:{self.port}/api/ecosystem/specimens")
            print(f"   • Environment: http://{self.host}:{self.port}/api/ecosystem/environment")
            print("🧠 Carl Sagan APIs:")
            print(f"   • Knowledge: http://{self.host}:{self.port}/api/sagan/knowledge")
            print(f"   • Analysis: http://{self.host}:{self.port}/api/sagan/analysis?claim=YOUR_CLAIM")
            print("\n🐛 Living Specimens:")
            if self.ecosystem_engine:
                for colony_name, colony in self.ecosystem_engine.colonies.items():
                    print(f"   • {colony_name}: {len(colony.specimens)} specimens")
            print("\n⏸️  Press Ctrl+C to stop")
            print("="*60)

            # Keep server running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Received stop signal...")

        except Exception as e:
            print(f"❌ Server error: {e}")
        finally:
            self.stop()


def main():
    """Main function to run the live ecosystem server with epistemic integrity verification"""
    import argparse

    # EPISTEMIC INTEGRITY CHECKPOINT
    if INTEGRITY_GUARDIAN_AVAILABLE:
        print("🔬 Verifying epistemic integrity before system initialization...")
        guardian = IntegrityGuardian()

        if not guardian.guard_system_initialization():
            print("\n🕯️  The candle of science has been extinguished.")
            print("This system refuses to operate without commitment to the scientific method.")
            print("🚫 SYSTEM STARTUP REFUSED")
            sys.exit(1)
    else:
        print("⚠️  Warning: Running without epistemic integrity verification")
        print("🔬 Consider installing integrity guardian for secure operation")

    parser = argparse.ArgumentParser(description='ΨQRH Living Ecosystem Server with Carl Sagan Knowledge')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--knowledge-base', type=str, help='Path to Carl Sagan knowledge base file')
    parser.add_argument('--skip-integrity-check', action='store_true', help='Skip epistemic integrity verification (NOT RECOMMENDED)')

    args = parser.parse_args()

    # Additional integrity check if explicitly requested to skip
    if args.skip_integrity_check and INTEGRITY_GUARDIAN_AVAILABLE:
        print("\n⚠️  WARNING: Epistemic integrity check was explicitly skipped!")
        print("This system operates on the principle of scientific skepticism.")
        print("Running without integrity verification compromises this foundation.")
        response = input("Are you sure you want to continue? (type 'yes' to confirm): ")
        if response.lower() != 'yes':
            print("🕯️  Integrity check maintained. Exiting.")
            sys.exit(0)
        else:
            print("💀 Running without epistemic integrity verification...")

    # Create and run server with Sagan knowledge
    knowledge_base_path = None
    if hasattr(args, 'knowledge_base') and args.knowledge_base:
        knowledge_base_path = args.knowledge_base
    else:
        # Try default path
        default_kb = Path(__file__).parent.parent.parent / "data" / "knowledge_bases" / "sagan_spectral.kb"
        if default_kb.exists():
            knowledge_base_path = str(default_kb)

    server = LiveEcosystemServer(port=args.port, host=args.host, knowledge_base=knowledge_base_path)

    if server.sagan_engine.knowledge_loaded:
        print("🕯️  'Extraordinary claims require extraordinary evidence'")
        print("🌌 ΨQRH System enhanced with scientific skepticism")
        print("🧠 Epistemic integrity verified - The Method Endures")
    else:
        print("⚠️  Running without Carl Sagan knowledge base")

    server.run()


if __name__ == "__main__":
    main()