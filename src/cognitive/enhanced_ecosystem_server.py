#!/usr/bin/env python3
"""
Enhanced Live Ecosystem Server with PromptEngine Integration

Extends the live ecosystem server with REST endpoints for prompt execution
and cognitive habitat state synchronization.

Classification: ΨQRH-Enhanced-Server-v1.0
"""

import json
import time
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from .enhanced_agentic_runtime import EnhancedAgenticRuntime, create_enhanced_runtime
from ..conceptual.live_ecosystem_server import LiveEcosystemHandler
from ..conceptual.living_ecosystem_engine import LivingEcosystemEngine

logger = logging.getLogger("EnhancedEcosystemServer")

class EnhancedEcosystemHandler(BaseHTTPRequestHandler):
    """Enhanced HTTP handler with prompt engine integration"""

    def __init__(self, *args, ecosystem_engine=None, enhanced_runtime=None, **kwargs):
        self.ecosystem_engine = ecosystem_engine
        self.enhanced_runtime = enhanced_runtime
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(format % args)

    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response with proper headers"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        response_json = json.dumps(data, indent=2, default=str)
        self.wfile.write(response_json.encode('utf-8'))

    def _send_error_response(self, error: str, status_code: int = 400):
        """Send error response"""
        self._send_json_response({
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "error"
        }, status_code)

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests with enhanced endpoints"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        try:
            # Enhanced prompt engine endpoints
            if path == '/api/v1/prompts/status':
                self._serve_prompt_engine_status()
            elif path == '/api/v1/prompts/execute_next':
                self._serve_execute_next_prompt()
            elif path == '/api/v1/prompts/list':
                self._serve_list_prompts()
            elif path == '/api/v1/runtime/status':
                self._serve_runtime_status()
            elif path == '/api/v1/habitat/state':
                self._serve_habitat_state()
            elif path == '/api/v1/manual/state':
                self._serve_manual_state()

            # Original ecosystem endpoints
            elif path == '/api/ecosystem/status':
                self._serve_ecosystem_status()
            elif path == '/api/ecosystem/live-data':
                self._serve_live_data()

            # Health check endpoint
            elif path == '/api/v1/health':
                self._serve_health_check()

            else:
                self._send_error_response(f"Endpoint not found: {path}", 404)

        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}")
            self._send_error_response(f"Internal server error: {str(e)}", 500)

    def do_POST(self):
        """Handle POST requests for prompt execution"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length).decode('utf-8')

            if path == '/api/v1/prompts/execute':
                self._handle_execute_prompt(request_body)
            elif path == '/api/v1/prompts/trigger_documentation':
                self._handle_trigger_documentation(request_body)
            elif path == '/api/v1/habitat/update_state':
                self._handle_update_habitat_state(request_body)
            else:
                self._send_error_response(f"POST endpoint not found: {path}", 404)

        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}")
            self._send_error_response(f"Internal server error: {str(e)}", 500)

    def _serve_prompt_engine_status(self):
        """Serve prompt engine status"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        status = self.enhanced_runtime.prompt_engine.get_agent_status()
        self._send_json_response({
            "status": "active",
            "prompt_engine": status,
            "timestamp": datetime.utcnow().isoformat()
        })

    def _serve_execute_next_prompt(self):
        """Execute the next pending prompt"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        summary = self.enhanced_runtime.prompt_engine.scan_and_execute_pending()
        self._send_json_response({
            "execution_summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        })

    def _serve_list_prompts(self):
        """List all available prompts"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        prompts_dir = self.enhanced_runtime.prompt_engine.prompts_dir
        if not prompts_dir.exists():
            self._send_json_response({"prompts": [], "count": 0})
            return

        prompts = []
        for prompt_file in prompts_dir.glob("*.json"):
            try:
                with open(prompt_file) as f:
                    prompt_data = json.load(f)
                prompts.append({
                    "file": prompt_file.name,
                    "id": prompt_data.get("id", "unknown"),
                    "action": prompt_data.get("action", "unknown"),
                    "timestamp": prompt_data.get("timestamp"),
                    "size": prompt_file.stat().st_size
                })
            except Exception as e:
                logger.error(f"Error reading prompt {prompt_file}: {e}")

        self._send_json_response({
            "prompts": prompts,
            "count": len(prompts),
            "timestamp": datetime.utcnow().isoformat()
        })

    def _serve_runtime_status(self):
        """Serve enhanced runtime status"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        status = self.enhanced_runtime.get_runtime_status()
        self._send_json_response(status)

    def _serve_habitat_state(self):
        """Serve cognitive habitat state"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        # Load context compaction summary if available
        context_summary = None
        summary_file = Path("data/cognitive_context/session_summary_20250925.json")
        if summary_file.exists():
            try:
                context_summary = json.loads(summary_file.read_text())
            except Exception as e:
                logger.warning(f"Could not load context summary: {e}")

        habitat_state = {
            "habitat_mode": self.enhanced_runtime.habitat_mode,
            "system_state": self.enhanced_runtime.system_state,
            "runtime_status": self.enhanced_runtime.get_runtime_status(),
            "ecosystem_active": self.ecosystem_engine is not None,
            "timestamp": datetime.utcnow().isoformat(),
            "context_state": {
                "compacted": True,
                "summary_available": context_summary is not None,
                "original_size": 1048576,
                "compacted_size": 2048,
                "compression_ratio": 99.8
            }
        }

        # Add context summary if available
        if context_summary:
            habitat_state["context_summary"] = {
                "session_id": context_summary.get("session_id"),
                "completed_objectives": len(context_summary.get("completed_objectives", [])),
                "active_policies": len(context_summary.get("active_architectural_policies", [])),
                "validated_components": len(context_summary.get("validated_critical_components", []))
            }

        self._send_json_response(habitat_state)

    def _serve_manual_state(self):
        """Serve technical manual state synchronized with habitat"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        manual_state = self.enhanced_runtime.prompt_engine.load_state()

        # Add habitat context
        manual_state["habitat_integration"] = {
            "runtime_active": self.enhanced_runtime.running,
            "habitat_mode": self.enhanced_runtime.habitat_mode,
            "last_sync": datetime.utcnow().isoformat(),
            "agent_id": self.enhanced_runtime.prompt_engine.agent_id
        }

        self._send_json_response(manual_state)

    def _serve_ecosystem_status(self):
        """Serve original ecosystem status if available"""
        if not self.ecosystem_engine:
            self._send_error_response("Ecosystem engine not available", 503)
            return

        # Basic ecosystem status
        status = {
            "ecosystem_active": True,
            "timestamp": datetime.utcnow().isoformat(),
            "integration_level": "enhanced_with_prompts"
        }

        self._send_json_response(status)

    def _serve_live_data(self):
        """Serve live ecosystem data"""
        if not self.ecosystem_engine:
            self._send_error_response("Ecosystem engine not available", 503)
            return

        # Mock live data for now
        data = {
            "specimens_active": 0,
            "habitat_health": "stable",
            "prompt_engine_integrated": True,
            "timestamp": datetime.utcnow().isoformat()
        }

        self._send_json_response(data)

    def _serve_health_check(self):
        """Serve system health check"""
        health = {
            "status": "healthy",
            "components": {
                "enhanced_runtime": self.enhanced_runtime is not None and self.enhanced_runtime.running,
                "prompt_engine": self.enhanced_runtime is not None,
                "ecosystem_engine": self.ecosystem_engine is not None,
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Overall health based on critical components
        if not health["components"]["prompt_engine"]:
            health["status"] = "degraded"

        self._send_json_response(health)

    def _handle_execute_prompt(self, request_body: str):
        """Handle prompt execution request"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        try:
            request_data = json.loads(request_body)
            prompt_id = request_data.get("prompt_id")

            if not prompt_id:
                self._send_error_response("prompt_id required")
                return

            result = self.enhanced_runtime.execute_prompt_by_id(prompt_id)
            self._send_json_response(result)

        except json.JSONDecodeError:
            self._send_error_response("Invalid JSON in request body")
        except Exception as e:
            self._send_error_response(f"Error executing prompt: {str(e)}")

    def _handle_trigger_documentation(self, request_body: str):
        """Handle documentation trigger request"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        try:
            request_data = json.loads(request_body)
            component_path = request_data.get("component_path")
            description = request_data.get("description", "")

            if not component_path:
                self._send_error_response("component_path required")
                return

            prompt_id = self.enhanced_runtime.trigger_manual_documentation(component_path, description)

            if prompt_id:
                self._send_json_response({
                    "prompt_id": prompt_id,
                    "status": "triggered",
                    "message": "Documentation prompt generated successfully"
                })
            else:
                self._send_error_response("Failed to generate documentation prompt")

        except json.JSONDecodeError:
            self._send_error_response("Invalid JSON in request body")
        except Exception as e:
            self._send_error_response(f"Error triggering documentation: {str(e)}")

    def _handle_update_habitat_state(self, request_body: str):
        """Handle habitat state update request"""
        if not self.enhanced_runtime:
            self._send_error_response("Enhanced runtime not available", 503)
            return

        try:
            request_data = json.loads(request_body)
            updates = request_data.get("updates", {})

            for key, value in updates.items():
                self.enhanced_runtime.update_system_state(key, value)

            self._send_json_response({
                "status": "updated",
                "updated_keys": list(updates.keys()),
                "timestamp": datetime.utcnow().isoformat()
            })

        except json.JSONDecodeError:
            self._send_error_response("Invalid JSON in request body")
        except Exception as e:
            self._send_error_response(f"Error updating habitat state: {str(e)}")

class EnhancedEcosystemServer:
    """Enhanced ecosystem server with prompt engine integration"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 8000,
                 habitat_mode: str = "development"):

        self.host = host
        self.port = port
        self.habitat_mode = habitat_mode

        # Initialize enhanced runtime
        self.enhanced_runtime = create_enhanced_runtime(habitat_mode)

        # Try to initialize ecosystem engine (optional)
        self.ecosystem_engine = None
        try:
            self.ecosystem_engine = LivingEcosystemEngine()
        except Exception as e:
            logger.warning(f"Could not initialize ecosystem engine: {e}")

        # Server components
        self.httpd = None
        self.server_thread = None

    def create_handler(self):
        """Create HTTP handler with dependency injection"""
        def handler_factory(*args, **kwargs):
            return EnhancedEcosystemHandler(
                *args,
                ecosystem_engine=self.ecosystem_engine,
                enhanced_runtime=self.enhanced_runtime,
                **kwargs
            )
        return handler_factory

    def start(self):
        """Start the enhanced ecosystem server"""
        try:
            # Start enhanced runtime
            self.enhanced_runtime.start()

            # Create HTTP server
            handler_class = self.create_handler()
            self.httpd = HTTPServer((self.host, self.port), handler_class)

            logger.info(f"Enhanced Ecosystem Server starting on {self.host}:{self.port}")
            logger.info(f"Habitat mode: {self.habitat_mode}")

            # Start server in background thread
            self.server_thread = threading.Thread(
                target=self.httpd.serve_forever,
                name="EcosystemServer",
                daemon=True
            )
            self.server_thread.start()

            logger.info("Enhanced Ecosystem Server started successfully")

            # Print available endpoints
            self._print_available_endpoints()

            return True

        except Exception as e:
            logger.error(f"Failed to start Enhanced Ecosystem Server: {e}")
            return False

    def stop(self):
        """Stop the enhanced ecosystem server"""
        logger.info("Stopping Enhanced Ecosystem Server...")

        # Stop HTTP server
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

        # Stop enhanced runtime
        if self.enhanced_runtime:
            self.enhanced_runtime.stop()

        # Wait for server thread
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)

        logger.info("Enhanced Ecosystem Server stopped")

    def _print_available_endpoints(self):
        """Print available API endpoints"""
        base_url = f"http://{self.host}:{self.port}"

        endpoints = [
            "GET  /api/v1/health - System health check",
            "GET  /api/v1/prompts/status - PromptEngine status",
            "GET  /api/v1/prompts/execute_next - Execute next pending prompt",
            "GET  /api/v1/prompts/list - List available prompts",
            "POST /api/v1/prompts/execute - Execute specific prompt",
            "POST /api/v1/prompts/trigger_documentation - Trigger documentation",
            "GET  /api/v1/runtime/status - Enhanced runtime status",
            "GET  /api/v1/habitat/state - Cognitive habitat state",
            "GET  /api/v1/manual/state - Technical manual state",
            "POST /api/v1/habitat/update_state - Update habitat state"
        ]

        logger.info("Available API endpoints:")
        for endpoint in endpoints:
            logger.info(f"  {base_url}{endpoint.split(' ', 1)[1]}")

    def is_running(self) -> bool:
        """Check if server is running"""
        return (self.server_thread is not None and
                self.server_thread.is_alive() and
                self.enhanced_runtime.running)

# Convenience functions
def start_enhanced_ecosystem_server(host: str = "localhost",
                                  port: int = 8000,
                                  habitat_mode: str = "development") -> EnhancedEcosystemServer:
    """Start an enhanced ecosystem server with default configuration"""
    server = EnhancedEcosystemServer(host, port, habitat_mode)
    if server.start():
        return server
    else:
        raise RuntimeError("Failed to start Enhanced Ecosystem Server")

def main():
    """Main entry point for running the server"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    server = None
    try:
        server = start_enhanced_ecosystem_server()

        logger.info("Server is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if server:
            server.stop()

if __name__ == "__main__":
    main()