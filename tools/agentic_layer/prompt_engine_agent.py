#!/usr/bin/env python3
"""
PromptEngine Cognitive Agent Integration

Integrates the self-managing prompt engine with the ΨQRH cognitive runtime,
enabling automatic documentation and reactive system behavior.

Classification: ΨQRH-Cognitive-Integration-v1.0
"""

import json
import os
import uuid
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import hashlib

from .navigator_agent import NavigatorAgent
# Note: ΨQRH import removed - agentic layer is now decoupled from core

@dataclass
class PromptExecutionContext:
    """Context for prompt execution with cognitive awareness"""
    prompt_id: str
    action: str
    dependencies: List[str]
    habitat_mode: str = "development"
    production_safe: bool = False
    cognitive_priority: int = 0
    expected_outcome: str = ""

class PromptEngineAgent:
    """
    Cognitive Agent wrapper for the prompt engine system

    This agent bridges the self-managing prompt system with the ΨQRH
    cognitive runtime, providing:
    - Navigator integration for execution tracking
    - Audit logging for all prompt operations
    - Production safety filtering
    - Reactive documentation capabilities
    """

    def __init__(self,
                 manual_dir: str = "construction_technical_manual",
                 audit_log_path: str = "data/audit_logs/prompt_engine.jsonl",
                 habitat_mode: str = "development"):

        self.manual_dir = Path(manual_dir)
        self.prompts_dir = self.manual_dir / "prompts"
        self.state_file = self.manual_dir / "state.json"
        self.manual_file = self.manual_dir / "manual.md"
        self.structure_file = Path("estrutura_diretorios.txt")

        # Cognitive integration
        self.navigator = NavigatorAgent()
        self.audit_log_path = Path(audit_log_path)
        self.habitat_mode = habitat_mode

        # Ensure directories exist
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

        # Agent state
        self.agent_id = str(uuid.uuid4())
        self.execution_count = 0

        print(f"🤖 PromptEngine Agent initialized (ID: {self.agent_id[:8]})")

    def _validate_output_path(self, prompt: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate output paths mentioned in a prompt against architectural policies."""
        # This is a simplified check. A more robust implementation would traverse the prompt structure.
        for key, value in prompt.items():
            if isinstance(value, str) and ('test_' in value or '_test.py' in value) and value.endswith('.py'):
                # This looks like a test file path
                path = Path(value)
                # Check if the path is inside the 'tests/' directory.
                # We assume the path is relative to the project root.
                if not str(path).startswith('tests/'):
                    error_msg = f"Architectural policy violation: Test file '{value}' must be in 'tests/' directory."
                    return False, error_msg
        return True, None

    def load_state(self) -> Dict[str, Any]:
        """Load current manual construction state"""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {
            "executed_prompts": [],
            "manual_version": "1.0.0",
            "system_status": "active",
            "last_execution": None
        }

    def save_state(self, state: Dict[str, Any]):
        """Save updated state with cognitive context"""
        state["last_execution"] = datetime.utcnow().isoformat()

        # Add context compaction tracking
        if "context_compactions" not in state:
            state["context_compactions"] = []

        state["context_compactions"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "context_compaction",
            "summary_file": f"data/cognitive_context/session_summary_{datetime.utcnow().strftime('%Y%m%d')}.json"
        })
        state["agent_id"] = self.agent_id
        state["execution_count"] = self.execution_count

        self.state_file.write_text(json.dumps(state, indent=2))

    def clear_context_buffer(self):
        """Clear cognitive agent context buffer, preserving only summary"""
        # Clear execution tracking while preserving essential state
        state = self.load_state()

        # Preserve only essential information
        preserved_state = {
            "manual_version": state.get("manual_version", "1.0.0"),
            "system_status": "active_compacted",
            "last_execution": datetime.utcnow().isoformat(),
            "context_compactions": state.get("context_compactions", []),
            "agent_id": self.agent_id,
            "execution_count": self.execution_count,
            "compaction_timestamp": datetime.utcnow().isoformat()
        }

        # Save compacted state
        self.state_file.write_text(json.dumps(preserved_state, indent=2))

        # Log the context compaction
        self._log_audit_event(
            action_type="context_compaction",
            prompt_id="system_context_compaction",
            outcome="success",
            artifacts_generated=["session_summary_20250925.json"],
            error=None
        )

        print(f"🧹 Cognitive context buffer cleared. Context compacted to summary.")

    def log_audit_entry(self,
                       prompt_id: str,
                       action_type: str,
                       outcome: str,
                       artifacts: List[str] = None,
                       error: str = None):
        """Log prompt execution to audit system with cognitive metadata"""

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": "PromptEngine",
            "agent_id": self.agent_id,
            "action_type": action_type,
            "prompt_id": prompt_id,
            "outcome": outcome,
            "artifacts_generated": artifacts or [],
            "habitat_mode": self.habitat_mode,
            "execution_count": self.execution_count,
            "continuity_seal": "Ω∞Ω",  # Standard ΨQRH seal
        }

        if error:
            entry["error"] = error

        # Add cognitive context from navigator
        if hasattr(self.navigator, 'dyad_mode'):
            entry["active_dyad"] = self.navigator.dyad_mode
            entry["RG"] = self.navigator.target_rg

        # Calculate SHA256 for audit chain
        entry_str = json.dumps(entry, sort_keys=True)
        entry["entry_sha256"] = hashlib.sha256(entry_str.encode()).hexdigest()

        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def update_directory_structure(self):
        """Update directory structure with navigator oversight"""
        try:
            result = subprocess.run(["tree", "-d", "-L", "3"],
                                  capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                self.structure_file.write_text(result.stdout)
                print(f"✅ Navigator: Updated {self.structure_file}")
                return True
            else:
                print(f"⚠️ Navigator: tree command failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Navigator: Failed to update structure: {e}")
            return False

    def is_production_safe(self, prompt: Dict[str, Any]) -> bool:
        """Determine if prompt is safe for production execution"""

        # Check explicit production safety flag
        if prompt.get("production_safe", False):
            return True

        # Check action type safety
        safe_actions = [
            "document_component",
            "validate_system",
            "generate_report",
            "update_structure"
        ]

        if prompt.get("action") in safe_actions:
            return True

        # Check for dangerous operations
        dangerous_keywords = [
            "delete", "remove", "destroy", "overwrite",
            "system_restart", "production_deploy", "migrate"
        ]

        instructions = prompt.get("instructions", "").lower()
        for keyword in dangerous_keywords:
            if keyword in instructions:
                return False

        return False

    def execute_prompt(self, prompt_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Execute a single prompt with full cognitive integration"""

        try:
            # Load and validate prompt for placeholders
            prompt_content = prompt_path.read_text()
            if '{' in prompt_content or '}' in prompt_content:
                error_msg = f"Prompt {prompt_path.name} contains unresolved placeholders and cannot be executed."
                self.log_audit_entry(prompt_path.stem, "prompt_execution", "blocked_placeholder_violation", error=error_msg)
                return False, {"error": error_msg}

            prompt = json.loads(prompt_content)
            prompt_id = prompt.get("id", "unknown")

            # Architectural policy validation
            is_valid, error_msg = self._validate_output_path(prompt)
            if not is_valid:
                self.log_audit_entry(prompt_id, "prompt_execution", "blocked_arch_violation", error=error_msg)
                return False, {"error": error_msg}

            print(f"🚀 PromptEngine: Executing {prompt_id}")

            # Create execution context
            context = PromptExecutionContext(
                prompt_id=prompt_id,
                action=prompt.get("action", "unknown"),
                dependencies=prompt.get("dependencies", []),
                habitat_mode=self.habitat_mode,
                production_safe=self.is_production_safe(prompt),
                expected_outcome=prompt.get("expected_outcome", "")
            )

            # Production safety check
            if self.habitat_mode == "production" and not context.production_safe:
                error_msg = f"Prompt {prompt_id} not production-safe, execution blocked"
                self.log_audit_entry(prompt_id, "prompt_execution", "blocked_unsafe", error=error_msg)
                return False, {"error": error_msg, "context": context}

            # Pre-execution Navigator check
            if not self.navigator.pre_execution_check(prompt):
                error_msg = "Navigator pre-execution check failed"
                self.log_audit_entry(prompt_id, "prompt_execution", "failed_precheck", error=error_msg)
                return False, {"error": error_msg, "context": context}

            start_time = datetime.utcnow()
            artifacts_generated = []

            # Execute based on action type
            if prompt["action"] == "document_component":
                success, artifacts = self._execute_documentation(prompt)
                artifacts_generated.extend(artifacts)

            elif prompt["action"] == "execute_validation_suite":
                success, artifacts = self._execute_validation(prompt)
                artifacts_generated.extend(artifacts)

            elif prompt["action"] == "integrate_systems":
                success, artifacts = self._execute_integration(prompt)
                artifacts_generated.extend(artifacts)

            else:
                # Generic execution - append to manual
                success, artifacts = self._execute_generic(prompt)
                artifacts_generated.extend(artifacts)

            # Post-execution hooks
            if prompt.get("post_execution_hook") == "update_directory_structure":
                if self.update_directory_structure():
                    artifacts_generated.append("estrutura_diretorios.txt")

            # Auto-delete if requested
            if prompt.get("auto_delete", False):
                prompt_path.unlink()
                print(f"🗑️ Auto-deleted {prompt_path.name}")

            # Update execution count and state
            self.execution_count += 1
            state = self.load_state()
            state["executed_prompts"].append({
                "id": prompt_id,
                "timestamp": start_time.isoformat(),
                "status": "completed" if success else "failed",
                "artifacts": artifacts_generated
            })
            self.save_state(state)

            # Log successful execution
            outcome = "success" if success else "failed"
            self.log_audit_entry(prompt_id, "prompt_execution", outcome, artifacts_generated)

            return success, {
                "context": context,
                "artifacts": artifacts_generated,
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            }

        except Exception as e:
            error_msg = f"Prompt execution failed: {str(e)}"
            self.log_audit_entry(prompt.get("id", "unknown"), "prompt_execution", "error", error=error_msg)
            return False, {"error": error_msg}

    def _execute_documentation(self, prompt: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Execute documentation prompt"""
        try:
            output_section = prompt.get("output_section", "Generated Documentation")
            target_file = prompt.get("target_file", "unknown")

            with open(self.manual_file, "a") as f:
                f.write(f"\n## {output_section}\n")
                f.write(f"**Target File**: `{target_file}`\n\n")
                f.write(f"{prompt['instructions']}\n\n")
                f.write("---\n\n")

            return True, [str(self.manual_file)]

        except Exception as e:
            print(f"❌ Documentation execution failed: {e}")
            return False, []

    def _execute_validation(self, prompt: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Execute validation suite prompt"""
        # This would trigger the validation system
        # For now, just document the request
        return self._execute_documentation(prompt)

    def _execute_integration(self, prompt: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Execute system integration prompt"""
        # This would trigger integration processes
        # For now, just document the request
        return self._execute_documentation(prompt)

    def _execute_generic(self, prompt: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Generic prompt execution"""
        return self._execute_documentation(prompt)

    def scan_and_execute_pending(self) -> Dict[str, Any]:
        """Scan for pending prompts and execute them"""

        if not self.prompts_dir.exists():
            return {"executed": 0, "failed": 0, "results": []}

        results = []
        executed = 0
        failed = 0

        # Get all JSON prompt files
        prompt_files = list(self.prompts_dir.glob("*.json"))
        prompt_files.sort()  # Execute in alphabetical order

        for prompt_file in prompt_files:
            success, result = self.execute_prompt(prompt_file)

            results.append({
                "file": prompt_file.name,
                "success": success,
                "result": result
            })

            if success:
                executed += 1
            else:
                failed += 1

        summary = {
            "executed": executed,
            "failed": failed,
            "total_prompts": len(prompt_files),
            "results": results
        }

        print(f"📊 PromptEngine: {executed} executed, {failed} failed of {len(prompt_files)} prompts")
        return summary

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status for habitat integration"""
        state = self.load_state()

        return {
            "agent_id": self.agent_id,
            "agent_type": "PromptEngine",
            "habitat_mode": self.habitat_mode,
            "execution_count": self.execution_count,
            "system_status": state.get("system_status", "active"),
            "last_execution": state.get("last_execution"),
            "pending_prompts": len(list(self.prompts_dir.glob("*.json"))) if self.prompts_dir.exists() else 0,
            "manual_sections": len(state.get("manual_sections", [])),
            "audit_log_path": str(self.audit_log_path)
        }

# Factory function for easy instantiation
def create_prompt_engine_agent(habitat_mode: str = "development") -> PromptEngineAgent:
    """Create a PromptEngine agent with cognitive integration"""
    return PromptEngineAgent(habitat_mode=habitat_mode)