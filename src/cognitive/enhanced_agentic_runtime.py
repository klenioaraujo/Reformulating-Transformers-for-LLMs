#!/usr/bin/env python3
"""
Enhanced Agentic Runtime with PromptEngine Integration

Extends the base agentic runtime with reactive prompt execution capabilities,
allowing the system to automatically generate documentation and respond to
changes in the cognitive habitat.

Classification: ΨQRH-Enhanced-Runtime-v1.0
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import threading
import queue
import logging

from .agentic_runtime import *
from .prompt_engine_agent import PromptEngineAgent, create_prompt_engine_agent

logger = logging.getLogger("EnhancedAgenticRuntime")

class ReactivePromptTrigger:
    """Defines conditions that trigger automatic prompt generation"""

    def __init__(self,
                 trigger_id: str,
                 condition: Callable[[Dict[str, Any]], bool],
                 prompt_template: Dict[str, Any],
                 priority: int = 0):
        self.trigger_id = trigger_id
        self.condition = condition
        self.prompt_template = prompt_template
        self.priority = priority
        self.last_triggered = None
        self.trigger_count = 0

class EnhancedAgenticRuntime:
    """
    Enhanced runtime that combines cognitive processing with reactive documentation

    This runtime extends the base agentic runtime with:
    - Automatic prompt generation and execution
    - Reactive documentation based on system changes
    - Integration with the cognitive habitat
    - Production-safe operation modes
    """

    def __init__(self,
                 habitat_mode: str = "development",
                 auto_documentation: bool = True,
                 prompt_execution_interval: float = 30.0):

        # Initialize base runtime components
        self.habitat_mode = habitat_mode
        self.auto_documentation = auto_documentation
        self.prompt_execution_interval = prompt_execution_interval

        # Initialize prompt engine agent
        self.prompt_engine = create_prompt_engine_agent(habitat_mode)

        # Reactive system state
        self.reactive_triggers = {}
        self.system_state = {}
        self.change_queue = queue.Queue()
        self.running = False

        # Background thread for prompt processing
        self.prompt_thread = None

        # Initialize built-in triggers
        self._setup_builtin_triggers()

        logger.info(f"Enhanced Agentic Runtime initialized (mode: {habitat_mode})")

    def _setup_builtin_triggers(self):
        """Setup built-in reactive triggers"""

        # Trigger documentation when new components are detected
        self.add_reactive_trigger(
            "new_component_detected",
            lambda state: self._check_new_components(state),
            {
                "action": "document_component",
                "auto_delete": True,
                "production_safe": True,
                "priority": "high"
            }
        )

        # Trigger validation when core components change
        self.add_reactive_trigger(
            "core_system_changed",
            lambda state: self._check_core_changes(state),
            {
                "action": "execute_validation_suite",
                "auto_delete": True,
                "production_safe": False,  # Validation might affect production
                "priority": "critical"
            }
        )

        # Trigger structure update when directories change
        self.add_reactive_trigger(
            "directory_structure_changed",
            lambda state: self._check_directory_changes(state),
            {
                "action": "update_structure",
                "post_execution_hook": "update_directory_structure",
                "auto_delete": True,
                "production_safe": True,
                "priority": "low"
            }
        )

    def _check_new_components(self, state: Dict[str, Any]) -> bool:
        """Check if new components have been added to src/"""
        # This would scan src/ directories for new files
        # For now, return False (no new components detected)
        return False

    def _check_core_changes(self, state: Dict[str, Any]) -> bool:
        """Check if core ΨQRH components have changed"""
        # This would check modification times of core files
        # For now, return False
        return False

    def _check_directory_changes(self, state: Dict[str, Any]) -> bool:
        """Check if directory structure has changed"""
        # This would compare current structure to cached version
        # For now, return False
        return False

    def add_reactive_trigger(self,
                           trigger_id: str,
                           condition: Callable[[Dict[str, Any]], bool],
                           prompt_template: Dict[str, Any],
                           priority: int = 0):
        """Add a reactive trigger for automatic prompt generation"""

        trigger = ReactivePromptTrigger(
            trigger_id=trigger_id,
            condition=condition,
            prompt_template=prompt_template,
            priority=priority
        )

        self.reactive_triggers[trigger_id] = trigger
        logger.info(f"Added reactive trigger: {trigger_id}")

    def remove_reactive_trigger(self, trigger_id: str):
        """Remove a reactive trigger"""
        if trigger_id in self.reactive_triggers:
            del self.reactive_triggers[trigger_id]
            logger.info(f"Removed reactive trigger: {trigger_id}")

    def update_system_state(self, key: str, value: Any):
        """Update system state and check for reactive triggers"""
        old_value = self.system_state.get(key)
        self.system_state[key] = value

        if old_value != value:
            # State change detected, queue for evaluation
            self.change_queue.put({
                "timestamp": datetime.utcnow(),
                "key": key,
                "old_value": old_value,
                "new_value": value
            })

            logger.debug(f"State change: {key} = {value}")

    def _evaluate_reactive_triggers(self):
        """Evaluate all reactive triggers against current state"""
        triggered_prompts = []

        for trigger_id, trigger in self.reactive_triggers.items():
            try:
                if trigger.condition(self.system_state):
                    # Trigger condition met, generate prompt
                    prompt = self._generate_prompt_from_trigger(trigger)
                    if prompt:
                        triggered_prompts.append((trigger, prompt))

                    trigger.last_triggered = datetime.utcnow()
                    trigger.trigger_count += 1

            except Exception as e:
                logger.error(f"Error evaluating trigger {trigger_id}: {e}")

        return triggered_prompts

    def _generate_prompt_from_trigger(self, trigger: ReactivePromptTrigger) -> Optional[Dict[str, Any]]:
        """Generate a prompt from a reactive trigger"""
        try:
            # Create unique prompt ID
            prompt_id = f"{trigger.trigger_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

            # Build prompt from template
            prompt = {
                "id": prompt_id,
                "version": "1.0",
                "timestamp": datetime.utcnow().isoformat(),
                "action": trigger.prompt_template.get("action", "document_component"),
                "auto_delete": trigger.prompt_template.get("auto_delete", True),
                "dependencies": [],
                "trigger_source": trigger.trigger_id,
                "generated_by": "EnhancedAgenticRuntime",
                **trigger.prompt_template
            }

            # Add reactive context
            prompt["instructions"] = f"""
Reactive prompt generated by trigger: {trigger.trigger_id}

System state at trigger time:
{json.dumps(self.system_state, indent=2)}

Original template: {trigger.prompt_template.get('action', 'document_component')}

This prompt was automatically generated in response to system changes.
Execute the appropriate action based on the current state and trigger context.
"""

            return prompt

        except Exception as e:
            logger.error(f"Error generating prompt from trigger {trigger.trigger_id}: {e}")
            return None

    def _save_generated_prompt(self, prompt: Dict[str, Any]) -> bool:
        """Save a generated prompt to the prompts directory"""
        try:
            prompt_file = self.prompt_engine.prompts_dir / f"{prompt['id']}.json"

            with open(prompt_file, 'w') as f:
                json.dump(prompt, f, indent=2)

            logger.info(f"Generated prompt saved: {prompt_file.name}")
            return True

        except Exception as e:
            logger.error(f"Error saving prompt {prompt['id']}: {e}")
            return False

    def _background_prompt_processor(self):
        """Background thread for processing prompts and triggers"""
        logger.info("Background prompt processor started")

        while self.running:
            try:
                # Process any state changes
                while not self.change_queue.empty():
                    try:
                        change = self.change_queue.get_nowait()
                        logger.debug(f"Processing state change: {change}")

                        # Evaluate triggers after state change
                        triggered_prompts = self._evaluate_reactive_triggers()

                        # Generate and save triggered prompts
                        for trigger, prompt in triggered_prompts:
                            if self._save_generated_prompt(prompt):
                                logger.info(f"Reactive prompt generated: {prompt['id']}")

                    except queue.Empty:
                        break

                # Execute any pending prompts
                if self.auto_documentation:
                    execution_summary = self.prompt_engine.scan_and_execute_pending()
                    if execution_summary["executed"] > 0:
                        logger.info(f"Executed {execution_summary['executed']} prompts")

                # Wait before next iteration
                time.sleep(self.prompt_execution_interval)

            except Exception as e:
                logger.error(f"Error in background prompt processor: {e}")
                time.sleep(5)  # Wait before retrying

        logger.info("Background prompt processor stopped")

    def start(self):
        """Start the enhanced runtime"""
        if self.running:
            logger.warning("Runtime already running")
            return

        self.running = True

        # Start background prompt processing thread
        self.prompt_thread = threading.Thread(
            target=self._background_prompt_processor,
            name="PromptProcessor",
            daemon=True
        )
        self.prompt_thread.start()

        logger.info("Enhanced Agentic Runtime started")

    def stop(self):
        """Stop the enhanced runtime"""
        if not self.running:
            return

        self.running = False

        # Wait for background thread to finish
        if self.prompt_thread and self.prompt_thread.is_alive():
            self.prompt_thread.join(timeout=5)

        logger.info("Enhanced Agentic Runtime stopped")

    def execute_prompt_by_id(self, prompt_id: str) -> Dict[str, Any]:
        """Execute a specific prompt by ID"""
        prompt_file = self.prompt_engine.prompts_dir / f"{prompt_id}.json"

        if not prompt_file.exists():
            return {"error": f"Prompt {prompt_id} not found"}

        success, result = self.prompt_engine.execute_prompt(prompt_file)

        return {
            "prompt_id": prompt_id,
            "success": success,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

    def get_runtime_status(self) -> Dict[str, Any]:
        """Get comprehensive runtime status"""
        prompt_agent_status = self.prompt_engine.get_agent_status()

        return {
            "runtime_type": "EnhancedAgenticRuntime",
            "habitat_mode": self.habitat_mode,
            "running": self.running,
            "auto_documentation": self.auto_documentation,
            "reactive_triggers": {
                trigger_id: {
                    "priority": trigger.priority,
                    "last_triggered": trigger.last_triggered.isoformat() if trigger.last_triggered else None,
                    "trigger_count": trigger.trigger_count
                }
                for trigger_id, trigger in self.reactive_triggers.items()
            },
            "system_state_keys": list(self.system_state.keys()),
            "change_queue_size": self.change_queue.qsize(),
            "prompt_engine": prompt_agent_status
        }

    def trigger_manual_documentation(self, component_path: str, description: str = "") -> str:
        """Manually trigger documentation for a specific component"""
        prompt_id = f"manual_doc_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        prompt = {
            "id": prompt_id,
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "action": "document_component",
            "target_file": component_path,
            "output_section": f"Manual Documentation: {component_path}",
            "auto_delete": True,
            "production_safe": True,
            "instructions": f"""
Generate technical documentation for the component at: {component_path}

{description}

This documentation was manually triggered and should include:
- Component purpose and functionality
- Key interfaces and dependencies
- Usage examples if applicable
- Integration points with other system components

Generated at: {datetime.utcnow().isoformat()}
            """.strip()
        }

        if self._save_generated_prompt(prompt):
            logger.info(f"Manual documentation prompt created: {prompt_id}")
            return prompt_id
        else:
            return None

# Factory function for easy instantiation
def create_enhanced_runtime(habitat_mode: str = "development") -> EnhancedAgenticRuntime:
    """Create an enhanced agentic runtime with prompt integration"""
    return EnhancedAgenticRuntime(habitat_mode=habitat_mode)