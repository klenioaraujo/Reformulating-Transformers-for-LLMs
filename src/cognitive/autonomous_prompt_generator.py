#!/usr/bin/env python3
"""
Autonomous Prompt Generator for ΨQRH System

Monitors file system changes and automatically generates prompts for documentation,
validation, and maintenance tasks. Integrates with the Enhanced Agentic Runtime
to provide self-maintaining capabilities.

Classification: ΨQRH-Autonomous-Generator-v1.0
"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import threading
import queue
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .prompt_engine_agent import PromptEngineAgent

logger = logging.getLogger("AutonomousPromptGenerator")

class ChangeDetector:
    """Tracks file hashes to detect actual content changes"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.hash_cache_path = project_root / "data" / "system_state" / "file_hashes.json"
        self.hash_cache = self._load_hash_cache()

    def _load_hash_cache(self) -> Dict[str, str]:
        """Load existing hash cache or create new one"""
        self.hash_cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.hash_cache_path.exists():
            try:
                with open(self.hash_cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load hash cache: {e}")

        return {}

    def _save_hash_cache(self):
        """Save hash cache to disk"""
        try:
            with open(self.hash_cache_path, 'w') as f:
                json.dump(self.hash_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save hash cache: {e}")

    def _calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calculate SHA256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Could not calculate hash for {file_path}: {e}")
            return None

    def detect_change(self, file_path: Path) -> Dict[str, Any]:
        """Detect if file has actually changed and return change info"""
        file_str = str(file_path.relative_to(self.project_root))
        current_hash = self._calculate_file_hash(file_path)

        if current_hash is None:
            return {"changed": False, "error": "Could not calculate hash"}

        previous_hash = self.hash_cache.get(file_str)
        changed = previous_hash != current_hash

        # Update cache
        self.hash_cache[file_str] = current_hash
        if changed:
            self._save_hash_cache()

        return {
            "changed": changed,
            "previous_hash": previous_hash,
            "current_hash": current_hash,
            "file_path": file_str,
            "is_new": previous_hash is None
        }

class PrioritySystem:
    """Determines priority based on file location and type"""

    PRIORITY_RULES = {
        "src/core/": 90,
        "src/fractal/": 80,
        "src/cognitive/": 85,
        "src/conceptual/": 60,
        "construction_technical_manual/": 70,
        "experiments/": 30,
        "tests/": 40,
        "docs/": 50
    }

    FILE_TYPE_MODIFIERS = {
        ".py": 10,
        ".md": 5,
        ".json": 8,
        ".yaml": 6,
        ".txt": 2
    }

    @classmethod
    def calculate_priority(cls, file_path: Path, change_type: str = "modification") -> int:
        """Calculate priority score for a file change"""
        base_priority = 50  # default
        file_str = str(file_path)

        # Directory-based priority
        for dir_pattern, priority in cls.PRIORITY_RULES.items():
            if dir_pattern in file_str:
                base_priority = priority
                break

        # File type modifier
        suffix = file_path.suffix
        if suffix in cls.FILE_TYPE_MODIFIERS:
            base_priority += cls.FILE_TYPE_MODIFIERS[suffix]

        # Change type modifier
        if change_type == "creation":
            base_priority += 15
        elif change_type == "deletion":
            base_priority += 20

        return min(100, max(1, base_priority))

class PromptTemplate:
    """Manages prompt template loading and variable substitution"""

    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load all available prompt templates"""
        templates = {}

        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return templates

        for template_file in self.templates_dir.glob("*.tpl.json"):
            try:
                with open(template_file, 'r') as f:
                    template_name = template_file.stem
                    templates[template_name] = json.load(f)
                    logger.debug(f"Loaded template: {template_name}")
            except Exception as e:
                logger.error(f"Could not load template {template_file}: {e}")

        return templates

    def generate_prompt(self, template_name: str, variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a prompt from template with variable substitution"""
        if template_name not in self.templates:
            logger.error(f"Template not found: {template_name}")
            return None

        try:
            template = self.templates[template_name].copy()
            return self._substitute_variables(template, variables)
        except Exception as e:
            logger.error(f"Error generating prompt from template {template_name}: {e}")
            return None

    def _substitute_variables(self, obj: Any, variables: Dict[str, Any]) -> Any:
        """Recursively substitute variables in template"""
        if isinstance(obj, dict):
            return {k: self._substitute_variables(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_variables(item, variables) for item in obj]
        elif isinstance(obj, str):
            # Simple variable substitution
            for var_name, var_value in variables.items():
                placeholder = "{" + var_name + "}"
                if placeholder in obj:
                    obj = obj.replace(placeholder, str(var_value))
            return obj
        else:
            return obj

class FileSystemWatcher(FileSystemEventHandler):
    """Watches file system for changes and triggers prompt generation"""

    def __init__(self, generator: 'AutonomousPromptGenerator'):
        self.generator = generator
        super().__init__()

    def on_created(self, event):
        if not event.is_directory:
            self.generator.handle_file_change(Path(event.src_path), "creation")

    def on_modified(self, event):
        if not event.is_directory:
            self.generator.handle_file_change(Path(event.src_path), "modification")

    def on_deleted(self, event):
        if not event.is_directory:
            self.generator.handle_file_change(Path(event.src_path), "deletion")

class AutonomousPromptGenerator:
    """
    Autonomous prompt generation system for ΨQRH maintenance

    Monitors file system changes and automatically generates appropriate
    prompts for documentation, validation, and maintenance tasks.
    """

    def __init__(self, project_root: Path, prompt_engine: PromptEngineAgent):
        self.project_root = project_root
        self.prompt_engine = prompt_engine
        self.change_detector = ChangeDetector(project_root)
        self.prompt_templates = PromptTemplate(project_root / "construction_technical_manual" / "templates")

        # File system monitoring
        self.observer = Observer()
        self.handler = FileSystemWatcher(self)

        # Queue for prompt generation
        self.prompt_queue = queue.PriorityQueue()
        self.generation_thread = None
        self.running = False

        # Ignore patterns
        self.ignore_patterns = {
            '.git/',
            '__pycache__/',
            '.pytest_cache/',
            'data/audit/',
            'data/validation_reports/',
            '.venv/',
            'venv/',
            'node_modules/',
            '.DS_Store'
        }

        # Architectural validation rules
        self.architectural_rules = {
            "test_files_location": {
                "patterns": ["test_*.py", "*_test.py", "*test*.py"],
                "required_directory": "tests/",
                "forbidden_directories": ["src/", "experiments/", "./"]
            }
        }

    def should_ignore_path(self, file_path: Path) -> bool:
        """Check if path should be ignored"""
        path_str = str(file_path.relative_to(self.project_root))

        for pattern in self.ignore_patterns:
            if pattern in path_str:
                return True

        # Ignore temporary files
        if file_path.name.startswith('.') and file_path.suffix in ['.tmp', '.swp', '.lock']:
            return True

        return False

    def validate_architectural_rules(self, file_path: Path) -> Dict[str, Any]:
        """Validate file against architectural rules and return violation info"""
        violations = []
        rel_path = file_path.relative_to(self.project_root)
        file_name = file_path.name

        # Check test files location rule
        test_rule = self.architectural_rules["test_files_location"]
        is_test_file = any(
            file_name.lower().startswith(pattern.replace("*", "")) or
            file_name.lower().endswith(pattern.replace("*", "")) or
            pattern.replace("*", "") in file_name.lower()
            for pattern in test_rule["patterns"]
        )

        if is_test_file:
            # Check if test file is in forbidden directory
            current_dir = str(rel_path.parent) + "/"
            if any(current_dir.startswith(forbidden) for forbidden in test_rule["forbidden_directories"]):
                violations.append({
                    "rule": "test_files_location",
                    "severity": "critical",
                    "message": f"Test file {file_name} found in forbidden directory {current_dir}",
                    "current_path": str(rel_path),
                    "required_directory": test_rule["required_directory"],
                    "suggested_path": f"{test_rule['required_directory']}{file_name}"
                })

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "is_test_file": is_test_file
        }

    def handle_file_change(self, file_path: Path, change_type: str):
        """Handle detected file system change"""
        try:
            # Convert to relative path and check if should ignore
            if not file_path.is_relative_to(self.project_root):
                return

            if self.should_ignore_path(file_path):
                return

            # Validate architectural rules
            validation_result = self.validate_architectural_rules(file_path)
            if not validation_result["valid"]:
                # Log architectural violations
                for violation in validation_result["violations"]:
                    logger.critical(f"ARCHITECTURAL VIOLATION: {violation['message']}")
                    logger.critical(f"Required action: Move {violation['current_path']} to {violation['suggested_path']}")

                    # Log audit entry for violation
                    self.prompt_engine.log_audit_entry(
                        f"arch_violation_{int(time.time())}",
                        "architectural_violation",
                        "blocked",
                        [str(file_path)]
                    )

                # Skip processing files that violate architectural rules
                logger.warning(f"Skipping prompt generation for {file_path.relative_to(self.project_root)} due to architectural violations")
                return

            # Detect actual content changes (for modifications)
            change_info = self.change_detector.detect_change(file_path) if file_path.exists() else None

            # Skip if file didn't actually change
            if change_type == "modification" and change_info and not change_info.get("changed", True):
                return

            # Calculate priority
            priority = PrioritySystem.calculate_priority(file_path, change_type)

            # Add validation info to change event
            change_event = {
                "file_path": file_path,
                "change_type": change_type,
                "priority": priority,
                "timestamp": datetime.utcnow().isoformat(),
                "change_info": change_info,
                "validation_result": validation_result
            }

            self.prompt_queue.put((-priority, time.time(), change_event))
            logger.info(f"Queued {change_type} of {file_path.relative_to(self.project_root)} (priority: {priority})")

        except Exception as e:
            logger.error(f"Error handling file change {file_path}: {e}")

    def _generate_prompt_for_change(self, change_event: Dict[str, Any]) -> Optional[str]:
        """Generate appropriate prompt for a file change"""
        file_path = change_event["file_path"]
        change_type = change_event["change_type"]
        priority = change_event["priority"]
        timestamp = change_event["timestamp"]

        # Determine appropriate template
        template_name = self._select_template(file_path, change_type)
        if not template_name:
            logger.warning(f"No suitable template for {file_path} ({change_type})")
            return None

        # Prepare template variables
        variables = self._prepare_template_variables(file_path, change_event)

        # Generate prompt
        prompt_data = self.prompt_templates.generate_prompt(template_name, variables)
        if not prompt_data:
            return None

        # Save prompt to prompts directory
        prompt_id = prompt_data.get("id", f"auto_{int(time.time())}")
        prompt_path = self.prompt_engine.prompts_dir / f"{prompt_id}.json"

        try:
            with open(prompt_path, 'w') as f:
                json.dump(prompt_data, f, indent=2)

            logger.info(f"Generated prompt {prompt_id} for {file_path.relative_to(self.project_root)}")

            # Log audit entry
            self.prompt_engine.log_audit_entry(
                prompt_id,
                "prompt_generated",
                "success",
                [str(prompt_path)]
            )

            return prompt_id

        except Exception as e:
            logger.error(f"Could not save generated prompt: {e}")
            return None

    def _select_template(self, file_path: Path, change_type: str) -> Optional[str]:
        """Select appropriate template based on file and change type"""
        path_str = str(file_path.relative_to(self.project_root))

        # File creation/modification in src directories
        if change_type in ["creation", "modification"] and path_str.startswith("src/"):
            if change_type == "creation":
                return "document_new_component"
            else:
                return "validate_modified_layer"

        # Documentation changes
        if "construction_technical_manual" in path_str and path_str.endswith(".md"):
            return "update_manual_section"

        # Default for other significant files
        if file_path.suffix == ".py":
            return "document_new_component" if change_type == "creation" else "validate_modified_layer"

        return None

    def _prepare_template_variables(self, file_path: Path, change_event: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare variables for template substitution"""
        rel_path = file_path.relative_to(self.project_root)
        change_info = change_event.get("change_info", {})

        # Base variables
        variables = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "target_file": str(rel_path),
            "component_name": file_path.stem,
            "component_display_name": file_path.stem.replace("_", " ").title(),
            "layer_name": file_path.stem,
            "layer_display_name": file_path.stem.replace("_", " ").title(),
            "section_id": file_path.stem,
            "section_display_name": file_path.stem.replace("_", " ").title(),
            "priority": change_event["priority"],
            "detection_time": change_event["timestamp"],
            "detected_directory": str(rel_path.parent),
            "component_classification": self._classify_component(file_path),
            "component_type": file_path.suffix,
            "previous_hash": change_info.get("previous_hash", "unknown"),
            "current_hash": change_info.get("current_hash", "unknown"),
            "modification_type": "creation" if change_info.get("is_new") else "modification",
            "related_files": [],  # Could be enhanced to find related files
            "modified_files": [str(rel_path)],
            "new_components": [str(rel_path)] if change_info.get("is_new") else [],
            "removed_components": [],
            "target_section": rel_path.stem,
            "change_summary": f"{change_event['change_type']} of {rel_path}",
            "update_type": change_event["change_type"],
            "impact_level": "medium"
        }

        return variables

    def _classify_component(self, file_path: Path) -> str:
        """Classify component based on its location and name"""
        path_str = str(file_path.relative_to(self.project_root))

        if "src/core/" in path_str:
            return "ΨQRH-Core-Component-v1.0"
        elif "src/cognitive/" in path_str:
            return "ΨQRH-Cognitive-Component-v1.0"
        elif "src/fractal/" in path_str:
            return "ΨQRH-Fractal-Component-v1.0"
        elif "src/conceptual/" in path_str:
            return "ΨQRH-Conceptual-Component-v1.0"
        else:
            return "ΨQRH-General-Component-v1.0"

    def _process_prompt_queue(self):
        """Background thread to process prompt generation queue"""
        logger.info("Prompt generation thread started")

        while self.running:
            try:
                # Get next item from queue (with timeout)
                neg_priority, timestamp, change_event = self.prompt_queue.get(timeout=1.0)

                # Generate prompt
                prompt_id = self._generate_prompt_for_change(change_event)

                if prompt_id:
                    logger.info(f"Successfully generated prompt: {prompt_id}")
                else:
                    logger.warning(f"Failed to generate prompt for {change_event['file_path']}")

                self.prompt_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing prompt queue: {e}")

    def start_monitoring(self):
        """Start file system monitoring and prompt generation"""
        if self.running:
            logger.warning("Autonomous prompt generator already running")
            return

        try:
            # Start background processing thread
            self.running = True
            self.generation_thread = threading.Thread(
                target=self._process_prompt_queue,
                name="PromptGenerator",
                daemon=True
            )
            self.generation_thread.start()

            # Start file system monitoring
            self.observer.schedule(self.handler, str(self.project_root), recursive=True)
            self.observer.start()

            logger.info(f"Autonomous prompt generation started, monitoring {self.project_root}")
            return True

        except Exception as e:
            logger.error(f"Failed to start autonomous prompt generator: {e}")
            self.stop_monitoring()
            return False

    def stop_monitoring(self):
        """Stop file system monitoring and prompt generation"""
        logger.info("Stopping autonomous prompt generator...")

        self.running = False

        # Stop file system monitoring
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=5)

        # Wait for generation thread
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=5)

        logger.info("Autonomous prompt generator stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the autonomous prompt generator"""
        return {
            "running": self.running,
            "queue_size": self.prompt_queue.qsize(),
            "observer_active": self.observer.is_alive() if self.observer else False,
            "generation_thread_active": self.generation_thread.is_alive() if self.generation_thread else False,
            "templates_loaded": len(self.prompt_templates.templates),
            "ignored_patterns": list(self.ignore_patterns)
        }

def create_autonomous_prompt_generator(project_root: Path,
                                     prompt_engine: PromptEngineAgent) -> AutonomousPromptGenerator:
    """Create and configure autonomous prompt generator"""
    return AutonomousPromptGenerator(project_root, prompt_engine)