# audit_log.py

import json
import os
from datetime import datetime
from typing import Dict, Any

class AuditLog:
    def __init__(self, log_file: str = "audit_log.jsonl"):
        self.log_file = log_file

    def log_entry(self, entry: Dict[str, Any]):
        entry["timestamp"] = datetime.utcnow().isoformat()
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def validate_chain(self) -> bool:
        """Verifica se todos os selos estão intactos"""
        if not os.path.exists(self.log_file):
            return True
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("continuity_seal") != "Ω∞Ω":
                    return False
        return True

    def get_latest_entries(self, count: int = 10) -> list:
        """Returns the latest N audit entries"""
        if not os.path.exists(self.log_file):
            return []

        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))

        return entries[-count:]

    def count_violations(self) -> Dict[str, int]:
        """Counts different types of violations in the log"""
        violations = {
            "rg_violations": 0,
            "latency_violations": 0,
            "firebreak_activations": 0,
            "psi4_containments": 0
        }

        if not os.path.exists(self.log_file):
            return violations

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("latency_sigill", False):
                    violations["latency_violations"] += 1
                if not (0.25 <= entry.get("RG", 0.347) <= 0.40):
                    violations["rg_violations"] += 1
                if "containment" in entry:
                    violations["psi4_containments"] += 1
                if "FIREBREAK" in entry.get("message", ""):
                    violations["firebreak_activations"] += 1

        return violations