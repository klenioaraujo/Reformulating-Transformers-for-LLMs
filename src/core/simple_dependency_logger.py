#!/usr/bin/env python3
"""
Sistema Simplificado de Logging de Dependências - ΨQRH Framework
================================================================

Versão simplificada do DependencyLogger para evitar dependências circulares.
"""

import os
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any, List


class SimpleDependencyLogger:
    """
    Sistema simplificado de logging de dependências sem imports externos complexos.
    """

    def __init__(self, log_dir: str = "logs/dependencies"):
        """Inicializar logger simplificado."""
        self.log_dir = log_dir
        self.session_id = f"psiqrh_simple_{int(time.time())}_{hex(id(self))[2:10]}"
        self.start_time = datetime.now()
        self.function_context = "default"

        # Log básico
        self.init_log = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "dependencies": {},
            "conflicts": [],
            "function_dependencies": {},
            "system_info": {
                "python_executable": sys.executable,
                "python_path": sys.path[:5]  # Primeiros 5 para evitar logs gigantes
            }
        }

    def set_function_context(self, context: str):
        """Definir contexto da função atual."""
        self.function_context = context

    def log_function_dependency(self, function_name: str, required_libs: Dict[str, str]):
        """Log de dependência de função específica."""
        self.init_log["function_dependencies"][self.function_context] = {
            "function_name": function_name,
            "required_libraries": required_libs,
            "timestamp": datetime.now().isoformat()
        }

    def generate_compatibility_report(self) -> str:
        """Gerar relatório de compatibilidade simplificado."""
        deps_count = len(self.init_log.get("dependencies", {}))
        conflicts_count = len(self.init_log.get("conflicts", []))

        report = f"""🔍 ΨQRH DEPENDENCY COMPATIBILITY REPORT (SIMPLIFIED)
==================================================
Session ID: {self.session_id}
Timestamp: {datetime.now().isoformat()}
Dependencies loaded: {deps_count}
Conflicts detected: {conflicts_count}

{'✅ NO CONFLICTS DETECTED' if conflicts_count == 0 else f'⚠️ {conflicts_count} CONFLICTS FOUND'}

📊 SUMMARY:
Total dependencies: {deps_count}
Function dependencies: {len(self.init_log.get('function_dependencies', {}))}
"""
        return report

    def save_log(self):
        """Salvar log simplificado."""
        os.makedirs(self.log_dir, exist_ok=True)

        self.init_log["end_time"] = datetime.now().isoformat()

        # Salvar JSON
        json_path = os.path.join(self.log_dir, f"simple_dependency_log_{self.session_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.init_log, f, indent=2, ensure_ascii=False)

        # Salvar relatório
        report_path = os.path.join(self.log_dir, f"simple_compatibility_report_{self.session_id}.txt")
        report = self.generate_compatibility_report()
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📝 Simple dependency log saved: {json_path}")
        print(f"📊 Simple compatibility report: {report_path}")

    def get_cross_reference_data(self) -> Dict[str, Any]:
        """Obter dados para referência cruzada."""
        return {
            "session_id": self.session_id,
            "dependencies": self.init_log.get("dependencies", {}),
            "conflicts": self.init_log.get("conflicts", []),
            "function_contexts": list(self.init_log.get("function_dependencies", {}).keys())
        }