"""
Sistema de Log de Inicialização - ΨQRH Framework
================================================================
Sistema para detectar e resolver conflitos de versões de bibliotecas.

Autores: Claude Code & ΨQRH Team
Versão: 2.0.0 (Desacoplado da camada agêntica)
"""

import sys
import os
import json
import time
import hashlib
import importlib
import pkg_resources
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback
import warnings


@dataclass
class DependencyInfo:
    """Informações sobre uma dependência."""
    name: str
    version: str
    location: str
    imported_by: str
    import_time: float
    function_context: str
    requirements: List[str]
    conflicts: List[str] = None


@dataclass
class ConflictReport:
    """Relatório de conflito entre dependências."""
    library: str
    required_versions: List[str]
    installed_version: str
    conflicting_functions: List[str]
    severity: str  # 'critical', 'warning', 'info'
    resolution_suggestion: str


class DependencyLogger:
    """
    Logger de dependências para monitoramento de conflitos de versão.

    Sistema autônomo que detecta conflitos de versão e sugere resoluções
    usando análise baseada em regras (sem dependências externas).
    """

    def __init__(self, log_dir: str = "logs/dependencies"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = self._generate_session_id()
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.conflicts: List[ConflictReport] = []
        self.import_stack: List[str] = []
        self.function_context: str = "system_init"

        # Log de inicialização
        self.log_file = self.log_dir / f"dependency_log_{self.session_id}.json"
        self.init_log = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform": sys.platform,
            "dependencies": {},
            "conflicts": [],
            "analysis": {},
            "resolution_history": []
        }

        self._setup_import_hooks()
        self._log_system_info()

    def _generate_session_id(self) -> str:
        """Gera ID único para a sessão."""
        timestamp = str(int(time.time()))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"psiqrh_{timestamp}_{random_part}"

    def _setup_import_hooks(self):
        """Configura hooks para monitorar imports."""
        original_import = __builtins__['__import__']

        def tracked_import(name, globals=None, locals=None, fromlist=(), level=0):
            start_time = time.time()
            try:
                module = original_import(name, globals, locals, fromlist, level)
                import_time = time.time() - start_time

                # Log da importação
                self._log_import(name, import_time, module)
                return module
            except ImportError as e:
                self._log_import_error(name, str(e))
                raise

        __builtins__['__import__'] = tracked_import

    def _log_system_info(self):
        """Log informações do sistema."""
        self.init_log["system_info"] = {
            "python_executable": sys.executable,
            "python_path": sys.path[:5],  # Primeiros 5 caminhos
            "installed_packages": self._get_installed_packages(),
            "environment_variables": {
                k: v for k, v in os.environ.items()
                if k.startswith(('PYTHON', 'PIP', 'CONDA', 'VIRTUAL_ENV'))
            }
        }

    def _get_installed_packages(self) -> Dict[str, str]:
        """Obtém lista de pacotes instalados."""
        try:
            installed = {}
            for dist in pkg_resources.working_set:
                installed[dist.project_name] = dist.version
            return installed
        except Exception:
            return {"error": "Unable to enumerate packages"}

    def set_function_context(self, context: str):
        """Define o contexto da função atual."""
        self.function_context = context
        self.import_stack.append(context)

    def exit_function_context(self):
        """Sai do contexto da função atual."""
        if self.import_stack:
            self.import_stack.pop()
            self.function_context = self.import_stack[-1] if self.import_stack else "system"

    def _log_import(self, module_name: str, import_time: float, module):
        """Log de importação bem-sucedida."""
        try:
            # Obter informações da versão
            version = "unknown"
            location = "unknown"

            if hasattr(module, '__version__'):
                version = module.__version__
            elif hasattr(module, 'VERSION'):
                version = str(module.VERSION)

            if hasattr(module, '__file__'):
                location = str(module.__file__)

            # Verificar se é uma dependência conhecida
            try:
                dist = pkg_resources.get_distribution(module_name)
                version = dist.version
                location = dist.location
            except:
                pass

            dep_info = DependencyInfo(
                name=module_name,
                version=version,
                location=location,
                imported_by=self.function_context,
                import_time=import_time,
                function_context=self.function_context,
                requirements=self._get_module_requirements(module_name)
            )

            # Verificar conflitos
            self._check_version_conflicts(dep_info)

            self.dependencies[module_name] = dep_info
            self.init_log["dependencies"][module_name] = asdict(dep_info)

            # Log detalhado se import demorou muito
            if import_time > 1.0:
                self._log_slow_import(module_name, import_time)

        except Exception as e:
            print(f"Erro ao logar import {module_name}: {e}")

    def _log_import_error(self, module_name: str, error: str):
        """Log de erro de importação."""
        self.init_log.setdefault("import_errors", []).append({
            "module": module_name,
            "error": error,
            "context": self.function_context,
            "timestamp": datetime.now().isoformat()
        })

    def _get_module_requirements(self, module_name: str) -> List[str]:
        """Obtém requirements do módulo."""
        try:
            dist = pkg_resources.get_distribution(module_name)
            return [str(req) for req in dist.requires()]
        except:
            return []

    def _check_version_conflicts(self, dep_info: DependencyInfo):
        """Verifica conflitos de versão."""
        module_name = dep_info.name
        current_version = dep_info.version

        # Verificar se já temos uma versão diferente
        existing_deps = [
            dep for dep in self.dependencies.values()
            if dep.name == module_name and dep.version != current_version
        ]

        if existing_deps:
            conflict = ConflictReport(
                library=module_name,
                required_versions=[dep.version for dep in existing_deps] + [current_version],
                installed_version=current_version,
                conflicting_functions=[dep.function_context for dep in existing_deps] + [dep_info.function_context],
                severity=self._assess_conflict_severity(module_name),
                resolution_suggestion=self._generate_resolution_suggestion(module_name, existing_deps, dep_info)
            )

            self.conflicts.append(conflict)
            self.init_log["conflicts"].append(asdict(conflict))

    def _assess_conflict_severity(self, module_name: str) -> str:
        """Avalia severidade do conflito."""
        critical_modules = {
            'torch', 'tensorflow', 'numpy', 'scipy', 'pandas',
            'matplotlib', 'sklearn', 'transformers', 'accelerate'
        }

        if module_name.lower() in critical_modules:
            return 'critical'
        elif module_name.startswith('psiqrh') or 'qrh' in module_name.lower():
            return 'critical'
        else:
            return 'warning'

    def _generate_resolution_suggestion(self,
                                       module_name: str,
                                       existing_deps: List[DependencyInfo],
                                       new_dep: DependencyInfo) -> str:
        """Gera sugestão de resolução baseada em regras."""
        suggestions = []

        # Análise básica
        all_versions = [dep.version for dep in existing_deps] + [new_dep.version]
        unique_versions = list(set(all_versions))

        if len(unique_versions) > 1:
            suggestions.append(f"Unificar versão do {module_name}")
            suggestions.append(f"Versões encontradas: {', '.join(unique_versions)}")

            # Sugestão de versão mais recente
            try:
                sorted_versions = sorted(unique_versions, key=lambda x: tuple(map(int, x.split('.'))))
                latest = sorted_versions[-1]
                suggestions.append(f"Sugestão: usar versão {latest}")
            except:
                suggestions.append("Verificar compatibilidade manual necessária")

        return " | ".join(suggestions)

    def _log_slow_import(self, module_name: str, import_time: float):
        """Log especial para imports lentos."""
        self.init_log.setdefault("slow_imports", []).append({
            "module": module_name,
            "time": import_time,
            "context": self.function_context,
            "timestamp": datetime.now().isoformat()
        })

    def log_function_dependency(self, function_name: str, required_libs: Dict[str, str]):
        """
        Log manual de dependências específicas de uma função.

        Args:
            function_name: Nome da função
            required_libs: Dict {library_name: required_version}
        """
        self.init_log.setdefault("function_dependencies", {})[function_name] = {
            "required_libraries": required_libs,
            "timestamp": datetime.now().isoformat()
        }

        # Verificar se as versões batem
        for lib_name, required_version in required_libs.items():
            if lib_name in self.dependencies:
                installed_version = self.dependencies[lib_name].version
                if installed_version != required_version and required_version != "*":
                    self._create_manual_conflict(function_name, lib_name,
                                                required_version, installed_version)

    def _create_manual_conflict(self, function_name: str, lib_name: str,
                               required_version: str, installed_version: str):
        """Cria conflito para dependências manuais."""
        conflict = ConflictReport(
            library=lib_name,
            required_versions=[required_version],
            installed_version=installed_version,
            conflicting_functions=[function_name],
            severity=self._assess_conflict_severity(lib_name),
            resolution_suggestion=f"Função {function_name} requer {lib_name}=={required_version}, instalada: {installed_version}"
        )

        self.conflicts.append(conflict)
        self.init_log["conflicts"].append(asdict(conflict))

    def generate_compatibility_report(self) -> str:
        """Gera relatório de compatibilidade completo."""
        report = []
        report.append("ΨQRH DEPENDENCY COMPATIBILITY REPORT")
        report.append("=" * 50)
        report.append(f"Session ID: {self.session_id}")
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Dependencies loaded: {len(self.dependencies)}")
        report.append(f"Conflicts detected: {len(self.conflicts)}")
        report.append("")

        if self.conflicts:
            report.append("CONFLICTS DETECTED:")
            report.append("-" * 30)

            for conflict in self.conflicts:
                report.append(f"📚 {conflict.library}")
                report.append(f"   Severity: {conflict.severity.upper()}")
                report.append(f"   Required: {conflict.required_versions}")
                report.append(f"   Installed: {conflict.installed_version}")
                report.append(f"   Functions: {conflict.conflicting_functions}")
                report.append(f"   Resolution: {conflict.resolution_suggestion}")
                report.append("")
        else:
            report.append("NO CONFLICTS DETECTED")

        report.append("\nSUMMARY:")
        report.append(f"Total dependencies: {len(self.dependencies)}")

        critical_conflicts = [c for c in self.conflicts if c.severity == 'critical']
        if critical_conflicts:
            report.append(f"🚨 Critical conflicts: {len(critical_conflicts)}")

        slow_imports = self.init_log.get("slow_imports", [])
        if slow_imports:
            report.append(f"🐌 Slow imports: {len(slow_imports)}")

        return "\n".join(report)

    def save_log(self):
        """Salva log completo."""
        self.init_log["end_time"] = datetime.now().isoformat()
        self.init_log["duration"] = time.time() - time.mktime(
            datetime.fromisoformat(self.init_log["start_time"]).timetuple()
        )

        with open(self.log_file, 'w') as f:
            json.dump(self.init_log, f, indent=2, default=str)

        # Salvar relatório texto
        report_file = self.log_dir / f"compatibility_report_{self.session_id}.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_compatibility_report())

        print(f"Dependency log saved: {self.log_file}")
        print(f"Compatibility report: {report_file}")

    def get_cross_reference_data(self) -> Dict[str, Any]:
        """Retorna dados para cruzamento entre sessões."""
        return {
            "session_id": self.session_id,
            "dependencies": {name: info.version for name, info in self.dependencies.items()},
            "conflicts": [asdict(conflict) for conflict in self.conflicts],
            "function_contexts": list(set(dep.function_context for dep in self.dependencies.values())),
            "timestamp": datetime.now().isoformat()
        }

    def analyze_historical_conflicts(self, log_dir: Optional[str] = None) -> Dict[str, Any]:
        """Analisa conflitos históricos para padrões."""
        if log_dir is None:
            log_dir = self.log_dir

        historical_data = []
        log_files = Path(log_dir).glob("dependency_log_*.json")

        for log_file in log_files:
            try:
                with open(log_file) as f:
                    data = json.load(f)
                    historical_data.append(data)
            except Exception as e:
                print(f"Erro ao ler {log_file}: {e}")

        # Análise de padrões
        all_conflicts = []
        recurring_conflicts = {}

        for data in historical_data:
            for conflict in data.get("conflicts", []):
                all_conflicts.append(conflict)
                lib = conflict["library"]
                recurring_conflicts[lib] = recurring_conflicts.get(lib, 0) + 1

        analysis = {
            "total_sessions": len(historical_data),
            "total_conflicts": len(all_conflicts),
            "recurring_conflicts": {
                lib: count for lib, count in recurring_conflicts.items()
                if count > 1
            },
            "most_problematic": max(recurring_conflicts.items(), key=lambda x: x[1])
                                if recurring_conflicts else None
        }

        return analysis


# Instância global do logger
_global_logger: Optional[DependencyLogger] = None


def get_dependency_logger() -> DependencyLogger:
    """Obtém instância global do logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = DependencyLogger()
    return _global_logger


def log_function_dependencies(function_name: str, dependencies: Dict[str, str]):
    """
    Função para logar dependências de uma função.

    Usage:
        log_function_dependencies("neural_network_training", {
            "torch": "2.1.0",
            "numpy": "1.24.0"
        })
    """
    logger = get_dependency_logger()
    logger.log_function_dependency(function_name, dependencies)


def function_context(context_name: str):
    """
    Context manager para definir contexto de função.

    Usage:
        with function_context("data_processing"):
            import pandas as pd
            import numpy as np
    """
    class FunctionContext:
        def __enter__(self):
            get_dependency_logger().set_function_context(context_name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            get_dependency_logger().exit_function_context()

    return FunctionContext()


if __name__ == "__main__":
    # Demo do sistema
    print("ΨQRH Dependency Logger Demo")
    logger = DependencyLogger()

    # Simular função com dependências específicas
    with function_context("neural_network_setup"):
        log_function_dependencies("neural_network_setup", {
            "torch": "2.1.0",
            "numpy": "1.26.0"
        })

    with function_context("data_analysis"):
        log_function_dependencies("data_analysis", {
            "pandas": "2.0.0",
            "numpy": "1.24.0"  # Versão diferente!
        })

    # Gerar relatório
    print(logger.generate_compatibility_report())

    # Salvar logs
    logger.save_log()

    print("\nDemo concluída!")
