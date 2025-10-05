"""
Provenance Metadata Utilities

Utilities for tracking and recording provenance metadata in reports
to ensure reproducibility and FAIR compliance.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
Project: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs
"""

import platform
import sys
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
import torch


def get_git_commit() -> Optional[str]:
    """
    Get the current git commit hash.

    Returns:
        Commit hash (7 chars) or None if not in a git repository
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_hardware_info() -> Dict[str, str]:
    """
    Get hardware information.

    Returns:
        Dictionary with CPU, GPU, RAM, and device info
    """
    info = {
        'cpu': platform.processor() or platform.machine(),
        'ram': 'Unknown',
        'device': 'cpu'
    }

    # GPU information
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['device'] = 'cuda'
        info['cuda_version'] = torch.version.cuda
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['gpu'] = 'Apple Metal Performance Shaders'
        info['device'] = 'mps'
    else:
        info['gpu'] = 'None'

    # RAM information (Linux)
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    kb = int(line.split()[1])
                    gb = kb / (1024 ** 2)
                    info['ram'] = f"{gb:.1f} GB"
                    break
    except FileNotFoundError:
        pass

    return info


def get_execution_environment() -> Dict[str, str]:
    """
    Get execution environment details.

    Returns:
        Dictionary with Python, PyTorch, and system info
    """
    return {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pytorch_version': torch.__version__,
        'numpy_version': _get_package_version('numpy'),
        'scipy_version': _get_package_version('scipy'),
        'platform': platform.system(),
        'os': f"{platform.system()} {platform.release()}",
        'architecture': platform.machine()
    }


def _get_package_version(package_name: str) -> str:
    """Get version of an installed package."""
    try:
        mod = __import__(package_name)
        return mod.__version__
    except (ImportError, AttributeError):
        return 'Not installed'


def compute_data_hash(data: Any, algorithm: str = 'sha256') -> str:
    """
    Compute hash of input data for reproducibility.

    Args:
        data: Data to hash (tensor, string, or bytes)
        algorithm: Hash algorithm ('sha256' or 'md5')

    Returns:
        Hash string in format 'algorithm:hexdigest'
    """
    if algorithm == 'sha256':
        hasher = hashlib.sha256()
    elif algorithm == 'md5':
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Convert data to bytes
    if isinstance(data, torch.Tensor):
        data_bytes = data.cpu().numpy().tobytes()
    elif isinstance(data, str):
        data_bytes = data.encode('utf-8')
    elif isinstance(data, bytes):
        data_bytes = data
    else:
        data_bytes = str(data).encode('utf-8')

    hasher.update(data_bytes)
    return f"{algorithm}:{hasher.hexdigest()}"


def create_provenance_metadata(
    software_version: str = "1.0.0",
    input_data: Optional[Any] = None,
    random_seed: Optional[int] = None,
    execution_time: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create complete provenance metadata for a report.

    Args:
        software_version: Version of the ΨQRH software
        input_data: Input data to hash (optional)
        random_seed: Random seed used (optional)
        execution_time: Execution time in seconds (optional)
        config: Configuration parameters (optional)

    Returns:
        Dictionary with complete provenance metadata
    """
    metadata = {
        'software_version': software_version,
        'git_commit': get_git_commit(),
        'hardware': get_hardware_info(),
        'execution_environment': get_execution_environment(),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }

    if input_data is not None:
        metadata['input_data_hash'] = compute_data_hash(input_data)

    if random_seed is not None:
        metadata['random_seed'] = random_seed

    if execution_time is not None:
        metadata['execution_time'] = execution_time

    if config is not None:
        metadata['configuration'] = config

    return metadata


def add_provenance_to_report(
    report: Dict[str, Any],
    software_version: str = "1.0.0",
    input_data: Optional[Any] = None,
    random_seed: Optional[int] = None,
    execution_time: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add provenance metadata to an existing report.

    Args:
        report: Existing report dictionary
        software_version: Version of the ΨQRH software
        input_data: Input data to hash (optional)
        random_seed: Random seed used (optional)
        execution_time: Execution time in seconds (optional)
        config: Configuration parameters (optional)

    Returns:
        Report with added provenance metadata
    """
    report['provenance'] = create_provenance_metadata(
        software_version=software_version,
        input_data=input_data,
        random_seed=random_seed,
        execution_time=execution_time,
        config=config
    )

    # Add schema reference if not present
    if '$schema' not in report:
        report['$schema'] = (
            "https://raw.githubusercontent.com/klenioaraujo/"
            "Reformulating-Transformers-for-LLMs/master/schemas/report_schema.json"
        )

    # Add metadata if not present
    if 'metadata' not in report:
        report['metadata'] = {
            'project': 'ΨQRH Transformer',
            'version': software_version,
            'doi': 'https://zenodo.org/records/17171112',
            'license': 'GPL-3.0-or-later'
        }

    return report


class ProvenanceTracker:
    """
    Context manager for tracking execution provenance.

    Example:
        with ProvenanceTracker() as tracker:
            result = run_experiment()

        report = tracker.create_report(result)
    """

    def __init__(self, software_version: str = "1.0.0"):
        self.software_version = software_version
        self.start_time = None
        self.end_time = None
        self.input_data_hash = None
        self.random_seed = None
        self.config = None

    def __enter__(self):
        """Start tracking."""
        import time
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop tracking."""
        import time
        self.end_time = time.time()

    def set_input_data(self, data: Any):
        """Set input data for hashing."""
        self.input_data_hash = compute_data_hash(data)

    def set_random_seed(self, seed: int):
        """Set random seed."""
        self.random_seed = seed

    def set_config(self, config: Dict[str, Any]):
        """Set configuration."""
        self.config = config

    def get_execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def create_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a report with provenance metadata.

        Args:
            results: Experiment results

        Returns:
            Complete report with provenance
        """
        return add_provenance_to_report(
            report=results,
            software_version=self.software_version,
            random_seed=self.random_seed,
            execution_time=self.get_execution_time(),
            config=self.config
        )


# Example usage
if __name__ == "__main__":
    print("ΨQRH Provenance Utilities")
    print("=" * 60)

    # Get system info
    print("\nHardware Information:")
    for key, value in get_hardware_info().items():
        print(f"  {key}: {value}")

    print("\nExecution Environment:")
    for key, value in get_execution_environment().items():
        print(f"  {key}: {value}")

    # Create provenance metadata
    print("\nCreating provenance metadata...")
    metadata = create_provenance_metadata(
        software_version="1.0.0",
        random_seed=42
    )

    print(f"Git commit: {metadata.get('git_commit', 'N/A')}")
    print(f"Timestamp: {metadata['timestamp']}")

    # Example with ProvenanceTracker
    print("\nExample with ProvenanceTracker:")
    with ProvenanceTracker() as tracker:
        tracker.set_random_seed(42)
        tracker.set_config({'model': 'psiqrh', 'layers': 6})

        # Simulate some work
        import time
        time.sleep(0.1)

        results = {'accuracy': 0.95, 'loss': 0.05}

    report = tracker.create_report(results)
    print(f"Execution time: {report['provenance']['execution_time']:.3f}s")
    print(f"Random seed: {report['provenance']['random_seed']}")

    print("\n✓ Provenance utilities ready!")