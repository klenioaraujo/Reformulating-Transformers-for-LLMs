# seal_protocol.py

import hashlib
import time
from typing import Dict, Any, Tuple

class SealProtocol:
    OMEGA_SEAL = "Ω∞Ω"

    @staticmethod
    def compute_sha256(data: str) -> str:
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    @staticmethod
    def validate_rg(rg_value: float) -> bool:
        return 0.25 <= rg_value <= 0.40

    @staticmethod
    def validate_latency(latency_ms: float, tier: str = "B") -> bool:
        threshold = 120 if tier == "A" else 250
        return latency_ms <= threshold

    @staticmethod
    def activate_dyad(mode: str = "Σ7↔Nyx") -> str:
        # Σ7 = factual mode, Nyx = bounded creativity
        return mode

    @staticmethod
    def generate_seal(
        continuity_sha: str,
        response_sha: str,
        qz_sha: str,
        rg_value: float = 0.347,
        active_dyad: str = "Σ7↔Nyx"
    ) -> Dict[str, Any]:
        return {
            "continuity_sha256": continuity_sha,
            "response_sha256": response_sha,
            "qz_sha256": qz_sha,
            "epsilon_cover": 1.0,  # assumindo cobertura total por enquanto
            "latency_sigill": False,  # será definido dinamicamente
            "RG": rg_value,
            "active_dyad": active_dyad,
            "continuity_seal": SealProtocol.OMEGA_SEAL
        }

    @staticmethod
    def firebreak_check(seal: Dict[str, Any]) -> bool:
        """
        FIREBREAK: trava o sistema se algo estiver fora dos limites
        """
        if not SealProtocol.validate_rg(seal["RG"]):
            return False
        if seal.get("latency_sigill", True):  # se latência estourou
            return False
        if seal["continuity_seal"] != SealProtocol.OMEGA_SEAL:
            return False
        return True

    @staticmethod
    def trigger_psi4_containment(reason: str = "RG_VIOLATION") -> Dict[str, Any]:
        """
        Ψ4 containment mode - locks down system when parameters exceed limits
        """
        return {
            "mode": "Ψ4_CONTAINMENT",
            "trigger_reason": reason,
            "timestamp": time.time(),
            "status": "LOCKED"
        }