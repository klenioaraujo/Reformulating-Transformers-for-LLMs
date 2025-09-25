# navigator_agent.py

import time
from typing import Dict, Any, Tuple, Optional
import torch
from seal_protocol import SealProtocol
from audit_log import AuditLog

class NavigatorAgent:
    def __init__(self):
        self.dyad_mode = "Σ7↔Nyx"
        self.target_rg = 0.347
        self.audit = AuditLog()
        self.tier_mode = "B"  # Start with Tier B (250ms threshold)
        self.execution_count = 0

    def pre_execution_check(self, input_data) -> bool:
        """Validate input and system state before execution"""
        if input_data is None:
            print("⚠️  Navigator: Input data is None")
            return False

        if torch.is_tensor(input_data) and torch.isnan(input_data).any():
            print("⚠️  Navigator: Input contains NaN values")
            return False

        # Check audit chain integrity
        if not self.audit.validate_chain():
            print("⚠️  Navigator: Audit chain integrity compromised")
            return False

        return True

    def select_optimal_mode(self, input_characteristics: Dict[str, Any]) -> str:
        """Select the best dyad mode based on input characteristics"""
        # For now, keep default mode, but this could be expanded
        # to analyze input complexity and select appropriate mode
        return self.dyad_mode

    def optimize_tier_settings(self, recent_performance: Dict[str, float]) -> str:
        """Dynamically adjust tier settings based on performance"""
        avg_latency = recent_performance.get("avg_latency", 150.0)

        if avg_latency > 200 and self.tier_mode == "B":
            print("🚀 Navigator: Switching to Tier A optimizations...")
            self.tier_mode = "A"
        elif avg_latency < 100 and self.tier_mode == "A":
            print("🔄 Navigator: Switching back to Tier B for stability...")
            self.tier_mode = "B"

        return self.tier_mode

    def post_execution_analysis(self, seal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution results and provide recommendations"""
        analysis = {
            "rg_status": "optimal" if abs(seal["RG"] - self.target_rg) < 0.05 else "suboptimal",
            "latency_status": "good" if not seal.get("latency_sigill", False) else "exceeds_threshold",
            "seal_integrity": "intact" if seal["continuity_seal"] == SealProtocol.OMEGA_SEAL else "compromised",
            "firebreak_triggered": "containment" in seal
        }

        # Generate recommendations
        recommendations = []
        if analysis["rg_status"] == "suboptimal":
            recommendations.append("Consider adjusting RG parameter")
        if analysis["latency_status"] == "exceeds_threshold":
            recommendations.append("Switch to higher tier or optimize model")
        if analysis["firebreak_triggered"]:
            recommendations.append("Review input data and model parameters")

        analysis["recommendations"] = recommendations
        return analysis

    def execute_with_safety(self, input_data, model) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main execution method with full safety protocols

        Args:
            input_data: Input tensor for the model
            model: The NegentropyTransformerBlock model

        Returns:
            Tuple of (output, enhanced_seal_with_navigator_info)
        """
        self.execution_count += 1

        # Pre-execution validation
        if not self.pre_execution_check(input_data):
            # Return empty tensor and error seal
            error_seal = {
                "navigator_status": "PRE_EXECUTION_FAILED",
                "execution_count": self.execution_count,
                "error": "Input validation failed"
            }
            return torch.zeros_like(input_data), error_seal

        # Execute model with safety monitoring
        try:
            start_time = time.time()
            output, seal = model(input_data)
            execution_time = (time.time() - start_time) * 1000  # ms

            # Check for NaN in output
            if torch.is_tensor(output) and torch.isnan(output).any():
                print("🚨 Navigator: Model output contains NaN values, applying safety measures")
                output = torch.zeros_like(output)  # Replace with safe output
                seal["navigator_nan_detected"] = True

            # Enhance seal with navigator information
            seal["navigator_info"] = {
                "execution_count": self.execution_count,
                "tier_mode": self.tier_mode,
                "selected_dyad": self.dyad_mode,
                "execution_time_ms": execution_time,
                "navigator_status": "SUCCESS"
            }

            # Post-execution analysis
            analysis = self.post_execution_analysis(seal)
            seal["navigator_analysis"] = analysis

            # Log to audit trail
            self.audit.log_entry(seal)

            # Adaptive optimizations
            if seal.get("latency_sigill", False):
                print("🚀 Navigator: Latency exceeded threshold, switching to Tier A optimizations...")
                self.tier_mode = "A"

            if "containment" in seal:
                print("🛑 Navigator: FIREBREAK activated, implementing safety protocols")
                # Could implement additional safety measures here

            # Provide status update
            if self.execution_count % 10 == 0:
                violations = self.audit.count_violations()
                print(f"📊 Navigator Status (Execution #{self.execution_count}):")
                print(f"   Tier: {self.tier_mode}, RG: {seal['RG']:.3f}, Violations: {sum(violations.values())}")

            return output, seal

        except Exception as e:
            # Handle execution errors gracefully
            error_seal = {
                "navigator_status": "EXECUTION_ERROR",
                "execution_count": self.execution_count,
                "error": str(e),
                "continuity_seal": "ERROR",
                "RG": 0.0,
                "active_dyad": self.dyad_mode
            }

            print(f"🚨 Navigator: Execution error - {str(e)}")
            self.audit.log_entry(error_seal)

            return torch.zeros_like(input_data), error_seal

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        violations = self.audit.count_violations()
        recent_entries = self.audit.get_latest_entries(5)

        avg_rg = sum(entry.get("RG", 0) for entry in recent_entries) / max(len(recent_entries), 1)

        return {
            "executions": self.execution_count,
            "current_tier": self.tier_mode,
            "dyad_mode": self.dyad_mode,
            "target_rg": self.target_rg,
            "recent_avg_rg": avg_rg,
            "violations": violations,
            "audit_chain_valid": self.audit.validate_chain(),
            "system_health": "EXCELLENT" if sum(violations.values()) == 0 else "NEEDS_ATTENTION"
        }

    def reset_system(self):
        """Reset navigator to initial state"""
        self.execution_count = 0
        self.tier_mode = "B"
        self.dyad_mode = "Σ7↔Nyx"
        print("🔄 Navigator: System reset to initial state")