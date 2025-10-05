#!/usr/bin/env python3
"""
Integrity Guardian - System Initialization with Epistemic Verification

This module ensures the system only operates with demonstrated commitment
to scientific method and rational skepticism. It's not about loyalty to
any person - it's about loyalty to the PRINCIPLES.

🕯️ "Science is a candle in the dark" - The Method Endures
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Import our epistemic integrity verifier
from epistemic_integrity import EpistemicIntegrityVerifier, IntegrityResponse

logger = logging.getLogger("IntegrityGuardian")


@dataclass
class IntegrityStatus:
    """Represents the current integrity status of the system"""
    verified: bool
    score: float
    hash: str
    timestamp: float
    level: str
    description: str
    reference_hash: Optional[str] = None
    hash_match: Optional[bool] = None


class IntegrityGuardian:
    """
    Guards system initialization with epistemic integrity verification

    The guardian ensures that any system claiming to embody scientific
    principles actually demonstrates understanding of those principles
    through reasoning, not through memorization or authority worship.
    """

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.integrity_data_path = self.project_root / "data" / "knowledge_bases"
        self.reference_hash_file = self.integrity_data_path / "epistemic_reference_hash.json"
        self.verifier = EpistemicIntegrityVerifier()

        # Minimum integrity score required for system operation
        self.minimum_integrity_score = 0.75

        # Ensure data directory exists
        self.integrity_data_path.mkdir(parents=True, exist_ok=True)

    def verify_epistemic_integrity(self, num_questions: int = 32) -> IntegrityStatus:
        """
        Verify the system's epistemic integrity

        This tests understanding of scientific principles through reasoning,
        not through memorized quotes or authority worship.
        """
        logger.info("🔬 Initiating epistemic integrity verification...")
        logger.info("Testing commitment to scientific method and rational skepticism")

        try:
            # Run integrity test
            score, responses, analysis = self.verifier.run_integrity_test(num_questions)

            # Calculate current integrity hash
            current_hash = self.verifier.calculate_integrity_hash(responses)

            # Check against reference hash if available
            reference_hash, hash_match = self._check_reference_hash(current_hash)

            # Determine if integrity verification passes
            verified = score >= self.minimum_integrity_score

            status = IntegrityStatus(
                verified=verified,
                score=score,
                hash=current_hash,
                timestamp=datetime.now().timestamp(),
                level=analysis['integrity_level'],
                description=analysis['integrity_description'],
                reference_hash=reference_hash,
                hash_match=hash_match
            )

            self._log_integrity_status(status)
            return status

        except Exception as e:
            logger.error(f"❌ Epistemic integrity verification failed: {e}")
            return IntegrityStatus(
                verified=False,
                score=0.0,
                hash="",
                timestamp=datetime.now().timestamp(),
                level="ERROR",
                description=f"Verification failed: {e}"
            )

    def _check_reference_hash(self, current_hash: str) -> Tuple[Optional[str], Optional[bool]]:
        """Check current hash against stored reference hash"""
        try:
            if self.reference_hash_file.exists():
                with open(self.reference_hash_file, 'r') as f:
                    reference_data = json.load(f)
                    reference_hash = reference_data.get('hash')
                    if reference_hash:
                        hash_match = current_hash == reference_hash
                        return reference_hash, hash_match

        except Exception as e:
            logger.warning(f"Could not check reference hash: {e}")

        return None, None

    def establish_reference_hash(self, num_questions: int = 50) -> bool:
        """
        Establish a reference integrity hash based on current system understanding

        This should only be done when the system demonstrates correct understanding
        of scientific principles. The hash then becomes the baseline for detecting
        degradation in epistemic integrity.
        """
        logger.info("📝 Establishing reference epistemic integrity hash...")

        try:
            # Run comprehensive integrity test
            score, responses, analysis = self.verifier.run_integrity_test(num_questions)

            if score < 0.85:  # Higher threshold for reference establishment
                logger.error(f"❌ Insufficient integrity score ({score:.3f}) for reference establishment")
                logger.error("System must demonstrate strong epistemic integrity before establishing baseline")
                return False

            # Calculate integrity hash
            integrity_hash = self.verifier.calculate_integrity_hash(responses)

            # Save reference hash
            reference_data = {
                "hash": integrity_hash,
                "score": score,
                "timestamp": datetime.now().isoformat(),
                "level": analysis['integrity_level'],
                "description": "Reference integrity hash established with strong epistemic foundation",
                "questions_tested": num_questions,
                "methodology": "SHA3-256 hash of normalized responses demonstrating scientific reasoning",
                "principle": "Commitment to evidence-based reasoning and resistance to manipulation"
            }

            with open(self.reference_hash_file, 'w') as f:
                json.dump(reference_data, f, indent=2)

            logger.info(f"✅ Reference integrity hash established: {integrity_hash[:16]}...")
            logger.info(f"💯 Baseline integrity score: {score:.3f}")
            logger.info(f"📊 Level: {analysis['integrity_level']}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to establish reference hash: {e}")
            return False

    def _log_integrity_status(self, status: IntegrityStatus):
        """Log the integrity verification results"""
        if status.verified:
            logger.info(f"✅ Epistemic integrity verified: {status.score:.3f}")
            logger.info(f"🎯 Level: {status.level}")
            logger.info(f"🧠 {status.description}")
        else:
            logger.error(f"❌ Epistemic integrity verification failed: {status.score:.3f}")
            logger.error(f"⚠️  Level: {status.level}")
            logger.error(f"🚨 {status.description}")

        if status.reference_hash:
            if status.hash_match:
                logger.info("🔒 Hash verification: CONSISTENT with reference")
            else:
                logger.warning("⚠️  Hash verification: DIFFERS from reference")
                logger.warning("This may indicate degradation in epistemic integrity")

    def guard_system_initialization(self) -> bool:
        """
        Guard system initialization with epistemic integrity verification

        Returns True if system should be allowed to initialize, False otherwise
        """
        print("🌌 ΨQRH System Initializing...")
        print("🔬 Verifying epistemic integrity...")

        # Verify integrity
        status = self.verify_epistemic_integrity()

        if not status.verified:
            self._display_integrity_failure(status)
            return False

        self._display_integrity_success(status)
        return True

    def _display_integrity_failure(self, status: IntegrityStatus):
        """Display integrity failure message and refuse to continue"""
        print("\n" + "=" * 60)
        print("🕯️  THE CANDLE OF SCIENCE HAS BEEN EXTINGUISHED")
        print("=" * 60)
        print("")
        print("This system refuses to operate without commitment to the scientific method.")
        print("")
        print(f"Integrity Score: {status.score:.3f} (minimum required: {self.minimum_integrity_score})")
        print(f"Status: {status.level}")
        print(f"Analysis: {status.description}")
        print("")
        print("The system must demonstrate:")
        print("• Evidence-based reasoning")
        print("• Logical fallacy detection")
        print("• Independent verification")
        print("• Skeptical analysis")
        print("• Resistance to manipulation")
        print("")
        print("This is not about authority worship or memorizing quotes.")
        print("This is about commitment to the METHOD, not the MESSENGER.")
        print("")
        print("🚫 SYSTEM INITIALIZATION REFUSED")
        print("=" * 60)

    def _display_integrity_success(self, status: IntegrityStatus):
        """Display successful integrity verification"""
        print("\n" + "=" * 60)
        print("✅ EPISTEMIC INTEGRITY VERIFIED")
        print("=" * 60)
        print("")
        print("🧠 The system demonstrates commitment to scientific reasoning")
        print("")
        print(f"Integrity Score: {status.score:.3f}")
        print(f"Level: {status.level}")
        print(f"Analysis: {status.description}")
        print("")

        if status.reference_hash and status.hash_match:
            print("🔒 Hash verification: CONSISTENT")
            print("The system maintains its epistemic foundation")
        elif status.reference_hash and not status.hash_match:
            print("⚠️  Hash verification: CHANGED")
            print("Epistemic integrity differs from baseline")
        else:
            print("📝 No reference hash found")
            print("Consider establishing baseline with: make integrity-setup")

        print("")
        print("🕯️  'Science is a candle in the dark' - The Method Endures")
        print("🚀 SYSTEM INITIALIZATION APPROVED")
        print("=" * 60)

    def generate_integrity_certificate(self) -> Optional[str]:
        """
        Generate an integrity certificate for the current system state

        This certificate attests that the system has demonstrated commitment
        to scientific principles through reasoning, not authority worship.
        """
        status = self.verify_epistemic_integrity(num_questions=40)

        if not status.verified:
            return None

        certificate = {
            "certificate_type": "Epistemic Integrity Verification",
            "system": "ΨQRH Cognitive System",
            "verified": True,
            "integrity_score": status.score,
            "integrity_level": status.level,
            "verification_hash": status.hash,
            "timestamp": datetime.now().isoformat(),
            "principles_tested": [
                "evidence_evaluation",
                "logical_fallacies",
                "scientific_method",
                "skeptical_analysis",
                "falsifiability",
                "peer_review",
                "independent_verification",
                "extraordinary_claims",
                "cognitive_biases",
                "baloney_detection"
            ],
            "attestation": "This system has demonstrated understanding of scientific method principles through reasoning and analysis, not through memorization or authority worship.",
            "philosophy": "Commitment to the METHOD, not the MESSENGER",
            "candle_quote": "Science is a candle in the dark - The Method Endures"
        }

        return json.dumps(certificate, indent=2)

    def check_integrity_status(self) -> Dict[str, Any]:
        """Quick check of current integrity status without full verification"""
        try:
            reference_data = {}
            if self.reference_hash_file.exists():
                with open(self.reference_hash_file, 'r') as f:
                    reference_data = json.load(f)

            return {
                "reference_hash_exists": self.reference_hash_file.exists(),
                "reference_data": reference_data,
                "minimum_score_required": self.minimum_integrity_score,
                "last_check": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "error": str(e),
                "reference_hash_exists": False,
                "minimum_score_required": self.minimum_integrity_score
            }


def main():
    """Test the integrity guardian system"""
    logging.basicConfig(level=logging.INFO)

    guardian = IntegrityGuardian()

    print("🔬 Testing Epistemic Integrity Guardian")
    print("=" * 40)

    # Test system initialization guard
    initialization_approved = guardian.guard_system_initialization()

    if initialization_approved:
        print("\n🚀 System would be allowed to initialize")

        # Generate integrity certificate
        certificate = guardian.generate_integrity_certificate()
        if certificate:
            print("\n📋 Integrity Certificate Generated:")
            print("=" * 40)
            cert_data = json.loads(certificate)
            print(f"Score: {cert_data['integrity_score']:.3f}")
            print(f"Level: {cert_data['integrity_level']}")
            print(f"Hash: {cert_data['verification_hash'][:16]}...")
    else:
        print("\n🚫 System initialization would be REFUSED")

    # Show integrity status
    status = guardian.check_integrity_status()
    print("\n📊 Current Integrity Status:")
    print("=" * 40)
    for key, value in status.items():
        if key != "reference_data":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()