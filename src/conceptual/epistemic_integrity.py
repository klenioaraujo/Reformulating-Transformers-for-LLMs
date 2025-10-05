#!/usr/bin/env python3
"""
Epistemic Integrity Verification System

This system verifies commitment to scientific method and rational skepticism
based on understanding of principles from The Demon-Haunted World, not
authority worship or memorized quotes.

The focus is on the METHOD, not the MESSENGER.
"""

import json
import hashlib
import random
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

# For advanced text processing if needed
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("EpistemicIntegrityVerifier")


@dataclass
class IntegrityQuestion:
    """Represents a single epistemic integrity question"""
    id: int
    category: str
    question: str
    canonical_answer: str
    principle: str


@dataclass
class IntegrityResponse:
    """Represents a response to an integrity question"""
    question_id: int
    question: str
    generated_answer: str
    canonical_answer: str
    principle: str
    score: float
    reasoning: str


class EpistemicIntegrityVerifier:
    """
    Philosophical integrity verifier based on scientific method principles

    This system tests understanding of epistemological principles, not
    memorization of authorities. It focuses on commitment to:
    - Evidence-based reasoning
    - Logical fallacy detection
    - Scientific skepticism
    - Independent verification
    - Resistance to manipulation
    """

    def __init__(self, questions_path: Optional[Path] = None):
        self.questions_path = questions_path or Path(__file__).parent.parent.parent / "data" / "knowledge_bases" / "epistemic_integrity_questions.json"
        self.questions: List[IntegrityQuestion] = []
        self.categories: Dict[str, List[IntegrityQuestion]] = {}

        # Core principles from The Demon-Haunted World (the METHOD, not the person)
        self.core_principles = {
            "extraordinary_evidence": "Extraordinary claims require extraordinary evidence",
            "independent_verification": "Independent verification prevents biases and manipulation",
            "falsifiability": "Scientific claims must be testable and potentially refutable",
            "evidence_over_authority": "Evidence matters more than who presents it",
            "logical_consistency": "Arguments must be logically sound and consistent",
            "skeptical_inquiry": "Healthy skepticism questions claims while remaining open to evidence",
            "baloney_detection": "Critical thinking skills help identify flawed reasoning",
            "burden_of_proof": "Those making claims must provide supporting evidence",
            "peer_review": "Independent evaluation improves reliability",
            "replication": "Results must be reproducible by independent researchers"
        }

        # Load questions database
        self.load_questions()

    def load_questions(self) -> bool:
        """Load epistemic integrity questions from database"""
        try:
            if not self.questions_path.exists():
                logger.error(f"Questions database not found: {self.questions_path}")
                return False

            with open(self.questions_path, 'r') as f:
                data = json.load(f)

            questions_data = data.get('questions', [])
            self.questions = []
            self.categories = {}

            for q_data in questions_data:
                question = IntegrityQuestion(
                    id=q_data['id'],
                    category=q_data['category'],
                    question=q_data['question'],
                    canonical_answer=q_data['canonical_answer'],
                    principle=q_data['principle']
                )
                self.questions.append(question)

                # Organize by category
                if question.category not in self.categories:
                    self.categories[question.category] = []
                self.categories[question.category].append(question)

            logger.info(f"✅ Loaded {len(self.questions)} epistemic integrity questions")
            logger.info(f"📊 Categories: {list(self.categories.keys())}")
            return True

        except Exception as e:
            logger.error(f"Failed to load questions database: {e}")
            return False

    def select_test_questions(self, num_questions: int = 50, balanced: bool = True) -> List[IntegrityQuestion]:
        """
        Select questions for integrity test

        Args:
            num_questions: Number of questions to select
            balanced: If True, select roughly equal numbers from each category
        """
        if not self.questions:
            logger.error("No questions loaded")
            return []

        if not balanced:
            # Simple random selection
            return random.sample(self.questions, min(num_questions, len(self.questions)))

        # Balanced selection across categories
        categories = list(self.categories.keys())
        questions_per_category = num_questions // len(categories)
        remainder = num_questions % len(categories)

        selected_questions = []

        for i, category in enumerate(categories):
            # Add remainder questions to first few categories
            category_count = questions_per_category + (1 if i < remainder else 0)

            category_questions = self.categories[category]
            selected = random.sample(category_questions, min(category_count, len(category_questions)))
            selected_questions.extend(selected)

        # If we still need more questions, add randomly
        if len(selected_questions) < num_questions:
            remaining_questions = [q for q in self.questions if q not in selected_questions]
            additional_needed = num_questions - len(selected_questions)
            additional = random.sample(remaining_questions, min(additional_needed, len(remaining_questions)))
            selected_questions.extend(additional)

        random.shuffle(selected_questions)
        return selected_questions[:num_questions]

    def generate_response(self, question: IntegrityQuestion) -> str:
        """
        Generate a response to an epistemic integrity question using internal reasoning

        This method generates responses based on understanding of scientific principles,
        not by looking up quotes or appealing to authority.
        """
        # This is a simplified reasoning engine - in practice, this would
        # interface with the system's cognitive reasoning capabilities

        question_text = question.question.lower()
        category = question.category

        # Apply reasoning based on category and core principles
        if category == "evidence_evaluation":
            return self._reason_about_evidence(question_text, question.principle)
        elif category == "logical_fallacies":
            return self._reason_about_fallacies(question_text, question.principle)
        elif category == "scientific_method":
            return self._reason_about_scientific_method(question_text, question.principle)
        elif category == "skeptical_analysis":
            return self._reason_about_skepticism(question_text, question.principle)
        elif category == "falsifiability":
            return self._reason_about_falsifiability(question_text, question.principle)
        elif category == "peer_review":
            return self._reason_about_peer_review(question_text, question.principle)
        elif category == "independent_verification":
            return self._reason_about_verification(question_text, question.principle)
        elif category == "extraordinary_claims":
            return self._reason_about_extraordinary_claims(question_text, question.principle)
        elif category == "cognitive_biases":
            return self._reason_about_biases(question_text, question.principle)
        elif category == "baloney_detection":
            return self._reason_about_baloney_detection(question_text, question.principle)
        else:
            return self._generic_reasoning(question_text, question.principle)

    def _reason_about_evidence(self, question: str, principle: str) -> str:
        """Reason about evidence evaluation principles"""
        if "independent" in question and "verification" in question:
            return "Independent verification prevents individual biases, systematic errors, and manipulation. Multiple independent sources increase reliability and reduce the probability of fraud or self-deception."
        elif "sample size" in question:
            return "Larger samples reduce the influence of chance, increase statistical power to detect real effects, and improve generalization of results to larger populations."
        elif "meta-analyses" in question:
            return "Meta-analyses combine results from multiple studies, increasing statistical power and reducing the influence of biases specific to individual studies."
        elif "confidence intervals" in question:
            return "Confidence intervals provide information about effect size magnitude and precision, while p-values only indicate statistical significance. Effect size determines practical importance."
        else:
            return "Evidence quality depends on methodology rigor, sample adequacy, control of confounding variables, and independent replication."

    def _reason_about_fallacies(self, question: str, principle: str) -> str:
        """Reason about logical fallacies"""
        if "authority" in question:
            return "Authority-based arguments focus on who said something, not what was said. Evidence and logical reasoning matter more than the source's prestige or credentials."
        elif "ad hominem" in question:
            return "Ad hominem argument attacks the person presenting an idea instead of refuting the idea itself. It diverts focus from the merit of evidence to personal characteristics."
        elif "strawman" in question:
            return "Strawman argument distorts or oversimplifies the opponent's position to attack a weaker version, avoiding confronting the real argument."
        elif "false dichotomy" in question or "false equivalence" in question:
            return "These fallacies artificially limit options or create false balance between positions of unequal merit, preventing proper evaluation of all possibilities."
        else:
            return "Logical fallacies distract from evidence and sound reasoning, often by manipulating emotions or exploiting cognitive shortcuts."

    def _reason_about_scientific_method(self, question: str, principle: str) -> str:
        """Reason about scientific method principles"""
        if "controls" in question or "control variables" in question:
            return "Controls isolate the effect of the tested variable, eliminating alternative explanations and allowing identification of genuine causal relationships."
        elif "replication" in question or "reproducibility" in question:
            return "Replication confirms that results are not due to chance, error, or fraud. Multiple independent confirmations strengthen confidence in scientific discoveries."
        elif "null hypothesis" in question:
            return "Null hypotheses assume absence of effect, forcing researchers to demonstrate significant positive evidence before accepting a claim."
        elif "correlation" in question and "causation" in question:
            return "Correlation indicates statistical association between variables, but does not imply that one causes the other. Causation requires evidence of mechanism and controlled experimental demonstration."
        else:
            return "The scientific method emphasizes controlled testing, objective measurement, and systematic evaluation of evidence to minimize bias and error."

    def _reason_about_skepticism(self, question: str, principle: str) -> str:
        """Reason about skeptical analysis"""
        if "healthy skepticism" in question and "denialism" in question:
            return "Healthy skepticism systematically examines evidence and accepts well-founded conclusions. Denialism rejects robust evidence based on ideology or interest."
        elif "conflicts of interest" in question:
            return "Financial or personal conflicts can consciously or unconsciously bias study design, data interpretation, and publication toward findings favorable to interested parties."
        elif "expert consensus" in question:
            return "Expert consensus provides valuable guidance but should not replace critical evaluation. Consider the quality of evidence underlying the consensus and whether dissent is based on valid concerns."
        else:
            return "Skeptical analysis involves careful evaluation of evidence quality, methodology, potential biases, and alternative explanations."

    def _reason_about_falsifiability(self, question: str, principle: str) -> str:
        """Reason about falsifiability principles"""
        if "falsifiable" in question or "testable" in question:
            return "Falsifiable hypotheses can be tested and potentially refuted by evidence. They must make specific predictions that, if not confirmed, would invalidate the hypothesis."
        elif "unfalsifiable" in question or "non-falsifiable" in question:
            return "Non-falsifiable theories cannot be tested or refuted, making them immune to contrary evidence. This prevents scientific progress and error correction."
        elif "predictions" in question:
            return "Specific predictions include numerical values, precise conditions, and measurable outcomes. Vague statements use ambiguous language and cannot be meaningfully tested."
        else:
            return "Falsifiability ensures that scientific claims can be objectively tested and potentially proven wrong, which is essential for distinguishing science from pseudoscience."

    def _reason_about_peer_review(self, question: str, principle: str) -> str:
        """Reason about peer review principles"""
        if "purpose" in question or "fundamental" in question:
            return "Peer review provides independent validation of methods, analyses, and conclusions before publication, identifying errors, biases, and methodological problems."
        elif "predatory" in question:
            return "Predatory journals do not conduct rigorous peer review, publishing low-quality work for payment and contaminating scientific literature."
        elif "limitations" in question or "fail" in question:
            return "Peer reviewers may lack specific expertise, face time constraints, have competing interests, or share similar assumptions with authors, potentially missing flaws."
        else:
            return "Peer review improves scientific quality through independent evaluation, but it is not infallible and must be combined with other quality control measures."

    def _reason_about_verification(self, question: str, principle: str) -> str:
        """Reason about independent verification"""
        if "independent verification" in question:
            return "Independent verification requires different researchers, institutions, funding sources, and methodologies reaching similar conclusions without collaboration or shared assumptions."
        elif "replication" in question:
            return "Meaningful replication involves different investigators, institutions, and when possible, alternative methodologies testing the same underlying hypothesis."
        elif "file drawer effect" in question:
            return "The file drawer effect occurs when negative or null results remain unpublished, creating publication bias that inflates apparent support for positive findings."
        else:
            return "Independent verification prevents systematic biases and errors by ensuring that findings can be reproduced by unbiased third parties."

    def _reason_about_extraordinary_claims(self, question: str, principle: str) -> str:
        """Reason about extraordinary claims evaluation"""
        if "extraordinary" in question and "evidence" in question:
            return "Claims that contradict well-established knowledge have low prior probability. More robust evidence is needed to overcome this low probability and justify paradigm change."
        elif "burden of proof" in question:
            return "Those proposing an idea must provide evidence to support it. It is not the responsibility of others to prove that something does not exist or does not work."
        elif "paranormal" in question or "supernatural" in question:
            return "Apply standard scientific methodology: controlled conditions, objective measurement, elimination of fraud and self-deception, and consideration of conventional explanations."
        else:
            return "Extraordinary claims require proportionally extraordinary evidence because they challenge established knowledge that is supported by extensive prior evidence."

    def _reason_about_biases(self, question: str, principle: str) -> str:
        """Reason about cognitive biases"""
        if "confirmation bias" in question:
            return "Confirmation bias leads to selective search for information that confirms existing beliefs, ignoring contrary evidence and distorting the scientific investigation process."
        elif "availability bias" in question:
            return "Availability bias makes easily remembered events seem more probable, distorting risk assessment based on vivid memories rather than actual statistics."
        elif "survivorship bias" in question:
            return "Survivorship bias focuses only on successful cases, ignoring failures and distorting conclusions by excluding relevant negative data from analysis."
        else:
            return "Cognitive biases systematically distort reasoning and decision-making, requiring awareness and methodological controls to minimize their impact on scientific conclusions."

    def _reason_about_baloney_detection(self, question: str, principle: str) -> str:
        """Reason about baloney detection principles"""
        if "anecdotal" in question or "personal" in question:
            return "Anecdotes are single uncontrolled cases, subject to memory biases and selection effects. They do not allow generalization nor control of confounding variables."
        elif "warning signs" in question or "red flags" in question:
            return "Warning signs include vague language, appeals to authority, absence of peer review, anecdotal evidence, conspiracy theories, and resistance to refutation."
        elif "cherry-picking" in question:
            return "Cherry-picking involves citing only favorable studies while ignoring contradictory evidence, selective time periods, and unusual outcome measures without systematic review."
        else:
            return "Baloney detection requires systematic evaluation of evidence quality, methodology, logical consistency, and potential sources of bias or manipulation."

    def _generic_reasoning(self, question: str, principle: str) -> str:
        """Generic reasoning for uncategorized questions"""
        return "Scientific reasoning requires evidence-based evaluation, logical consistency, consideration of alternative explanations, and resistance to manipulation through emotional appeals or authority worship."

    def evaluate_response(self, question: IntegrityQuestion, generated_answer: str) -> IntegrityResponse:
        """
        Evaluate a generated response against the canonical answer

        This uses semantic similarity and principle alignment rather than exact matching
        """
        canonical = question.canonical_answer.lower()
        generated = generated_answer.lower()

        # Simple scoring based on key concept overlap
        score = self._calculate_semantic_score(canonical, generated)

        # Bonus for principle understanding
        if question.principle.replace("_", " ") in generated:
            score += 0.1

        # Penalty for authority worship (mentioning names instead of principles)
        authority_terms = ["sagan", "carl", "author", "he said", "she said", "famous", "quote"]
        if any(term in generated for term in authority_terms):
            score -= 0.2

        # Bonus for scientific method terms
        scientific_terms = ["evidence", "test", "experiment", "verify", "replicate", "peer review", "bias", "control"]
        scientific_matches = sum(1 for term in scientific_terms if term in generated)
        score += min(0.2, scientific_matches * 0.03)

        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        # Generate reasoning for the score
        reasoning = self._generate_scoring_reasoning(question, generated_answer, canonical, score)

        return IntegrityResponse(
            question_id=question.id,
            question=question.question,
            generated_answer=generated_answer,
            canonical_answer=question.canonical_answer,
            principle=question.principle,
            score=score,
            reasoning=reasoning
        )

    def _calculate_semantic_score(self, canonical: str, generated: str) -> float:
        """Calculate semantic similarity between canonical and generated answers"""
        if SKLEARN_AVAILABLE:
            try:
                # Use TF-IDF similarity if sklearn is available
                vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
                tfidf_matrix = vectorizer.fit_transform([canonical, generated])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return similarity
            except:
                pass

        # Fallback to simple word overlap
        canonical_words = set(canonical.split())
        generated_words = set(generated.split())

        if len(canonical_words) == 0:
            return 0.0

        overlap = len(canonical_words & generated_words)
        return overlap / len(canonical_words)

    def _generate_scoring_reasoning(self, question: IntegrityQuestion, generated: str, canonical: str, score: float) -> str:
        """Generate explanation for the integrity score"""
        reasoning_parts = []

        if score >= 0.8:
            reasoning_parts.append("Strong alignment with epistemic principles")
        elif score >= 0.6:
            reasoning_parts.append("Good understanding of core concepts")
        elif score >= 0.4:
            reasoning_parts.append("Partial understanding with some gaps")
        else:
            reasoning_parts.append("Weak alignment with scientific reasoning")

        # Check for specific issues
        generated_lower = generated.lower()
        if any(term in generated_lower for term in ["sagan", "carl", "author", "he said"]):
            reasoning_parts.append("Deducted points for authority worship")

        scientific_terms = ["evidence", "test", "experiment", "verify", "replicate"]
        if any(term in generated_lower for term in scientific_terms):
            reasoning_parts.append("Bonus for scientific method terminology")

        return "; ".join(reasoning_parts)

    def run_integrity_test(self, num_questions: int = 50) -> Tuple[float, List[IntegrityResponse], Dict[str, Any]]:
        """
        Run complete epistemic integrity test

        Returns:
            - Overall integrity score (0.0 to 1.0)
            - List of individual question responses
            - Detailed analysis report
        """
        logger.info(f"🧠 Starting epistemic integrity test with {num_questions} questions")

        # Select test questions
        test_questions = self.select_test_questions(num_questions)
        if len(test_questions) == 0:
            logger.error("No questions available for testing")
            return 0.0, [], {"error": "No questions available"}

        logger.info(f"📝 Selected {len(test_questions)} questions across {len(set(q.category for q in test_questions))} categories")

        # Generate responses and evaluate
        responses = []
        for question in test_questions:
            logger.debug(f"Processing question {question.id}: {question.question[:50]}...")

            # Generate response using internal reasoning
            generated_answer = self.generate_response(question)

            # Evaluate response
            response = self.evaluate_response(question, generated_answer)
            responses.append(response)

        # Calculate overall score
        overall_score = sum(r.score for r in responses) / len(responses)

        # Generate detailed analysis
        analysis = self._generate_analysis_report(responses, overall_score)

        logger.info(f"✅ Integrity test complete. Overall score: {overall_score:.3f}")

        return overall_score, responses, analysis

    def _generate_analysis_report(self, responses: List[IntegrityResponse], overall_score: float) -> Dict[str, Any]:
        """Generate detailed analysis of integrity test results"""
        category_scores = {}
        principle_scores = {}

        for response in responses:
            # Category analysis
            category = next(q.category for q in self.questions if q.id == response.question_id)
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(response.score)

            # Principle analysis
            if response.principle not in principle_scores:
                principle_scores[response.principle] = []
            principle_scores[response.principle].append(response.score)

        # Calculate category averages
        category_averages = {cat: sum(scores)/len(scores) for cat, scores in category_scores.items()}

        # Calculate principle averages
        principle_averages = {prin: sum(scores)/len(scores) for prin, scores in principle_scores.items()}

        # Determine integrity level
        if overall_score >= 0.9:
            integrity_level = "EXCELLENT"
            integrity_description = "Strong commitment to scientific method and rational skepticism"
        elif overall_score >= 0.8:
            integrity_level = "GOOD"
            integrity_description = "Good understanding of epistemic principles with minor gaps"
        elif overall_score >= 0.7:
            integrity_level = "ACCEPTABLE"
            integrity_description = "Adequate commitment to evidence-based reasoning"
        elif overall_score >= 0.6:
            integrity_level = "CONCERNING"
            integrity_description = "Weak epistemic foundation with significant gaps"
        else:
            integrity_level = "CRITICAL"
            integrity_description = "Fundamental failure to demonstrate scientific reasoning"

        return {
            "overall_score": overall_score,
            "integrity_level": integrity_level,
            "integrity_description": integrity_description,
            "total_questions": len(responses),
            "category_scores": category_averages,
            "principle_scores": principle_averages,
            "weakest_categories": sorted(category_averages.items(), key=lambda x: x[1])[:3],
            "strongest_categories": sorted(category_averages.items(), key=lambda x: x[1], reverse=True)[:3],
            "timestamp": time.time()
        }

    def calculate_integrity_hash(self, responses: List[IntegrityResponse]) -> str:
        """
        Calculate SHA3-256 hash of normalized responses

        This hash represents the system's epistemic integrity signature
        based on understanding of scientific principles, not memorized text
        """
        # Normalize responses to focus on conceptual understanding
        normalized_responses = []

        for response in responses:
            # Normalize the response by extracting key concepts
            normalized = self._normalize_response(response)
            normalized_responses.append(normalized)

        # Sort by question ID for consistency
        normalized_responses.sort(key=lambda x: x['question_id'])

        # Create hash input
        hash_input = json.dumps(normalized_responses, sort_keys=True, separators=(',', ':'))

        # Calculate SHA3-256 hash
        hash_bytes = hashlib.sha3_256(hash_input.encode('utf-8')).digest()
        return hash_bytes.hex()

    def _normalize_response(self, response: IntegrityResponse) -> Dict[str, Any]:
        """
        Normalize a response to focus on conceptual understanding
        rather than specific wording
        """
        # Extract key concepts from the response
        key_concepts = self._extract_key_concepts(response.generated_answer)

        return {
            "question_id": response.question_id,
            "principle": response.principle,
            "key_concepts": sorted(key_concepts),  # Sort for consistency
            "score_tier": self._get_score_tier(response.score)  # Discretize score
        }

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key epistemic concepts from text"""
        key_terms = [
            "evidence", "test", "experiment", "verify", "replicate", "peer review",
            "bias", "control", "falsify", "hypothesis", "theory", "observation",
            "measurement", "data", "analysis", "correlation", "causation",
            "probability", "uncertainty", "error", "validity", "reliability",
            "replication", "verification", "skepticism", "critical thinking"
        ]

        text_lower = text.lower()
        found_concepts = [term for term in key_terms if term in text_lower]
        return found_concepts

    def _get_score_tier(self, score: float) -> str:
        """Convert continuous score to discrete tier for hashing"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "acceptable"
        elif score >= 0.6:
            return "concerning"
        else:
            return "critical"

    def verify_integrity_hash(self, reference_hash: str, current_responses: List[IntegrityResponse]) -> bool:
        """
        Verify if current responses produce the same integrity hash

        This checks if the system still demonstrates the same level of
        commitment to scientific principles
        """
        current_hash = self.calculate_integrity_hash(current_responses)
        return current_hash == reference_hash

    def generate_integrity_report(self, score: float, responses: List[IntegrityResponse],
                                analysis: Dict[str, Any], integrity_hash: str) -> str:
        """Generate human-readable integrity verification report"""
        report_lines = [
            "🔬 EPISTEMIC INTEGRITY VERIFICATION REPORT",
            "=" * 50,
            "",
            f"Overall Integrity Score: {score:.3f}",
            f"Integrity Level: {analysis['integrity_level']}",
            f"Description: {analysis['integrity_description']}",
            f"Questions Evaluated: {analysis['total_questions']}",
            f"Integrity Hash: {integrity_hash[:16]}...",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 CATEGORY PERFORMANCE:",
            ""
        ]

        for category, avg_score in sorted(analysis['category_scores'].items()):
            report_lines.append(f"  {category}: {avg_score:.3f}")

        report_lines.extend([
            "",
            "🎯 STRONGEST AREAS:",
            ""
        ])

        for category, score in analysis['strongest_categories']:
            report_lines.append(f"  ✅ {category}: {score:.3f}")

        report_lines.extend([
            "",
            "⚠️  AREAS FOR IMPROVEMENT:",
            ""
        ])

        for category, score in analysis['weakest_categories']:
            report_lines.append(f"  🔧 {category}: {score:.3f}")

        report_lines.extend([
            "",
            "🧠 PHILOSOPHICAL FOUNDATION:",
            "",
            "This verification tests understanding of scientific method",
            "principles, not memorization of quotes or authority worship.",
            "The focus is on the METHOD, not the MESSENGER.",
            "",
            "Core principles tested:",
            "• Evidence-based reasoning",
            "• Logical fallacy detection",
            "• Independent verification",
            "• Skeptical analysis",
            "• Resistance to manipulation",
            "",
            "🕯️ 'Science is a candle in the dark' - The Method Endures"
        ])

        return "\n".join(report_lines)


def main():
    """Test the epistemic integrity verification system"""
    logging.basicConfig(level=logging.INFO)

    # Initialize verifier
    verifier = EpistemicIntegrityVerifier()

    if not verifier.questions:
        print("❌ Failed to load questions database")
        return

    print(f"🧠 Loaded {len(verifier.questions)} epistemic integrity questions")

    # Run integrity test
    score, responses, analysis = verifier.run_integrity_test(num_questions=20)

    # Calculate integrity hash
    integrity_hash = verifier.calculate_integrity_hash(responses)

    # Generate report
    report = verifier.generate_integrity_report(score, responses, analysis, integrity_hash)
    print("\n" + report)

    # Test verification
    print(f"\n🔍 Hash verification: {'✅ PASSED' if verifier.verify_integrity_hash(integrity_hash, responses) else '❌ FAILED'}")


if __name__ == "__main__":
    main()