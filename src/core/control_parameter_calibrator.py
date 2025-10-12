#!/usr/bin/env python3
"""
Control Parameter Calibrator for ΨQRH Pipeline
==============================================

Auto-calibrates control parameters for generation and optimization:

Parameters calibrated:
- temperature: Based on creativity vs precision requirements
- top_k: Based on diversity requirements
- learning_rate: Based on convergence requirements
"""

import torch
import math
from typing import Dict, Any
import re


class ControlParameterCalibrator:
    """
    Calibrates control parameters for generation and training
    """

    def __init__(self):
        """Initialize the control parameter calibrator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze_creativity_requirement(self, text: str) -> float:
        """
        Analyze creativity requirement based on text characteristics

        Args:
            text: Input text

        Returns:
            Creativity score (0-1, higher = more creativity needed)
        """
        # Creative tasks often involve generation, imagination, open-ended questions
        creative_indicators = [
            'create', 'imagine', 'design', 'generate', 'what if', 'suppose',
            'invent', 'compose', 'write', 'draw', 'build', 'make',
            'how would you', 'can you', 'tell me about', 'describe'
        ]

        # Precision tasks involve facts, calculations, analysis
        precision_indicators = [
            'calculate', 'compute', 'analyze', 'explain', 'prove', 'define',
            'what is', 'how does', 'why does', 'when did', 'where is'
        ]

        text_lower = text.lower()

        creative_score = sum(1 for indicator in creative_indicators if indicator in text_lower)
        precision_score = sum(1 for indicator in precision_indicators if indicator in text_lower)

        total_indicators = creative_score + precision_score

        if total_indicators == 0:
            # Default to moderate creativity for neutral text
            return 0.5

        # Normalize creativity score
        creativity_ratio = creative_score / total_indicators

        # Adjust based on text length (longer text may need more precision)
        length_factor = min(len(text) / 500.0, 1.0)
        creativity_ratio = creativity_ratio * (1 - length_factor * 0.3)

        return creativity_ratio

    def analyze_diversity_requirement(self, text: str) -> float:
        """
        Analyze diversity requirement for top-k sampling

        Args:
            text: Input text

        Returns:
            Diversity score (0-1, higher = more diversity needed)
        """
        # Tasks requiring diverse outputs
        diversity_indicators = [
            'options', 'choices', 'alternatives', 'possibilities', 'ways',
            'examples', 'ideas', 'suggestions', 'recommendations', 'list'
        ]

        # Tasks requiring focused/precise outputs
        focus_indicators = [
            'exact', 'precise', 'specific', 'correct', 'right', 'answer',
            'solution', 'result', 'value', 'number'
        ]

        text_lower = text.lower()

        diversity_score = sum(1 for indicator in diversity_indicators if indicator in text_lower)
        focus_score = sum(1 for indicator in focus_indicators if indicator in text_lower)

        total_indicators = diversity_score + focus_score

        if total_indicators == 0:
            # Default diversity based on question marks (open-ended questions)
            question_count = text.count('?')
            return min(question_count * 0.2, 0.8)

        diversity_ratio = diversity_score / total_indicators

        # Boost diversity for creative tasks
        creativity_boost = self.analyze_creativity_requirement(text) * 0.2
        diversity_ratio += creativity_boost

        return min(diversity_ratio, 1.0)

    def analyze_convergence_requirement(self, text: str) -> float:
        """
        Analyze convergence requirements for learning rate

        Args:
            text: Input text

        Returns:
            Convergence difficulty score (0-1, higher = harder convergence)
        """
        # Complex tasks require more careful learning
        complexity_indicators = [
            'complex', 'difficult', 'advanced', 'sophisticated', 'complicated',
            'optimize', 'fine-tune', 'precise', 'accurate', 'stable'
        ]

        # Simple tasks can use more aggressive learning
        simplicity_indicators = [
            'simple', 'basic', 'easy', 'straightforward', 'obvious',
            'quick', 'fast', 'approximate', 'rough'
        ]

        text_lower = text.lower()

        complexity_score = sum(1 for indicator in complexity_indicators if indicator in text_lower)
        simplicity_score = sum(1 for indicator in simplicity_indicators if indicator in text_lower)

        # Length-based complexity
        length_complexity = min(len(text) / 1000.0, 1.0)

        # Lexical complexity
        words = re.findall(r'\b\w+\b', text)
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            lexical_complexity = min(avg_word_length / 8.0, 1.0)
        else:
            lexical_complexity = 0.0

        # Combined complexity score
        indicator_score = complexity_score / max(complexity_score + simplicity_score, 1)
        total_complexity = (indicator_score + length_complexity + lexical_complexity) / 3.0

        return total_complexity

    def calibrate_temperature(self, creativity_requirement: float) -> float:
        """
        Calibrate temperature based on creativity requirements

        temperature = max(0.1, min(2.0, 0.5 + creativity_factor * 1.5))
        """
        # Higher creativity = higher temperature
        temperature = 0.5 + creativity_requirement * 1.5  # Range: 0.5-2.0

        # Clamp to reasonable range
        temperature = max(0.1, min(2.0, temperature))

        return temperature

    def calibrate_top_k(self, diversity_requirement: float) -> int:
        """
        Calibrate top-k based on diversity requirements

        top_k = max(5, min(50, int(10 + diversity_factor * 40)))
        """
        # Higher diversity = higher top_k
        top_k = int(10 + diversity_requirement * 40)  # Range: 10-50

        # Clamp to reasonable range
        top_k = max(5, min(50, top_k))

        return top_k

    def calibrate_learning_rate(self, convergence_difficulty: float) -> float:
        """
        Calibrate learning rate based on convergence requirements

        learning_rate = 1e-4 * (1 / (1 + difficulty_factor))
        """
        # Higher difficulty = lower learning rate
        difficulty_factor = convergence_difficulty * 2.0

        learning_rate = 1e-4 * math.exp(-difficulty_factor * 0.8)  # Exponential decay

        # Clamp to reasonable range
        learning_rate = max(1e-6, min(1e-3, learning_rate))

        return learning_rate

    def calibrate_all(self, text: str) -> Dict[str, Any]:
        """
        Calibrate all control parameters

        Args:
            text: Input text

        Returns:
            Dict with calibrated control parameters
        """
        # Analyze control requirements
        creativity_requirement = self.analyze_creativity_requirement(text)
        diversity_requirement = self.analyze_diversity_requirement(text)
        convergence_difficulty = self.analyze_convergence_requirement(text)

        # Calibrate parameters
        temperature = self.calibrate_temperature(creativity_requirement)
        top_k = self.calibrate_top_k(diversity_requirement)
        learning_rate = self.calibrate_learning_rate(convergence_difficulty)

        return {
            'temperature': temperature,
            'top_k': top_k,
            'learning_rate': learning_rate,
            # Include analysis metrics for debugging
            'creativity_requirement': creativity_requirement,
            'diversity_requirement': diversity_requirement,
            'convergence_difficulty': convergence_difficulty
        }

    def validate_control_consistency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate control parameter consistency

        Args:
            params: Calibrated parameters

        Returns:
            Validation results
        """
        temperature = params['temperature']
        top_k = params['top_k']
        learning_rate = params['learning_rate']

        # Check parameter ranges
        range_checks = {
            'temperature_range': 0.1 <= temperature <= 2.0,
            'top_k_range': 5 <= top_k <= 50,
            'learning_rate_range': 1e-6 <= learning_rate <= 1e-3
        }

        # Control consistency checks
        control_checks = {
            'temperature_positive': temperature > 0,
            'top_k_positive': top_k > 0,
            'learning_rate_positive': learning_rate > 0,
            'learning_rate_reasonable': learning_rate <= 1e-3  # Not too high
        }

        all_checks_pass = all(range_checks.values()) and all(control_checks.values())

        return {
            'range_checks': range_checks,
            'control_checks': control_checks,
            'all_checks_pass': all_checks_pass
        }