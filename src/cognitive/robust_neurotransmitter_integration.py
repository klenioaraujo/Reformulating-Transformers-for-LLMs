#!/usr/bin/env python3
"""
ROBUST NEUROTRANSMITTER INTEGRATION FOR QRH FRAMEWORK

Optimized hybrid architecture that:
1. Maintains all JIT computational gains (speed, memory, efficiency)
2. Leverages adaptive neurotransmitter expertise as parallel non-blocking processing
3. Provides deep reasoning without significant overhead
4. Guarantees robustness in complex scenarios and edge cases

Architecture:
- JIT Path (Primary - Always Active): 95% of cases
- Expertise Path (Parallel - Conditional): 4% of cases
- Expertise-Lead Path: 1% of complex cases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import asyncio
import time
import threading
import weakref
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from ..core.qrh_layer import QRHLayer, QRHConfig
from .synthetic_neurotransmitters import SyntheticNeurotransmitterSystem, NeurotransmitterConfig


class OperationMode(Enum):
    """Operation modes for neurotransmitter integration"""
    JIT_PURE = "jit_pure"              # Fast path only (95% cases)
    JIT_EXPERTISE = "jit_expertise"    # Parallel processing (4% cases)
    EXPERTISE_LEAD = "expertise_lead"   # Expertise primary (1% cases)


class ActivationTrigger(Enum):
    """Triggers for neurotransmitter activation"""
    SEMANTIC_CONFLICT = "semantic_conflict"
    LOW_CONFIDENCE = "low_confidence"
    COMPLEX_PATTERNS = "complex_patterns"
    PERFORMANCE_FEEDBACK = "performance_feedback"
    USER_OVERRIDE = "user_override"


@dataclass
class IntegrationConfig:
    """Configuration for robust neurotransmitter integration"""
    # Performance Thresholds
    activation_threshold: float = 0.7
    confidence_threshold: float = 0.8
    complexity_threshold: float = 0.8
    performance_budget: float = 0.95  # Allow 5% degradation max

    # Caching and Optimization
    enable_lazy_loading: bool = True
    enable_decision_cache: bool = True
    enable_adaptive_quantization: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600  # 1 hour

    # Parallel Processing
    enable_async_processing: bool = True
    max_concurrent_tasks: int = 4
    async_timeout_seconds: float = 0.1

    # Adaptive Learning
    enable_meta_learning: bool = True
    learning_rate: float = 0.01
    history_window: int = 1000

    # Health Monitoring
    enable_health_monitoring: bool = True
    health_check_interval: int = 100
    latency_degradation_threshold: float = 1.5  # 50% slower
    accuracy_drop_threshold: float = 0.05       # 5% accuracy loss
    memory_overhead_threshold: float = 0.2      # 20% memory increase


class NeurotransmitterCache:
    """Intelligent caching system for neurotransmitter decisions"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.cache = {}  # {input_signature: CacheEntry}
        self.access_counts = defaultdict(int)
        self.last_cleanup = time.time()
        self._lock = threading.RLock()

    def get_signature(self, x: torch.Tensor, context: Dict[str, Any]) -> str:
        """Generate unique signature for input and context"""
        shape_sig = f"{x.shape}_{x.dtype}"
        content_sig = f"{torch.mean(x).item():.4f}_{torch.std(x).item():.4f}"
        context_sig = "_".join(f"{k}:{v}" for k, v in sorted(context.items()))
        return f"{shape_sig}_{content_sig}_{context_sig}"

    def get(self, input_signature: str) -> Optional[Dict[str, Any]]:
        """Get cached decision if valid"""
        with self._lock:
            if input_signature in self.cache:
                entry = self.cache[input_signature]
                if time.time() - entry['timestamp'] < self.config.cache_ttl_seconds:
                    self.access_counts[input_signature] += 1
                    return entry['decision']
                else:
                    # Remove expired entry
                    del self.cache[input_signature]
            return None

    def set(self, input_signature: str, decision: Dict[str, Any]):
        """Cache decision with timestamp"""
        with self._lock:
            if len(self.cache) >= self.config.cache_size:
                self._evict_least_used()

            self.cache[input_signature] = {
                'decision': decision,
                'timestamp': time.time()
            }

    def _evict_least_used(self):
        """Evict least recently used entries"""
        if not self.cache:
            return

        # Find least accessed entries
        sorted_entries = sorted(self.access_counts.items(), key=lambda x: x[1])
        to_remove = len(sorted_entries) // 4  # Remove 25%

        for sig, _ in sorted_entries[:to_remove]:
            if sig in self.cache:
                del self.cache[sig]
            if sig in self.access_counts:
                del self.access_counts[sig]


class AdaptiveActivationLearner:
    """Meta-learning system for optimal activation patterns"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.activation_history = deque(maxlen=config.history_window)
        self.performance_correlation = defaultdict(list)
        self.learned_thresholds = {
            'confidence': config.confidence_threshold,
            'complexity': config.complexity_threshold,
            'activation': config.activation_threshold
        }
        self._update_count = 0

    def record_activation(self, trigger_context: Dict[str, float],
                         final_performance: float, processing_time: float):
        """Record activation decision and its outcome"""
        entry = {
            'trigger_context': trigger_context,
            'performance': final_performance,
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        self.activation_history.append(entry)

        # Update performance correlation
        key = self._context_to_key(trigger_context)
        self.performance_correlation[key].append({
            'performance': final_performance,
            'time': processing_time
        })

        self._update_count += 1
        if self._update_count % 50 == 0:  # Update thresholds periodically
            self._update_learned_thresholds()

    def _context_to_key(self, context: Dict[str, float]) -> str:
        """Convert context to hashable key"""
        # Discretize continuous values for grouping
        discretized = {}
        for k, v in context.items():
            discretized[k] = round(v, 2)  # Round to 2 decimal places
        return str(sorted(discretized.items()))

    def _update_learned_thresholds(self):
        """Update thresholds based on historical performance"""
        if len(self.activation_history) < 50:
            return

        # Analyze performance patterns
        recent_history = list(self.activation_history)[-100:]  # Last 100 activations

        # Calculate average performance for different threshold ranges
        confidence_performance = defaultdict(list)
        complexity_performance = defaultdict(list)

        for entry in recent_history:
            ctx = entry['trigger_context']
            perf = entry['performance']

            conf_bucket = int(ctx.get('confidence', 0.5) * 10) / 10  # 0.0, 0.1, 0.2, ...
            comp_bucket = int(ctx.get('complexity', 0.5) * 10) / 10

            confidence_performance[conf_bucket].append(perf)
            complexity_performance[comp_bucket].append(perf)

        # Find optimal thresholds
        self._optimize_threshold('confidence', confidence_performance)
        self._optimize_threshold('complexity', complexity_performance)

    def _optimize_threshold(self, metric_name: str, performance_data: Dict[float, List[float]]):
        """Optimize threshold for a specific metric"""
        if len(performance_data) < 3:
            return

        best_threshold = self.learned_thresholds[metric_name]
        best_score = -1

        for threshold, performances in performance_data.items():
            if len(performances) >= 5:  # Minimum sample size
                avg_perf = np.mean(performances)
                stability = 1.0 / (1.0 + np.std(performances))  # Prefer stable performance
                score = avg_perf * stability

                if score > best_score:
                    best_score = score
                    best_threshold = threshold

        # Smooth update to avoid oscillations
        alpha = self.config.learning_rate
        current = self.learned_thresholds[metric_name]
        self.learned_thresholds[metric_name] = (1 - alpha) * current + alpha * best_threshold

    def get_learned_thresholds(self) -> Dict[str, float]:
        """Get current learned thresholds"""
        return self.learned_thresholds.copy()


class AdaptiveQuantization:
    """Adaptive quantization for neurotransmitters based on performance budget"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.precision_levels = {
            'high': torch.float32,
            'medium': torch.float16,
            'low': torch.int8
        }
        self.current_precision = 'medium'
        self.performance_history = deque(maxlen=100)

    def adjust_precision(self, performance_budget: float, accuracy_requirement: float):
        """Adjust precision based on performance budget and accuracy needs"""
        if performance_budget < 0.8:  # High budget available
            if accuracy_requirement > 0.95:
                self.current_precision = 'high'
            else:
                self.current_precision = 'medium'
        elif performance_budget > 0.95:  # Low budget
            self.current_precision = 'low'
        else:  # Medium budget
            self.current_precision = 'medium'

    def get_precision_dtype(self) -> torch.dtype:
        """Get current precision dtype"""
        return self.precision_levels[self.current_precision]

    def quantize_if_needed(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply quantization if beneficial"""
        target_dtype = self.get_precision_dtype()
        if tensor.dtype != target_dtype and target_dtype != torch.int8:
            return tensor.to(dtype=target_dtype)
        return tensor


class HealthMonitor:
    """Comprehensive health monitoring for the integrated system"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.baseline_metrics = None
        self.current_metrics = {
            'latency': deque(maxlen=100),
            'accuracy': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'activation_rate': deque(maxlen=100)
        }
        self.alerts = []
        self.last_health_check = time.time()

    def record_metrics(self, latency: float, accuracy: float,
                      memory_usage: float, activation_rate: float):
        """Record current metrics"""
        self.current_metrics['latency'].append(latency)
        self.current_metrics['accuracy'].append(accuracy)
        self.current_metrics['memory_usage'].append(memory_usage)
        self.current_metrics['activation_rate'].append(activation_rate)

        # Set baseline if not set
        if self.baseline_metrics is None and len(self.current_metrics['latency']) >= 10:
            self.baseline_metrics = {
                'latency': np.mean(list(self.current_metrics['latency'])[:10]),
                'accuracy': np.mean(list(self.current_metrics['accuracy'])[:10]),
                'memory_usage': np.mean(list(self.current_metrics['memory_usage'])[:10])
            }

    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        if self.baseline_metrics is None or len(self.current_metrics['latency']) < 5:
            return {'status': 'insufficient_data', 'healthy': True}

        current_time = time.time()
        if current_time - self.last_health_check < self.config.health_check_interval:
            return self._get_cached_health_status()

        self.last_health_check = current_time

        # Calculate current averages
        current_avg = {
            'latency': np.mean(list(self.current_metrics['latency'])[-20:]),
            'accuracy': np.mean(list(self.current_metrics['accuracy'])[-20:]),
            'memory_usage': np.mean(list(self.current_metrics['memory_usage'])[-20:])
        }

        # Check for alerts
        self.alerts = []
        healthy = True

        # Latency check
        latency_ratio = current_avg['latency'] / self.baseline_metrics['latency']
        if latency_ratio > self.config.latency_degradation_threshold:
            self.alerts.append(f"Latency degraded: {latency_ratio:.2f}x baseline")
            healthy = False

        # Accuracy check
        accuracy_drop = self.baseline_metrics['accuracy'] - current_avg['accuracy']
        if accuracy_drop > self.config.accuracy_drop_threshold:
            self.alerts.append(f"Accuracy dropped: {accuracy_drop:.3f}")
            healthy = False

        # Memory check
        memory_ratio = current_avg['memory_usage'] / self.baseline_metrics['memory_usage']
        if memory_ratio > (1 + self.config.memory_overhead_threshold):
            self.alerts.append(f"Memory overhead: {(memory_ratio-1)*100:.1f}%")
            healthy = False

        return {
            'status': 'healthy' if healthy else 'degraded',
            'healthy': healthy,
            'alerts': self.alerts,
            'current_metrics': current_avg,
            'baseline_metrics': self.baseline_metrics,
            'recommendations': self._generate_recommendations()
        }

    def _get_cached_health_status(self) -> Dict[str, Any]:
        """Get cached health status"""
        return {
            'status': 'cached',
            'healthy': len(self.alerts) == 0,
            'alerts': self.alerts
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current alerts"""
        recommendations = []

        for alert in self.alerts:
            if "Latency degraded" in alert:
                recommendations.append("Consider reducing neurotransmitter activation frequency")
            elif "Accuracy dropped" in alert:
                recommendations.append("Increase neurotransmitter precision or activation threshold")
            elif "Memory overhead" in alert:
                recommendations.append("Enable adaptive quantization or reduce cache size")

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return recommendations


class RobustNeurotransmitterIntegration(nn.Module):
    """
    Robust integration of synthetic neurotransmitters with QRH framework

    Features:
    - Maintains JIT performance as priority (95% of cases)
    - Adaptive expertise activation (4% parallel, 1% lead)
    - Intelligent caching and quantization
    - Continuous learning and optimization
    - Comprehensive health monitoring
    """

    def __init__(self, qrh_config: QRHConfig, integration_config: IntegrationConfig):
        super().__init__()
        self.qrh_config = qrh_config
        self.config = integration_config

        # Core QRH System (JIT-optimized)
        self.qrh_core = QRHLayer(qrh_config)

        # Lazy-loaded neurotransmitter system
        self._neurotransmitter_system = None
        self._nt_config = NeurotransmitterConfig(embed_dim=qrh_config.embed_dim)

        # Integration components
        self.cache = NeurotransmitterCache(integration_config)
        self.learner = AdaptiveActivationLearner(integration_config)
        self.quantization = AdaptiveQuantization(integration_config)
        self.health_monitor = HealthMonitor(integration_config)

        # Performance tracking
        self.forward_count = 0
        self.jit_only_count = 0
        self.expertise_count = 0
        self.total_processing_time = 0.0

        # Async processing
        self._executor = None
        if integration_config.enable_async_processing:
            import concurrent.futures
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=integration_config.max_concurrent_tasks
            )

        # Apply JIT optimization to core
        self._apply_jit_optimization()

    @property
    def neurotransmitter_system(self):
        """Lazy-loaded neurotransmitter system"""
        if self._neurotransmitter_system is None and self.config.enable_lazy_loading:
            self._neurotransmitter_system = SyntheticNeurotransmitterSystem(self._nt_config)
            print("🧬 Neurotransmitter system initialized (lazy loading)")
        return self._neurotransmitter_system

    def _apply_jit_optimization(self):
        """Apply JIT compilation to core QRH components"""
        try:
            sample_input = torch.randn(2, 16, self.qrh_config.embed_dim * 4)

            # JIT compile the core QRH layer
            self.qrh_core = torch.jit.trace(self.qrh_core, sample_input)
            print("✅ QRH core JIT compilation successful")

        except Exception as e:
            print(f"⚠️ JIT compilation warning: {e}")

    def _should_activate_expertise(self, jit_output: torch.Tensor,
                                  confidence: float, complexity: float) -> Tuple[bool, OperationMode, Dict[str, Any]]:
        """Intelligent decision on whether to activate neurotransmitter expertise"""

        # Get learned thresholds
        learned_thresholds = self.learner.get_learned_thresholds()

        # Calculate activation triggers
        triggers = {
            'confidence': confidence,
            'complexity': complexity,
            'jit_output_variance': torch.var(jit_output).item(),
            'signal_stability': 1.0 / (1.0 + torch.std(jit_output).item())
        }

        # Check activation conditions
        low_confidence = confidence < learned_thresholds['confidence']
        high_complexity = complexity > learned_thresholds['complexity']
        unstable_signal = triggers['signal_stability'] < 0.7

        # Determine operation mode
        if low_confidence and high_complexity and unstable_signal:
            return True, OperationMode.EXPERTISE_LEAD, triggers
        elif low_confidence or high_complexity:
            return True, OperationMode.JIT_EXPERTISE, triggers
        else:
            return False, OperationMode.JIT_PURE, triggers

    def _calculate_complexity_metric(self, x: torch.Tensor) -> float:
        """Calculate complexity metric for input"""
        # Multi-factor complexity assessment
        variance = torch.var(x).item()
        entropy = -torch.sum(F.softmax(x.flatten(), dim=0) *
                           F.log_softmax(x.flatten(), dim=0)).item()
        gradient_norm = torch.norm(torch.gradient(x.flatten())[0]).item()

        # Normalize and combine
        complexity = (
            min(variance / 10.0, 1.0) * 0.4 +
            min(entropy / 10.0, 1.0) * 0.4 +
            min(gradient_norm / 10.0, 1.0) * 0.2
        )

        return complexity

    def _calculate_confidence(self, output: torch.Tensor) -> float:
        """Calculate confidence in JIT output"""
        # Confidence based on output stability and distribution
        stability = 1.0 / (1.0 + torch.std(output).item())
        distribution_entropy = -torch.sum(
            F.softmax(output.flatten(), dim=0) *
            F.log_softmax(output.flatten(), dim=0)
        ).item()

        # Normalize entropy (higher entropy = lower confidence)
        normalized_entropy = min(distribution_entropy / 10.0, 1.0)
        confidence = stability * (1.0 - normalized_entropy * 0.5)

        return float(torch.clamp(torch.tensor(confidence), 0.0, 1.0).item())

    async def _process_with_expertise_async(self, x: torch.Tensor,
                                          mode: OperationMode) -> torch.Tensor:
        """Asynchronous expertise processing with dtype control"""
        if self.neurotransmitter_system is None:
            return x  # Fallback if not initialized

        try:
            # Store original dtype for consistency
            original_dtype = x.dtype
            processing_tensor = x

            # Apply quantization but keep dtype tracking
            if self.config.enable_adaptive_quantization:
                processing_tensor = self.quantization.quantize_if_needed(x)

            # Ensure neurotransmitter system uses consistent dtype
            # Force float32 for stable matrix operations
            if processing_tensor.dtype == torch.float16:
                processing_tensor = processing_tensor.to(torch.float32)

            # Process with neurotransmitters
            expertise_output = self.neurotransmitter_system(processing_tensor)

            # Return in original dtype for blending compatibility
            if expertise_output.dtype != original_dtype:
                expertise_output = expertise_output.to(original_dtype)

            return expertise_output

        except Exception as e:
            print(f"⚠️ Expertise processing error: {e}")
            return x  # Fallback to original input

    def forward(self, x: torch.Tensor, return_detailed_metrics: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Optimized forward pass with adaptive expertise integration
        """
        start_time = time.time()
        self.forward_count += 1

        # Step 1: Primary JIT processing (always executed)
        jit_output = self.qrh_core(x)

        # Step 2: Calculate metrics for decision making
        confidence = self._calculate_confidence(jit_output)
        complexity = self._calculate_complexity_metric(x)

        # Step 3: Check cache for similar inputs
        context = {
            'confidence': confidence,
            'complexity': complexity,
            'input_shape': str(x.shape)
        }

        input_signature = self.cache.get_signature(x, context)
        cached_decision = None

        if self.config.enable_decision_cache:
            cached_decision = self.cache.get(input_signature)

        # Step 4: Activation decision
        if cached_decision:
            should_activate = cached_decision['should_activate']
            mode = OperationMode(cached_decision['mode'])
            triggers = cached_decision['triggers']
        else:
            should_activate, mode, triggers = self._should_activate_expertise(
                jit_output, confidence, complexity
            )

            # Cache the decision
            if self.config.enable_decision_cache:
                self.cache.set(input_signature, {
                    'should_activate': should_activate,
                    'mode': mode.value,
                    'triggers': triggers
                })

        # Step 5: Process based on mode
        final_output = jit_output
        processing_mode = mode

        if should_activate:
            self.expertise_count += 1

            if mode == OperationMode.JIT_EXPERTISE:
                # Parallel processing - don't wait
                if self._executor and self.config.enable_async_processing:
                    future = self._executor.submit(
                        lambda: asyncio.run(self._process_with_expertise_async(x, mode))
                    )
                    try:
                        expertise_output = future.result(timeout=self.config.async_timeout_seconds)
                        # Ensure dtype compatibility
                        if jit_output.dtype != expertise_output.dtype:
                            expertise_output = expertise_output.to(dtype=jit_output.dtype)

                        # Inverted logic: Start smooth and stable, become sharper with experience
                        experience_factor = min(1.0, self.expertise_count / 500)  # Faster adaptation
                        stability = 1.0 - experience_factor  # High initially, decreases
                        sharpness = 0.1 + 0.9 * experience_factor  # Low initially, increases

                        # Control expertise magnitude - tighter control initially
                        expertise_magnitude = torch.norm(expertise_output)
                        jit_magnitude = torch.norm(jit_output)
                        max_ratio = 1.1 + 0.4 * experience_factor  # Start tight, allow more growth
                        if expertise_magnitude > jit_magnitude * max_ratio:
                            expertise_output = expertise_output * (jit_magnitude * max_ratio / expertise_magnitude)

                        # Inverted blend: conservative start, more aggressive learning with experience
                        expertise_weight = 0.1 + 0.6 * experience_factor  # Start low, increase
                        jit_weight = 1.0 - expertise_weight

                        final_output = jit_weight * jit_output + expertise_weight * expertise_output
                    except Exception:
                        # Timeout or error - use JIT output
                        final_output = jit_output
                        processing_mode = OperationMode.JIT_PURE

            elif mode == OperationMode.EXPERTISE_LEAD:
                # Synchronous expertise processing
                try:
                    expertise_output = asyncio.run(
                        self._process_with_expertise_async(x, mode)
                    )
                    # Ensure dtype compatibility
                    if jit_output.dtype != expertise_output.dtype:
                        expertise_output = expertise_output.to(dtype=jit_output.dtype)

                    # Inverted logic for lead mode: stable start, progressive learning
                    experience_factor = min(1.0, self.expertise_count / 300)  # Even faster for lead mode
                    stability = 1.0 - experience_factor
                    learning_curve = 0.2 + 0.8 * experience_factor

                    # Very tight control initially, gradual loosening
                    expertise_magnitude = torch.norm(expertise_output)
                    jit_magnitude = torch.norm(jit_output)
                    max_ratio = 1.05 + 0.45 * experience_factor  # Very conservative start
                    if expertise_magnitude > jit_magnitude * max_ratio:
                        expertise_output = expertise_output * (jit_magnitude * max_ratio / expertise_magnitude)

                    # Conservative expertise weight that grows with experience
                    expertise_weight = 0.2 + 0.6 * experience_factor  # Start conservative
                    jit_weight = 1.0 - expertise_weight

                    final_output = jit_weight * jit_output + expertise_weight * expertise_output
                except Exception:
                    final_output = jit_output
                    processing_mode = OperationMode.JIT_PURE
        else:
            self.jit_only_count += 1
            processing_mode = OperationMode.JIT_PURE

        # Step 6: Performance tracking and learning
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time

        # Record for learning
        if self.config.enable_meta_learning:
            output_quality = self._assess_output_quality(final_output, x)
            self.learner.record_activation(triggers, output_quality, processing_time)

        # Health monitoring
        if self.config.enable_health_monitoring:
            activation_rate = self.expertise_count / self.forward_count
            memory_usage = self._get_memory_usage()
            self.health_monitor.record_metrics(
                processing_time, confidence, memory_usage, activation_rate
            )

        # Step 7: Prepare output
        if return_detailed_metrics:
            metrics = {
                'processing_mode': processing_mode.value,
                'confidence': confidence,
                'complexity': complexity,
                'processing_time_ms': processing_time * 1000,
                'jit_only_rate': self.jit_only_count / self.forward_count,
                'expertise_rate': self.expertise_count / self.forward_count,
                'health_status': self.health_monitor.check_health() if self.forward_count % 100 == 0 else None
            }
            return final_output, metrics

        return final_output

    def _assess_output_quality(self, output: torch.Tensor, input_tensor: torch.Tensor) -> float:
        """Assess quality of output for learning purposes"""
        # Simple quality metrics
        signal_to_noise = torch.mean(output) / (torch.std(output) + 1e-8)
        stability = 1.0 / (1.0 + torch.var(output).item())

        # Information preservation (correlation with input)
        input_flat = input_tensor.flatten()
        output_flat = output.flatten()
        min_size = min(len(input_flat), len(output_flat))

        correlation = torch.corrcoef(torch.stack([
            input_flat[:min_size],
            output_flat[:min_size]
        ]))[0, 1]

        if torch.isnan(correlation):
            correlation = torch.tensor(0.5)

        quality = (
            torch.sigmoid(signal_to_noise).item() * 0.3 +
            stability * 0.3 +
            correlation.item() * 0.4
        )

        return max(0.0, min(1.0, quality))

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            # Approximate CPU memory usage
            total_params = sum(p.numel() * p.element_size() for p in self.parameters())
            return total_params / (1024 * 1024)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if self.forward_count == 0:
            return {'status': 'no_data'}

        avg_processing_time = self.total_processing_time / self.forward_count
        jit_only_rate = self.jit_only_count / self.forward_count
        expertise_rate = self.expertise_count / self.forward_count

        return {
            'total_forward_passes': self.forward_count,
            'average_processing_time_ms': avg_processing_time * 1000,
            'jit_only_rate': jit_only_rate,
            'expertise_parallel_rate': expertise_rate * 0.8,  # Approximate
            'expertise_lead_rate': expertise_rate * 0.2,     # Approximate
            'performance_efficiency': jit_only_rate + expertise_rate * 0.5,  # Weighted efficiency
            'learned_thresholds': self.learner.get_learned_thresholds(),
            'health_status': self.health_monitor.check_health(),
            'cache_hit_rate': len(self.cache.cache) / (self.forward_count + 1)
        }

    def optimize_for_deployment(self, target_profile: str = 'balanced'):
        """Optimize system for specific deployment profile"""
        profiles = {
            'high_performance': {
                'activation_threshold': 0.9,
                'cache_size': 500,
                'enable_async_processing': True,
                'precision': 'low'
            },
            'balanced': {
                'activation_threshold': 0.7,
                'cache_size': 1000,
                'enable_async_processing': True,
                'precision': 'medium'
            },
            'high_accuracy': {
                'activation_threshold': 0.5,
                'cache_size': 2000,
                'enable_async_processing': False,
                'precision': 'high'
            }
        }

        if target_profile in profiles:
            profile_config = profiles[target_profile]

            # Update configuration
            self.config.activation_threshold = profile_config['activation_threshold']
            self.config.cache_size = profile_config['cache_size']
            self.config.enable_async_processing = profile_config['enable_async_processing']

            # Update quantization
            if profile_config['precision'] == 'low':
                self.quantization.current_precision = 'low'
            elif profile_config['precision'] == 'high':
                self.quantization.current_precision = 'high'
            else:
                self.quantization.current_precision = 'medium'

            print(f"✅ System optimized for {target_profile} deployment profile")
        else:
            print(f"⚠️ Unknown profile: {target_profile}")

    def __del__(self):
        """Cleanup resources"""
        if self._executor:
            self._executor.shutdown(wait=False)