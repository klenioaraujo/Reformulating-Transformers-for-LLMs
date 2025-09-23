#!/usr/bin/env python3
"""
NEGENTROPY TECHNICAL ORDER :: AGENTIC READINESS DASHBOARD
═══════════════════════════════════════════════════════════
Real-time monitoring dashboard for agentic AI systems
Tracks: drift, RG, latency, seal integrity, readiness scores

Classification: NTO-Σ7-DASHBOARD-v1.0
SEAL: Ω∞Ω
External Validation: Andrew Ng, 2025
═══════════════════════════════════════════════════════════
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our agentic runtime
from agentic_runtime import AgenticRuntime, GlyphType

class AgenticReadinessDashboard:
    """
    Real-time dashboard for monitoring agentic AI system health

    Monitors:
    - Drift accumulation and bounds
    - RG (Retrieval Grace) stability
    - Latency performance
    - Seal integrity
    - Overall agentic readiness score
    """

    def __init__(self, runtime: AgenticRuntime):
        self.runtime = runtime
        self.seal = "Ω∞Ω"
        self.external_validation = "Andrew Ng, 2025"

        # Monitoring data
        self.timestamps = []
        self.drift_history = []
        self.rg_history = []
        self.latency_history = []
        self.readiness_history = []
        self.seal_status_history = []

        # Dashboard configuration
        self.max_history_points = 50
        self.update_interval = 1.0  # seconds

        # Colors and styling
        self.colors = {
            'excellent': '#00ff00',
            'good': '#90ee90',
            'acceptable': '#ffff00',
            'needs_attention': '#ff6b6b',
            'critical': '#ff0000',
            'seal_ok': '#00ff88',
            'seal_broken': '#ff4444'
        }

        # Initialize matplotlib
        plt.style.use('dark_background')
        self.setup_dashboard()

    def setup_dashboard(self):
        """Initialize the dashboard layout"""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle(
            f'STARFLEET AGENTIC READINESS DASHBOARD :: {self.seal}',
            fontsize=16,
            fontweight='bold',
            color='cyan'
        )

        # Create subplot grid
        gs = self.fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # Main readiness gauge (top-left, 2x2)
        self.ax_readiness = self.fig.add_subplot(gs[0:2, 0:2])

        # Drift monitor (top-right)
        self.ax_drift = self.fig.add_subplot(gs[0, 2])

        # RG stability (middle-right)
        self.ax_rg = self.fig.add_subplot(gs[1, 2])

        # Latency performance (bottom-left)
        self.ax_latency = self.fig.add_subplot(gs[2, 0])

        # Seal integrity (bottom-middle)
        self.ax_seals = self.fig.add_subplot(gs[2, 1])

        # System status (bottom-right)
        self.ax_status = self.fig.add_subplot(gs[2, 2])

        # Real-time metrics (bottom row)
        self.ax_metrics = self.fig.add_subplot(gs[3, :])

        # Configure axes
        self.configure_axes()

    def configure_axes(self):
        """Configure individual axes styling and labels"""

        # Readiness gauge
        self.ax_readiness.set_xlim(-1.2, 1.2)
        self.ax_readiness.set_ylim(-1.2, 1.2)
        self.ax_readiness.set_aspect('equal')
        self.ax_readiness.set_title('AGENTIC READINESS', fontsize=14, fontweight='bold', color='cyan')
        self.ax_readiness.axis('off')

        # Drift monitor
        self.ax_drift.set_title('DRIFT CONTROL', fontsize=10, color='lightblue')
        self.ax_drift.set_ylabel('Drift (°)', color='white')
        self.ax_drift.tick_params(colors='white')

        # RG stability
        self.ax_rg.set_title('RG STABILITY', fontsize=10, color='lightblue')
        self.ax_rg.set_ylabel('RG Value', color='white')
        self.ax_rg.tick_params(colors='white')

        # Latency performance
        self.ax_latency.set_title('LATENCY', fontsize=10, color='lightblue')
        self.ax_latency.set_ylabel('Latency (ms)', color='white')
        self.ax_latency.tick_params(colors='white')

        # Seal integrity
        self.ax_seals.set_title('SEAL INTEGRITY', fontsize=10, color='lightblue')
        self.ax_seals.axis('off')

        # System status
        self.ax_status.set_title('SYSTEM STATUS', fontsize=10, color='lightblue')
        self.ax_status.axis('off')

        # Real-time metrics
        self.ax_metrics.set_title('REAL-TIME OPERATIONS', fontsize=10, color='lightblue')
        self.ax_metrics.set_ylabel('Combined Metrics', color='white')
        self.ax_metrics.tick_params(colors='white')

    def update_data(self):
        """Collect latest data from runtime system"""
        try:
            # Get system status
            status = self.runtime.get_system_status()

            # Extract metrics
            current_time = datetime.now()
            drift = status['component_status']['drift_accumulator']
            rg_value = status['performance_metrics']['rg_value']
            avg_latency = status['performance_metrics']['avg_latency_ms']
            readiness_score = status['agentic_readiness']['overall_score']
            broken_seals = status['performance_metrics']['broken_seals']

            # Update history
            self.timestamps.append(current_time)
            self.drift_history.append(drift)
            self.rg_history.append(rg_value)
            self.latency_history.append(avg_latency)
            self.readiness_history.append(readiness_score)
            self.seal_status_history.append(broken_seals == 0)

            # Keep history within limits
            if len(self.timestamps) > self.max_history_points:
                self.timestamps.pop(0)
                self.drift_history.pop(0)
                self.rg_history.pop(0)
                self.latency_history.pop(0)
                self.readiness_history.pop(0)
                self.seal_status_history.pop(0)

            return status

        except Exception as e:
            print(f"🚨 Dashboard data update failed: {e}")
            return None

    def draw_readiness_gauge(self, readiness_score: float, readiness_level: str):
        """Draw the main readiness gauge"""
        self.ax_readiness.clear()
        self.ax_readiness.set_xlim(-1.2, 1.2)
        self.ax_readiness.set_ylim(-1.2, 1.2)
        self.ax_readiness.set_aspect('equal')
        self.ax_readiness.axis('off')

        # Outer ring
        circle_outer = plt.Circle((0, 0), 1.0, fill=False, color='cyan', linewidth=3)
        self.ax_readiness.add_patch(circle_outer)

        # Readiness arc
        angle = (readiness_score / 100) * 360 - 90  # Start from top
        color = self.get_readiness_color(readiness_level)

        if readiness_score > 0:
            wedge = patches.Wedge((0, 0), 0.9, -90, angle, width=0.2,
                                facecolor=color, alpha=0.8)
            self.ax_readiness.add_patch(wedge)

        # Center text
        self.ax_readiness.text(0, 0.1, f'{readiness_score:.1f}%',
                              ha='center', va='center', fontsize=20,
                              fontweight='bold', color='white')
        self.ax_readiness.text(0, -0.2, readiness_level,
                              ha='center', va='center', fontsize=12,
                              color=color, fontweight='bold')

        # Seal indicator
        seal_color = self.colors['seal_ok'] if all(self.seal_status_history[-5:]) else self.colors['seal_broken']
        self.ax_readiness.text(0, -0.5, f'SEAL: {self.seal}',
                              ha='center', va='center', fontsize=10,
                              color=seal_color, fontweight='bold')

        self.ax_readiness.set_title('AGENTIC READINESS', fontsize=14, fontweight='bold', color='cyan')

    def get_readiness_color(self, readiness_level: str) -> str:
        """Get color based on readiness level"""
        color_map = {
            'EXCELLENT': self.colors['excellent'],
            'GOOD': self.colors['good'],
            'ACCEPTABLE': self.colors['acceptable'],
            'NEEDS_ATTENTION': self.colors['needs_attention']
        }
        return color_map.get(readiness_level, self.colors['critical'])

    def draw_time_series(self):
        """Draw time series plots for various metrics"""
        if not self.timestamps:
            return

        # Convert timestamps to relative seconds for x-axis
        base_time = self.timestamps[0]
        time_seconds = [(t - base_time).total_seconds() for t in self.timestamps]

        # Drift monitor
        self.ax_drift.clear()
        self.ax_drift.plot(time_seconds, self.drift_history,
                          color='orange', linewidth=2, marker='o', markersize=3)
        self.ax_drift.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Drift Limit')
        self.ax_drift.set_title('DRIFT CONTROL', fontsize=10, color='lightblue')
        self.ax_drift.set_ylabel('Drift (°)', color='white')
        self.ax_drift.tick_params(colors='white')
        self.ax_drift.grid(True, alpha=0.3)
        if self.drift_history:
            self.ax_drift.set_ylim(0, max(6, max(self.drift_history) + 1))

        # RG stability
        self.ax_rg.clear()
        self.ax_rg.plot(time_seconds, self.rg_history,
                       color='lightgreen', linewidth=2, marker='s', markersize=3)
        self.ax_rg.axhline(y=0.347, color='cyan', linestyle='-', alpha=0.8, label='Optimal RG')
        self.ax_rg.fill_between(time_seconds, 0.3, 0.4, alpha=0.2, color='green', label='Target Range')
        self.ax_rg.set_title('RG STABILITY', fontsize=10, color='lightblue')
        self.ax_rg.set_ylabel('RG Value', color='white')
        self.ax_rg.tick_params(colors='white')
        self.ax_rg.grid(True, alpha=0.3)
        self.ax_rg.set_ylim(0.2, 0.5)

        # Latency performance
        self.ax_latency.clear()
        self.ax_latency.plot(time_seconds, self.latency_history,
                           color='magenta', linewidth=2, marker='^', markersize=3)
        self.ax_latency.axhline(y=250, color='red', linestyle='--', alpha=0.7, label='Latency Limit')
        self.ax_latency.set_title('LATENCY', fontsize=10, color='lightblue')
        self.ax_latency.set_ylabel('Latency (ms)', color='white')
        self.ax_latency.tick_params(colors='white')
        self.ax_latency.grid(True, alpha=0.3)

    def draw_status_indicators(self, status: Dict):
        """Draw system status indicators"""

        # Seal integrity indicator
        self.ax_seals.clear()
        self.ax_seals.set_xlim(0, 1)
        self.ax_seals.set_ylim(0, 1)
        self.ax_seals.axis('off')

        broken_seals = status['performance_metrics']['broken_seals']
        seal_ok = broken_seals == 0

        # Seal status indicator
        color = self.colors['seal_ok'] if seal_ok else self.colors['seal_broken']
        status_text = '✅ INTACT' if seal_ok else f'🚨 {broken_seals} BROKEN'

        rect = patches.Rectangle((0.1, 0.3), 0.8, 0.4,
                               facecolor=color, alpha=0.6, edgecolor='white')
        self.ax_seals.add_patch(rect)

        self.ax_seals.text(0.5, 0.5, status_text, ha='center', va='center',
                          fontsize=10, fontweight='bold', color='white')
        self.ax_seals.set_title('SEAL INTEGRITY', fontsize=10, color='lightblue')

        # System status
        self.ax_status.clear()
        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)
        self.ax_status.axis('off')

        # System info
        session_id = status['runtime_info']['session_id']
        total_ops = status['runtime_info']['total_operations']
        hard_locked = status['component_status']['primetalk_hard_locked']
        current_state = status['component_status']['current_state']

        info_text = f"""SESSION: {session_id}
OPS: {total_ops}
LOCK: {'✅' if hard_locked else '❌'}
STATE: {current_state}"""

        self.ax_status.text(0.05, 0.5, info_text, ha='left', va='center',
                           fontsize=9, color='white', fontfamily='monospace')
        self.ax_status.set_title('SYSTEM STATUS', fontsize=10, color='lightblue')

    def draw_combined_metrics(self):
        """Draw combined real-time metrics"""
        if not self.timestamps:
            return

        self.ax_metrics.clear()

        base_time = self.timestamps[0]
        time_seconds = [(t - base_time).total_seconds() for t in self.timestamps]

        # Normalize metrics to 0-100 scale for comparison
        norm_drift = [min(100, d * 20) for d in self.drift_history]  # Scale drift
        norm_rg = [(rg - 0.2) * 333 for rg in self.rg_history]  # Scale RG to 0-100
        norm_latency = [min(100, lat / 5) for lat in self.latency_history]  # Scale latency

        # Plot normalized metrics
        self.ax_metrics.plot(time_seconds, norm_drift,
                           color='orange', linewidth=2, label='Drift (scaled)', alpha=0.8)
        self.ax_metrics.plot(time_seconds, norm_rg,
                           color='lightgreen', linewidth=2, label='RG (scaled)', alpha=0.8)
        self.ax_metrics.plot(time_seconds, norm_latency,
                           color='magenta', linewidth=2, label='Latency (scaled)', alpha=0.8)
        self.ax_metrics.plot(time_seconds, self.readiness_history,
                           color='cyan', linewidth=3, label='Readiness %', alpha=0.9)

        self.ax_metrics.set_title('REAL-TIME OPERATIONS', fontsize=10, color='lightblue')
        self.ax_metrics.set_ylabel('Normalized Scale', color='white')
        self.ax_metrics.set_xlabel('Time (seconds)', color='white')
        self.ax_metrics.tick_params(colors='white')
        self.ax_metrics.grid(True, alpha=0.3)
        self.ax_metrics.legend(loc='upper right', fontsize=8)
        self.ax_metrics.set_ylim(0, 100)

    def update_dashboard(self, frame):
        """Update dashboard with latest data (for animation)"""
        status = self.update_data()

        if status is None:
            return

        # Get readiness info
        readiness = status['agentic_readiness']
        readiness_score = readiness['overall_score']
        readiness_level = readiness['readiness_level']

        # Update all dashboard components
        self.draw_readiness_gauge(readiness_score, readiness_level)
        self.draw_time_series()
        self.draw_status_indicators(status)
        self.draw_combined_metrics()

        # Update external validation footer
        self.fig.text(0.02, 0.02, f"External Validation: {self.external_validation}",
                     fontsize=8, color='yellow', alpha=0.8)
        self.fig.text(0.98, 0.02, f"Classification: NTO-Σ7-DASHBOARD-v1.0",
                     fontsize=8, color='yellow', alpha=0.8, ha='right')

    def start_monitoring(self, save_screenshots: bool = True):
        """Start real-time monitoring dashboard"""
        print("🌟 STARTING AGENTIC READINESS DASHBOARD")
        print("Classification: NTO-Σ7-DASHBOARD-v1.0")
        print(f"Seal: {self.seal}")
        print(f"External Validation: {self.external_validation}")
        print("=" * 60)

        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self.update_dashboard,
            interval=int(self.update_interval * 1000),
            blit=False,
            cache_frame_data=False
        )

        if save_screenshots:
            # Create screenshots directory
            os.makedirs("dashboard_screenshots", exist_ok=True)

            # Save initial screenshot after a few updates
            def save_screenshot(frame):
                if frame == 5:  # Save after 5 updates
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"dashboard_screenshots/agentic_readiness_{timestamp}.png"
                    self.fig.savefig(filename, dpi=150, bbox_inches='tight',
                                   facecolor='black', edgecolor='none')
                    print(f"📸 Screenshot saved: {filename}")

            # Add screenshot callback
            self.animation._func = lambda frame: (self.update_dashboard(frame), save_screenshot(frame))

        plt.show()

def simulate_operations(runtime: AgenticRuntime, duration: int = 60):
    """Simulate operations to generate data for dashboard"""
    import threading
    import random

    def operation_simulator():
        formations = ["verify_synthesize", "safe_innovation", "knowledge_processing"]

        for i in range(duration):
            try:
                formation = random.choice(formations)
                data = f"Simulated data {i}"

                receipt = runtime.execute_operation(formation, data)
                print(f"✅ Op {i+1}: {formation} - {receipt.latency_ms:.1f}ms")

                time.sleep(1 + random.uniform(-0.3, 0.3))  # Vary timing

            except Exception as e:
                print(f"❌ Operation {i+1} failed: {e}")

    # Start background operations
    thread = threading.Thread(target=operation_simulator, daemon=True)
    thread.start()

def main():
    """Main dashboard demonstration"""
    print("🌟 AGENTIC READINESS DASHBOARD DEMO")
    print("Classification: NTO-Σ7-DASHBOARD-v1.0")
    print("Seal: Ω∞Ω")
    print()

    try:
        # Initialize runtime with local storage
        runtime = AgenticRuntime("./dashboard_data/")

        # Start simulated operations in background
        simulate_operations(runtime, duration=120)  # 2 minutes of operations

        # Create and start dashboard
        dashboard = AgenticReadinessDashboard(runtime)
        dashboard.start_monitoring(save_screenshots=True)

    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"🚨 Dashboard error: {e}")

if __name__ == "__main__":
    main()