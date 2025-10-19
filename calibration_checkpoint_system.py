#!/usr/bin/env python3
"""
Calibration Checkpoint System for ΨQRH Pipeline
==============================================

Implements checkpoint-based auto-calibration with mount points:
- Saves calibrated parameters as mount points
- Uses mount points until recalibration is needed
- Auto-recalibrates when mount point fails
- Critical systems always recalibrate from scratch
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib


class CalibrationCheckpointSystem:
    """
    Manages calibration checkpoints (mount points) for ΨQRH system

    Features:
    - Save calibrated parameters as mount points
    - Load and validate mount points
    - Auto-recalibrate on mount point failure
    - Critical recalibration for critical systems
    """

    def __init__(self, checkpoint_dir: str = "models/calibration_checkpoints"):
        """Initialize checkpoint system"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Current mount point
        self.current_mount_point = None
        self.mount_point_hash = None

        # Recalibration counters
        self.recalibration_count = 0
        self.last_recalibration = None

        print(f"🔧 Calibration Checkpoint System initialized: {self.checkpoint_dir}")

    def save_mount_point(self, calibrated_params: Dict[str, Any],
                        text_hash: str = None) -> str:
        """
        Save calibrated parameters as mount point

        Args:
            calibrated_params: All calibrated parameters
            text_hash: Hash of input text (for tracking)

        Returns:
            Mount point filepath
        """
        # Generate unique mount point ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if text_hash:
            mount_id = f"mount_{timestamp}_{text_hash[:8]}"
        else:
            mount_id = f"mount_{timestamp}"

        mount_file = self.checkpoint_dir / f"{mount_id}.json"

        # Add metadata
        mount_data = {
            'calibrated_params': calibrated_params,
            'metadata': {
                'mount_id': mount_id,
                'timestamp': timestamp,
                'text_hash': text_hash,
                'system_version': 'ΨQRH_v1.0',
                'validation_status': 'VALIDATED'
            }
        }

        # Save mount point
        with open(mount_file, 'w') as f:
            json.dump(mount_data, f, indent=2, ensure_ascii=False)

        # Update current mount point
        self.current_mount_point = mount_file
        self.mount_point_hash = self._compute_mount_hash(mount_data)

        print(f"💾 Mount point saved: {mount_file}")
        print(f"   📊 Parameters: {len(calibrated_params)} categories")

        return str(mount_file)

    def load_mount_point(self, mount_file: str = None) -> Tuple[Dict[str, Any], bool]:
        """
        Load mount point and validate integrity

        Args:
            mount_file: Specific mount file to load (None = use current)

        Returns:
            Tuple of (calibrated_params, is_valid)
        """
        if mount_file is None:
            if self.current_mount_point is None:
                return None, False
            mount_file = self.current_mount_point

        mount_path = Path(mount_file)

        if not mount_path.exists():
            print(f"❌ Mount point not found: {mount_file}")
            return None, False

        try:
            # Load mount point
            with open(mount_path, 'r') as f:
                mount_data = json.load(f)

            # Validate structure
            if 'calibrated_params' not in mount_data:
                print(f"❌ Invalid mount point structure: {mount_file}")
                return None, False

            # Validate hash integrity
            current_hash = self._compute_mount_hash(mount_data)
            if self.mount_point_hash and current_hash != self.mount_point_hash:
                print(f"❌ Mount point integrity check failed: {mount_file}")
                return None, False

            # Validate parameter ranges
            is_valid = self._validate_mount_point(mount_data['calibrated_params'])

            if is_valid:
                print(f"✅ Mount point loaded successfully: {mount_file}")
                self.current_mount_point = mount_path
                self.mount_point_hash = current_hash
                return mount_data['calibrated_params'], True
            else:
                print(f"❌ Mount point validation failed: {mount_file}")
                return mount_data['calibrated_params'], False

        except Exception as e:
            print(f"❌ Error loading mount point {mount_file}: {e}")
            return None, False

    def should_recalibrate(self, text: str, is_critical: bool = False) -> bool:
        """
        Determine if recalibration is needed

        Args:
            text: Input text for analysis
            is_critical: Whether this is a critical system

        Returns:
            True if recalibration needed
        """
        # Critical systems always recalibrate
        if is_critical:
            print("🔴 CRITICAL SYSTEM: Forcing recalibration")
            return True

        # No mount point available
        if self.current_mount_point is None:
            print("🟡 No mount point available: Recalibrating")
            return True

        # Mount point validation failed
        params, is_valid = self.load_mount_point()
        if not is_valid:
            print("🟡 Mount point invalid: Recalibrating")
            return True

        # Text characteristics changed significantly
        if self._text_characteristics_changed(text, params):
            print("🟡 Text characteristics changed: Recalibrating")
            return True

        # Too many recalibrations recently
        if self._too_many_recalibrations():
            print("🟡 Too many recalibrations: Using mount point")
            return False

        # Use existing mount point
        print("🟢 Using existing mount point")
        return False

    def auto_recalibrate_if_needed(self, text: str, calibration_system,
                                 is_critical: bool = False) -> Dict[str, Any]:
        """
        Auto-recalibrate if needed, otherwise use mount point

        Args:
            text: Input text
            calibration_system: CompleteAutoCalibrationSystem instance
            is_critical: Whether this is critical

        Returns:
            Calibrated parameters
        """
        if self.should_recalibrate(text, is_critical):
            print("🔄 Auto-recalibrating...")

            # Perform calibration
            calibrated_params = calibration_system.calibrate_all_parameters(
                text=text,
                fractal_signal=None,
                D_fractal=None
            )

            # Save as new mount point
            text_hash = self._compute_text_hash(text)
            self.save_mount_point(calibrated_params, text_hash)

            # Update recalibration tracking
            self.recalibration_count += 1
            self.last_recalibration = datetime.now()

            return calibrated_params
        else:
            # Use existing mount point
            params, _ = self.load_mount_point()
            return params

    def _compute_mount_hash(self, mount_data: Dict[str, Any]) -> str:
        """Compute hash for mount point integrity"""
        hash_data = json.dumps(mount_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(hash_data.encode('utf-8')).hexdigest()

    def _compute_text_hash(self, text: str) -> str:
        """Compute hash for text tracking"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _validate_mount_point(self, params: Dict[str, Any]) -> bool:
        """Validate mount point parameter ranges"""
        try:
            # Physical parameters
            phys = params['physical_params']
            if not (0.1 <= phys['I0'] <= 5.0):
                return False
            if not (0.1 <= phys['omega'] <= 10.0):
                return False
            if not (0.5 <= phys['k'] <= 5.0):
                return False

            # Architecture parameters
            arch = params['architecture_params']
            if not (32 <= arch['embed_dim'] <= 256):
                return False
            if not (4 <= arch['num_heads'] <= 16):
                return False

            # Processing parameters
            proc = params['processing_params']
            if not (0.05 <= proc['dropout'] <= 0.3):
                return False

            return True

        except (KeyError, TypeError):
            return False

    def _text_characteristics_changed(self, text: str, params: Dict[str, Any]) -> bool:
        """Check if text characteristics changed significantly"""
        if 'input_analysis' not in params:
            return True

        old_analysis = params['input_analysis']

        # Simple length-based change detection
        old_length = old_analysis.get('text_length', 0)
        new_length = len(text)

        # Significant length change (>50%)
        if old_length > 0:
            length_ratio = abs(new_length - old_length) / old_length
            if length_ratio > 0.5:
                return True

        return False

    def _too_many_recalibrations(self) -> bool:
        """Check if too many recalibrations occurred recently"""
        # Limit recalibrations to prevent thrashing
        if self.recalibration_count > 10:
            return True

        # Reset counter if last recalibration was long ago
        if self.last_recalibration:
            time_diff = datetime.now() - self.last_recalibration
            if time_diff.total_seconds() > 3600:  # 1 hour
                self.recalibration_count = 0

        return False

    def get_mount_point_info(self) -> Dict[str, Any]:
        """Get information about current mount point"""
        if self.current_mount_point is None:
            return {'status': 'NO_MOUNT_POINT'}

        try:
            with open(self.current_mount_point, 'r') as f:
                mount_data = json.load(f)

            return {
                'status': 'MOUNT_POINT_ACTIVE',
                'mount_file': str(self.current_mount_point),
                'timestamp': mount_data['metadata']['timestamp'],
                'text_hash': mount_data['metadata']['text_hash'],
                'recalibration_count': self.recalibration_count
            }
        except:
            return {'status': 'MOUNT_POINT_CORRUPTED'}

    def cleanup_old_mount_points(self, keep_count: int = 5):
        """Clean up old mount points, keeping only recent ones"""
        mount_files = list(self.checkpoint_dir.glob("mount_*.json"))

        if len(mount_files) <= keep_count:
            return

        # Sort by modification time
        mount_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old mount points
        for mount_file in mount_files[keep_count:]:
            try:
                mount_file.unlink()
                print(f"🗑️  Cleaned up old mount point: {mount_file}")
            except Exception as e:
                print(f"⚠️  Failed to clean up {mount_file}: {e}")


def create_calibration_checkpoint_system(checkpoint_dir: str = "models/calibration_checkpoints") -> CalibrationCheckpointSystem:
    """
    Factory function for calibration checkpoint system

    Args:
        checkpoint_dir: Directory for mount points

    Returns:
        CalibrationCheckpointSystem instance
    """
    return CalibrationCheckpointSystem(checkpoint_dir)