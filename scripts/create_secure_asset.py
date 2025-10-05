#!/usr/bin/env python3
"""
Secure Asset Creation Script for ΨQRH Framework

Creates encrypted .Ψcws assets with certification and manifest files.
Implements security levels: personal, enterprise, government
"""

import argparse
import json
import os
import hashlib
import datetime
import sys
from pathlib import Path

# Add project root to path to import ΨQRH modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# from src.core.quaternion_operations import QuaternionOperations
# from src.core.qrh_layer import QRHProcessor


class SecureAssetCreator:
    def __init__(self):
        self.security_levels = ['personal', 'enterprise', 'government']
        self.audit_log_path = Path('data/secure_assets/audit_log.jsonl')
        self.assets_dir = Path('data/secure_assets/Ψcws')
        self.certificates_dir = Path('data/secure_assets/certificates')
        self.manifests_dir = Path('data/secure_assets/manifests')

        # Ensure directories exist
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.certificates_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def generate_certificate(self, asset_name, security_level, key, author):
        """Generate a digital certificate for the asset"""
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Create certificate data
        cert_data = {
            "asset_name": asset_name,
            "security_level": security_level,
            "author": author,
            "creation_timestamp": timestamp,
            "system_id": "PSIQRH_SECURE_SYSTEM",
            "certificate_version": "1.0"
        }

        # Generate certificate hash (this would be signed with private key in production)
        cert_string = json.dumps(cert_data, sort_keys=True)
        cert_hash = hashlib.sha256(cert_string.encode()).hexdigest()

        # In production, this would be signed with a private key
        cert_data["certificate_hash"] = cert_hash

        return cert_data

    def create_secure_asset(self, source_file, asset_name, security_level, key,
                          author="Unknown", description="", classification=""):
        """
        Create a secure .Ψcws asset with manifest and certification
        """

        # Validate security level
        if security_level not in self.security_levels:
            raise ValueError(f"Invalid security level: {security_level}. Must be one of {self.security_levels}")

        # Validate key requirements
        if security_level in ['enterprise', 'government'] and not key:
            raise ValueError(f"Key is required for {security_level} security level")

        # Use default key for personal level if not provided
        if security_level == 'personal' and not key:
            key = "PSIQRH_SECURE_SYSTEM"

        # Calculate source file hash
        source_hash = self.calculate_file_hash(source_file)

        # Generate asset file path
        asset_path = self.assets_dir / f"{asset_name}.Ψcws"

        # Create the .Ψcws file using ΨQRH transformation
        # This is a simplified version - in practice you'd use the actual ΨQRH transformation
        print(f"🔒 Creating secure asset: {asset_path}")

        # Read source file
        with open(source_file, 'r', encoding='utf-8') as f:
            source_content = f.read()

        # Apply ΨQRH transformation with security key
        # In practice, this would use the actual ΨQRH spectral transformation
        transformed_data = self._apply_psiqrh_transformation(source_content, key)

        # Save the encrypted asset
        with open(asset_path, 'wb') as f:
            f.write(transformed_data)

        # Calculate integrity hash
        integrity_hash = self.calculate_file_hash(asset_path)

        # Generate certificate
        certificate = self.generate_certificate(asset_name, security_level, key, author)

        # Create manifest
        manifest = {
            "packageName": asset_name,
            "sourceFileHash": source_hash,
            "creationTimestamp": certificate["creation_timestamp"],
            "author": author,
            "description": description,
            "securityLevel": security_level,
            "classification": classification,
            "integrityHash": integrity_hash,
            "certificate": certificate
        }

        # Save manifest
        manifest_path = self.manifests_dir / f"{asset_name}.manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Save certificate
        cert_path = self.certificates_dir / f"{asset_name}.certificate.json"
        with open(cert_path, 'w', encoding='utf-8') as f:
            json.dump(certificate, f, indent=2, ensure_ascii=False)

        # Log to audit log for enterprise and government levels
        if security_level in ['enterprise', 'government']:
            self._log_to_audit(asset_name, security_level, author, source_hash, integrity_hash)

        print(f"✅ Secure asset created successfully!")
        print(f"   Asset: {asset_path}")
        print(f"   Manifest: {manifest_path}")
        print(f"   Certificate: {cert_path}")

        return asset_path, manifest_path, cert_path

    def _apply_psiqrh_transformation(self, content, key):
        """
        Apply ΨQRH transformation with security key
        This is a placeholder - in practice, use the actual ΨQRH spectral transformation
        """
        # Combine content with key for transformation
        secure_content = f"{key}:{content}"

        # Apply hashing as a simple transformation (in practice, use ΨQRH spectral transform)
        transformed = hashlib.sha256(secure_content.encode()).hexdigest().encode()

        return transformed

    def _log_to_audit(self, asset_name, security_level, author, source_hash, integrity_hash):
        """Log asset creation to audit log"""
        audit_entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action": "CREATE_SECURE_ASSET",
            "asset_name": asset_name,
            "security_level": security_level,
            "author": author,
            "source_hash": source_hash,
            "integrity_hash": integrity_hash
        }

        with open(self.audit_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(audit_entry) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Create secure .Ψcws assets for ΨQRH framework')
    parser.add_argument('--source', required=True, help='Path to source text file')
    parser.add_argument('--name', required=True, help='Name of the secure asset')
    parser.add_argument('--level', required=True, choices=['personal', 'enterprise', 'government'],
                       help='Security level')
    parser.add_argument('--key', help='Encryption key (required for enterprise/government)')
    parser.add_argument('--author', default='Unknown', help='Author name')
    parser.add_argument('--description', default='', help='Asset description')
    parser.add_argument('--classification', default='', help='Security classification')

    args = parser.parse_args()

    # Validate source file exists
    if not os.path.exists(args.source):
        print(f"❌ Source file not found: {args.source}")
        sys.exit(1)

    creator = SecureAssetCreator()

    try:
        creator.create_secure_asset(
            source_file=args.source,
            asset_name=args.name,
            security_level=args.level,
            key=args.key,
            author=args.author,
            description=args.description,
            classification=args.classification
        )
    except Exception as e:
        print(f"❌ Error creating secure asset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()