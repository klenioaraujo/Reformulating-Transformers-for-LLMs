#!/usr/bin/env python3
"""
Secure Asset Validator for ΨQRH Framework

Validates .Ψcws assets and their certificates to ensure only certified files
can be used by the ΨQRH system.
"""

import json
import hashlib
import os
import sys
from pathlib import Path


class SecureAssetValidator:
    def __init__(self):
        self.assets_dir = Path('data/secure_assets/Ψcws')
        self.certificates_dir = Path('data/secure_assets/certificates')
        self.manifests_dir = Path('data/secure_assets/manifests')
        self.system_id = "PSIQRH_SECURE_SYSTEM"

    def validate_asset(self, asset_name, key=None):
        """
        Validate a secure asset and its certificate
        Returns True if valid, False otherwise
        """
        asset_path = self.assets_dir / f"{asset_name}.Ψcws"
        manifest_path = self.manifests_dir / f"{asset_name}.manifest.json"
        cert_path = self.certificates_dir / f"{asset_name}.certificate.json"

        # Check if all required files exist
        if not asset_path.exists():
            print(f"❌ Asset file not found: {asset_path}")
            return False

        if not manifest_path.exists():
            print(f"❌ Manifest file not found: {manifest_path}")
            return False

        if not cert_path.exists():
            print(f"❌ Certificate file not found: {cert_path}")
            return False

        try:
            # Load manifest
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            # Load certificate
            with open(cert_path, 'r', encoding='utf-8') as f:
                certificate = json.load(f)

            # Validate certificate
            if not self._validate_certificate(certificate):
                print("❌ Invalid certificate")
                return False

            # Validate integrity hash
            current_integrity_hash = self._calculate_file_hash(asset_path)
            if current_integrity_hash != manifest.get('integrityHash'):
                print("❌ Integrity hash mismatch - file may be corrupted")
                return False

            # Validate security level requirements
            security_level = manifest.get('securityLevel', 'personal')
            if security_level in ['enterprise', 'government'] and not key:
                print(f"❌ Key required for {security_level} security level")
                return False

            # If key provided, validate it matches certificate
            if key:
                # In practice, this would validate against the actual encryption
                # For now, we'll check if the key matches the expected pattern
                if security_level == 'enterprise' and len(key) < 8:
                    print("❌ Enterprise key too weak")
                    return False

            print(f"✅ Asset '{asset_name}' validated successfully")
            print(f"   Security Level: {security_level}")
            print(f"   Author: {manifest.get('author', 'Unknown')}")
            print(f"   Created: {manifest.get('creationTimestamp', 'Unknown')}")

            return True

        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False

    def _validate_certificate(self, certificate):
        """Validate the digital certificate"""
        # Check required fields
        required_fields = ['asset_name', 'security_level', 'author', 'creation_timestamp',
                          'system_id', 'certificate_version', 'certificate_hash']

        for field in required_fields:
            if field not in certificate:
                print(f"❌ Missing certificate field: {field}")
                return False

        # Validate system ID
        if certificate['system_id'] != self.system_id:
            print("❌ Invalid system ID in certificate")
            return False

        # Recalculate certificate hash for validation
        cert_copy = certificate.copy()
        stored_hash = cert_copy.pop('certificate_hash')

        cert_string = json.dumps(cert_copy, sort_keys=True)
        calculated_hash = hashlib.sha256(cert_string.encode()).hexdigest()

        if stored_hash != calculated_hash:
            print("❌ Certificate hash validation failed")
            return False

        return True

    def _calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def list_secure_assets(self):
        """List all available secure assets"""
        assets = []

        for manifest_file in self.manifests_dir.glob("*.manifest.json"):
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)

                asset_name = manifest.get('packageName', manifest_file.stem.replace('.manifest', ''))
                assets.append({
                    'name': asset_name,
                    'security_level': manifest.get('securityLevel', 'unknown'),
                    'author': manifest.get('author', 'Unknown'),
                    'created': manifest.get('creationTimestamp', 'Unknown'),
                    'description': manifest.get('description', '')
                })
            except Exception as e:
                print(f"Warning: Could not read manifest {manifest_file}: {e}")

        return assets


def main():
    parser = argparse.ArgumentParser(description='Validate secure .Ψcws assets for ΨQRH framework')
    parser.add_argument('--name', help='Name of asset to validate')
    parser.add_argument('--key', help='Encryption key (if required)')
    parser.add_argument('--list', action='store_true', help='List all secure assets')

    args = parser.parse_args()

    validator = SecureAssetValidator()

    if args.list:
        assets = validator.list_secure_assets()
        if not assets:
            print("📭 No secure assets found")
        else:
            print("📦 Available Secure Assets:")
            for asset in assets:
                print(f"  • {asset['name']} ({asset['security_level']})")
                print(f"    Author: {asset['author']}")
                print(f"    Created: {asset['created']}")
                if asset['description']:
                    print(f"    Description: {asset['description']}")
                print()
    elif args.name:
        success = validator.validate_asset(args.name, args.key)
        sys.exit(0 if success else 1)
    else:
        print("❌ Please specify --name to validate an asset or --list to see all assets")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    main()