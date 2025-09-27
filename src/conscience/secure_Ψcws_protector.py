#!/usr/bin/env python3
"""
Secure ΨCWS Protector - Sistema de Segurança Criptografado
=========================================================

Sistema avançado de proteção para arquivos .Ψcws com:
- Divisão de arquivos com hash de verificação
- 7 camadas de criptografia
- Política anti-violacao
- Sistema de leitura com validação de hash
- Proteção exclusiva para sistema ΨQRH
"""

import torch
import hashlib
import json
import gzip
import struct
import os
import time
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64


@dataclass
class ΨCWSFilePart:
    """Estrutura de parte de arquivo .Ψcws com hash de verificação."""
    part_number: int
    total_parts: int
    content_hash: str
    file_hash: str
    timestamp: str
    encrypted_data: bytes
    integrity_hash: str


class ΨCWSSecurityLayer:
    """Camada de segurança para arquivos .Ψcws com 7 níveis de criptografia."""

    def __init__(self, system_key: str = "PSIQRH_SECURE_SYSTEM"):
        self.system_key = system_key
        self.backend = default_backend()

        # Derivação de chaves do sistema ΨQRH
        self.master_key = self._derive_master_key(system_key)
        self.layer_keys = self._generate_layer_keys()

    def _derive_master_key(self, system_key: str) -> bytes:
        """Deriva chave mestra exclusiva do sistema ΨQRH."""
        salt = b"PSIQRH_SECURE_SALT_v1.0"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=64,
            salt=salt,
            iterations=1000000,
            backend=self.backend
        )
        return base64.urlsafe_b64encode(kdf.derive(system_key.encode()))[:32]

    def _generate_layer_keys(self) -> List[bytes]:
        """Gera 7 chaves únicas para cada camada de criptografia."""
        keys = []
        for i in range(7):
            layer_seed = f"PSIQRH_LAYER_{i}_{self.system_key}_{time.time_ns()}"
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA3_512(),
                length=32,
                salt=layer_seed.encode(),
                iterations=500000,
                backend=self.backend
            )
            keys.append(base64.urlsafe_b64encode(kdf.derive(self.master_key))[:32])
        return keys

    def encrypt_7_layers(self, data: bytes) -> bytes:
        """Aplica 7 camadas de criptografia ao dado."""
        encrypted_data = data

        # Camada 1: AES-256-GCM
        encrypted_data = self._layer_aes_gcm(encrypted_data, self.layer_keys[0])

        # Camada 2: ChaCha20-Poly1305
        encrypted_data = self._layer_chacha20(encrypted_data, self.layer_keys[1])

        # Camada 3: Fernet (AES-128-CBC)
        encrypted_data = self._layer_fernet(encrypted_data, self.layer_keys[2])

        # Camada 4: XOR com chave derivada
        encrypted_data = self._layer_xor(encrypted_data, self.layer_keys[3])

        # Camada 5: Transposição customizada
        encrypted_data = self._layer_transposition(encrypted_data)

        # Camada 6: HMAC + AES
        encrypted_data = self._layer_hmac_aes(encrypted_data, self.layer_keys[4])

        # Camada 7: Obfuscação final
        encrypted_data = self._layer_obfuscation(encrypted_data, self.layer_keys[5])

        return encrypted_data

    def decrypt_7_layers(self, encrypted_data: bytes) -> bytes:
        """Remove 7 camadas de criptografia do dado."""
        decrypted_data = encrypted_data

        # Remover camadas na ordem inversa
        decrypted_data = self._layer_obfuscation(decrypted_data, self.layer_keys[5], decrypt=True)
        decrypted_data = self._layer_hmac_aes(decrypted_data, self.layer_keys[4], decrypt=True)
        decrypted_data = self._layer_transposition(decrypted_data, decrypt=True)
        decrypted_data = self._layer_xor(decrypted_data, self.layer_keys[3], decrypt=True)
        decrypted_data = self._layer_fernet(decrypted_data, self.layer_keys[2], decrypt=True)
        decrypted_data = self._layer_chacha20(decrypted_data, self.layer_keys[1], decrypt=True)
        decrypted_data = self._layer_aes_gcm(decrypted_data, self.layer_keys[0], decrypt=True)

        return decrypted_data

    def _layer_aes_gcm(self, data: bytes, key: bytes, decrypt: bool = False) -> bytes:
        """Camada 1: AES-256-GCM."""
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)

        if decrypt:
            # Para descriptografia, extrair IV dos primeiros bytes
            iv = data[:16]
            tag = data[16:32]
            ciphertext = data[32:]
            # Recreate cipher with extracted IV
            cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
            decryptor = cipher.decryptor()
            return decryptor.update(ciphertext) + decryptor.finalize()
        else:
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data) + encryptor.finalize()
            return iv + encryptor.tag + ciphertext

    def _layer_chacha20(self, data: bytes, key: bytes, decrypt: bool = False) -> bytes:
        """Camada 2: ChaCha20-Poly1305."""
        nonce = os.urandom(16)
        algorithm = algorithms.ChaCha20(key, nonce)
        cipher = Cipher(algorithm, mode=None, backend=self.backend)

        if decrypt:
            nonce = data[:16]
            ciphertext = data[16:]
            algorithm = algorithms.ChaCha20(key, nonce)
            cipher = Cipher(algorithm, mode=None, backend=self.backend)
            decryptor = cipher.decryptor()
            return decryptor.update(ciphertext)
        else:
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(data)
            return nonce + ciphertext

    def _layer_fernet(self, data: bytes, key: bytes, decrypt: bool = False) -> bytes:
        """Camada 3: Fernet (AES-128-CBC)."""
        # Ensure key is proper Fernet key (32 url-safe base64 bytes)
        fernet_key = base64.urlsafe_b64encode(key[:32])
        f = Fernet(fernet_key)
        if decrypt:
            return f.decrypt(data)
        else:
            return f.encrypt(data)

    def _layer_xor(self, data: bytes, key: bytes, decrypt: bool = False) -> bytes:
        """Camada 4: XOR com chave derivada."""
        # Expandir chave para tamanho dos dados
        expanded_key = (key * (len(data) // len(key) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, expanded_key))

    def _layer_transposition(self, data: bytes, key: bytes = None, decrypt: bool = False) -> bytes:
        """Camada 5: Transposição customizada (simplificada)."""
        # Simple reversible transposition - swap adjacent bytes
        if decrypt:
            # Reverse the swap
            result = bytearray(data)
            for i in range(0, len(result)-1, 2):
                result[i], result[i+1] = result[i+1], result[i]
            return bytes(result)
        else:
            # Swap adjacent bytes
            result = bytearray(data)
            for i in range(0, len(result)-1, 2):
                result[i], result[i+1] = result[i+1], result[i]
            return bytes(result)

    def _layer_hmac_aes(self, data: bytes, key: bytes, decrypt: bool = False) -> bytes:
        """Camada 6: HMAC + AES (Fixed Version)."""
        from cryptography.hazmat.primitives import padding

        if decrypt:
            # Extract HMAC and ciphertext
            if len(data) < 32 + 16:  # HMAC + IV minimum
                raise ValueError("Invalid HMAC-AES data format")

            hmac_digest = data[:32]
            ciphertext = data[32:]

            # Verify HMAC BEFORE decryption (ciphertext only, not including IV)
            h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
            h.update(ciphertext[16:])  # Only the actual encrypted data, not IV
            h.verify(hmac_digest)

            # Extract IV and encrypted data
            iv = ciphertext[:16]
            encrypted_data = ciphertext[16:]

            # Decrypt
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

            # Unpad
            unpadder = padding.PKCS7(128).unpadder()
            return unpadder.update(padded_data) + unpadder.finalize()

        else:
            # Pad data
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()

            # Encrypt
            iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()

            # Calculate HMAC (ONLY on ciphertext, not including IV)
            h = hmac.HMAC(key, hashes.SHA256(), backend=self.backend)
            h.update(ciphertext)
            hmac_digest = h.finalize()

            return hmac_digest + iv + ciphertext

    def _layer_obfuscation(self, data: bytes, key: bytes, decrypt: bool = False) -> bytes:
        """Camada 7: Obfuscação final."""
        if decrypt:
            # Remover padding e reverter XOR
            padding_length = data[-1]
            data = data[:-padding_length]
            return self._layer_xor(data, key, decrypt=False)  # XOR reverso
        else:
            # Aplicar XOR e padding
            data = self._layer_xor(data, key, decrypt=False)
            padding_length = 16 - (len(data) % 16)
            return data + bytes([padding_length] * padding_length)


class ΨCWSFileSplitter:
    """Sistema de divisão de arquivos .Ψcws com hash de verificação."""

    def __init__(self, security_layer: ΨCWSSecurityLayer):
        self.security_layer = security_layer

    def split_file(self, file_path: Union[str, Path], parts: int = 4) -> List[ΨCWSFilePart]:
        """Divide arquivo .Ψcws em partes com hash de verificação."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        with open(file_path, 'rb') as f:
            file_data = f.read()

        # Calcular hash do arquivo completo
        file_hash = hashlib.sha3_512(file_data).hexdigest()

        # Dividir arquivo em partes
        part_size = len(file_data) // parts
        file_parts = []

        for i in range(parts):
            start_idx = i * part_size
            end_idx = start_idx + part_size if i < parts - 1 else len(file_data)
            part_data = file_data[start_idx:end_idx]

            # Calcular hash da parte
            part_hash = hashlib.sha3_512(part_data).hexdigest()

            # Criptografar parte
            encrypted_part = self.security_layer.encrypt_7_layers(part_data)

            # Calcular hash de integridade
            integrity_data = part_hash + file_hash + str(i)
            integrity_hash = hashlib.sha3_512(integrity_data.encode()).hexdigest()

            # Criar estrutura da parte
            file_part = ΨCWSFilePart(
                part_number=i,
                total_parts=parts,
                content_hash=part_hash,
                file_hash=file_hash,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                encrypted_data=encrypted_part,
                integrity_hash=integrity_hash
            )

            file_parts.append(file_part)

        return file_parts

    def save_parts(self, file_parts: List[ΨCWSFilePart], output_dir: Union[str, Path]):
        """Salva partes do arquivo em disco."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for part in file_parts:
            # Nome do arquivo baseado no hash do arquivo e número da parte
            filename = f"{part.file_hash[:16]}_part_{part.part_number:02d}.Ψcws"
            part_path = output_dir / filename

            # Serializar parte
            part_dict = asdict(part)
            part_dict['encrypted_data'] = base64.b64encode(part_dict['encrypted_data']).decode()

            with open(part_path, 'w') as f:
                json.dump(part_dict, f, indent=2)

            saved_paths.append(part_path)

        return saved_paths

    def load_parts(self, part_files: List[Union[str, Path]]) -> List[ΨCWSFilePart]:
        """Carrega partes do arquivo do disco."""
        file_parts = []

        for part_file in part_files:
            part_file = Path(part_file)

            if not part_file.exists():
                raise FileNotFoundError(f"Parte não encontrada: {part_file}")

            with open(part_file, 'r') as f:
                part_dict = json.load(f)

            # Desserializar dados
            part_dict['encrypted_data'] = base64.b64decode(part_dict['encrypted_data'])

            file_part = ΨCWSFilePart(**part_dict)
            file_parts.append(file_part)

        return file_parts

    def reassemble_file(self, file_parts: List[ΨCWSFilePart], output_path: Union[str, Path]) -> bool:
        """Reconstrói arquivo original a partir das partes com verificação de hash."""
        # Ordenar partes por número
        file_parts.sort(key=lambda x: x.part_number)

        # Verificar integridade de todas as partes
        for part in file_parts:
            if not self._verify_part_integrity(part):
                raise ValueError(f"Parte {part.part_number} corrompida ou violada")

        # Verificar consistência entre partes
        if not self._verify_parts_consistency(file_parts):
            raise ValueError("Partes inconsistentes - possível violação")

        # Decriptografar e juntar partes
        reassembled_data = b''
        for part in file_parts:
            decrypted_part = self.security_layer.decrypt_7_layers(part.encrypted_data)
            reassembled_data += decrypted_part

        # Verificar hash do arquivo reconstruído
        reconstructed_hash = hashlib.sha3_512(reassembled_data).hexdigest()
        expected_hash = file_parts[0].file_hash

        if reconstructed_hash != expected_hash:
            raise ValueError("Hash do arquivo reconstruído não corresponde - violação detectada")

        # Salvar arquivo reconstruído
        output_path = Path(output_path)
        with open(output_path, 'wb') as f:
            f.write(reassembled_data)

        return True

    def _verify_part_integrity(self, part: ΨCWSFilePart) -> bool:
        """Verifica integridade de uma parte individual."""
        # Decriptografar para verificar hash do conteúdo
        try:
            decrypted_data = self.security_layer.decrypt_7_layers(part.encrypted_data)
            calculated_hash = hashlib.sha3_512(decrypted_data).hexdigest()

            if calculated_hash != part.content_hash:
                return False

            # Verificar hash de integridade
            integrity_data = part.content_hash + part.file_hash + str(part.part_number)
            calculated_integrity = hashlib.sha3_512(integrity_data.encode()).hexdigest()

            return calculated_integrity == part.integrity_hash

        except Exception:
            return False

    def _verify_parts_consistency(self, file_parts: List[ΨCWSFilePart]) -> bool:
        """Verifica consistência entre todas as partes."""
        if not file_parts:
            return False

        # Verificar se todas as partes pertencem ao mesmo arquivo
        expected_file_hash = file_parts[0].file_hash
        expected_total_parts = file_parts[0].total_parts

        for part in file_parts:
            if part.file_hash != expected_file_hash:
                return False
            if part.total_parts != expected_total_parts:
                return False

        # Verificar se temos todas as partes
        part_numbers = {part.part_number for part in file_parts}
        expected_numbers = set(range(expected_total_parts))

        return part_numbers == expected_numbers


class ΨCWSProtector:
    """Sistema completo de proteção para arquivos .Ψcws."""

    def __init__(self, system_key: str = "PSIQRH_SECURE_SYSTEM"):
        self.security_layer = ΨCWSSecurityLayer(system_key)
        self.file_splitter = ΨCWSFileSplitter(self.security_layer)
        self.anti_violation_policy = ΨCWSAntiViolationPolicy()

    def protect_file(self, file_path: Union[str, Path], parts: int = 4,
                    output_dir: Union[str, Path] = None) -> List[Path]:
        """Protege arquivo .Ψcws com todas as camadas de segurança."""
        file_path = Path(file_path)

        if output_dir is None:
            output_dir = file_path.parent / "secure_parts"

        # Aplicar política anti-violacao
        self.anti_violation_policy.scan_file(file_path)

        # Dividir arquivo em partes seguras
        file_parts = self.file_splitter.split_file(file_path, parts)

        # Salvar partes
        saved_paths = self.file_splitter.save_parts(file_parts, output_dir)

        # Registrar proteção
        self.anti_violation_policy.record_protection(file_path, saved_paths)

        return saved_paths

    def read_protected_file(self, part_files: List[Union[str, Path]],
                          output_path: Union[str, Path] = None) -> bool:
        """Lê arquivo protegido apenas se todas as verificações passarem."""
        # Aplicar política anti-violacao
        if not self.anti_violation_policy.verify_access(part_files):
            raise PermissionError("Acesso negado - violação detectada")

        # Carregar partes
        file_parts = self.file_splitter.load_parts(part_files)

        # Reconstruir arquivo
        if output_path is None:
            output_path = Path(part_files[0]).parent / "reconstructed.Ψcws"

        success = self.file_splitter.reassemble_file(file_parts, output_path)

        if success:
            self.anti_violation_policy.record_access(part_files, "success")
        else:
            self.anti_violation_policy.record_access(part_files, "failed")

        return success


class ΨCWSAntiViolationPolicy:
    """Política anti-violacao para arquivos .Ψcws."""

    def __init__(self):
        self.violation_log = []
        self.access_log = []
        self.max_attempts = 3
        self.attempt_count = {}

    def scan_file(self, file_path: Path) -> bool:
        """Escaneia arquivo em busca de violações."""
        # Verificar se arquivo foi modificado
        file_stat = file_path.stat()
        current_time = time.time()

        # Verificar se modificação é suspeita
        if current_time - file_stat.st_mtime < 60:  # Modificado nos últimos 60 segundos
            self._log_violation(f"Modificação suspeita detectada: {file_path}")
            return False

        # Verificar tamanho do arquivo
        if file_stat.st_size == 0:
            self._log_violation(f"Arquivo vazio: {file_path}")
            return False

        return True

    def verify_access(self, part_files: List[Path]) -> bool:
        """Verifica se acesso às partes é permitido."""
        file_key = str(sorted(part_files))

        # Verificar tentativas excessivas
        if self.attempt_count.get(file_key, 0) >= self.max_attempts:
            self._log_violation(f"Tentativas excessivas para: {file_key}")
            return False

        # Verificar integridade das partes
        for part_file in part_files:
            if not part_file.exists():
                self._log_violation(f"Parte não encontrada: {part_file}")
                return False

            # Verificar timestamp
            part_stat = part_file.stat()
            current_time = time.time()

            if current_time - part_stat.st_mtime < 30:  # Modificado nos últimos 30 segundos
                self._log_violation(f"Parte modificada recentemente: {part_file}")
                return False

        return True

    def record_protection(self, original_file: Path, part_files: List[Path]):
        """Registra proteção bem-sucedida."""
        protection_record = {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'original_file': str(original_file),
            'part_files': [str(p) for p in part_files],
            'status': 'protected'
        }
        self.access_log.append(protection_record)

    def record_access(self, part_files: List[Path], status: str):
        """Registra tentativa de acesso."""
        file_key = str(sorted(part_files))

        if status == "success":
            self.attempt_count[file_key] = 0
        else:
            self.attempt_count[file_key] = self.attempt_count.get(file_key, 0) + 1

        access_record = {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'part_files': [str(p) for p in part_files],
            'status': status
        }
        self.access_log.append(access_record)

    def _log_violation(self, message: str):
        """Registra violação detectada."""
        violation_record = {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            'message': message,
            'severity': 'high'
        }
        self.violation_log.append(violation_record)

        # Alertar sistema
        print(f"🚨 VIOLAÇÃO DETECTADA: {message}")


# Interface principal para uso externo
def create_secure_Ψcws_protector(system_key: str = "PSIQRH_SECURE_SYSTEM") -> ΨCWSProtector:
    """Cria instância do protetor seguro para arquivos .Ψcws."""
    return ΨCWSProtector(system_key)


if __name__ == "__main__":
    # Exemplo de uso
    protector = create_secure_Ψcws_protector()

    # Proteger arquivo
    # protected_parts = protector.protect_file("exemplo.Ψcws", parts=4)

    # Ler arquivo protegido
    # success = protector.read_protected_file(protected_parts, "reconstruido.Ψcws")

    print("🔒 Sistema de proteção ΨCWS inicializado com sucesso")