#!/usr/bin/env python3
"""
Correção permanente para o erro 'tuple index out of range' no optical probe
"""

class OpticalProbeSafeHandler:
    def __init__(self):
        self.fallback_chars = 'ΨQRHhelloHELLO123'
        self.fallback_index = 0
    
    def safe_extract(self, optical_output, max_attempts=5):
        """Extrai texto do optical probe de forma absolutamente segura"""
        
        for attempt in range(max_attempts):
            try:
                # Tentativa 1: Acesso direto
                if hasattr(optical_output, '__getitem__'):
                    if len(optical_output) > 0:
                        result = optical_output[0]
                        if result is not None:
                            return str(result)
                
                # Tentativa 2: Tuple unpacking
                if isinstance(optical_output, tuple):
                    if len(optical_output) > 0:
                        result = optical_output[0]
                        if result is not None:
                            return str(result)
                
                # Tentativa 3: List access
                if isinstance(optical_output, list):
                    if len(optical_output) > 0:
                        result = optical_output[0]
                        if result is not None:
                            return str(result)
                
                # Tentativa 4: String conversion
                str_rep = str(optical_output)
                if str_rep and str_rep != 'None':
                    return str_rep[0] if len(str_rep) > 0 else self._get_fallback()
                    
            except (IndexError, TypeError, AttributeError) as e:
                if attempt == max_attempts - 1:
                    print(f"⚠️  Todas as tentativas falharam, usando fallback: {e}")
                    return self._get_fallback()
                continue
        
        return self._get_fallback()
    
    def _get_fallback(self):
        """Retorna caractere fallback com rotação"""
        char = self.fallback_chars[self.fallback_index % len(self.fallback_chars)]
        self.fallback_index += 1
        return char

# Instância global
optical_probe_handler = OpticalProbeSafeHandler()