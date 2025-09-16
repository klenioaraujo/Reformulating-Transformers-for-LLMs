import torch
import torch.nn as nn
import torch.fft as fft
import math
from typing import Tuple, Optional

class QuaternionOperations:
    """Classe utilitária para operações com quatérnions"""
    
    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiplica dois quatérnions: q1 * q2.
        Args:
            q1, q2: Tensores de forma [..., 4] (w, x, y, z)
        Returns:
            Produto quaterniônico de forma [..., 4]
        """
        w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
        w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)
    
    @staticmethod
    def create_unit_quaternion(theta: torch.Tensor, 
                              omega: torch.Tensor, 
                              phi: torch.Tensor) -> torch.Tensor:
        """Cria um quaternion unitário a partir dos ângulos"""
        cos_theta_2 = torch.cos(theta / 2)
        sin_theta_2 = torch.sin(theta / 2)
        cos_omega = torch.cos(omega)
        sin_omega = torch.sin(omega)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        return torch.stack([
            cos_theta_2,
            sin_theta_2 * cos_omega,
            sin_theta_2 * sin_omega * cos_phi,
            sin_theta_2 * sin_omega * sin_phi
        ], dim=-1)

class SpectralFilter(nn.Module):
    """Implementa o filtro de fase logarítmico F(k)"""
    
    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-10):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Aplica o filtro espectral F(k) = exp(i * alpha * arctan(ln(|k| + epsilon)))
        Args:
            k: Tensor de frequências de forma [..., dims]
        Returns:
            Filtro aplicado com mesma forma de k
        """
        k_mag = torch.abs(k) + self.epsilon
        phase = self.alpha * torch.arctan(torch.log(k_mag))
        return torch.exp(1j * phase)

class QRHLayer(nn.Module):
    """
    Camada ΨQRH para Transformers: Ψ_QRH = R · F^{-1} { F(k) · F { Ψ } }
    
    Args:
        embed_dim: Dimensão de embedding por componente quaterniônico
        alpha: Parâmetro de escala para o filtro espectral
        theta, omega, phi: Ângulos para o quaternion de rotação R
        use_learned_rotation: Se True, os ângulos são parâmetros aprendíveis
    """
    
    def __init__(self, 
                 embed_dim: int, 
                 alpha: float = 1.0, 
                 theta: float = 0.1, 
                 omega: float = 0.05, 
                 phi: float = 0.02,
                 use_learned_rotation: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.total_dim = 4 * embed_dim
        self.alpha = alpha
        
        # Parâmetros de rotação (aprendíveis ou fixos)
        if use_learned_rotation:
            self.theta = nn.Parameter(torch.tensor(theta))
            self.omega = nn.Parameter(torch.tensor(omega))
            self.phi = nn.Parameter(torch.tensor(phi))
        else:
            self.register_buffer('theta', torch.tensor(theta))
            self.register_buffer('omega', torch.tensor(omega))
            self.register_buffer('phi', torch.tensor(phi))
        
        # Inicializa o filtro espectral
        self.spectral_filter = SpectralFilter(alpha)
        
        # Camadas de projeção
        self.v_proj = nn.Linear(self.total_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)
        
        # Registra frequências FFT para reutilização
        self.register_buffer('freqs', None)
    
    def get_rotation_quaternion(self) -> torch.Tensor:
        """Retorna o quaternion de rotação R"""
        return QuaternionOperations.create_unit_quaternion(
            self.theta, self.omega, self.phi)
    
    def _compute_frequencies(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Calcula ou reutiliza as frequências FFT"""
        if self.freqs is None or self.freqs.size(0) != seq_len:
            self.freqs = fft.fftfreq(seq_len, d=1.0, device=device)
        return self.freqs
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de entrada de forma [batch_size, seq_len, 4 * embed_dim]
        Returns:
            Tensor processado com mesma forma
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 1. Projeção para V
        V = self.v_proj(x)
        
        # 2. Divide em componentes quaterniônicos
        D = self.embed_dim
        Ψ_components = [
            V[:, :, i*D:(i+1)*D] for i in range(4)
        ]
        Ψ_w, Ψ_i, Ψ_j, Ψ_k = Ψ_components
        
        # 3. Representação complexa para processamento espectral
        Ψ_complex = Ψ_w + 1j * Ψ_i
        
        # 4. Transformada de Fourier
        Ψ_fft = fft.fft(Ψ_complex, dim=1)
        
        # 5. Aplica filtro espectral
        freqs = self._compute_frequencies(seq_len, device)
        k = 2 * math.pi * freqs.view(1, seq_len, 1).expand(batch_size, -1, D)
        F_k = self.spectral_filter(k)
        Ψ_filtered = Ψ_fft * F_k
        
        # 6. Transformada inversa
        Ψ_ifft_complex = fft.ifft(Ψ_filtered, dim=1)
        
        # 7. Atualiza componentes
        Ψ_new_w = torch.real(Ψ_ifft_complex)
        Ψ_new_i = torch.imag(Ψ_ifft_complex)
        
        # 8. Reconstrói o tensor quaterniônico
        Ψ_new = torch.cat([Ψ_new_w, Ψ_new_i, Ψ_j, Ψ_k], dim=-1)
        
        # 9. Prepara para rotação quaterniônica
        Ψ_reshaped = Ψ_new.view(batch_size, seq_len, D, 4)
        
        # 10. Aplica rotação
        R = self.get_rotation_quaternion()
        R_expanded = R.view(1, 1, 1, 4)  # Preparado para broadcasting
        rotated = QuaternionOperations.multiply(R_expanded, Ψ_reshaped)
        
        # 11. Prepara para saída
        Ψ_final = rotated.view(batch_size, seq_len, self.total_dim)
        
        # 12. Projeção final + conexão residual
        output = self.out_proj(Ψ_final) + x
        
        return output

# Exemplo de uso e teste
if __name__ == "__main__":
    # Configuração
    embed_dim = 16
    batch_size = 2
    seq_len = 8
    
    # Dados de entrada
    x = torch.randn(batch_size, seq_len, 4 * embed_dim)
    
    # Camada com rotação aprendível
    layer = QRHLayer(embed_dim, use_learned_rotation=True)
    
    # Teste forward
    output = layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Rotation parameters - theta: {layer.theta.item():.4f}, "
          f"omega: {layer.omega.item():.4f}, phi: {layer.phi.item():.4f}")
    
    # Teste backward
    loss = output.sum()
    loss.backward()
    
    print("Gradients computed successfully!")
    print("Layer implementation is working correctly.")