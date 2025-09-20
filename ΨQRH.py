import torch
import torch.nn as nn
import torch.fft as fft
import math
from typing import Tuple, Optional, Dict
from torch.cuda.amp import autocast

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
    """Implementa o filtro de fase logarítmico F(k) para filtragem de negentropia com estabilidade numérica aprimorada"""

    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-10, use_stable_activation: bool = True):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.use_stable_activation = use_stable_activation

        # Parâmetros para estabilização
        self.register_buffer('k_min', torch.tensor(1e-8))
        self.register_buffer('k_max', torch.tensor(1e4))

    def forward(self, k_mag: torch.Tensor) -> torch.Tensor:
        """
        Aplica o filtro espectral com estabilização numérica

        Args:
            k_mag: Magnitude do vetor de onda de forma [..., dims]
        Returns:
            Filtro aplicado com mesma forma de k_mag
        """
        # Clamp para evitar valores extremos
        k_mag_clamped = torch.clamp(k_mag, self.k_min, self.k_max)

        if self.use_stable_activation:
            # Versão estabilizada usando GELU em vez de arctan
            # F(k) = exp(i * alpha * GELU(log(k_mag + epsilon)))
            log_k = torch.log(k_mag_clamped + self.epsilon)

            # Normalização mais robusta para tensores multi-dimensionais
            log_k_mean = log_k.mean(dim=-1, keepdim=True)
            log_k_std = log_k.std(dim=-1, keepdim=True) + self.epsilon
            log_k_normalized = (log_k - log_k_mean) / log_k_std

            # Usa GELU para suavização estável
            phase = self.alpha * torch.nn.functional.gelu(log_k_normalized)
        else:
            # Versão original com melhorias
            log_k = torch.log(k_mag_clamped + self.epsilon)
            phase = self.alpha * torch.arctan(log_k)

        # Aplica filtro com verificação de NaN/Inf
        filter_response = torch.exp(1j * phase)

        # Substitui valores inválidos por identidade
        invalid_mask = torch.isnan(filter_response) | torch.isinf(filter_response)
        filter_response = torch.where(invalid_mask, torch.ones_like(filter_response), filter_response)

        return filter_response

class QRHLayer(nn.Module):
    """
    Camada ΨQRH para Transformers: Ψ_QRH = R_left · F^{-1} { F(k) · F { Ψ } } · R_right

    Implementa rotações SO(4) corretas usando dois quatérnions independentes
    para obter o grupo completo de rotações 4D.

    Args:
        embed_dim: Dimensão de embedding por componente quaterniônico
        alpha: Parâmetro de escala para o filtro espectral
        theta_left, omega_left, phi_left: Ângulos para o quaternion esquerdo
        theta_right, omega_right, phi_right: Ângulos para o quaternion direito
        use_learned_rotation: Se True, os ângulos são parâmetros aprendíveis
    """

    def __init__(self,
                 embed_dim: int,
                 alpha: float = 1.0,
                 theta_left: float = 0.1,
                 omega_left: float = 0.05,
                 phi_left: float = 0.02,
                 theta_right: float = 0.08,
                 omega_right: float = 0.03,
                 phi_right: float = 0.015,
                 use_learned_rotation: bool = False,
                 spatial_dims: Tuple[int, ...] = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.total_dim = 4 * embed_dim
        self.alpha = alpha
        self.spatial_dims = spatial_dims if spatial_dims is not None else None

        # Parâmetros de rotação esquerda (aprendíveis ou fixos)
        if use_learned_rotation:
            self.theta_left = nn.Parameter(torch.tensor(theta_left, dtype=torch.float32, requires_grad=True))
            self.omega_left = nn.Parameter(torch.tensor(omega_left, dtype=torch.float32, requires_grad=True))
            self.phi_left = nn.Parameter(torch.tensor(phi_left, dtype=torch.float32, requires_grad=True))
            self.theta_right = nn.Parameter(torch.tensor(theta_right, dtype=torch.float32, requires_grad=True))
            self.omega_right = nn.Parameter(torch.tensor(omega_right, dtype=torch.float32, requires_grad=True))
            self.phi_right = nn.Parameter(torch.tensor(phi_right, dtype=torch.float32, requires_grad=True))
        else:
            self.register_buffer('theta_left', torch.tensor(theta_left))
            self.register_buffer('omega_left', torch.tensor(omega_left))
            self.register_buffer('phi_left', torch.tensor(phi_left))
            self.register_buffer('theta_right', torch.tensor(theta_right))
            self.register_buffer('omega_right', torch.tensor(omega_right))
            self.register_buffer('phi_right', torch.tensor(phi_right))

        # Inicializa o filtro espectral
        self.spectral_filter = SpectralFilter(alpha)

        # Camadas de projeção
        self.v_proj = nn.Linear(self.total_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)

        # Registra frequências FFT para reutilização
        self.register_buffer('freqs', None)
    
    def get_rotation_quaternions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna os quatérnions de rotação esquerda e direita para SO(4)"""
        q_left = QuaternionOperations.create_unit_quaternion(
            self.theta_left, self.omega_left, self.phi_left)
        q_right = QuaternionOperations.create_unit_quaternion(
            self.theta_right, self.omega_right, self.phi_right)
        return q_left, q_right
    
    def _compute_frequencies(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcula ou reutiliza as frequências FFT para filtragem multi-dimensional"""
        # Para compatibilidade, se spatial_dims não foi definido, usa dimensão da sequência
        if self.spatial_dims is None or (isinstance(self.spatial_dims, (list, tuple)) and len(self.spatial_dims) == 1 and self.spatial_dims[0] == seq_len):
            # Caso 1D (compatibilidade com implementação anterior)
            if self.freqs is None or self.freqs.size(0) != seq_len:
                self.freqs = fft.fftfreq(seq_len, d=1.0, device=device)
            k = 2 * math.pi * self.freqs.view(1, seq_len, 1)
            k_mag = torch.abs(k) + 1e-10
            return k, k_mag
        else:
            # Caso multi-dimensional
            k_vecs = [fft.fftfreq(n, d=1.0, device=device) for n in self.spatial_dims]
            k_mesh = torch.meshgrid(*k_vecs, indexing='ij')

            # Calcula magnitude do vetor de onda
            k_squared = sum(k_i**2 for k_i in k_mesh)
            k_mag = torch.sqrt(k_squared + 1e-10)  # Soft floor para evitar log(0)

            return k_mesh, k_mag
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de entrada de forma [batch_size, seq_len, 4 * embed_dim]
        Returns:
            Tensor processado com mesma forma
        """
        with autocast(enabled=torch.cuda.is_available()):
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
            k_mesh, k_mag = self._compute_frequencies(seq_len, device)
            F_k = self.spectral_filter(k_mag)
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

            # 10. Aplica rotação SO(4) com dois quatérnions
            q_left, q_right = self.get_rotation_quaternions()
            q_left_expanded = q_left.view(1, 1, 1, 4)   # Preparado para broadcasting
            q_right_expanded = q_right.view(1, 1, 1, 4) # Preparado para broadcasting

            # Aplicar rotação: q_left * Ψ * q_right
            temp = QuaternionOperations.multiply(q_left_expanded, Ψ_reshaped)
            rotated = QuaternionOperations.multiply(temp, q_right_expanded)

            # 11. Prepara para saída
            Ψ_final = rotated.view(batch_size, seq_len, self.total_dim)

            # 12. Projeção final + conexão residual
            output = self.out_proj(Ψ_final) + x

            return output

class GateController:
    """
    Controlador de gate baseado em "recibos" numéricos para controle de fluxo

    Implementa mecanismo ABSTAIN/DELIVER/CLARIFY baseado em:
    - Erro de ortogonalidade
    - Razão de energia removida
    - Ângulo de deriva
    """

    def __init__(self,
                 orthogonal_threshold: float = 1e-6,
                 energy_threshold: float = 0.1,
                 drift_threshold: float = 0.1):
        self.orthogonal_threshold = orthogonal_threshold
        self.energy_threshold = energy_threshold
        self.drift_threshold = drift_threshold

    def calculate_receipts(self,
                          input_tensor: torch.Tensor,
                          output_tensor: torch.Tensor,
                          rotation_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Calcula os recibos numéricos para tomada de decisão de gate

        Args:
            input_tensor: Tensor de entrada
            output_tensor: Tensor de saída
            rotation_params: Parâmetros de rotação atuais

        Returns:
            Dicionário com os valores dos recibos
        """
        receipts = {}

        # 1. Erro de ortogonalidade (preservação de norma)
        input_norm = torch.norm(input_tensor, dim=-1)
        output_norm = torch.norm(output_tensor, dim=-1)
        receipts['orthogonal_error'] = torch.mean(torch.abs(input_norm - output_norm)).item()

        # 2. Razão de energia removida pelo filtro espectral
        input_energy = torch.mean(input_tensor ** 2)
        output_energy = torch.mean(output_tensor ** 2)
        receipts['energy_ratio'] = (input_energy - output_energy) / (input_energy + 1e-10)
        receipts['energy_ratio'] = receipts['energy_ratio'].item()

        # 3. Ângulo de deriva (mudança nos parâmetros de rotação)
        if all(key in rotation_params for key in ['theta_left', 'omega_left', 'phi_left', 'theta_right', 'omega_right', 'phi_right']):
            drift_angle = torch.sqrt(
                rotation_params['theta_left'].detach()**2 +
                rotation_params['omega_left'].detach()**2 +
                rotation_params['phi_left'].detach()**2 +
                rotation_params['theta_right'].detach()**2 +
                rotation_params['omega_right'].detach()**2 +
                rotation_params['phi_right'].detach()**2
            ).item()
        else:
            drift_angle = 0.0
        receipts['drift_angle'] = drift_angle

        return receipts

    def decide_gate(self, receipts: Dict[str, float]) -> str:
        """
        Toma decisão de gate baseada nos recibos

        Returns:
            'ABSTAIN': Recusar processamento (erro muito alto)
            'DELIVER': Entregar resultado (processamento bem-sucedido)
            'CLARIFY': Pedir esclarecimento (resultados incertos)
        """
        orthogonal_error = receipts['orthogonal_error']
        energy_ratio = abs(receipts['energy_ratio'])
        drift_angle = receipts['drift_angle']

        # ABSTAIN: Se erro de ortogonalidade muito alto
        if orthogonal_error > self.orthogonal_threshold:
            return 'ABSTAIN'

        # DELIVER: Se todos os critérios estão dentro dos limites
        if (orthogonal_error <= self.orthogonal_threshold * 0.1 and
            energy_ratio <= self.energy_threshold and
            drift_angle <= self.drift_threshold):
            return 'DELIVER'

        # CLARIFY: Caso intermediário
        return 'CLARIFY'

    def apply_gate_policy(self,
                         gate_decision: str,
                         input_tensor: torch.Tensor,
                         output_tensor: torch.Tensor) -> torch.Tensor:
        """
        Aplica política de gate baseada na decisão

        Args:
            gate_decision: Decisão do gate ('ABSTAIN', 'DELIVER', 'CLARIFY')
            input_tensor: Tensor de entrada original
            output_tensor: Tensor de saída da camada 4D

        Returns:
            Tensor final após aplicação da política
        """
        if gate_decision == 'ABSTAIN':
            # Retorna entrada original (sem modificação)
            return input_tensor
        elif gate_decision == 'DELIVER':
            # Entrega saída processada
            return output_tensor
        elif gate_decision == 'CLARIFY':
            # Mistura entrada e saída com pesos
            confidence_weight = 0.5  # Pode ser adaptativo
            return confidence_weight * output_tensor + (1 - confidence_weight) * input_tensor
        else:
            # Fallback para entrega
            return output_tensor


class NegentropyTransformerBlock(nn.Module):
    """
    Bloco Transformer com integração da camada 4D Unitary (U) Layer

    Combina attention padrão, processamento 4D quaterniônico, e mecanismo de gate
    baseado em "recibos" numéricos para controle de fluxo.
    """

    def __init__(self,
                 d_model: int,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 qrh_embed_dim: int = 64,
                 alpha: float = 1.0,
                 use_learned_rotation: bool = True,
                 enable_gate: bool = True):
        super().__init__()

        self.d_model = d_model
        self.enable_gate = enable_gate

        # Projeções para mapear d_model -> 4 * qrh_embed_dim e vice-versa
        self.input_proj = nn.Linear(d_model, 4 * qrh_embed_dim)
        self.output_proj = nn.Linear(4 * qrh_embed_dim, d_model)

        # Componentes do transformer padrão
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Camada 4D quaterniônica
        self.qrh_layer = QRHLayer(
            embed_dim=qrh_embed_dim,
            alpha=alpha,
            use_learned_rotation=use_learned_rotation
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Mecanismo de gate
        if enable_gate:
            self.gate_controller = GateController()
        else:
            self.gate_controller = None

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass do bloco transformer com integração 4D e mixed precision

        Args:
            src: Tensor de entrada [batch_size, seq_len, d_model]
            src_mask: Máscara de atenção opcional

        Returns:
            Tensor processado [batch_size, seq_len, d_model]
        """
        with autocast(enabled=torch.cuda.is_available()):
            # Self-attention com residual connection
            src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # Projeção para espaço 4D quaterniônico
            qrh_input = self.input_proj(src)  # [batch, seq, 4 * qrh_embed_dim]

            # Processamento 4D quaterniônico
            qrh_output = self.qrh_layer(qrh_input)

            # Aplicar mecanismo de gate se habilitado
            if self.gate_controller is not None:
                # Calcular recibos
                rotation_params = {
                    'theta_left': self.qrh_layer.theta_left,
                    'omega_left': self.qrh_layer.omega_left,
                    'phi_left': self.qrh_layer.phi_left,
                    'theta_right': self.qrh_layer.theta_right,
                    'omega_right': self.qrh_layer.omega_right,
                    'phi_right': self.qrh_layer.phi_right
                }

                receipts = self.gate_controller.calculate_receipts(
                    qrh_input, qrh_output, rotation_params
                )

                # Tomar decisão de gate
                gate_decision = self.gate_controller.decide_gate(receipts)

                # Aplicar política de gate
                gated_output = self.gate_controller.apply_gate_policy(
                    gate_decision, qrh_input, qrh_output
                )
            else:
                gated_output = qrh_output

            # Projeção de volta para espaço d_model
            projected_output = self.output_proj(gated_output)

            # Residual connection e normalização
            src = src + self.dropout(projected_output)
            src = self.norm2(src)

            # Feed-forward network
            src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm3(src)

            return src


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
    print(f"Rotation parameters - theta_left: {layer.theta_left.item():.4f}, "
          f"omega_left: {layer.omega_left.item():.4f}, phi_left: {layer.phi_left.item():.4f}")
    print(f"Rotation parameters - theta_right: {layer.theta_right.item():.4f}, "
          f"omega_right: {layer.omega_right.item():.4f}, phi_right: {layer.phi_right.item():.4f}")
    
    # Teste backward
    loss = output.sum()
    loss.backward()
    
    print("Gradients computed successfully!")
    print("Layer implementation is working correctly.")