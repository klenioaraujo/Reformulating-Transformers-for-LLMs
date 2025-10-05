"""
Tensor Shape Validator - Engine avançada para validação e manipulação segura de tensores

Desenvolvido para o Sistema ΨQRH com validação científica completa
"""

import torch
from typing import Union, Tuple, Optional
import logging


def validate_and_reshape(x: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Valida e redimensiona tensor garantindo compatibilidade dimensional.

    Args:
        x (torch.Tensor): Tensor de entrada
        target_shape (tuple): Shape alvo desejado

    Returns:
        torch.Tensor: Tensor redimensionado

    Raises:
        ValueError: Se houver incompatibilidade dimensional
    """
    total_elements = x.numel()
    target_elements = torch.prod(torch.tensor(target_shape)).item()

    if total_elements != target_elements:
        raise ValueError(
            f"Shape mismatch: input has {total_elements} elements, "
            f"but target shape {target_shape} requires {target_elements}."
        )
    return x.reshape(target_shape)


class TensorShapeValidator:
    """
    Engine avançada para validação e manipulação segura de formas de tensor
    """

    def __init__(self, auto_adjust: bool = True, logger: Optional[logging.Logger] = None):
        self.auto_adjust = auto_adjust
        self.logger = logger or logging.getLogger(__name__)

    def validate_and_reshape(self,
                           x: torch.Tensor,
                           target_shape: Union[tuple, list],
                           operation_name: str = "unknown") -> torch.Tensor:
        """
        Valida e redimensiona tensor com opções avançadas.

        Args:
            x: Tensor de entrada
            target_shape: Shape alvo
            operation_name: Nome da operação para logging

        Returns:
            Tensor redimensionado
        """
        total_elements = x.numel()
        target_elements = torch.prod(torch.tensor(target_shape)).item()

        # Verificação básica
        if total_elements == target_elements:
            return x.reshape(target_shape)

        # Log do erro
        error_msg = (
            f"Shape mismatch in {operation_name}: "
            f"input has {total_elements} elements, "
            f"target shape {target_shape} requires {target_elements}"
        )

        if self.auto_adjust:
            # Tentar ajuste automático
            adjusted_shape = self._auto_adjust_shape(x, target_shape)
            if adjusted_shape:
                self.logger.warning(f"{error_msg}. Auto-adjusting to {adjusted_shape}")
                return x.reshape(adjusted_shape)

        # Se não puder ajustar, levantar erro
        raise ValueError(error_msg)

    def _auto_adjust_shape(self,
                          x: torch.Tensor,
                          target_shape: Union[tuple, list]) -> Optional[tuple]:
        """
        Tenta ajustar automaticamente o shape mantendo a estrutura.
        """
        total_elements = x.numel()

        # Handle -1 dimensions (infer dimension)
        if -1 in target_shape:
            # Calculate inferred dimension
            known_dims = [dim for dim in target_shape if dim != -1 and dim > 0]
            known_product = 1
            for dim in known_dims:
                known_product *= dim

            if known_product > 0 and total_elements % known_product == 0:
                inferred_dim = total_elements // known_product
                adjusted_shape = tuple(inferred_dim if dim == -1 else dim for dim in target_shape)

                # Verify the adjusted shape is valid
                if torch.prod(torch.tensor(adjusted_shape)).item() == total_elements:
                    return adjusted_shape

        # Caso 1: Ajustar apenas a dimensão de features
        if len(target_shape) >= 2:
            batch_dim = target_shape[0]
            if batch_dim > 0 and total_elements % batch_dim == 0:
                feature_dim = total_elements // batch_dim
                adjusted_shape = (batch_dim, feature_dim) + target_shape[2:]

                # Verificar se o shape ajustado é válido
                if torch.prod(torch.tensor(adjusted_shape)).item() == total_elements:
                    return adjusted_shape

        # Caso 2: Shape 4D comum [batch, channels, height, width]
        if len(target_shape) == 4:
            batch, channels, height, width = target_shape

            # Tentar ajustar channels
            if all(dim > 0 for dim in [batch, height, width]) and total_elements % (batch * height * width) == 0:
                adjusted_channels = total_elements // (batch * height * width)
                return (batch, adjusted_channels, height, width)

            # Tentar ajustar spatial dimensions
            if all(dim > 0 for dim in [batch, channels]) and total_elements % (batch * channels) == 0:
                spatial_elements = total_elements // (batch * channels)
                # Encontrar divisores para height e width
                for h in range(1, int(spatial_elements**0.5) + 1):
                    if spatial_elements % h == 0:
                        w = spatial_elements // h
                        return (batch, channels, h, w)

        return None

    def safe_reshape_chain(self,
                          x: torch.Tensor,
                          operations: list,
                          context: dict = None) -> torch.Tensor:
        """
        Executa uma cadeia de operações de reshape com validação.

        Args:
            x: Tensor inicial
            operations: Lista de tuplas (target_shape, operation_name)
            context: Contexto adicional para logging

        Returns:
            Tensor após todas as operações
        """
        current_tensor = x

        for i, (target_shape, op_name) in enumerate(operations):
            try:
                current_tensor = self.validate_and_reshape(
                    current_tensor, target_shape, f"{op_name}_step_{i}"
                )
            except ValueError as e:
                self.logger.error(f"Failed at operation {i} ({op_name}): {e}")
                if context:
                    self.logger.error(f"Context: {context}")
                raise

        return current_tensor


# Função de conveniência
def safe_reshape(x: torch.Tensor,
                target_shape: Union[tuple, list],
                auto_adjust: bool = True,
                operation_name: str = "reshape") -> torch.Tensor:
    """
    Função de conveniência para reshape seguro.
    """
    validator = TensorShapeValidator(auto_adjust=auto_adjust)
    return validator.validate_and_reshape(x, target_shape, operation_name)


if __name__ == "__main__":
    # Testes básicos
    print("Testing TensorShapeValidator...")

    # Teste 1: Caso de sucesso
    x = torch.randn(4096)
    validator = TensorShapeValidator(auto_adjust=True)

    try:
        result = validator.validate_and_reshape(x, (64, 64), "test_success")
        print(f"✓ Teste sucesso: {x.shape} -> {result.shape}")
    except Exception as e:
        print(f"✗ Teste sucesso falhou: {e}")

    # Teste 2: Ajuste automático
    try:
        result = validator.validate_and_reshape(x, (1, 128, 1, 1), "test_auto_adjust")
        print(f"✓ Ajuste automático: {x.shape} -> {result.shape}")
    except Exception as e:
        print(f"✗ Ajuste automático falhou: {e}")

    print("Testes completos!")


class ScientificTensorValidator(TensorShapeValidator):
    """
    Validador especializado para operações científicas do ΨQRH
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Configurações específicas para ΨQRH
        self.expected_shapes = {
            "energy_conservation": [1, 128, 1, 1],
            "spectral_filter": [1, 128, 32, 32],
            "quaternion_processing": [1, 128, 8, 8]
        }

    def validate_for_operation(self, x: torch.Tensor, operation_type: str) -> torch.Tensor:
        """
        Valida tensor para operação específica do ΨQRH.
        """
        if operation_type not in self.expected_shapes:
            raise ValueError(f"Operação desconhecida: {operation_type}")

        target_shape = self.expected_shapes[operation_type]
        return self.validate_and_reshape(x, target_shape, operation_type)