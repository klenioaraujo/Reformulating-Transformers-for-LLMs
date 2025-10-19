"""
Testes para componentes de física: PadilhaEquation, QuaternionOps, SpectralFiltering
"""

import unittest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ΨQRHSystem.physics.PadilhaEquation import PadilhaEquation
from ΨQRHSystem.physics.QuaternionOps import QuaternionOps
from ΨQRHSystem.physics.SpectralFiltering import SpectralFiltering


class TestPadilhaEquation(unittest.TestCase):
    """Testes para PadilhaEquation"""

    def setUp(self):
        """Configuração inicial para testes"""
        self.eq = PadilhaEquation(I0=1.0, alpha=1.0, beta=0.5, k=2.0, omega=1.0)

    def test_wave_function_computation(self):
        """Testa computação da função de onda"""
        wavelength = torch.linspace(0.1, 1.0, 10)
        time = torch.linspace(0.0, 1.0, 10)

        wave = self.eq.compute_wave_function(wavelength, time)

        # Verificar forma
        self.assertEqual(wave.shape, (10,))
        # Verificar que é complexo
        self.assertTrue(torch.is_complex(wave))

    def test_spectral_components(self):
        """Testa computação de componentes espectrais"""
        wavelength = torch.linspace(0.1, 1.0, 5)
        time = torch.linspace(0.0, 1.0, 5)

        components = self.eq.compute_spectral_components(wavelength, time)

        # Verificar componentes obrigatórios
        required_keys = ['wave_function', 'oscillatory_term', 'dispersion_linear',
                        'dispersion_quadratic', 'total_phase', 'magnitude', 'phase']
        for key in required_keys:
            self.assertIn(key, components)

    def test_fractal_dimension(self):
        """Testa cálculo de dimensão fractal"""
        # Criar sinal de teste
        wave = torch.randn(100) + 1j * torch.randn(100)

        D = self.eq.compute_fractal_dimension(wave)

        # Verificar range físico
        self.assertGreaterEqual(D, 1.0)
        self.assertLessEqual(D, 2.0)

    def test_energy_conservation_validation(self):
        """Testa validação de conservação de energia"""
        input_energy = 100.0
        output_wave = torch.randn(50) + 1j * torch.randn(50)

        is_conserved = self.eq.validate_energy_conservation(input_energy, output_wave)
        self.assertIsInstance(is_conserved, bool)

    def test_optical_probe_output(self):
        """Testa saída da sonda óptica"""
        wave = torch.randn(20) + 1j * torch.randn(20)

        text = self.eq.get_optical_probe_output(wave)

        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_parameter_updates(self):
        """Testa atualização de parâmetros"""
        self.eq.update_parameters({'I0': 2.0, 'alpha': 1.5})

        self.assertEqual(self.eq.I0, 2.0)
        self.assertEqual(self.eq.alpha, 1.5)


class TestQuaternionOps(unittest.TestCase):
    """Testes para QuaternionOps"""

    def setUp(self):
        """Configuração inicial para testes"""
        self.qops = QuaternionOps()

    def test_hamilton_product(self):
        """Testa produto de Hamilton"""
        q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Quaternion unitário
        q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # Quaternion i

        result = self.qops.hamilton_product(q1, q2)

        # Produto de 1 * i = i
        expected = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)

    def test_quaternion_conjugate(self):
        """Testa conjugado quaterniônico"""
        q = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        conj = self.qops.quaternion_conjugate(q)

        expected = torch.tensor([[1.0, -2.0, -3.0, -4.0]])
        torch.testing.assert_close(conj, expected)

    def test_quaternion_norm(self):
        """Testa norma quaterniônica"""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Unitário
        norm = self.qops.quaternion_norm(q)

        self.assertAlmostEqual(norm.item(), 1.0, places=6)

    def test_normalize_quaternion(self):
        """Testa normalização de quaternion"""
        q = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
        normalized = self.qops.normalize_quaternion(q)

        # Deve ser unitário
        norm = self.qops.quaternion_norm(normalized)
        self.assertAlmostEqual(norm.item(), 1.0, places=6)

    def test_so4_rotation(self):
        """Testa rotações SO(4)"""
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # Estado inicial
        angles = torch.tensor([0.1, 0.05, 0.02])   # Ângulos de rotação

        rotated = self.qops.so4_rotation(q.unsqueeze(0), angles.unsqueeze(0))

        # Verificar que mantém a forma
        self.assertEqual(rotated.shape, (1, 1, 4))

        # Verificar que é unitário (aproximadamente)
        norm = self.qops.quaternion_norm(rotated.squeeze(0))
        self.assertAlmostEqual(norm.item(), 1.0, places=3)

    def test_create_unit_quaternion(self):
        """Testa criação de quaternion unitário"""
        q = self.qops.create_unit_quaternion((5, 3, 4))

        self.assertEqual(q.shape, (5, 3, 4, 4))
        self.assertTrue(torch.allclose(q[..., 0], torch.ones_like(q[..., 0])))
        self.assertTrue(torch.allclose(q[..., 1:], torch.zeros_like(q[..., 1:])))

    def test_quaternion_exponential(self):
        """Testa exponencial quaterniônico"""
        q = torch.tensor([[0.0, 0.5, 0.0, 0.0]])  # log(i)
        exp_q = self.qops.quaternion_exponential(q)

        # e^(i*π/2) = i, mas aqui é diferente
        self.assertEqual(exp_q.shape, q.shape)


class TestSpectralFiltering(unittest.TestCase):
    """Testes para SpectralFiltering"""

    def setUp(self):
        """Configuração inicial para testes"""
        self.filter = SpectralFiltering(alpha=1.0, epsilon=1e-10)

    def test_filter_application(self):
        """Testa aplicação do filtro espectral"""
        # Criar estado quântico de teste
        psi = torch.randn(1, 5, 16, 4)

        filtered = self.filter.apply_filter(psi)

        # Verificar forma preservada
        self.assertEqual(filtered.shape, psi.shape)

        # Verificar que não é idêntico (filtragem aplicada)
        self.assertFalse(torch.allclose(filtered, psi, atol=1e-6))

    def test_energy_conservation(self):
        """Testa conservação de energia no filtro"""
        psi = torch.randn(1, 3, 8, 4)

        filtered = self.filter.apply_filter(psi)

        # Calcular energias
        energy_in = torch.sum(psi.abs() ** 2)
        energy_out = torch.sum(filtered.abs() ** 2)

        # Verificar conservação (dentro de 5%)
        conservation_ratio = abs(energy_in - energy_out) / energy_in
        self.assertLess(conservation_ratio, 0.05)

    def test_filter_unitarity_validation(self):
        """Testa validação de unitariedade do filtro"""
        is_unitary = self.filter.validate_filter_unitarity(embed_dim=16)
        self.assertIsInstance(is_unitary.item(), bool)

    def test_get_filter_response(self):
        """Testa obtenção da resposta do filtro"""
        response = self.filter.get_filter_response(embed_dim=16)

        required_keys = ['frequencies', 'k_values', 'filter_magnitude',
                        'filter_phase', 'alpha', 'epsilon']
        for key in required_keys:
            self.assertIn(key, response)

    def test_parameter_updates(self):
        """Testa atualização de parâmetros do filtro"""
        self.filter.update_parameters(alpha=2.0, epsilon=1e-8)

        self.assertEqual(self.filter.alpha, 2.0)
        self.assertEqual(self.filter.epsilon, 1e-8)

    def test_adaptive_filtering(self):
        """Testa filtragem adaptativa"""
        psi = torch.randn(1, 3, 8, 4)
        spectral_characteristics = {'spectral_centroid': 0.7}

        filtered = self.filter.apply_adaptive_filtering(psi, spectral_characteristics)

        self.assertEqual(filtered.shape, psi.shape)

    def test_numerical_stability(self):
        """Testa estabilidade numérica"""
        psi = torch.randn(1, 2, 4, 4)

        is_stable = self.filter.validate_numerical_stability(psi)
        self.assertTrue(is_stable)


if __name__ == '__main__':
    unittest.main()