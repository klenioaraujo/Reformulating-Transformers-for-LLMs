import torch
import os
import sys

# Adicionar diretório base ao path para encontrar os módulos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tests.human_testing.test_advanced_chat import AdvancedTestModel

def run_simple_test():
    """
    Executa um teste simples do AdvancedTestModel para demonstrar um ciclo de entrada e saída básico.
    """
    print("--- Iniciando Teste de Chat Simples ---")

    # 1. Inicializa o modelo com uma configuração compacta.
    # Usando um modelo menor para um teste mais rápido e simples.
    model = AdvancedTestModel(embed_dim=32, num_layers=2, seq_len=128)
    model.eval()  # Coloca o modelo em modo de avaliação.

    # 2. Define um texto de entrada simples.
    # Este texto será processado pelo modelo.
    input_text = "Explique o conceito de um quatérnion."

    # 3. Converte o texto de entrada para um tensor.
    # O modelo requer um tensor numérico como entrada, não texto puro.
    input_tensor = model.text_to_tensor(input_text)

    # 4. Processa a entrada através do modelo para obter a saída.
    # Usa o método de geração wiki-appropriate para produzir texto compreensível em inglês.
    # Define informações do prompt para categoria matemática.
    prompt_info = {
        'category': 'Mathematical_Concept',
        'domain': 'Mathematics',
        'content': input_text
    }
    output_text = model.generate_wiki_appropriate_response(input_text, prompt_info)


    # 5. Imprime os resultados em um formato legível por humanos.
    print("\n--- Resultados do Teste ---")
    print(f"Texto de Entrada:  '{input_text}'")
    print(f"Texto de Saída: '{output_text}'")
    print("\n--- Teste Finalizado ---")

if __name__ == "__main__":
    run_simple_test()
