import torch
import os
import sys

# Add base directory to path to find modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tests.human_testing.test_advanced_chat import AdvancedTestModel

import torch
import os
import sys

# Add base directory to path to find modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from tests.human_testing.test_advanced_chat import AdvancedTestModel

def run_graduated_complexity_test():
    """
    Runs a test with 10 questions of increasing complexity to demonstrate
    the framework's ability to analyze different concepts.
    """
    print("--- Starting Graduated Complexity Test ---")

    # 1. Initialize the model with a standard configuration.
    model = AdvancedTestModel(embed_dim=32, num_layers=2, seq_len=256)
    model.eval()

    # 2. Define 10 questions with increasing complexity and varied domains.
    test_prompts = [
        # Level 1: Simple mathematical concept
        {"content": "What is a prime number?", "category": "Mathematical_Concept", "domain": "Mathematics"},
        # Level 2: Basic programming concept
        {"content": "Explain the difference between a list and a tuple in Python.", "category": "Code_Explanation", "domain": "Programming"},
        # Level 3: Fundamental law of physics
        {"content": "What is Newton's first law of motion?", "category": "Scientific_Question", "domain": "Physics"},
        # Level 4: Literary structure
        {"content": "Describe the structure of a sonnet.", "category": "Creative_Writing", "domain": "Literature"},
        # Level 5: Intermediate technical concept
        {"content": "What is the importance of the Fourier Transform in signal processing?", "category": "Technical_Explanation", "domain": "Engineering"},
        # Level 6: Concept connecting two fields
        {"content": "Explain the concept of recursion from a computational and mathematical perspective.", "category": "Mathematical_Concept", "domain": "Computer Science"},
        # Level 7: Applied mathematics
        {"content": "How can a differential equation model population growth?", "category": "Scientific_Question", "domain": "Applied Mathematics"},
        # Level 8: Abstract linguistic concept
        {"content": "Analyze the linguistic concept of semantic satiation.", "category": "Sarcasm_Irony", "domain": "Linguistics"}, # Adapted category
        # Level 9: Complex, cross-disciplinary concept
        {"content": "Discuss the relationship between entropy in thermodynamics and information theory.", "category": "Scientific_Question", "domain": "Physics"},
        # Level 10: Advanced physics topic
        {"content": "Elaborate on the geometric interpretation of gauge theories in particle physics.", "category": "Scientific_Question", "domain": "Particle Physics"}
    ]

    # 3. Iterate over each question, process, and print the result.
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Question {i+1}/10: {prompt['domain']} ---")
        
        input_text = prompt['content']
        
        prompt_info = {
            'category': prompt['category'],
            'domain': prompt['domain'],
            'content': input_text
        }

        # Process the input using the real ΨQRH pipeline.
        output_text = model.generate_wiki_appropriate_response(input_text, prompt_info)

        # Print the results in a readable format.
        print(f"Input:  '{input_text}'")
        print(f"Analytical Output:\n{output_text}")

    print("\n--- Graduated Complexity Test Finished ---")

if __name__ == "__main__":
    run_graduated_complexity_test()
