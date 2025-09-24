import torch
import os
import sys

# Add base directory to path to find the model module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

# Conditional import for the test block
if __name__ == "__main__":
    from tests.human_testing.test_advanced_chat import AdvancedTestModel

def design_signal_for_min_complexity(seq_len: int) -> str:
    """
    Designs a string that minimizes Spectral Complexity.

    A signal with minimum complexity is a constant (DC) signal. Its spectrum
    has energy only at frequency k=0.

    Args:
        seq_len: The desired length of the string.

    Returns:
        A string of `seq_len` identical characters.
    """
    # A constant signal has zero variance, minimizing complexity.
    # We choose 'A' as the constant character.
    return 'A' * seq_len

def design_signal_for_max_complexity(seq_len: int) -> str:
    """
    Designs a string that maximizes Spectral Complexity.

    A signal with maximum complexity has high variance and is rich in high
    frequencies. This is achieved by alternating between the lowest and highest
    printable ASCII characters.

    Args:
        seq_len: The desired length of the string.

    Returns:
        A string of `seq_len` alternating characters.
    """
    min_char = ' '  # ASCII 32
    max_char = '~'  # ASCII 126
    result = []
    for i in range(seq_len):
        result.append(max_char if i % 2 == 0 else min_char)
    return "".join(result)

def design_signal_from_target_metrics(
    target_centroid: float, 
    target_complexity: str = 'low', 
    seq_len: int = 32
) -> str:
    """
    Designs a signal from target spectral metrics using inverse FFT.

    Args:
        target_centroid: The desired spectral centroid (0.0 to 0.5).
                         A value of 0.1 means energy is concentrated at low frequencies.
                         A value of 0.4 means energy is at high frequencies.
        target_complexity: 'low' or 'high'. 'low' concentrates energy around the
                           centroid, 'high' spreads it out.
        seq_len: The desired length of the string.

    Returns:
        A string designed to produce the target metrics.
    """
    if not (0.0 <= target_centroid <= 0.5):
        raise ValueError("Target centroid must be between 0.0 and 0.5.")

    # 1. Construct the target spectrum X[k]
    spectrum = torch.zeros(seq_len, dtype=torch.complex64)
    
    # Determine the main frequency bin for the centroid
    k_target = int(target_centroid * (seq_len // 2)) * 2

    if target_complexity == 'low':
        # For low complexity, create a single sharp peak at the target frequency.
        # We set the magnitude to 1.0. Phase is 0.
        spectrum[k_target] = 1.0 + 0j
    elif target_complexity == 'high':
        # For high complexity, create a wider band of energy around the centroid.
        spread = max(2, seq_len // 8)  # Create a band of frequencies
        for i in range(-spread, spread + 1):
            k_i = k_target + i
            if 0 <= k_i < seq_len:
                spectrum[k_i] = 1.0 + 0j
    else:
        raise ValueError("target_complexity must be 'low' or 'high'.")

    # 2. Enforce Hermitian symmetry to ensure the output signal is real-valued.
    # X[k] = conj(X[N-k]). Since our phases are 0, this means X[k] = X[N-k].
    for k in range(1, seq_len // 2):
        spectrum[seq_len - k] = spectrum[k].conj()

    # 3. Apply Inverse FFT
    # The result is a real-valued signal in the time domain.
    signal = torch.fft.ifft(spectrum).real

    # 4. Normalize the signal to the printable ASCII range [32, 126]
    min_val, max_val = torch.min(signal), torch.max(signal)
    if max_val == min_val:
        # Handle constant signal case to avoid division by zero
        normalized_signal = torch.full_like(signal, 77) # 'M' for Middle
    else:
        scale = (signal - min_val) / (max_val - min_val)
        normalized_signal = scale * (126 - 32) + 32
    
    # Round to nearest integer to get valid ASCII codes
    int_signal = torch.round(normalized_signal).to(torch.int)

    # 5. Convert integer codes back to a string
    return "".join([chr(c) for c in int_signal])


if __name__ == "__main__":
    """
    Validation block: Generates signals, processes them through the ΨQRH model,
    and prints the resulting analytical metrics to verify the design.
    """
    
    print("--- Initializing ΨQRH Model for Validation ---")
    # We assume the identity embedding, so model parameters don't affect the logic,
    # but we need the model's processing pipeline.
    SEQ_LEN_FOR_TEST = 32
    model = AdvancedTestModel(embed_dim=64, num_layers=2, seq_len=SEQ_LEN_FOR_TEST)
    model.eval()
    print("Model initialized.")

    def validate_string(description: str, text: str):
        """Helper function to process a string and print its analysis."""
        print(f"\n--- Validating: {description} ---")
        print(f"Generated String ({len(text)} chars): " + repr(text))
        
        prompt_info = {
            'category': 'Technical_Explanation',
            'domain': 'Signal Processing',
            'content': text
        }
        
        # Use the model's analysis pipeline
        output_analysis = model.generate_wiki_appropriate_response(text, prompt_info)
        
        # Extract and print key metrics from the report
        print("\nΨQRH Analysis Report:")
        for line in output_analysis.split('\n'):
            if "Spectral Complexity" in line or "Frequency Distribution" in line or "Dynamic Range" in line:
                print(line.strip())

    # --- Test Case 1: Minimum Complexity ---
    min_complexity_str = design_signal_for_min_complexity(SEQ_LEN_FOR_TEST)
    validate_string("Minimum Complexity Signal", min_complexity_str)

    # --- Test Case 2: Maximum Complexity ---
    max_complexity_str = design_signal_for_max_complexity(SEQ_LEN_FOR_TEST)
    validate_string("Maximum Complexity Signal", max_complexity_str)

    # --- Test Case 3: Low Complexity, Low Centroid ---
    low_low_str = design_signal_from_target_metrics(
        target_centroid=0.1, target_complexity='low', seq_len=SEQ_LEN_FOR_TEST
    )
    validate_string("Low Complexity, Low Centroid (0.1)", low_low_str)

    # --- Test Case 4: High Complexity, High Centroid ---
    high_high_str = design_signal_from_target_metrics(
        target_centroid=0.4, target_complexity='high', seq_len=SEQ_LEN_FOR_TEST
    )
    validate_string("High Complexity, High Centroid (0.4)", high_high_str)
