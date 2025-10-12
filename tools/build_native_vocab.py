import json
from collections import Counter
import argparse

def build_vocab(corpus_path, min_freq=1):
    """Builds a vocabulary from a text corpus."""
    word_counts = Counter()
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            word_counts.update(line.strip().split())

    # Filter words by minimum frequency
    words = [word for word, count in word_counts.items() if count >= min_freq]

    # Create vocab with special tokens
    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab

def main():
    parser = argparse.ArgumentParser(description="Native Vocabulary Builder for ΨQRH")
    parser.add_argument("--corpus", type=str, default="data/train.txt", help="Path to the training corpus.")
    parser.add_argument("--output", type=str, default="data/native_vocab.json", help="Path to save the native vocabulary.")
    parser.add_argument("--min-freq", type=int, default=1, help="Minimum frequency for a word to be included in the vocab.")
    args = parser.parse_args()

    print(f"Building vocabulary from {args.corpus}...")
    native_vocab = build_vocab(args.corpus, args.min_freq)

    output_data = {
        "vocab_size": len(native_vocab),
        "tokens": native_vocab
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Native vocabulary with {len(native_vocab)} tokens saved to {args.output}")

if __name__ == "__main__":
    main()