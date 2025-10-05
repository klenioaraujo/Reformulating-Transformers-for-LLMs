#!/usr/bin/env python3
"""
Extrator de Perfil de Atenção do GPT-2
======================================

Extrai estatísticas de atenção do modelo base GPT-2 para calibrar
os parâmetros da sonda óptica no ΨQRH.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path


def extract_attention_profile(
    gpt2_model: torch.nn.Module,
    calibration_corpus: List[str],
    tokenizer,
    max_length: int = 128
) -> Dict[str, float]:
    """
    Extrai estatísticas de atenção do GPT-2 para calibrar a sonda óptica.

    Args:
        gpt2_model: Modelo GPT-2
        calibration_corpus: Lista de textos para calibração
        tokenizer: Tokenizer do GPT-2
        max_length: Comprimento máximo dos textos

    Returns:
        Dict com estatísticas de atenção
    """
    attention_stats = {
        'mean_attention_entropy': [],
        'max_attention_weight': [],
        'attention_sparsity': [],
        'attention_concentration': []
    }

    print(f"🔍 Extraindo perfil de atenção de {len(calibration_corpus)} textos...")

    for i, text in enumerate(calibration_corpus):
        if i % 10 == 0:
            print(f"   Processando texto {i+1}/{len(calibration_corpus)}...")

        try:
            inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True)

            with torch.no_grad():
                outputs = gpt2_model(**inputs, output_attentions=True)

            if outputs.attentions is None:
                continue

            # Média das camadas e cabeças [n_layers, n_heads, seq_len, seq_len]
            attentions = torch.stack(outputs.attentions)  # [n_layers, n_heads, seq_len, seq_len]

            # Média por camada e cabeça
            layer_head_avg = attentions.mean(dim=(0, 1))  # [seq_len, seq_len]

            # Entropia da distribuição de atenção
            # Para cada posição de destino, calcular entropia sobre as fontes
            entropy = -torch.sum(layer_head_avg * torch.log(layer_head_avg + 1e-10), dim=-1).mean().item()
            attention_stats['mean_attention_entropy'].append(entropy)

            # Peso máximo de atenção
            max_weight = layer_head_avg.max().item()
            attention_stats['max_attention_weight'].append(max_weight)

            # Esparsidade (quantos pesos > 0.1)
            sparsity = (layer_head_avg > 0.1).float().mean().item()
            attention_stats['attention_sparsity'].append(sparsity)

            # Concentração (razão entre top-1 e soma)
            top1_ratio = (layer_head_avg.max(dim=-1)[0] / layer_head_avg.sum(dim=-1)).mean().item()
            attention_stats['attention_concentration'].append(top1_ratio)

        except Exception as e:
            print(f"   ⚠️  Erro no texto {i+1}: {e}")
            continue

    # Calcular estatísticas agregadas
    if not attention_stats['mean_attention_entropy']:
        print("⚠️  Nenhuma estatística de atenção extraída")
        return {
            'entropy_mean': 2.0,  # Valor padrão
            'entropy_std': 0.5,
            'max_weight_mean': 0.8,
            'sparsity_mean': 0.3,
            'concentration_mean': 0.6
        }

    profile = {
        'entropy_mean': float(np.mean(attention_stats['mean_attention_entropy'])),
        'entropy_std': float(np.std(attention_stats['mean_attention_entropy'])),
        'max_weight_mean': float(np.mean(attention_stats['max_attention_weight'])),
        'sparsity_mean': float(np.mean(attention_stats['attention_sparsity'])),
        'concentration_mean': float(np.mean(attention_stats['attention_concentration']))
    }

    print(f"✅ Perfil de atenção extraído:")
    print(f"   • Entropia média: {profile['entropy_mean']:.4f}")
    print(f"   • Esparsidade: {profile['sparsity_mean']:.4f}")
    print(f"   • Concentração: {profile['concentration_mean']:.4f}")
    print(f"   • Peso máximo: {profile['max_weight_mean']:.4f}")

    return profile


def load_calibration_corpus(corpus_path: Optional[Path] = None) -> List[str]:
    """
    Carrega corpus de calibração (100 frases simples).

    Args:
        corpus_path: Caminho opcional para arquivo de corpus

    Returns:
        Lista de textos para calibração
    """
    if corpus_path and corpus_path.exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    # Corpus padrão (100 frases simples)
    default_corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning algorithms can recognize patterns.",
        "Natural language processing helps computers understand text.",
        "Deep learning models require large amounts of data.",
        "Transformers have revolutionized natural language processing.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Quantum computing promises exponential speedups for certain problems.",
        "Neural networks are inspired by the human brain.",
        "Reinforcement learning agents learn through trial and error.",
        "Computer vision systems can identify objects in images.",
        "Speech recognition technology has improved dramatically.",
        "Robotics combines mechanics, electronics, and computer science.",
        "Autonomous vehicles use sensors and AI to navigate roads.",
        "Virtual reality creates immersive digital environments.",
        "Augmented reality overlays digital information on the real world.",
        "Blockchain technology enables secure decentralized transactions.",
        "Cryptocurrencies use cryptography for secure financial transactions.",
        "Cloud computing provides scalable computing resources.",
        "Internet of Things connects physical devices to the internet.",
        "Big data analytics extracts insights from large datasets.",
        "Data science combines statistics, programming, and domain knowledge.",
        "Cybersecurity protects computer systems from attacks.",
        "Software engineering involves designing and building applications.",
        "Web development creates websites and web applications.",
        "Mobile apps run on smartphones and tablets.",
        "Operating systems manage computer hardware and software.",
        "Databases store and organize information efficiently.",
        "Networks connect computers and devices together.",
        "Algorithms are step-by-step procedures for solving problems.",
        "Programming languages allow humans to communicate with computers.",
        "Compilers translate source code into machine code.",
        "Debugging involves finding and fixing errors in code.",
        "Testing ensures software works correctly.",
        "Version control tracks changes to source code.",
        "Agile development emphasizes flexibility and collaboration.",
        "User interface design focuses on user experience.",
        "Accessibility ensures technology is usable by everyone.",
        "Performance optimization improves software speed and efficiency.",
        "Security vulnerabilities can be exploited by attackers.",
        "Encryption protects sensitive information from unauthorized access.",
        "Authentication verifies the identity of users.",
        "Authorization determines what users can access.",
        "Backup systems protect against data loss.",
        "Disaster recovery plans ensure business continuity.",
        "Project management coordinates teams and resources.",
        "Quality assurance maintains high standards for products.",
        "Documentation helps users understand how to use software.",
        "Technical support assists users with problems.",
        "Customer feedback provides valuable insights for improvement.",
        "Market research identifies customer needs and preferences.",
        "Business strategy defines long-term goals and plans.",
        "Financial planning manages budgets and resources.",
        "Risk management identifies and mitigates potential problems.",
        "Change management handles organizational transitions.",
        "Leadership inspires and guides teams toward success.",
        "Communication skills are essential for effective collaboration.",
        "Problem-solving abilities help overcome challenges.",
        "Critical thinking enables logical analysis of information.",
        "Creativity generates innovative ideas and solutions.",
        "Adaptability allows adjustment to changing circumstances.",
        "Resilience helps recover from setbacks and failures.",
        "Empathy enables understanding of others' perspectives.",
        "Integrity involves honesty and ethical behavior.",
        "Accountability means taking responsibility for actions.",
        "Transparency builds trust through open communication.",
        "Collaboration brings together diverse skills and perspectives.",
        "Innovation drives progress and improvement.",
        "Sustainability considers long-term environmental impact.",
        "Diversity includes people with different backgrounds and experiences.",
        "Inclusion ensures everyone feels valued and respected.",
        "Equality provides fair treatment and opportunities for all.",
        "Justice involves fair and impartial decision-making.",
        "Freedom allows individuals to make their own choices.",
        "Democracy gives citizens a voice in government.",
        "Education provides knowledge and skills for personal growth.",
        "Healthcare maintains and improves physical and mental well-being.",
        "Science seeks to understand the natural world through observation.",
        "Technology applies scientific knowledge to practical problems.",
        "Engineering designs and builds structures, machines, and systems.",
        "Mathematics studies numbers, shapes, and patterns.",
        "Physics explores matter, energy, and their interactions.",
        "Chemistry examines the composition and properties of substances.",
        "Biology studies living organisms and their processes.",
        "Geology investigates the Earth's structure and history.",
        "Astronomy observes celestial objects and phenomena.",
        "Meteorology studies weather and atmospheric conditions.",
        "Oceanography explores the world's oceans and seas.",
        "Ecology examines relationships between organisms and their environment.",
        "Psychology studies human behavior and mental processes.",
        "Sociology analyzes human societies and social behavior.",
        "Anthropology investigates human cultures and societies.",
        "History records and interprets past events.",
        "Geography studies Earth's landscapes and environments.",
        "Economics analyzes production, distribution, and consumption.",
        "Political science examines government and political systems.",
        "Philosophy explores fundamental questions about existence.",
        "Literature expresses ideas and emotions through written works.",
        "Art creates visual, auditory, or performing works.",
        "Music organizes sound in time to create expressive compositions.",
        "Dance uses movement to express ideas and emotions.",
        "Theater presents stories through live performance.",
        "Film combines visual and auditory elements to tell stories.",
        "Photography captures images using light and cameras.",
        "Architecture designs buildings and other physical structures.",
        "Fashion creates clothing and accessories.",
        "Cuisine prepares and presents food in creative ways.",
        "Sports involve physical activity and competition.",
        "Games provide entertainment and challenge through structured play."
    ]

    print(f"✅ Corpus de calibração carregado: {len(default_corpus)} frases")
    return default_corpus


def save_attention_profile(profile: Dict, output_path: Path):
    """
    Salva perfil de atenção em arquivo JSON.

    Args:
        profile: Dicionário com estatísticas de atenção
        output_path: Caminho do arquivo de saída
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(profile, f, indent=2)

    print(f"✅ Perfil de atenção salvo: {output_path}")


def load_attention_profile(profile_path: Path) -> Dict:
    """
    Carrega perfil de atenção de arquivo JSON.

    Args:
        profile_path: Caminho do arquivo de perfil

    Returns:
        Dicionário com estatísticas de atenção
    """
    if not profile_path.exists():
        raise FileNotFoundError(f"Perfil de atenção não encontrado: {profile_path}")

    with open(profile_path, 'r') as f:
        return json.load(f)