# 🚀 Dynamic Quantum Vocabulary System for ΨQRH

## 📋 Overview

This system implements a **dynamic quantum vocabulary** with **word weights** and **model alignment** for the ΨQRH pipeline. It replaces the static character-to-word mapping with a comprehensive English vocabulary database containing **1,346+ words** with quantum references.

## 🎯 Key Features

### 🔬 **Dynamic Vocabulary Expansion**
- **1,346+ English words** organized by categories
- **Scientific, common, and advanced vocabulary**
- **Multiple words per character** with weighted selection
- **Quantum references** for each word

### ⚖️ **Word Weight System**
- **Model vocabulary alignment** - higher weights for model-compatible words
- **Word frequency weighting** - common words get preference
- **Scientific term bonuses** - quantum/physics terms get extra weight
- **Length optimization** - medium-length words preferred

### 🔗 **Quantum State Integration**
- **Quantum energy levels** influence word selection
- **Coherence factors** affect scientific vs common word preference
- **Entropy measurements** influence word complexity
- **Real-time quantum state analysis**

### 📊 **Comprehensive Vocabulary Coverage**

#### **Scientific Vocabulary (300+ terms)**
```
quantum, physics, electron, proton, neutron, photon, atom, molecule, energy,
wave, particle, field, force, gravity, electromagnetic, nuclear, relativity,
entropy, momentum, spin, charge, mass, velocity, acceleration, frequency,
wavelength, amplitude, phase, coherence, interference, diffraction, resonance,
oscillation, vibration, rotation, translation, symmetry, invariance,
conservation, evolution, dynamics, statistics, probability, measurement,
observation, collapse, superposition, entanglement, teleportation,
computation, algorithm, complexity, information, entropy, bit, qubit
```

#### **Common Vocabulary (500+ terms)**
```
the, and, for, are, but, not, you, all, can, had, her, was, one, our, out,
get, has, him, his, how, man, new, now, old, see, two, way, who, boy, did,
its, let, put, say, she, too, use, any, ask, big, buy, got, hot, its, may,
run, set, try, act, add, age, air, arm, art, bad, bag, bar, bed, bet, bid,
big, bit, box, bus, but, buy, can, car, cat, cup, cut, day, die, dig, dog,
dry, eat, egg, end, eye, fan, far, fat, fee, few, fit, fix, fly, fun, gas,
get, god, gun, guy, hat, hit, hot, how, ice, ill, ink, job, key, kid, lab,
law, lay, leg, let, lie, lip, log, lot, low, mad, man, map, mat, may, men,
mix, mom, mud, net, new, nod, nor, not, now, nut, oak, odd, off, oil, old,
one, our, out, owe, own, pad, pan, pay, pen, pet, pie, pig, pin, pit, pop,
pot, put, raw, red, rid, rip, rob, rod, rot, row, rub, run, sad, say, sea,
see, set, sew, she, shy, sin, sit, six, ski, sky, sly, son, sow, spy, sub,
sun, tap, tax, tea, ten, the, tie, tin, tip, toe, ton, too, top, toy, try,
tub, two, use, van, vet, via, war, was, wax, way, web, wet, who, why, win,
yes, yet, you, zip
```

#### **Advanced Vocabulary (500+ terms)**
```
algorithm, analysis, application, architecture, artificial, automation,
behavior, biological, calculation, capability, characteristic, classification,
cognitive, commercial, communication, computation, conceptual, configuration,
consciousness, constraint, construction, contemporary, contextual, coordination,
correlation, cryptographic, decentralized, deterministic, development, dimensional,
distribution, dynamical, ecological, economic, efficiency, electromagnetic,
emergence, engineering, environmental, equilibrium, evaluation, evolutionary,
experimental, exponential, expression, extraction, fabrication, fundamental,
generation, geometric, governance, hierarchical, implementation, information,
infrastructure, integration, intelligence, interaction, interpretation,
investigation, knowledge, linguistic, mathematical, measurement, mechanical,
methodology, minimization, modification, molecular, multidimensional,
navigation, negotiation, networking, neurological, observation, optimization,
organization, parameter, perception, performance, phenomenon, philosophical,
physical, prediction, probability, processing, production, programming,
progression, quantitative, realization, recognition, reconstruction,
regulation, representation, reproduction, requirement, resolution,
scientific, simulation, specification, stability, statistical, structural,
synthesis, systematic, technological, theoretical, transformation, validation,
verification, visualization
```

## 🛠️ Installation & Usage

### 1. **Basic Usage with Dynamic Vocabulary**

```python
from dynamic_quantum_vocabulary import DynamicQuantumVocabulary

# Initialize vocabulary
vocab = DynamicQuantumVocabulary()

# Create quantum prompt
text = "Hello Quantum World"
quantum_prompt = vocab.create_quantum_prompt(text, verbose=True)

print(f"Quantum Prompt: {quantum_prompt}")
```

### 2. **Integration with ΨQRH Pipeline**

```python
from update_psiqrh_with_dynamic_vocab import DynamicΨQRHPipeline

# Create enhanced pipeline
pipeline = DynamicΨQRHPipeline(
    task="text-generation",
    device="cpu"
)

# Process text with dynamic vocabulary
result = pipeline.process_text("Your input text here")

# Get vocabulary statistics
stats = pipeline.get_vocabulary_stats()
print(f"Vocabulary stats: {stats}")
```

### 3. **Direct Integration (Advanced)**

```python
from quantum_vocab_integration import QuantumVocabularyIntegrator

# Create integrator
integrator = QuantumVocabularyIntegrator()

# Integrate with existing pipeline
integrator.update_pipeline_method(your_pipeline_instance)
```

## 📊 Weight Calculation System

### **Weight Factors:**

1. **Base Weight**: `+1.0` (all words)
2. **Length Optimization**: `+0.5` (prefer 6-character words)
3. **Model Alignment**: `+2.0` (if word in model vocabulary)
4. **Frequency Bonus**: `+1.5` (common words like "the", "and")
5. **Scientific Bonus**: `+1.0` (quantum/physics terms)

### **Quantum Influence Factors:**

1. **Energy Level**: High energy prefers scientific terms
2. **Coherence**: High coherence prefers complex words
3. **Entropy**: High entropy prefers longer words

## 🔬 Example Outputs

### **Input: "The movie was"**
```
Original: The movie was
Quantum: TAXS hat ecologicaling state maies observation validationed ice economiced particle waring acting sayed
```

### **Input: "Hello World!"**
```
Original: Hello World!
Quantum: HATING ecological labing lawed oaking state WARING oak recognitioned labs decentralized quantum
```

### **Input: "Quantum Physics"**
```
Original: Quantum Physics
Quantum: QUANTUM uses added neted tea useed map quantum PANING had yet say illing calculations sading
```

## 📈 Performance Metrics

- **Vocabulary Size**: 1,346 words
- **Character Coverage**: 95 characters (ASCII + punctuation)
- **Words per Character**: 6.6 (average)
- **Quantum References**: 1,346 unique quantum concepts
- **Weight Range**: 0.0 - 5.0 (normalized)

## 🎯 Use Cases

1. **Enhanced Quantum Text Generation** - More diverse and meaningful quantum prompts
2. **Model Alignment** - Better compatibility with language model vocabularies
3. **Scientific Communication** - Rich quantum terminology for technical texts
4. **Educational Tools** - Quantum physics vocabulary for learning systems
5. **Research Applications** - Advanced quantum computing and physics terminology

## 🔧 Customization

### **Adding Custom Vocabulary**

```python
# Extend the vocabulary
custom_vocab = DynamicQuantumVocabulary()
custom_vocab.english_words.extend(["your", "custom", "words"])
```

### **Modifying Weight Factors**

```python
# Override weight calculation
class CustomVocabulary(DynamicQuantumVocabulary):
    def _calculate_word_weight(self, word, char):
        # Your custom weight logic
        return custom_weight
```

## 📁 File Structure

```
├── dynamic_quantum_vocabulary.py     # Main vocabulary system
├── quantum_vocab_integration.py      # ΨQRH pipeline integration
├── update_psiqrh_with_dynamic_vocab.py # Pipeline updater
├── dynamic_quantum_vocabulary.json   # Saved vocabulary data
└── README_DYNAMIC_VOCAB.md          # This documentation
```

## 🚀 Next Steps

1. **Integrate with ΨQRH** - Replace the current static vocabulary
2. **Expand Vocabulary** - Add more domain-specific terms
3. **Optimize Weights** - Fine-tune based on model performance
4. **Add Multilingual Support** - Extend to other languages
5. **Real-time Learning** - Dynamic vocabulary updates during training

## 📞 Support

For issues or feature requests, please refer to the main ΨQRH documentation or contact the development team.

---

**🔬 Developed for ΨQRH Quantum Language Processing System**