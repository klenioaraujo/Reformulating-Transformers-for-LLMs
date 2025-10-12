# ΨQRH Model Semantic Evaluation Report
============================================================

## Model Information
- **Model**: models/checkpoints/best_model.pt
- **Device**: cpu
- **Semantic Decoder Beam Width**: 5

## Summary Metrics
- **Test Cases Evaluated**: 5/5
.3f
.3f
.3f
.3f
.1f
.1f

## Detailed Test Case Results

### Test Case 1
**Input**: quantum mechanics
**Reference**: Quantum mechanics provides the foundation for understanding physical phenomena at atomic scales.
**Generated**: {'status': 'error', 'error': "Erro no pipeline físico ΨQRH: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.", 'task': 'text-generation', 'device': 'cpu', 'processing_time': 0.009788274765014648, 'mathematical_validation': False}
.3f
.3f
.1f

### Test Case 2
**Input**: uncertainty principle
**Reference**: The uncertainty principle states that it is impossible to simultaneously know both position and momentum with arbitrary precision.
**Generated**: {'status': 'error', 'error': "Erro no pipeline físico ΨQRH: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.", 'task': 'text-generation', 'device': 'cpu', 'processing_time': 0.00894021987915039, 'mathematical_validation': False}
.3f
.3f
.1f

### Test Case 3
**Input**: wave function
**Reference**: Wave functions describe the quantum state of systems, evolving according to the Schrödinger equation.
**Generated**: {'status': 'error', 'error': "Erro no pipeline físico ΨQRH: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.", 'task': 'text-generation', 'device': 'cpu', 'processing_time': 0.008471012115478516, 'mathematical_validation': False}
.3f
.3f
.1f

### Test Case 4
**Input**: superposition
**Reference**: Superposition allows particles to exist in multiple states simultaneously until measured.
**Generated**: {'status': 'error', 'error': "Erro no pipeline físico ΨQRH: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.", 'task': 'text-generation', 'device': 'cpu', 'processing_time': 0.008418798446655273, 'mathematical_validation': False}
.3f
.3f
.1f

### Test Case 5
**Input**: entanglement
**Reference**: Entanglement creates correlations between particles that persist regardless of distance.
**Generated**: {'status': 'error', 'error': "Erro no pipeline físico ΨQRH: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.", 'task': 'text-generation', 'device': 'cpu', 'processing_time': 0.009884119033813477, 'mathematical_validation': False}
.3f
.3f
.1f
