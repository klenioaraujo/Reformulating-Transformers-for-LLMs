#!/usr/bin/env python3
"""
Context Length Optimizer for ΨQRH Transformer - Enhanced Version
===============================================================

Optimized for large code files like psiqrh.py with intelligent chunking strategies.
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
import ast


class EnhancedContextOptimizer:
    """Enhanced optimizer with intelligent code analysis"""

    def __init__(self, model_name: str = "gpt-4"):
        self.chars_per_token = 4
        self.max_context_length = 131072
        self.max_file_tokens = 25000  # From the error message
        self.max_file_size_bytes = 256 * 1024
        self.safety_margin = 4096
        self.actual_max = self.max_context_length - self.safety_margin

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // self.chars_per_token)

    def analyze_code_structure(self, file_path: Path) -> Dict:
        """Analyze Python code structure to find logical breakpoints"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to understand code structure
            try:
                tree = ast.parse(content)
                structure = {
                    'classes': [],
                    'functions': [],
                    'imports': [],
                    'global_vars': []
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        structure['classes'].append({
                            'name': node.name,
                            'line': node.lineno,
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        })
                    elif isinstance(node, ast.FunctionDef):
                        structure['functions'].append({
                            'name': node.name,
                            'line': node.lineno,
                            'is_async': isinstance(node, ast.AsyncFunctionDef)
                        })
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        structure['imports'].append(ast.get_source_segment(content, node))
                
                return structure
            except SyntaxError:
                # Fallback for files with syntax errors
                return self._fallback_structure_analysis(content)
                
        except Exception as e:
            return {'error': str(e)}

    def _fallback_structure_analysis(self, content: str) -> Dict:
        """Fallback analysis using regex for files with syntax issues"""
        lines = content.split('\n')
        structure = {
            'classes': [],
            'functions': [],
            'imports': [],
            'global_vars': []
        }
        
        class_pattern = re.compile(r'^class\s+(\w+)')
        function_pattern = re.compile(r'^def\s+(\w+)')
        import_pattern = re.compile(r'^(import|from)\s+')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if class_match := class_pattern.match(line):
                structure['classes'].append({'name': class_match.group(1), 'line': i})
            elif function_match := function_pattern.match(line):
                structure['functions'].append({'name': function_match.group(1), 'line': i})
            elif import_pattern.match(line):
                structure['imports'].append(line)
                
        return structure

    def get_optimal_chunk_breaks(self, file_path: Path, target_chunk_size: int = 20000) -> List[int]:
        """Find optimal line numbers to break the file into chunks"""
        structure = self.analyze_code_structure(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.split('\n')
        
        # Collect potential break points
        break_points = set()
        
        # Add class definitions
        for cls in structure.get('classes', []):
            break_points.add(cls['line'] - 1)  # Convert to 0-indexed
        
        # Add function definitions
        for func in structure.get('functions', []):
            break_points.add(func['line'] - 1)
        
        # Add blank lines that separate logical sections
        for i, line in enumerate(lines):
            if i > 0 and i < len(lines) - 1:
                if (line.strip() == '' and 
                    lines[i-1].strip() != '' and 
                    lines[i+1].strip() != ''):
                    break_points.add(i)
        
        # Sort break points
        break_points = sorted(break_points)
        
        # Select optimal breaks based on target chunk size
        optimal_breaks = []
        current_pos = 0
        
        for bp in break_points:
            if bp <= current_pos:
                continue
                
            chunk_content = '\n'.join(lines[current_pos:bp])
            token_count = self.count_tokens(chunk_content)
            
            if token_count >= target_chunk_size * 0.7:  # Break when we reach 70% of target
                optimal_breaks.append(bp)
                current_pos = bp
        
        return optimal_breaks

    def chunk_python_file_intelligently(self, file_path: Path, max_tokens_per_chunk: int = 20000) -> List[Dict]:
        """Intelligently chunk Python files based on code structure"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        optimal_breaks = self.get_optimal_chunk_breaks(file_path, max_tokens_per_chunk)
        
        chunks = []
        start_line = 0
        
        for break_line in optimal_breaks:
            if break_line <= start_line:
                continue
                
            chunk_content = '\n'.join(lines[start_line:break_line])
            token_count = self.count_tokens(chunk_content)
            
            chunks.append({
                'chunk_number': len(chunks) + 1,
                'start_line': start_line + 1,
                'end_line': break_line,
                'token_count': token_count,
                'content': chunk_content,
                'chunking_method': 'STRUCTURE_BASED',
                'description': f"Lines {start_line + 1}-{break_line}"
            })
            
            start_line = break_line
        
        # Add final chunk
        if start_line < len(lines):
            chunk_content = '\n'.join(lines[start_line:])
            token_count = self.count_tokens(chunk_content)
            
            chunks.append({
                'chunk_number': len(chunks) + 1,
                'start_line': start_line + 1,
                'end_line': len(lines),
                'token_count': token_count,
                'content': chunk_content,
                'chunking_method': 'STRUCTURE_BASED',
                'description': f"Lines {start_line + 1}-{len(lines)}"
            })
        
        return chunks

    def generate_analysis_plan(self, file_path: Path) -> Dict:
        """Generate a strategic analysis plan for large Python files"""
        structure = self.analyze_code_structure(file_path)
        
        plan = {
            'file_info': {
                'path': str(file_path),
                'classes_count': len(structure.get('classes', [])),
                'functions_count': len(structure.get('functions', [])),
                'imports_count': len(structure.get('imports', []))
            },
            'recommended_approach': [],
            'chunking_strategy': 'STRUCTURE_BASED',
            'key_sections': []
        }
        
        # Add class analysis recommendations
        for cls in structure.get('classes', []):
            plan['key_sections'].append({
                'type': 'class',
                'name': cls['name'],
                'line': cls['line'],
                'analysis_focus': [
                    f"Analyze {cls['name']} class structure",
                    f"Examine methods: {', '.join(cls.get('methods', []))}",
                    "Review class dependencies and inheritance"
                ]
            })
        
        # Add function analysis recommendations
        for func in structure.get('functions', []):
            plan['key_sections'].append({
                'type': 'function',
                'name': func['name'],
                'line': func['line'],
                'analysis_focus': [
                    f"Analyze {func['name']} function purpose",
                    "Review parameters and return values",
                    "Check for dependencies and side effects"
                ]
            })
        
        # Generate strategic recommendations
        plan['recommended_approach'].extend([
            "1. Start with class and function definitions to understand architecture",
            "2. Analyze core transformation logic in small chunks",
            "3. Focus on main entry points and public methods",
            "4. Review import dependencies and module structure",
            "5. Examine configuration and initialization sections"
        ])
        
        return plan

    def create_focused_chunks(self, file_path: Path, focus_areas: List[str] = None) -> List[Dict]:
        """Create chunks focused on specific areas of interest"""
        if focus_areas is None:
            focus_areas = ['imports', 'classes', 'main_functions', 'configuration']
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        chunks = []
        
        structure = self.analyze_code_structure(file_path)
        
        # Chunk 1: Imports and global configuration
        if 'imports' in focus_areas:
            import_lines = []
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_lines.append(i)
            
            if import_lines:
                end_line = max(import_lines) + 1
                chunk_content = '\n'.join(lines[:end_line])
                chunks.append({
                    'chunk_number': len(chunks) + 1,
                    'start_line': 1,
                    'end_line': end_line,
                    'token_count': self.count_tokens(chunk_content),
                    'content': chunk_content,
                    'focus': 'imports_and_config',
                    'description': 'Module imports and global configuration'
                })
        
        # Chunk by classes
        if 'classes' in focus_areas:
            for cls in structure.get('classes', []):
                class_start = cls['line'] - 1
                # Find class end (next class or end of file)
                class_end = len(lines)
                for other_cls in structure.get('classes', []):
                    if other_cls['line'] > cls['line']:
                        class_end = other_cls['line'] - 2
                        break
                
                chunk_content = '\n'.join(lines[class_start:class_end])
                chunks.append({
                    'chunk_number': len(chunks) + 1,
                    'start_line': class_start + 1,
                    'end_line': class_end,
                    'token_count': self.count_tokens(chunk_content),
                    'content': chunk_content,
                    'focus': f"class_{cls['name']}",
                    'description': f"Class {cls['name']} implementation"
                })
        
        # Chunk by main functions
        if 'main_functions' in focus_areas:
            main_functions = [f for f in structure.get('functions', []) 
                            if not f['name'].startswith('_') or f['name'] == '__init__']
            
            for func in main_functions:
                func_start = func['line'] - 1
                # Find function end (next function/class or end of file)
                func_end = len(lines)
                for other_item in structure.get('classes', []) + structure.get('functions', []):
                    if other_item['line'] > func['line']:
                        func_end = other_item['line'] - 2
                        break
                
                chunk_content = '\n'.join(lines[func_start:func_end])
                chunks.append({
                    'chunk_number': len(chunks) + 1,
                    'start_line': func_start + 1,
                    'end_line': func_end,
                    'token_count': self.count_tokens(chunk_content),
                    'content': chunk_content,
                    'focus': f"function_{func['name']}",
                    'description': f"Function {func['name']} implementation"
                })
        
        return chunks


class PipelineOptimizer:
    """Optimizes the analysis pipeline to avoid freezing"""

    def __init__(self):
        self.context_optimizer = EnhancedContextOptimizer()
        self.dynamic_matrix = None
        self.performance_metrics = {}

    def optimize_psiqrh_analysis(self, file_path: Path) -> Dict:
        """Create optimized analysis strategy for psiqrh.py"""

        print("🔍 Analyzing psiqrh.py structure...")
        structure = self.context_optimizer.analyze_code_structure(file_path)

        print("📋 Generating intelligent chunks...")
        chunks = self.context_optimizer.chunk_python_file_intelligently(file_path)

        print("🎯 Creating analysis plan...")
        analysis_plan = self.context_optimizer.generate_analysis_plan(file_path)

        # Create focused analysis strategy
        focused_chunks = self.context_optimizer.create_focused_chunks(file_path, [
            'imports', 'classes', 'main_functions'
        ])

        return {
            'file_structure': structure,
            'chunks': chunks,
            'focused_chunks': focused_chunks,
            'analysis_plan': analysis_plan,
            'recommended_pipeline': self._generate_recommended_pipeline(structure, chunks)
        }

    def initialize_dynamic_quantum_matrix(self):
        """Initialize the Dynamic Quantum Character Matrix for semantic token extraction"""
        try:
            from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix
            self.dynamic_matrix = DynamicQuantumCharacterMatrix(vocab_size=50257, hidden_size=256)
            print("✅ Dynamic Quantum Character Matrix initialized")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Dynamic Quantum Matrix: {e}")
            return False

    def adapt_matrix_to_semantic_model(self, model_name: str) -> Dict:
        """Adapt the dynamic matrix to a specific semantic model and measure performance"""

        if self.dynamic_matrix is None:
            if not self.initialize_dynamic_quantum_matrix():
                return {'success': False, 'error': 'Matrix initialization failed'}

        import time
        start_time = time.time()

        try:
            print(f"🔧 Adapting Dynamic Quantum Matrix to: {model_name}")

            # Measure adaptation time
            adapt_start = time.time()
            success = self.dynamic_matrix.adapt_to_model(model_name)
            adapt_time = time.time() - adapt_start

            if success:
                # Test token extraction performance with shorter text
                test_text = "quantum"
                extract_start = time.time()
                encoded = self.dynamic_matrix.encode_text(test_text)
                extract_time = time.time() - extract_start

                # Measure memory usage (approximate)
                matrix_size = self.dynamic_matrix.quantum_matrix.numel() * self.dynamic_matrix.quantum_matrix.element_size()
                vocab_size = self.dynamic_matrix.vocab_size
                hidden_size = self.dynamic_matrix.hidden_size

                # Store performance metrics
                self.performance_metrics[model_name] = {
                    'adaptation_time': adapt_time,
                    'extraction_time': extract_time,
                    'matrix_size_mb': matrix_size / (1024 * 1024),
                    'vocab_size': vocab_size,
                    'hidden_size': hidden_size,
                    'total_time': time.time() - start_time
                }

                result = {
                    'success': True,
                    'model_name': model_name,
                    'adaptation_time': f"{adapt_time:.3f}s",
                    'extraction_time': f"{extract_time:.3f}s",
                    'matrix_size': f"{matrix_size / (1024 * 1024):.2f} MB",
                    'vocab_size': vocab_size,
                    'hidden_size': hidden_size,
                    'encoded_shape': encoded.shape,
                    'finite_values': torch.isfinite(encoded).all().item()
                }

                print("✅ Matrix adaptation completed:")
                print(f"   ⏱️  Adaptation time: {adapt_time:.3f}s")
                print(f"   ⚡ Extraction time: {extract_time:.3f}s")
                print(f"   💾 Matrix size: {matrix_size / (1024 * 1024):.2f} MB")
                print(f"   📊 Encoded shape: {encoded.shape}")

                return result
            else:
                return {'success': False, 'error': f'Adaptation failed for {model_name}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def extract_tokens_with_dynamic_matrix(self, semantic_models_dir: Path) -> Dict:
        """Extract tokens from all semantic models using Dynamic Quantum Matrix"""

        if self.dynamic_matrix is None:
            if not self.initialize_dynamic_quantum_matrix():
                return {'success': False, 'error': 'Matrix initialization failed'}

        import time
        import torch
        results = {
            'models_processed': 0,
            'total_tokens_extracted': 0,
            'performance_summary': {},
            'extraction_details': {}
        }

        semantic_models = list(semantic_models_dir.glob("*.pt"))
        if not semantic_models:
            return {'success': False, 'error': 'No semantic models found'}

        print(f"🔍 Processing {len(semantic_models)} semantic models with Dynamic Quantum Matrix")

        for model_path in semantic_models:
            model_name = model_path.stem.replace('psiqrh_semantic_', '')
            print(f"\n🎯 Processing model: {model_name}")

            # Adapt matrix to this model
            adapt_result = self.adapt_matrix_to_semantic_model(model_name)
            if not adapt_result['success']:
                print(f"   ❌ Adaptation failed: {adapt_result.get('error', 'Unknown error')}")
                continue

            try:
                # Load semantic model
                load_start = time.time()
                model_data = torch.load(model_path, map_location='cpu')
                load_time = time.time() - load_start

                # Extract embeddings
                embeddings = None
                if 'model_state_dict' in model_data:
                    state_dict = model_data['model_state_dict']
                    for key, value in state_dict.items():
                        if 'embed' in key.lower() and isinstance(value, torch.Tensor):
                            embeddings = value
                            break

                if embeddings is not None:
                    # Extract semantic tokens using Dynamic Quantum Matrix
                    print(f"   📊 Processing {embeddings.shape[0]} semantic tokens")

                    # Method 1: Dynamic matrix encoding (simplified for performance)
                    extract_start = time.time()
                    test_text = "q"  # Single character for faster processing
                    encoded_tokens = self.dynamic_matrix.encode_text(test_text)
                    extract_time = time.time() - extract_start

                    # Method 2: Hilbert space analysis
                    magnitudes = torch.norm(embeddings, dim=-1)
                    top_indices = torch.topk(magnitudes, k=min(10, len(magnitudes)))[1]

                    # Method 3: Quantum coherence analysis
                    if embeddings.shape[-1] >= 2:
                        complex_magnitudes = torch.sqrt(embeddings[..., 0]**2 + embeddings[..., 1]**2)
                        coherence_indices = torch.topk(complex_magnitudes, k=min(10, len(complex_magnitudes)))[1]

                        # Store results
                        model_results = {
                            'model_name': model_name,
                            'load_time': f"{load_time:.3f}s",
                            'extraction_time': f"{extract_time:.3f}s",
                            'vocab_size': embeddings.shape[0],
                            'embedding_dim': embeddings.shape[-1],
                            'top_magnitude_indices': top_indices.tolist(),
                            'top_coherence_indices': coherence_indices.tolist(),
                            'encoded_shape': encoded_tokens.shape,
                            'finite_encoded': torch.isfinite(encoded_tokens).all().item(),
                            'adaptation_metrics': adapt_result
                        }

                        results['extraction_details'][model_name] = model_results
                        results['models_processed'] += 1
                        results['total_tokens_extracted'] += len(top_indices)

                        print(f"   ✅ Extracted {len(top_indices)} semantic tokens")
                        print(f"   ⏱️  Load time: {load_time:.3f}s, Extract time: {extract_time:.3f}s")
                        print(f"   📐 Encoded shape: {encoded_tokens.shape}")

                    else:
                        print("   ⚠️  Embeddings dimension too low for coherence analysis")

                else:
                    print("   ⚠️  No embeddings found in model")

            except Exception as e:
                print(f"   ❌ Error processing model: {e}")
                import traceback
                traceback.print_exc()

        # Generate performance summary
        if self.performance_metrics:
            results['performance_summary'] = {
                'total_models': len(self.performance_metrics),
                'avg_adaptation_time': f"{sum(m['adaptation_time'] for m in self.performance_metrics.values()) / len(self.performance_metrics):.3f}s",
                'avg_extraction_time': f"{sum(m['extraction_time'] for m in self.performance_metrics.values()) / len(self.performance_metrics):.3f}s",
                'total_matrix_size': f"{sum(m['matrix_size_mb'] for m in self.performance_metrics.values()):.2f} MB",
                'avg_matrix_size': f"{sum(m['matrix_size_mb'] for m in self.performance_metrics.values()) / len(self.performance_metrics):.2f} MB"
            }

        print(f"\n✅ Dynamic Quantum Matrix token extraction completed!")
        print(f"   📊 Models processed: {results['models_processed']}")
        print(f"   🎯 Total tokens extracted: {results['total_tokens_extracted']}")

        return results
    
    def _generate_recommended_pipeline(self, structure: Dict, chunks: List[Dict]) -> List[str]:
        """Generate recommended pipeline steps"""
        pipeline = []
        
        pipeline.append("## Recommended Analysis Pipeline for psiqrh.py")
        pipeline.append("")
        pipeline.append("### Phase 1: Architecture Overview")
        pipeline.append("1. Analyze imports and module structure")
        pipeline.append("2. Review class hierarchy and main components")
        pipeline.append("3. Identify entry points and public API")
        pipeline.append("")
        
        pipeline.append("### Phase 2: Core Components")
        for cls in structure.get('classes', []):
            pipeline.append(f"4. Analyze {cls['name']} class:")
            for method in cls.get('methods', [])[:3]:  # Top 3 methods
                pipeline.append(f"   - {method} method")
        pipeline.append("")
        
        pipeline.append("### Phase 3: Implementation Details")
        pipeline.append("5. Examine key algorithms and transformations")
        pipeline.append("6. Review configuration and initialization")
        pipeline.append("7. Analyze data flow and processing logic")
        pipeline.append("")
        
        pipeline.append("### Chunking Strategy")
        pipeline.append(f"- Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:5], 1):  # First 5 chunks
            pipeline.append(f"- Chunk {i}: {chunk['description']} ({chunk['token_count']} tokens)")
        
        if len(chunks) > 5:
            pipeline.append(f"- ... and {len(chunks) - 5} more chunks")
        
        return pipeline


def main():
    """Main function with enhanced psiqrh.py optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Context Optimizer for Large Python Files')
    parser.add_argument('--file', type=str, required=True, help='Python file to analyze (e.g., psiqrh.py)')
    parser.add_argument('--strategy', choices=['smart', 'focused', 'full'], default='smart',
                       help='Chunking strategy: smart (structure-based), focused (key areas), full (complete)')
    parser.add_argument('--output', type=str, help='Output directory for chunks and analysis')
    parser.add_argument('--max-tokens', type=int, default=20000, help='Max tokens per chunk')
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    optimizer = PipelineOptimizer()
    
    print(f"🚀 Optimizing analysis for: {file_path}")
    print("=" * 60)
    
    result = optimizer.optimize_psiqrh_analysis(file_path)
    
    # Display structure overview
    structure = result['file_structure']
    print(f"\n📊 FILE STRUCTURE OVERVIEW:")
    print(f"   Classes: {len(structure.get('classes', []))}")
    print(f"   Functions: {len(structure.get('functions', []))}")
    print(f"   Imports: {len(structure.get('imports', []))}")
    
    # Display classes
    print(f"\n🏗️  CLASSES:")
    for cls in structure.get('classes', []):
        methods = cls.get('methods', [])
        print(f"   {cls['name']} (line {cls['line']}) - {len(methods)} methods")
        if methods:
            print(f"     Methods: {', '.join(methods[:3])}{'...' if len(methods) > 3 else ''}")
    
    # Display chunking plan
    print(f"\n📦 CHUNKING PLAN:")
    for chunk in result['chunks']:
        print(f"   Chunk {chunk['chunk_number']}: {chunk['description']}")
        print(f"     Tokens: {chunk['token_count']}, Lines: {chunk['start_line']}-{chunk['end_line']}")
    
    # Display recommended pipeline
    print(f"\n🎯 RECOMMENDED PIPELINE:")
    for step in result['recommended_pipeline']:
        print(f"   {step}")
    
    # Save results if output specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks as separate files
        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        for chunk in result['chunks']:
            chunk_file = chunks_dir / f"chunk_{chunk['chunk_number']}.py"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(f"# Chunk {chunk['chunk_number']}: {chunk['description']}\n")
                f.write(f"# Tokens: {chunk['token_count']}, Lines: {chunk['start_line']}-{chunk['end_line']}\n\n")
                f.write(chunk['content'])
        
        # Save analysis plan
        plan_file = output_dir / "analysis_plan.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            json.dump(result['analysis_plan'], f, indent=2, ensure_ascii=False)
        
        # Save pipeline
        pipeline_file = output_dir / "recommended_pipeline.md"
        with open(pipeline_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result['recommended_pipeline']))
        
        print(f"\n💾 Results saved to: {output_dir}")
        print(f"   - Chunks: {chunks_dir}/")
        print(f"   - Analysis plan: {plan_file}")
        print(f"   - Pipeline: {pipeline_file}")


if __name__ == "__main__":
    main()