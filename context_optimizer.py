#!/usr/bin/env python3
"""
Context Length Optimizer for ΨQRH Transformer
==============================================

This script helps optimize context length usage to avoid API errors when using
models with limited context windows (like DeepSeek's 131072 token limit).

Features:
- Analyzes token usage across files
- Provides recommendations for context reduction
- Implements chunking strategies for large files
- Monitors and optimizes token consumption
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ContextOptimizer:
    """Optimizes context length usage for API calls"""

    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize optimizer with tokenizer

        Args:
            model_name: Model name for tokenizer
        """
        # Simple token counting approximation (4 chars per token)
        self.chars_per_token = 4

        self.max_context_length = 131072  # DeepSeek limit
        self.safety_margin = 4096  # Reserve tokens for system messages
        self.actual_max = self.max_context_length - self.safety_margin

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using approximation"""
        # Simple approximation: 4 characters per token
        return max(1, len(text) // self.chars_per_token)

    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze token usage in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            token_count = self.count_tokens(content)
            lines = content.count('\n') + 1

            return {
                'file_path': str(file_path),
                'token_count': token_count,
                'line_count': lines,
                'avg_tokens_per_line': token_count / lines if lines > 0 else 0,
                'context_ratio': token_count / self.actual_max,
                'status': 'OK' if token_count <= self.actual_max else 'EXCEEDS_LIMIT'
            }
        except Exception as e:
            return {
                'file_path': str(file_path),
                'error': str(e),
                'token_count': 0,
                'status': 'ERROR'
            }

    def analyze_directory(self, directory: Path, patterns: List[str] = None) -> Dict:
        """Analyze token usage across directory"""
        if patterns is None:
            patterns = ['*.py', '*.md', '*.txt', '*.json', '*.yaml', '*.yml']

        results = {
            'total_files': 0,
            'total_tokens': 0,
            'files_exceeding_limit': [],
            'large_files': [],
            'file_analysis': []
        }

        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    analysis = self.analyze_file(file_path)
                    results['file_analysis'].append(analysis)
                    results['total_files'] += 1
                    results['total_tokens'] += analysis.get('token_count', 0)

                    if analysis.get('status') == 'EXCEEDS_LIMIT':
                        results['files_exceeding_limit'].append(analysis)

                    if analysis.get('token_count', 0) > 10000:  # Files > 10k tokens
                        results['large_files'].append(analysis)

        return results

    def chunk_file(self, file_path: Path, max_tokens_per_chunk: int = 30000) -> List[Dict]:
        """
        Split large file into manageable chunks

        Args:
            file_path: Path to file to chunk
            max_tokens_per_chunk: Maximum tokens per chunk

        Returns:
            List of chunks with metadata
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Simple line-based chunking for code files
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)

            # If adding this line would exceed limit, start new chunk
            if current_tokens + line_tokens > max_tokens_per_chunk and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    'chunk_number': len(chunks) + 1,
                    'start_line': i - len(current_chunk) + 1,
                    'end_line': i,
                    'token_count': current_tokens,
                    'content': chunk_content
                })
                current_chunk = []
                current_tokens = 0

            current_chunk.append(line)
            current_tokens += line_tokens

        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'chunk_number': len(chunks) + 1,
                'start_line': len(lines) - len(current_chunk) + 1,
                'end_line': len(lines),
                'token_count': current_tokens,
                'content': chunk_content
            })

        return chunks

    def get_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []

        # Check total token usage
        if analysis['total_tokens'] > self.actual_max:
            recommendations.append(
                f"Total context ({analysis['total_tokens']:,} tokens) exceeds limit "
                f"({self.actual_max:,} tokens). Consider using chunking."
            )

        # Check individual large files
        for file_analysis in analysis.get('large_files', []):
            if file_analysis['token_count'] > 50000:
                recommendations.append(
                    f"Large file detected: {file_analysis['file_path']} "
                    f"({file_analysis['token_count']:,} tokens). Consider splitting."
                )

        # Check files exceeding limit
        for file_analysis in analysis.get('files_exceeding_limit', []):
            recommendations.append(
                f"File exceeds context limit: {file_analysis['file_path']} "
                f"({file_analysis['token_count']:,} tokens). Must be chunked."
            )

        # General recommendations
        if len(recommendations) == 0:
            recommendations.append(
                "Context usage is within safe limits. No optimization needed."
            )

        return recommendations

    def create_optimization_plan(self, directory: Path) -> Dict:
        """Create comprehensive optimization plan"""
        analysis = self.analyze_directory(directory)
        recommendations = self.get_optimization_recommendations(analysis)

        # Identify files that need chunking
        files_to_chunk = []
        for file_analysis in analysis['file_analysis']:
            if file_analysis.get('token_count', 0) > 30000:  # Files > 30k tokens
                files_to_chunk.append(file_analysis)

        return {
            'analysis_summary': {
                'total_files': analysis['total_files'],
                'total_tokens': analysis['total_tokens'],
                'context_limit': self.actual_max,
                'utilization_percentage': (analysis['total_tokens'] / self.actual_max) * 100,
                'files_exceeding_limit': len(analysis['files_exceeding_limit']),
                'large_files': len(analysis['large_files'])
            },
            'recommendations': recommendations,
            'files_to_chunk': files_to_chunk,
            'optimization_strategy': self._generate_strategy(analysis)
        }

    def _generate_strategy(self, analysis: Dict) -> str:
        """Generate optimization strategy based on analysis"""
        total_tokens = analysis['total_tokens']

        if total_tokens <= self.actual_max * 0.5:
            return "LOW_PRIORITY: Context usage is low. No immediate action needed."
        elif total_tokens <= self.actual_max * 0.8:
            return "MEDIUM_PRIORITY: Monitor context usage. Consider light optimization."
        elif total_tokens <= self.actual_max:
            return "HIGH_PRIORITY: Approaching limit. Implement chunking for large files."
        else:
            return "CRITICAL: Exceeds limit. Must implement aggressive chunking strategy."


def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Optimize context length usage')
    parser.add_argument('--directory', type=str, default='.',
                       help='Directory to analyze (default: current directory)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze context usage')
    parser.add_argument('--chunk-file', type=str,
                       help='Chunk a specific file')
    parser.add_argument('--max-chunk-tokens', type=int, default=30000,
                       help='Maximum tokens per chunk (default: 30000)')
    parser.add_argument('--output', type=str,
                       help='Output file for analysis results')

    args = parser.parse_args()

    optimizer = ContextOptimizer()

    if args.chunk_file:
        file_path = Path(args.chunk_file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return

        chunks = optimizer.chunk_file(file_path, args.max_chunk_tokens)
        print(f"\n📁 File: {file_path}")
        print(f"📊 Split into {len(chunks)} chunks")

        for chunk in chunks:
            print(f"\n  Chunk {chunk['chunk_number']}:")
            print(f"    Lines: {chunk['start_line']}-{chunk['end_line']}")
            print(f"    Tokens: {chunk['token_count']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(chunks, f, indent=2)
            print(f"\n💾 Chunks saved to: {args.output}")

    elif args.analyze:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"❌ Directory not found: {directory}")
            return

        print(f"\n🔍 Analyzing context usage in: {directory}")
        print(f"📊 Context limit: {optimizer.actual_max:,} tokens")
        print("=" * 60)

        plan = optimizer.create_optimization_plan(directory)

        # Print summary
        summary = plan['analysis_summary']
        print(f"\n📈 SUMMARY:")
        print(f"  Total files: {summary['total_files']}")
        print(f"  Total tokens: {summary['total_tokens']:,}")
        print(f"  Context limit: {summary['context_limit']:,}")
        print(f"  Utilization: {summary['utilization_percentage']:.1f}%")
        print(f"  Files exceeding limit: {summary['files_exceeding_limit']}")
        print(f"  Large files (>10k tokens): {summary['large_files']}")

        # Print recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in plan['recommendations']:
            print(f"  • {rec}")

        # Print strategy
        print(f"\n🎯 STRATEGY:")
        print(f"  {plan['optimization_strategy']}")

        # Print files that need chunking
        if plan['files_to_chunk']:
            print(f"\n📁 FILES RECOMMENDED FOR CHUNKING:")
            for file_analysis in plan['files_to_chunk']:
                print(f"  • {file_analysis['file_path']} ({file_analysis['token_count']:,} tokens)")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(plan, f, indent=2)
            print(f"\n💾 Analysis saved to: {args.output}")

    else:
        print("\nUsage examples:")
        print("  python context_optimizer.py --analyze")
        print("  python context_optimizer.py --chunk-file large_script.py")
        print("  python context_optimizer.py --analyze --output analysis.json")


if __name__ == "__main__":
    main()