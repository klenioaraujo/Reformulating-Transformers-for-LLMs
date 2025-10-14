#!/usr/bin/env python3
"""
API Optimizer for DeepSeek Integration
======================================

This module provides optimized API calls for DeepSeek with automatic
context length management and chunking for large files.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path


class DeepSeekAPIOptimizer:
    """Optimizes API calls for DeepSeek with context length management"""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com"):
        """
        Initialize API optimizer

        Args:
            api_key: DeepSeek API key (optional, can use env var)
            base_url: Base URL for DeepSeek API
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url
        self.max_context_length = 131072
        self.safety_margin = 4096
        self.actual_max = self.max_context_length - self.safety_margin

    def count_tokens_approx(self, text: str) -> int:
        """Approximate token count (4 characters per token)"""
        return max(1, len(text) // 4)

    def optimize_messages(self, messages: List[Dict[str, str]], max_tokens: int = 120000) -> List[Dict[str, str]]:
        """
        Optimize messages to fit within token limits

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens allowed

        Returns:
            Optimized messages list
        """
        optimized_messages = []
        current_tokens = 0

        for message in messages:
            content = message.get('content', '')
            tokens = self.count_tokens_approx(content)

            # If adding this message would exceed limit, truncate it
            if current_tokens + tokens > max_tokens:
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only include if we have meaningful space
                    # Truncate content to fit remaining tokens
                    max_chars = remaining_tokens * 4
                    truncated_content = content[:max_chars] + "... [truncated]"
                    optimized_messages.append({
                        'role': message['role'],
                        'content': truncated_content
                    })
                break
            else:
                optimized_messages.append(message)
                current_tokens += tokens

        return optimized_messages

    def chunk_text(self, text: str, max_tokens_per_chunk: int = 25000) -> List[Dict[str, Any]]:
        """
        Split large text into manageable chunks

        Args:
            text: Text to chunk
            max_tokens_per_chunk: Maximum tokens per chunk

        Returns:
            List of chunks with metadata
        """
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0

        for i, line in enumerate(lines):
            line_tokens = self.count_tokens_approx(line)

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

    def call_api_safe(self, messages: List[Dict[str, str]], model: str = "deepseek-chat",
                     max_completion_tokens: int = 4000, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Make safe API call with automatic optimization

        Args:
            messages: List of message dictionaries
            model: Model to use
            max_completion_tokens: Maximum tokens for completion
            temperature: Sampling temperature

        Returns:
            API response
        """
        # Optimize messages to fit within context limit
        optimized_messages = self.optimize_messages(messages, self.actual_max - max_completion_tokens)

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": optimized_messages,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"API call failed: {str(e)}",
                "optimized_messages_count": len(optimized_messages),
                "total_tokens_approx": sum(self.count_tokens_approx(msg.get('content', '')) for msg in optimized_messages)
            }

    def process_large_file(self, file_path: Path, prompt: str,
                          max_chunk_tokens: int = 25000) -> List[Dict[str, Any]]:
        """
        Process large file by chunking and making multiple API calls

        Args:
            file_path: Path to file to process
            prompt: Prompt to use for each chunk
            max_chunk_tokens: Maximum tokens per chunk

        Returns:
            List of API responses for each chunk
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = self.chunk_text(content, max_chunk_tokens)
        results = []

        for chunk in chunks:
            chunk_prompt = f"""{prompt}

File chunk {chunk['chunk_number']} of {len(chunks)} (lines {chunk['start_line']}-{chunk['end_line']}):

```
{chunk['content']}
```"""

            messages = [
                {"role": "user", "content": chunk_prompt}
            ]

            result = self.call_api_safe(messages)
            results.append({
                "chunk_number": chunk['chunk_number'],
                "lines": f"{chunk['start_line']}-{chunk['end_line']}",
                "response": result
            })

        return results


# Example usage
def main():
    """Example usage of the API optimizer"""
    optimizer = DeepSeekAPIOptimizer()

    # Example 1: Safe API call with optimized messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "This is a test message."}
    ]

    result = optimizer.call_api_safe(messages)
    print("API Result:", json.dumps(result, indent=2))

    # Example 2: Process large file
    file_path = Path("psiqrh.py")
    if file_path.exists():
        prompt = "Please analyze this code chunk and provide insights:"
        results = optimizer.process_large_file(file_path, prompt)
        print(f"Processed {len(results)} chunks")


if __name__ == "__main__":
    main()