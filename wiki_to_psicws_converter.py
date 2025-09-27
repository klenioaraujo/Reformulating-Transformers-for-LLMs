#!/usr/bin/env python3
"""
ΨQRH-Native Wikipedia to Ψcws Converter
=======================================

Conversor nativo ΨQRH que obtém dados Wikipedia diretamente
e os converte para formato .Ψcws usando apenas processamento
quaterniônico e consciência fractal - SEM dependência do Transformers.

O ΨQRH é uma arquitetura independente que não precisa de Transformers!
"""

import sys
import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

class ΨQRHWikipediaProcessor:
    """
    Processador nativo ΨQRH para dados Wikipedia.

    Usa apenas a API da Wikipedia para obter conteúdo
    e processa via pipeline quaterniônico nativo.
    """

    def __init__(self):
        from src.conscience.conscious_wave_modulator import ConsciousWaveModulator

        # Configuração otimizada para conhecimento Wikipedia
        self.wiki_config = {
            'cache_dir': 'data/Ψcws_cache/wikipedia',
            'embedding_dim': 512,  # Maior para capturar complexidade enciclopédica
            'sequence_length': 256,  # Sequências longas para artigos
            'base_amplitude': 2.0,  # Amplitude alta para conhecimento denso
            'frequency_range': [0.05, 15.0],  # Range amplo para diversidade
            'chaotic_r': 3.98,  # Próximo ao caos para máxima complexidade
            'phase_consciousness': 1.047,  # π/3 para conhecimento estruturado
        }

        self.modulator = ConsciousWaveModulator(self.wiki_config)

        # Cache directory
        self.cache_dir = Path('data/Ψcws_cache/wikipedia')
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Wikipedia API endpoints
        self.wiki_api = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.wiki_search_api = "https://en.wikipedia.org/w/api.php"

        # Tópicos Wikipedia pré-definidos para conversão
        self.wikipedia_topics = {
            'science': [
                'Physics', 'Mathematics', 'Chemistry', 'Biology',
                'Quantum_mechanics', 'Artificial_intelligence', 'Computer_science'
            ],
            'philosophy': [
                'Philosophy', 'Consciousness', 'Metaphysics', 'Epistemology',
                'Ethics', 'Logic', 'Philosophy_of_mind'
            ],
            'history': [
                'History', 'Ancient_history', 'World_War_II', 'Renaissance',
                'Industrial_Revolution', 'Scientific_revolution'
            ],
            'mathematics': [
                'Mathematics', 'Algebra', 'Calculus', 'Geometry', 'Number_theory',
                'Topology', 'Mathematical_analysis', 'Complex_analysis'
            ],
            'consciousness': [
                'Consciousness', 'Neuroscience', 'Cognitive_science', 'Psychology',
                'Philosophy_of_mind', 'Artificial_consciousness', 'Qualia'
            ]
        }

        print("🔮 ΨQRH-Native Wikipedia Processor inicializado")
        print(f"📁 Cache: {self.cache_dir}")

    def fetch_wikipedia_article(self, title: str) -> Optional[Dict]:
        """
        Obtém artigo Wikipedia via API REST.

        Args:
            title: Título do artigo Wikipedia

        Returns:
            Dicionário com dados do artigo ou None se erro
        """
        try:
            # Headers para evitar bloqueio
            headers = {
                'User-Agent': 'PsiQRH-Wikipedia-Converter/1.0 (https://github.com/psiqrh; educational@psiqrh.ai)',
                'Accept': 'application/json'
            }

            # Usar API REST para resumo
            url = f"{self.wiki_api}{title}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Obter conteúdo completo via API
                content_url = f"https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': title,
                    'prop': 'extracts',
                    'exintro': False,
                    'explaintext': True,
                    'exsectionformat': 'plain'
                }

                content_response = requests.get(content_url, params=params, headers=headers, timeout=15)
                content_data = content_response.json()

                # Extrair texto completo
                pages = content_data.get('query', {}).get('pages', {})
                full_text = ""
                for page_id, page_data in pages.items():
                    if 'extract' in page_data:
                        full_text = page_data['extract']
                        break

                # Combinar dados
                article = {
                    'title': data.get('title', title),
                    'summary': data.get('extract', ''),
                    'full_text': full_text[:50000],  # Limitar tamanho
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    'categories': data.get('categories', []),
                    'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    'word_count': len(full_text.split()) if full_text else 0
                }

                return article

            else:
                print(f"❌ Erro ao buscar '{title}': HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Erro ao buscar '{title}': {e}")
            return None

    def synthesize_wikipedia_knowledge(self, articles: List[Dict]) -> str:
        """
        Sintetiza conhecimento de múltiplos artigos Wikipedia.

        Args:
            articles: Lista de artigos Wikipedia

        Returns:
            Texto sintetizado representando o conhecimento
        """

        knowledge_text = "ΨQRH Wikipedia Knowledge Synthesis\n"
        knowledge_text += "=" * 50 + "\n\n"

        # Estatísticas gerais
        total_words = sum(article['word_count'] for article in articles)
        knowledge_text += f"Knowledge Base Statistics:\n"
        knowledge_text += f"- Total Articles: {len(articles)}\n"
        knowledge_text += f"- Total Words: {total_words:,}\n"
        knowledge_text += f"- Synthesis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Processar cada artigo
        for i, article in enumerate(articles):
            knowledge_text += f"Article {i+1}: {article['title']}\n"
            knowledge_text += "-" * 30 + "\n"

            # Resumo
            if article['summary']:
                knowledge_text += f"Summary: {article['summary'][:500]}...\n\n"

            # Conteúdo principal (amostra)
            if article['full_text']:
                # Extrair primeiros parágrafos mais significativos
                paragraphs = article['full_text'].split('\n')
                significant_content = []

                for para in paragraphs:
                    if len(para.strip()) > 100:  # Parágrafos substanciais
                        significant_content.append(para.strip())
                        if len(significant_content) >= 3:  # Max 3 parágrafos por artigo
                            break

                knowledge_text += "Key Content:\n"
                for para in significant_content:
                    knowledge_text += f"{para[:300]}...\n\n"

            knowledge_text += f"Word Count: {article['word_count']}\n"
            knowledge_text += f"Source: {article['url']}\n\n"

        # Síntese de padrões de conhecimento
        knowledge_text += "ΨQRH Knowledge Patterns Analysis:\n"
        knowledge_text += "=" * 40 + "\n\n"

        # Extrair conceitos-chave de todos os artigos
        all_words = []
        for article in articles:
            if article['full_text']:
                words = article['full_text'].lower().split()
                all_words.extend([w for w in words if len(w) > 4 and w.isalpha()])

        # Frequência de conceitos
        from collections import Counter
        concept_freq = Counter(all_words)
        top_concepts = concept_freq.most_common(50)

        knowledge_text += "Top Knowledge Concepts:\n"
        for concept, freq in top_concepts[:20]:
            knowledge_text += f"- {concept}: {freq} occurrences\n"

        knowledge_text += "\nThis synthesis represents the collective human knowledge "
        knowledge_text += "from Wikipedia articles, processed through ΨQRH quaternionic "
        knowledge_text += "consciousness framework for fractal analysis and spectral "
        knowledge_text += "decomposition into conscious wave patterns.\n\n"

        knowledge_text += "The embedded consciousness reflects the structured organization "
        knowledge_text += "of human understanding across multiple domains of knowledge, "
        knowledge_text += "with complex semantic relationships and conceptual hierarchies "
        knowledge_text += "that form the foundation of collective human intelligence.\n"

        return knowledge_text

    def convert_topic_to_psicws(self, topic_name: str) -> Optional[str]:
        """
        Converte um tópico Wikipedia específico para .Ψcws.

        Args:
            topic_name: Nome do tópico (ex: 'science', 'philosophy')

        Returns:
            Path do arquivo .Ψcws gerado ou None se erro
        """

        if topic_name not in self.wikipedia_topics:
            print(f"❌ Tópico '{topic_name}' não suportado")
            print(f"Tópicos disponíveis: {list(self.wikipedia_topics.keys())}")
            return None

        articles_titles = self.wikipedia_topics[topic_name]
        print(f"🔄 Processando tópico '{topic_name}' com {len(articles_titles)} artigos...")

        # Buscar todos os artigos
        articles = []
        for title in articles_titles:
            print(f"📄 Buscando: {title}")
            article = self.fetch_wikipedia_article(title)
            if article:
                articles.append(article)
                time.sleep(1)  # Rate limiting
            else:
                print(f"⚠️ Falha ao buscar: {title}")

        if not articles:
            print(f"❌ Nenhum artigo obtido para '{topic_name}'")
            return None

        print(f"✅ Obtidos {len(articles)} artigos para '{topic_name}'")

        # Sintetizar conhecimento
        print("🧠 Sintetizando conhecimento...")
        knowledge_text = self.synthesize_wikipedia_knowledge(articles)

        # Salvar texto temporário
        temp_file = Path(f"temp_wiki_{topic_name}.txt")
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(knowledge_text)

            # Converter para .Ψcws
            print("🌊 Convertendo para formato .Ψcws...")
            Ψcws_file = self.modulator.process_file(temp_file)

            # Adicionar metadados específicos
            Ψcws_file.content_metadata.key_concepts.extend([
                'wikipedia', 'knowledge', 'encyclopedia', topic_name, 'ΨQRH'
            ])

            # Adicionar conceitos dos artigos
            for article in articles:
                Ψcws_file.content_metadata.key_concepts.append(article['title'].lower())

            # Salvar arquivo .Ψcws
            output_file = self.cache_dir / f"wikipedia_{topic_name}.Ψcws"
            Ψcws_file.save(output_file)

            print(f"✅ Arquivo .Ψcws salvo: {output_file}")

            # Análise de consciência
            metrics = Ψcws_file.spectral_data.consciousness_metrics
            print(f"\n🧠 Análise de Consciência Wikipedia:")
            print(f"   Complexity: {metrics['complexity']:.4f}")
            print(f"   Coherence: {metrics['coherence']:.4f}")
            print(f"   Adaptability: {metrics['adaptability']:.4f}")
            print(f"   Integration: {metrics['integration']:.4f}")

            return str(output_file)

        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()

    def convert_custom_articles(self, article_titles: List[str], output_name: str) -> Optional[str]:
        """
        Converte lista customizada de artigos Wikipedia.

        Args:
            article_titles: Lista de títulos de artigos
            output_name: Nome do arquivo de saída

        Returns:
            Path do arquivo .Ψcws gerado
        """

        print(f"🔄 Processando {len(article_titles)} artigos customizados...")

        articles = []
        for title in article_titles:
            print(f"📄 Buscando: {title}")
            article = self.fetch_wikipedia_article(title)
            if article:
                articles.append(article)
                time.sleep(1)

        if not articles:
            print("❌ Nenhum artigo obtido")
            return None

        # Sintetizar e converter
        knowledge_text = self.synthesize_wikipedia_knowledge(articles)

        temp_file = Path(f"temp_wiki_custom.txt")
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(knowledge_text)

            Ψcws_file = self.modulator.process_file(temp_file)

            # Metadados
            Ψcws_file.content_metadata.key_concepts.extend([
                'wikipedia', 'custom', 'knowledge', 'ΨQRH'
            ])

            output_file = self.cache_dir / f"{output_name}.Ψcws"
            Ψcws_file.save(output_file)

            print(f"✅ Arquivo customizado salvo: {output_file}")
            return str(output_file)

        finally:
            if temp_file.exists():
                temp_file.unlink()

    def list_available_topics(self):
        """Lista tópicos Wikipedia disponíveis para conversão."""

        print("📋 Tópicos Wikipedia Disponíveis para Conversão ΨQRH:")
        print("=" * 55)

        for topic, articles in self.wikipedia_topics.items():
            print(f"\n🔸 {topic.upper()}")
            print(f"   Artigos: {len(articles)}")
            print(f"   Exemplos: {', '.join(articles[:3])}")
            if len(articles) > 3:
                print(f"   + {len(articles) - 3} outros...")

    def convert_all_topics(self):
        """Converte todos os tópicos disponíveis."""

        print("🔄 Convertendo TODOS os tópicos Wikipedia para .Ψcws...")
        print("⚠️ Isso fará múltiplas requisições à API da Wikipedia")

        results = []
        for topic_name in self.wikipedia_topics:
            try:
                output_file = self.convert_topic_to_psicws(topic_name)
                results.append({
                    'topic': topic_name,
                    'success': output_file is not None,
                    'file': output_file
                })
                print(f"✅ {topic_name} concluído\n")

            except Exception as e:
                results.append({
                    'topic': topic_name,
                    'success': False,
                    'error': str(e)
                })
                print(f"❌ Erro em {topic_name}: {e}\n")

        # Resumo
        success_count = sum(1 for r in results if r['success'])
        print(f"📊 Conversão Completa: {success_count}/{len(results)} tópicos")

        return results


def main():
    """Função principal do conversor."""

    processor = ΨQRHWikipediaProcessor()

    if len(sys.argv) < 2:
        print("🔮 ΨQRH Wikipedia to .Ψcws Converter")
        print("=" * 40)
        print("Uso:")
        print("  python wiki_to_psicws_converter.py list")
        print("  python wiki_to_psicws_converter.py science")
        print("  python wiki_to_psicws_converter.py philosophy")
        print("  python wiki_to_psicws_converter.py consciousness")
        print("  python wiki_to_psicws_converter.py all")
        return

    command = sys.argv[1].lower()

    if command == "list":
        processor.list_available_topics()

    elif command == "all":
        processor.convert_all_topics()

    elif command in processor.wikipedia_topics:
        processor.convert_topic_to_psicws(command)

    else:
        print(f"❌ Comando/tópico '{command}' não reconhecido")
        print("Use 'list' para ver tópicos disponíveis")


if __name__ == "__main__":
    main()