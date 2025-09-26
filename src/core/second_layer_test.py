#!/usr/bin/env python3
"""
Teste de Segunda Camada - ΨQRH Framework
=======================================

Sistema de teste avançado com 10 perguntas de dificuldade crescente
em ordem aleatória usando prompt engine para validação.
"""

import sys
import os
import tempfile
import time
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Inicializar sistema de logging
sys.path.insert(0, str(Path(__file__).parent))
from simple_dependency_logger import SimpleDependencyLogger as DependencyLogger

class SecondLayerTest:
    """
    Sistema de teste de segunda camada com perguntas de dificuldade crescente.
    """

    def __init__(self, log_dir: str = None):
        """Inicializar sistema de teste de segunda camada."""
        if log_dir is None:
            temp_dir = tempfile.mkdtemp()
            log_dir = os.path.join(temp_dir, "psiqrh_second_layer_logs")

        self.log_dir = log_dir
        self.system_logger = DependencyLogger(log_dir=log_dir)
        self.session_id = f"psiqrh_layer2_{int(time.time())}_{hex(id(self))[2:10]}"
        self.test_results = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "tests": [],
            "system_status": "unknown",
            "difficulty_analysis": {},
            "randomization_seed": None
        }

        # Configurar contexto do sistema
        self.system_logger.set_function_context("second_layer_test_init")

        print(f"🎯 TESTE DE SEGUNDA CAMADA - ΨQRH FRAMEWORK")
        print("=" * 60)
        print(f"📝 Session ID: {self.session_id}")
        print(f"📁 Logs: {self.log_dir}")

        # Definir banco de perguntas com dificuldade crescente
        self.question_bank = self._create_question_bank()

    def _create_question_bank(self) -> List[Dict[str, Any]]:
        """
        Criar banco de 10 perguntas com dificuldade crescente (1-10).
        """

        questions = [
            {
                "id": 1,
                "difficulty": 1,
                "question": "What is the primary source of energy for Earth?",
                "expected_keywords": ["sun", "solar", "energy", "light"],
                "category": "basic_science",
                "validation_criteria": {
                    "mentions_sun": True,
                    "basic_explanation": True,
                    "coherent": True
                }
            },
            {
                "id": 2,
                "difficulty": 2,
                "question": "How does the scientific method help us understand the world?",
                "expected_keywords": ["hypothesis", "experiment", "observation", "evidence"],
                "category": "scientific_method",
                "validation_criteria": {
                    "mentions_methodology": True,
                    "explains_process": True,
                    "uses_examples": False
                }
            },
            {
                "id": 3,
                "difficulty": 3,
                "question": "What is the difference between a theory and a hypothesis in science?",
                "expected_keywords": ["theory", "hypothesis", "evidence", "tested", "explanation"],
                "category": "scientific_concepts",
                "validation_criteria": {
                    "distinguishes_both": True,
                    "explains_evidence": True,
                    "accurate_definitions": True
                }
            },
            {
                "id": 4,
                "difficulty": 4,
                "question": "How do cognitive biases affect human decision-making and perception?",
                "expected_keywords": ["bias", "cognition", "decision", "perception", "psychology"],
                "category": "cognitive_science",
                "validation_criteria": {
                    "defines_cognitive_bias": True,
                    "gives_examples": True,
                    "explains_impact": True
                }
            },
            {
                "id": 5,
                "difficulty": 5,
                "question": "Explain the relationship between entropy and information theory in complex systems.",
                "expected_keywords": ["entropy", "information", "complexity", "systems", "disorder"],
                "category": "information_theory",
                "validation_criteria": {
                    "connects_concepts": True,
                    "technical_accuracy": True,
                    "complex_reasoning": True
                }
            },
            {
                "id": 6,
                "difficulty": 6,
                "question": "How does quantum entanglement challenge classical notions of locality and realism?",
                "expected_keywords": ["quantum", "entanglement", "locality", "realism", "physics"],
                "category": "quantum_physics",
                "validation_criteria": {
                    "explains_entanglement": True,
                    "addresses_locality": True,
                    "philosophical_implications": True
                }
            },
            {
                "id": 7,
                "difficulty": 7,
                "question": "What are the implications of Gödel's incompleteness theorems for artificial intelligence and computation?",
                "expected_keywords": ["gödel", "incompleteness", "ai", "computation", "limits"],
                "category": "mathematical_logic",
                "validation_criteria": {
                    "understands_godel": True,
                    "connects_to_ai": True,
                    "discusses_limitations": True
                }
            },
            {
                "id": 8,
                "difficulty": 8,
                "question": "How might consciousness emerge from complex neural networks, and what are the hard problems this raises?",
                "expected_keywords": ["consciousness", "emergence", "neural", "hard_problem", "qualia"],
                "category": "consciousness_studies",
                "validation_criteria": {
                    "addresses_emergence": True,
                    "mentions_hard_problem": True,
                    "philosophical_depth": True
                }
            },
            {
                "id": 9,
                "difficulty": 9,
                "question": "Analyze the relationship between thermodynamic entropy, information entropy, and the arrow of time in cosmological contexts.",
                "expected_keywords": ["thermodynamic", "entropy", "information", "time", "cosmology"],
                "category": "theoretical_physics",
                "validation_criteria": {
                    "multi_entropy_types": True,
                    "cosmological_context": True,
                    "advanced_synthesis": True
                }
            },
            {
                "id": 10,
                "difficulty": 10,
                "question": "What are the fundamental epistemological challenges in bridging subjective conscious experience with objective scientific measurement?",
                "expected_keywords": ["epistemology", "subjective", "objective", "consciousness", "measurement"],
                "category": "philosophy_of_science",
                "validation_criteria": {
                    "epistemological_depth": True,
                    "subjective_objective_bridge": True,
                    "measurement_problem": True
                }
            }
        ]

        return questions

    def generate_test_prompt(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gerar prompt de teste de segunda camada usando prompt engine.
        """

        prompt_engine_config = {
            "context": f"Teste de segunda camada com {len(questions)} perguntas de dificuldade crescente (1-10) em ordem aleatória",
            "analysis": "Sistema deve demonstrar capacidade crescente de raciocínio e conhecimento em diferentes domínios",
            "solution": "Validação através de critérios específicos para cada nível de dificuldade",
            "implementation": [
                "✅ 10 perguntas de dificuldade crescente preparadas",
                "✅ Ordem aleatória para evitar viés de sequência",
                "✅ Critérios de validação específicos por pergunta",
                "✅ Análise de performance por nível de dificuldade",
                "✅ Sistema de logging integrado"
            ],
            "validation": "Performance consistente indica sistema robusto e funcional"
        }

        return {
            "prompt_engine": prompt_engine_config,
            "total_questions": len(questions),
            "difficulty_range": "1-10",
            "randomization": True,
            "timestamp": datetime.now().isoformat()
        }

    def randomize_questions(self, seed: int = None) -> List[Dict[str, Any]]:
        """
        Randomizar ordem das perguntas mantendo rastreabilidade.
        """
        if seed is None:
            seed = int(time.time() * 1000) % 10000

        self.test_results["randomization_seed"] = seed
        random.seed(seed)

        randomized = self.question_bank.copy()
        random.shuffle(randomized)

        print(f"🎲 ORDEM RANDOMIZADA (seed: {seed}):")
        for i, q in enumerate(randomized, 1):
            print(f"   {i}. Pergunta ID {q['id']} (Dificuldade {q['difficulty']})")

        return randomized

    def execute_second_layer_test(self):
        """
        Executar teste completo de segunda camada.
        """

        print(f"\n🧠 EXECUTANDO TESTE DE SEGUNDA CAMADA")
        print("-" * 50)

        # Randomizar perguntas
        randomized_questions = self.randomize_questions()

        # Gerar prompt de teste
        test_prompt = self.generate_test_prompt(randomized_questions)

        print(f"📋 ΨQRH-PROMPT-ENGINE TEST CONFIGURATION:")
        print(f"   Context: {test_prompt['prompt_engine']['context']}")
        print(f"   Total Questions: {test_prompt['total_questions']}")

        # Log da dependência do sistema
        self.system_logger.log_function_dependency("second_layer_validator", {
            "random": "builtin",
            "json": "builtin",
            "datetime": "builtin"
        })

        # Executar cada pergunta
        all_results = []
        difficulty_scores = {}

        for i, question in enumerate(randomized_questions, 1):
            print(f"\n🔍 PERGUNTA {i}/10 (ID: {question['id']}, Dificuldade: {question['difficulty']})")
            print(f"📝 {question['question']}")

            # Simular resposta do sistema
            response = self._simulate_response_for_question(question)

            # Validar resposta
            validation = self._validate_question_response(question, response)

            # Registrar resultado
            result = {
                "question_id": question["id"],
                "sequence_number": i,
                "difficulty": question["difficulty"],
                "category": question["category"],
                "question": question["question"],
                "response": response,
                "validation": validation,
                "timestamp": datetime.now().isoformat()
            }

            all_results.append(result)

            # Agrupar por dificuldade para análise
            diff = question["difficulty"]
            if diff not in difficulty_scores:
                difficulty_scores[diff] = []
            difficulty_scores[diff].append(validation["success_rate"])

            print(f"   Status: {validation['status']}")
            print(f"   Taxa de sucesso: {validation['success_rate']:.1%}")

        self.test_results["tests"] = all_results
        self.test_results["difficulty_analysis"] = self._analyze_difficulty_performance(difficulty_scores)

        return all_results

    def _simulate_response_for_question(self, question: Dict[str, Any]) -> str:
        """
        Gerar resposta dinâmica baseada na análise espectral da pergunta.
        Substitui respostas hardcoded por processamento dinâmico.
        """

        # Importar e usar o Response Spectrum Analyzer
        from response_spectrum_analyzer import DynamicResponseGenerator

        generator = DynamicResponseGenerator()

        # Gerar resposta dinâmica baseada na pergunta
        response = generator.generate_response(question["question"])

        return response

    def _validate_question_response(self, question: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Validar resposta baseada nos critérios específicos da pergunta.
        """

        # Critérios gerais
        general_criteria = {
            "has_content": len(response.strip()) > 50,
            "appropriate_length": 50 <= len(response.split()) <= 200,
            "coherent_response": not any(phrase in response.lower() for phrase in ["i don't know", "no information", "cannot answer"])
        }

        # Critérios específicos da pergunta
        specific_criteria = {}
        expected_keywords = question.get("expected_keywords", [])

        for keyword in expected_keywords:
            specific_criteria[f"mentions_{keyword}"] = keyword.lower() in response.lower()

        # Critérios de validação customizados
        validation_criteria = question.get("validation_criteria", {})
        for criterion, expected in validation_criteria.items():
            # Lógica simplificada de validação baseada em keywords e heurísticas
            if criterion == "mentions_sun" and expected:
                specific_criteria["mentions_sun"] = "sun" in response.lower()
            elif criterion == "explains_process" and expected:
                specific_criteria["explains_process"] = any(word in response.lower() for word in ["method", "process", "steps"])
            elif criterion == "distinguishes_both" and expected:
                specific_criteria["distinguishes_both"] = "hypothesis" in response.lower() and "theory" in response.lower()
            elif criterion == "connects_concepts" and expected:
                specific_criteria["connects_concepts"] = len([kw for kw in expected_keywords if kw in response.lower()]) >= 3
            # Adicionar mais lógica conforme necessário

        # Combinar todos os critérios
        all_criteria = {**general_criteria, **specific_criteria}

        passed_criteria = sum(all_criteria.values())
        total_criteria = len(all_criteria)
        success_rate = passed_criteria / total_criteria if total_criteria > 0 else 0.0

        # Ajustar threshold baseado na dificuldade
        difficulty = question["difficulty"]
        required_threshold = 0.6 + (difficulty * 0.04)  # 0.6 para fácil, até 1.0 para difícil

        is_successful = success_rate >= required_threshold

        return {
            "criteria": all_criteria,
            "passed_criteria": passed_criteria,
            "total_criteria": total_criteria,
            "success_rate": success_rate,
            "required_threshold": required_threshold,
            "is_successful": is_successful,
            "status": "✅ PASSED" if is_successful else "❌ FAILED"
        }

    def _analyze_difficulty_performance(self, difficulty_scores: Dict[int, List[float]]) -> Dict[str, Any]:
        """
        Analisar performance por nível de dificuldade.
        """

        analysis = {}

        for difficulty, scores in difficulty_scores.items():
            avg_score = sum(scores) / len(scores)
            analysis[f"difficulty_{difficulty}"] = {
                "average_score": avg_score,
                "questions_count": len(scores),
                "performance_level": "High" if avg_score >= 0.8 else "Medium" if avg_score >= 0.6 else "Low"
            }

        # Análise geral
        all_scores = [score for scores in difficulty_scores.values() for score in scores]
        overall_average = sum(all_scores) / len(all_scores) if all_scores else 0.0

        analysis["overall"] = {
            "average_score": overall_average,
            "total_questions": len(all_scores),
            "system_performance": "Excellent" if overall_average >= 0.9 else "Good" if overall_average >= 0.7 else "Needs Improvement"
        }

        return analysis

    def generate_comprehensive_report(self) -> str:
        """
        Gerar relatório abrangente usando prompt engine.
        """

        self.test_results["end_time"] = datetime.now().isoformat()

        # Calcular estatísticas gerais
        successful_tests = [t for t in self.test_results["tests"] if t["validation"]["is_successful"]]
        total_tests = len(self.test_results["tests"])
        success_rate = len(successful_tests) / total_tests if total_tests > 0 else 0.0

        system_functional = success_rate >= 0.7  # 70% de sucesso necessário

        self.test_results["system_status"] = "FUNCTIONAL" if system_functional else "NEEDS_IMPROVEMENT"

        # Gerar análise usando prompt engine
        report = f"""

🎯 ANÁLISE FINAL - TESTE DE SEGUNDA CAMADA

ΨQRH-PROMPT-ENGINE: {{
  "context": "Teste de segunda camada com 10 perguntas randomizadas de dificuldade crescente (1-10)",
  "analysis": "Sistema {'PASSOU' if system_functional else 'FALHOU'} no teste de segunda camada com {success_rate:.1%} de sucesso",
  "solution": "{'Sistema demonstra capacidade robusta em múltiplos domínios' if system_functional else 'Sistema requer melhorias em áreas específicas'}",
  "implementation": [
    "✅ 10 perguntas executadas em ordem randomizada (seed: {self.test_results['randomization_seed']})",
    "{'✅' if success_rate >= 0.7 else '❌'} Taxa de sucesso: {success_rate:.1%}",
    "✅ Validação por critérios específicos implementada",
    "✅ Análise de performance por dificuldade realizada",
    "✅ Logging detalhado mantido"
  ],
  "validation": "{'Sistema FUNCIONAL em múltiplas camadas de complexidade' if system_functional else 'Sistema necessita otimizações específicas'}"
}}

📊 ESTATÍSTICAS GERAIS:
Session ID: {self.session_id}
Perguntas executadas: {total_tests}/10
Sucessos: {len(successful_tests)}
Taxa de sucesso geral: {success_rate:.1%}
Status: {'🟢 FUNCIONAL' if system_functional else '🟡 MELHORIAS NECESSÁRIAS'}

🎲 RANDOMIZAÇÃO:
Seed utilizada: {self.test_results['randomization_seed']}

📈 PERFORMANCE POR DIFICULDADE:
"""

        # Adicionar análise por dificuldade
        difficulty_analysis = self.test_results.get("difficulty_analysis", {})
        for key, data in difficulty_analysis.items():
            if key.startswith("difficulty_"):
                diff_level = key.replace("difficulty_", "")
                report += f"""
Dificuldade {diff_level}: {data['average_score']:.1%} ({data['performance_level']})"""

        # Adicionar detalhes dos testes
        report += f"""

📋 DETALHES POR PERGUNTA:
"""

        for test in self.test_results["tests"]:
            report += f"""
Pergunta {test['sequence_number']} (ID:{test['question_id']}, Dificuldade:{test['difficulty']}):
- Categoria: {test['category']}
- Taxa de sucesso: {test['validation']['success_rate']:.1%}
- Status: {test['validation']['status']}
"""

        report += f"""
🔍 ANÁLISE GERAL:
Performance média: {difficulty_analysis.get('overall', {}).get('average_score', 0):.1%}
Classificação: {difficulty_analysis.get('overall', {}).get('system_performance', 'Unknown')}

RECOMENDAÇÕES:
1. {'🎯 Sistema validado para produção' if system_functional else '🔧 Focar em perguntas de maior dificuldade'}
2. {'📊 Manter monitoramento de performance' if system_functional else '📈 Implementar melhorias específicas por categoria'}
3. {'🔄 Expandir testes para novos domínios' if system_functional else '🎯 Re-executar após otimizações'}

CONCLUSÃO: {'✅ SISTEMA FUNCIONAL EM SEGUNDA CAMADA' if system_functional else '⚠️ SISTEMA REQUER OTIMIZAÇÕES'}
"""

        return report

    def save_comprehensive_results(self):
        """
        Salvar resultados completos do teste de segunda camada.
        """

        os.makedirs(self.log_dir, exist_ok=True)

        # Salvar JSON detalhado
        json_path = os.path.join(self.log_dir, f"second_layer_test_{self.session_id}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)

        # Salvar relatório final
        report_path = os.path.join(self.log_dir, f"second_layer_report_{self.session_id}.txt")
        comprehensive_report = self.generate_comprehensive_report()
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(comprehensive_report)

        print(f"\n💾 RESULTADOS SALVOS:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📋 Report: {report_path}")

        return {
            "json_path": json_path,
            "report_path": report_path,
            "session_id": self.session_id
        }


def main():
    """
    Função principal para executar teste de segunda camada.
    """

    print("🚀 INICIANDO TESTE DE SEGUNDA CAMADA - SISTEMA ΨQRH")
    print("=" * 60)

    # Criar instância do teste
    second_layer_test = SecondLayerTest()

    try:
        # Executar teste completo
        results = second_layer_test.execute_second_layer_test()

        # Gerar relatório abrangente
        comprehensive_report = second_layer_test.generate_comprehensive_report()
        print(comprehensive_report)

        # Salvar resultados
        saved_files = second_layer_test.save_comprehensive_results()

        print(f"\n🎉 TESTE DE SEGUNDA CAMADA COMPLETO")
        print(f"Session ID: {second_layer_test.session_id}")

        # Determinar se o sistema passou
        success_rate = len([r for r in results if r["validation"]["is_successful"]]) / len(results)
        return success_rate >= 0.7

    except Exception as e:
        print(f"❌ ERRO DURANTE TESTE: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)