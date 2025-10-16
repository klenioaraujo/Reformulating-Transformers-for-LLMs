import subprocess
import json

def run_benchmark(task_name, model_type, use_reformulated=False):
    """Executar benchmark para uma tarefa específica"""

    cmd = [
        "python", "custom_glue_runner.py",
        "--data_dir", "./glue_data",
        "--task_name", task_name,
        "--output_dir", f"./outputs/{task_name}_{model_type}",
        "--model_type", model_type,
    ]

    if use_reformulated:
        cmd.extend([
            "--use_reformulated",
            "--reformulated_config", "./reformulated_config.json"
        ])

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Processar resultados
    if result.returncode == 0:
        print(f"✅ {task_name} com {model_type} concluído")
        return parse_results(result.stdout)
    else:
        print(f"❌ Erro em {task_name}: {result.stderr}")
        return None

def parse_results(output):
    """Extrair métricas da saída"""
    # Implementar parsing específico baseado na saída do GLUE
    pass

# Executar para múltiplas tarefas
tasks = ["cola", "sst-2", "mrpc", "sts-b", "qqp", "mnli", "qnli", "rte"]

for task in tasks:
    # Baseline original
    run_benchmark(task, "bert")

    # Modelo reformulado
    run_benchmark(task, "reformulated", use_reformulated=True)